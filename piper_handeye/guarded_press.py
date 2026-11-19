"""Motor-current contact guard for a bounded Piper button press.

This module contains no robot commands.  It turns the Piper 0x251..0x256
high-speed CAN feedback into deterministic decisions that an OrangePi-local
trajectory executor can use to cancel forward motion and retract.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import json
import socket
import time
from typing import Iterable

import numpy as np


# Piper SDK V2 ``ArmMsgFeedbackHighSpd.cal_effort`` coefficients.  The CAN
# current field is converted to output-joint torque with these fixed gains.
PIPER_TORQUE_PER_AMP_NM = np.asarray(
    [1.18125, 1.18125, 1.18125, 0.95844, 0.95844, 0.95844], dtype=float
)


def piper_tcp_linear_jacobian(joints_rad: Iterable[float]) -> np.ndarray:
    """Return the official Piper TCP linear Jacobian without SciPy/ROS deps."""
    joints = np.asarray(joints_rad, dtype=float)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError("joints_rad must contain six finite values")
    origins = (
        ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416)),
        ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939)),
        ((-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),
        ((8.8259e-05, -0.091, 0.0), (1.5708, 0.0, 0.0)),
    )

    def fixed_transform(xyz, rpy):
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        result = np.eye(4)
        result[:3, :3] = rz @ ry @ rx
        result[:3, 3] = xyz
        return result

    transform = np.eye(4)
    axes, joint_origins = [], []
    for (xyz, rpy), angle in zip(origins, joints):
        before = transform @ fixed_transform(xyz, rpy)
        axes.append(before[:3, 2].copy())
        joint_origins.append(before[:3, 3].copy())
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.asarray(
            [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        transform = before @ rotation
    tcp = transform[:3, 3] + transform[:3, :3] @ np.asarray([0.0, 0.0, 0.14544])
    return np.cross(np.asarray(axes), tcp - np.asarray(joint_origins)).T


@dataclass(frozen=True)
class NormalForceEstimate:
    force_n: float
    sigma_n: float
    score: float
    alignment: float
    joint_torque_residual_nm: np.ndarray
    joint_force_signature_m: np.ndarray
    tangential_force_n: float = 0.0
    tangential_force_sigma_n: float = float("inf")
    tangential_force_score: float = 0.0
    total_force_n: float = 0.0


class CartesianForceKalman:
    """Estimate the three translational TCP-force components.

    This observer is an independent slip/contact guard. Unlike the normal-only
    press estimator, it does not discard a load merely because that load is
    poorly aligned with the expected panel normal. The input is still the
    external joint-torque residual supplied by the causal free-motion model;
    this remains an adaptation, not a complete identified dynamics model.
    """

    def __init__(self, *, process_noise_n2_per_sec: float = 80.0,
                 initial_variance_n2: float = 4.0):
        if process_noise_n2_per_sec <= 0 or initial_variance_n2 <= 0:
            raise ValueError("force observer covariance values must be positive")
        self.process_noise = float(process_noise_n2_per_sec)
        self.force_base_n = np.zeros(3, dtype=float)
        self.covariance_n2 = np.eye(3, dtype=float) * float(initial_variance_n2)
        self.last_time: float | None = None

    def update(self, *, joint_torque_residual_nm: Iterable[float],
               joint_torque_noise_nm: Iterable[float],
               joints_rad: Iterable[float], now: float) -> tuple[np.ndarray, np.ndarray]:
        torque = np.asarray(joint_torque_residual_nm, dtype=float)
        noise = np.asarray(joint_torque_noise_nm, dtype=float)
        if torque.shape != (6,) or noise.shape != (6,):
            raise ValueError("Cartesian force observer input dimensions are invalid")
        observation = piper_tcp_linear_jacobian(joints_rad).T
        dt = 0.005 if self.last_time is None else np.clip(
            float(now) - self.last_time, 0.001, 0.05
        )
        self.last_time = float(now)
        prior_covariance = self.covariance_n2 + np.eye(3) * self.process_noise * dt
        measurement_covariance = np.diag(np.square(np.maximum(noise, 1e-4)))
        innovation_covariance = (
            observation @ prior_covariance @ observation.T
            + measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance,
            observation @ prior_covariance,
        ).T
        innovation = torque - observation @ self.force_base_n
        self.force_base_n = self.force_base_n + gain @ innovation
        identity = np.eye(3)
        correction = identity - gain @ observation
        self.covariance_n2 = (
            correction @ prior_covariance @ correction.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance_n2 = 0.5 * (
            self.covariance_n2 + self.covariance_n2.T
        )
        return self.force_base_n.copy(), self.covariance_n2.copy()


class CartesianNormalForceKalman:
    """One-axis Cartesian force estimator adapted from Wahrburg et al.

    The paper augments generalized momentum with Cartesian wrench in a Kalman
    filter.  Piper exposes calibrated motor torque, joint position and speed,
    but this installation has no identified friction model.  Its causal
    free-motion predictor supplies the missing model torque.  The remaining
    external joint torque is related to the known panel-normal force by
    ``tau_ext = J_v(q).T * n * f`` and fused as a scalar Kalman state.
    """

    def __init__(self, *, process_noise_n2_per_sec: float = 80.0,
                 initial_variance_n2: float = 4.0):
        if process_noise_n2_per_sec <= 0 or initial_variance_n2 <= 0:
            raise ValueError("force observer covariance values must be positive")
        self.process_noise = float(process_noise_n2_per_sec)
        self.force_n = 0.0
        self.variance_n2 = float(initial_variance_n2)
        self.last_time: float | None = None

    def update(self, *, signed_current_residual_a: Iterable[float],
               current_noise_a: Iterable[float], joints_rad: Iterable[float],
               contact_normal_base: Iterable[float], now: float) -> NormalForceEstimate:
        residual = np.asarray(signed_current_residual_a, dtype=float)
        noise = np.asarray(current_noise_a, dtype=float)
        normal = np.asarray(contact_normal_base, dtype=float)
        if residual.shape != (6,) or noise.shape != (6,) or normal.shape != (3,):
            raise ValueError("force observer input dimensions are invalid")
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm < 0.9:
            raise ValueError("contact normal must be finite and nonzero")
        normal /= norm
        h = piper_tcp_linear_jacobian(joints_rad).T @ normal
        torque = PIPER_TORQUE_PER_AMP_NM * residual
        torque_sigma = np.maximum(
            PIPER_TORQUE_PER_AMP_NM * noise, 1e-4
        )
        dt = 0.005 if self.last_time is None else np.clip(
            float(now) - self.last_time, 0.001, 0.05
        )
        self.last_time = float(now)
        prior_variance = self.variance_n2 + self.process_noise * float(dt)
        inverse_r = 1.0 / np.square(torque_sigma)
        information = float(np.sum(np.square(h) * inverse_r))
        if information < 1e-9:
            return NormalForceEstimate(
                self.force_n, float(np.sqrt(prior_variance)), 0.0, 0.0,
                torque, h,
            )
        posterior_variance = 1.0 / (1.0 / prior_variance + information)
        posterior_force = posterior_variance * (
            self.force_n / prior_variance + float(np.sum(h * torque * inverse_r))
        )
        whitened_h = h / torque_sigma
        whitened_y = torque / torque_sigma
        denominator = float(np.linalg.norm(whitened_h) * np.linalg.norm(whitened_y))
        alignment = (
            0.0 if denominator < 1e-9
            else abs(float(whitened_h @ whitened_y)) / denominator
        )
        self.force_n = float(posterior_force)
        self.variance_n2 = float(posterior_variance)
        sigma = float(np.sqrt(posterior_variance))
        return NormalForceEstimate(
            self.force_n, sigma, abs(self.force_n) / max(sigma, 1e-9),
            alignment, torque, h,
        )


PIPER_HIGH_SPEED_FIRST_ID = 0x251
PIPER_HIGH_SPEED_LAST_ID = 0x256


def decode_piper_motor_current(can_id: int, payload: bytes) -> tuple[int, float]:
    """Decode one Piper high-speed motor frame.

    Bytes 2..3 are a signed, big-endian motor current in 0.001 A according to
    the Piper V2 protocol.  Returns a zero-based joint index and amperes.
    """

    can_id = int(can_id)
    if not PIPER_HIGH_SPEED_FIRST_ID <= can_id <= PIPER_HIGH_SPEED_LAST_ID:
        raise ValueError(f"not a Piper high-speed motor frame: 0x{can_id:x}")
    if len(payload) != 8:
        raise ValueError("Piper high-speed motor frame must contain 8 bytes")
    raw = int.from_bytes(payload[2:4], byteorder="big", signed=True)
    return can_id - PIPER_HIGH_SPEED_FIRST_ID, raw * 0.001


@dataclass(frozen=True)
class MotorCurrentSample:
    timestamp: float
    currents_a: np.ndarray


class PiperCurrentFrameAssembler:
    """Assemble six individual 200 Hz CAN frames into one coherent sample."""

    def __init__(self, max_skew_sec: float = 0.02):
        if max_skew_sec <= 0:
            raise ValueError("max_skew_sec must be positive")
        self.max_skew_sec = float(max_skew_sec)
        self._values = np.zeros(6, dtype=float)
        self._timestamps = np.full(6, -np.inf, dtype=float)
        self._generation = np.zeros(6, dtype=np.int64)
        self._emitted_generation = np.zeros(6, dtype=np.int64)

    def add(self, can_id: int, payload: bytes, timestamp: float) -> MotorCurrentSample | None:
        index, current_a = decode_piper_motor_current(can_id, payload)
        stamp = float(timestamp)
        if not np.isfinite(stamp):
            raise ValueError("CAN timestamp must be finite")
        self._values[index] = current_a
        self._timestamps[index] = stamp
        self._generation[index] += 1
        if np.any(self._generation <= self._emitted_generation):
            return None
        if float(np.max(self._timestamps) - np.min(self._timestamps)) > self.max_skew_sec:
            return None
        self._emitted_generation[:] = self._generation
        return MotorCurrentSample(float(np.max(self._timestamps)), self._values.copy())


@dataclass(frozen=True)
class RobustCurrentBaseline:
    median_a: np.ndarray
    sigma_a: np.ndarray
    sample_count: int

    @classmethod
    def fit(
        cls,
        samples: Iterable[Iterable[float]],
        *,
        minimum_samples: int = 40,
        noise_floor_a: float = 0.01,
    ) -> "RobustCurrentBaseline":
        values = np.asarray(list(samples), dtype=float)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError("baseline samples must have shape Nx6")
        if len(values) < minimum_samples:
            raise ValueError(f"need at least {minimum_samples} baseline samples")
        if not np.isfinite(values).all():
            raise ValueError("baseline samples must be finite")
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        sigma = np.maximum(1.4826 * mad, float(noise_floor_a))
        return cls(median, sigma, len(values))


@dataclass(frozen=True)
class GuardedPressConfig:
    # These values must be calibrated from this arm at the pressing pose.  The
    # controller intentionally has no guessed production defaults.
    contact_delta_a: np.ndarray
    hard_delta_a: np.ndarray
    absolute_current_limit_a: np.ndarray
    noise_sigma_multiplier: float = 5.0
    hard_noise_sigma_multiplier: float = 8.0
    median_window: int = 5
    persistence_samples: int = 5
    persistence_sec: float | None = None
    dynamic_reference_window_samples: int = 40
    contact_arm_travel_m: float = 0.0
    max_forward_travel_m: float = 0.020
    max_press_time_sec: float = 4.0
    max_sample_age_sec: float = 0.030
    # Zero-based joints whose sustained residual may confirm ordinary contact.
    # Absolute limits always remain active on all joints.
    contact_joint_indices: tuple[int, ...] | None = None
    # Zero-based joints whose *relative* current may trigger an instantaneous
    # hard stop.  Gravity-dominated shoulder/elbow axes can be excluded after
    # calibration; the absolute current limit still protects every joint.
    hard_relative_joint_indices: tuple[int, ...] | None = None
    contact_estimator: str = "joint_threshold"
    normal_force_threshold_n: float = 0.35
    normal_force_total_floor_n: float = 0.0
    normal_force_sigma_multiplier: float = 4.0
    normal_force_min_alignment: float = 0.60
    force_process_noise_n2_per_sec: float = 80.0
    tangential_force_threshold_n: float = 0.60
    tangential_force_sigma_multiplier: float = 4.0
    tangential_force_persistence_sec: float = 0.10
    relative_current_hard_stop_enabled: bool = True
    absolute_current_hard_stop_enabled: bool = True

    def __post_init__(self) -> None:
        contact = np.asarray(self.contact_delta_a, dtype=float)
        hard_delta = np.asarray(self.hard_delta_a, dtype=float)
        absolute = np.asarray(self.absolute_current_limit_a, dtype=float)
        if contact.shape != (6,) or hard_delta.shape != (6,) or absolute.shape != (6,):
            raise ValueError("current thresholds must contain six values")
        if not np.isfinite(contact).all() or np.any(contact <= 0):
            raise ValueError("contact current deltas must be finite and positive")
        if not np.isfinite(absolute).all() or np.any(absolute <= 0):
            raise ValueError("absolute current limits must be finite and positive")
        if not np.isfinite(hard_delta).all() or np.any(hard_delta <= contact):
            raise ValueError("hard current deltas must exceed contact deltas")
        if (
            not np.isfinite(self.noise_sigma_multiplier)
            or not np.isfinite(self.hard_noise_sigma_multiplier)
            or self.noise_sigma_multiplier <= 0
            or self.hard_noise_sigma_multiplier <= self.noise_sigma_multiplier
        ):
            raise ValueError("hard noise multiplier must exceed contact noise multiplier")
        if (
            self.persistence_samples < 2
            or self.median_window < 3
            or self.dynamic_reference_window_samples < self.median_window
        ):
            raise ValueError("contact filtering requires persistence and a median window")
        if (
            self.persistence_sec is not None
            and (not np.isfinite(self.persistence_sec) or self.persistence_sec <= 0)
        ):
            raise ValueError("contact persistence time must be finite and positive")
        if self.max_forward_travel_m <= 0 or self.max_press_time_sec <= 0:
            raise ValueError("press travel and time limits must be positive")
        if not 0 <= self.contact_arm_travel_m < self.max_forward_travel_m:
            raise ValueError("contact arm travel must be below maximum forward travel")
        contact_indices = (
            tuple(range(6))
            if self.contact_joint_indices is None
            else tuple(int(value) for value in self.contact_joint_indices)
        )
        if not contact_indices or len(set(contact_indices)) != len(contact_indices):
            raise ValueError("contact joint indices must be unique and nonempty")
        if any(value < 0 or value >= 6 for value in contact_indices):
            raise ValueError("contact joint indices must be in 0..5")
        hard_relative_indices = (
            tuple(range(6))
            if self.hard_relative_joint_indices is None
            else tuple(int(value) for value in self.hard_relative_joint_indices)
        )
        if (
            not hard_relative_indices
            or len(set(hard_relative_indices)) != len(hard_relative_indices)
        ):
            raise ValueError("hard relative joint indices must be unique and nonempty")
        if any(value < 0 or value >= 6 for value in hard_relative_indices):
            raise ValueError("hard relative joint indices must be in 0..5")
        if self.contact_estimator not in {
            "joint_threshold", "cartesian_normal_force_kalman"
        }:
            raise ValueError("unsupported contact estimator")
        if (
            not np.isfinite(self.normal_force_total_floor_n)
            or self.normal_force_total_floor_n < 0
            or self.normal_force_threshold_n <= 0
            or self.normal_force_sigma_multiplier <= 0
            or not 0.0 <= self.normal_force_min_alignment <= 1.0
            or self.force_process_noise_n2_per_sec <= 0
            or self.tangential_force_threshold_n <= 0
            or self.tangential_force_sigma_multiplier <= 0
            or self.tangential_force_persistence_sec <= 0
        ):
            raise ValueError("Cartesian force estimator settings are invalid")
        if not isinstance(self.relative_current_hard_stop_enabled, bool):
            raise ValueError("relative current hard-stop flag must be boolean")
        if not isinstance(self.absolute_current_hard_stop_enabled, bool):
            raise ValueError("absolute current hard-stop flag must be boolean")
        object.__setattr__(self, "contact_delta_a", contact)
        object.__setattr__(self, "hard_delta_a", hard_delta)
        object.__setattr__(self, "absolute_current_limit_a", absolute)
        object.__setattr__(self, "contact_joint_indices", contact_indices)
        object.__setattr__(self, "hard_relative_joint_indices", hard_relative_indices)


class GuardedPressDecision(str, Enum):
    CONTINUE = "CONTINUE"
    CONTACT = "CONTACT"
    HARD_STOP = "HARD_STOP"


@dataclass(frozen=True)
class GuardedPressResult:
    decision: GuardedPressDecision
    reason: str
    filtered_current_a: np.ndarray
    residual_a: np.ndarray
    contact_ratio: np.ndarray
    persistence: int
    persistence_sec: float = 0.0
    trigger_joint_index: int | None = None
    estimated_normal_force_n: float | None = None
    normal_force_sigma_n: float | None = None
    normal_force_score: float | None = None
    normal_force_alignment: float | None = None
    estimated_tangential_force_n: float | None = None
    tangential_force_sigma_n: float | None = None
    tangential_force_score: float | None = None
    estimated_total_force_n: float | None = None


class GuardedPressMonitor:
    """Detect sustained contact while enforcing independent hard limits."""

    def __init__(self, baseline: RobustCurrentBaseline, config: GuardedPressConfig,
                 *, contact_normal_base: Iterable[float] | None = None):
        self.baseline = baseline
        self.config = config
        self._window: deque[np.ndarray] = deque(maxlen=config.median_window)
        self._free_space_window: deque[np.ndarray] = deque(
            maxlen=config.dynamic_reference_window_samples
        )
        self._persistence = 0
        self._persistence_started_at: float | None = None
        self._tangential_persistence = 0
        self._tangential_persistence_started_at: float | None = None
        self._relative_reference_a = baseline.median_a.copy()
        self._relative_sigma_a = baseline.sigma_a.copy()
        self._was_armed = config.contact_arm_travel_m == 0.0
        self._contact_normal_base = (
            None if contact_normal_base is None
            else np.asarray(contact_normal_base, dtype=float)
        )
        self._force_observer = (
            CartesianNormalForceKalman(
                process_noise_n2_per_sec=config.force_process_noise_n2_per_sec
            )
            if config.contact_estimator == "cartesian_normal_force_kalman"
            else None
        )
        self._cartesian_force_observer = (
            CartesianForceKalman(
                process_noise_n2_per_sec=config.force_process_noise_n2_per_sec
            )
            if config.contact_estimator == "cartesian_normal_force_kalman"
            else None
        )

    def _predict_rolling_reference(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Predict the next free-motion current from the immediately prior window.

        A linear trend removes the normal gravity/acceleration drift that made
        a reference frozen at a depth-derived position fail at other button
        heights.  The current sample is deliberately excluded, so a contact
        change cannot move its own limit before it is tested.
        """
        values = np.asarray(self._free_space_window, dtype=float)
        minimum = max(2 * self.config.median_window, 20)
        if len(values) < minimum:
            return None
        t = np.arange(len(values), dtype=float)
        centered = t - float(np.mean(t))
        denominator = float(centered @ centered)
        slope = (centered[:, None] * values).sum(axis=0) / denominator
        intercept = np.mean(values, axis=0) - slope * float(np.mean(t))
        predicted = intercept + slope * float(len(values))
        fitted = intercept[None, :] + t[:, None] * slope[None, :]
        errors = values - fitted
        center = np.median(errors, axis=0)
        mad = np.median(np.abs(errors - center), axis=0)
        sigma = np.maximum(1.4826 * mad, 0.005)
        return predicted, sigma

    @property
    def relative_reference_a(self) -> np.ndarray:
        """Latest causal prediction of free-motion motor current."""
        return self._relative_reference_a.copy()

    @property
    def relative_sigma_a(self) -> np.ndarray:
        """Robust residual noise of the latest rolling free-motion fit."""
        return self._relative_sigma_a.copy()

    def update(
        self,
        sample: MotorCurrentSample,
        *,
        now: float,
        elapsed_sec: float,
        forward_travel_m: float,
        joints_rad: Iterable[float] | None = None,
    ) -> GuardedPressResult:
        currents = np.asarray(sample.currents_a, dtype=float)
        if currents.shape != (6,) or not np.isfinite(currents).all():
            return self._hard_stop("invalid_current_sample", currents)
        if float(now) - float(sample.timestamp) > self.config.max_sample_age_sec:
            return self._hard_stop("stale_current_sample", currents)
        if elapsed_sec > self.config.max_press_time_sec:
            return self._hard_stop("press_timeout", currents)
        if forward_travel_m > self.config.max_forward_travel_m:
            return self._hard_stop("maximum_forward_travel", currents)
        if (
            self.config.absolute_current_hard_stop_enabled
            and np.any(np.abs(currents) >= self.config.absolute_current_limit_a)
        ):
            return self._hard_stop("absolute_current_limit", currents)

        self._window.append(currents.copy())
        filtered = np.median(np.asarray(self._window), axis=0)
        armed = forward_travel_m >= self.config.contact_arm_travel_m
        rolling = self._predict_rolling_reference()
        reference_ready = rolling is not None or self.config.contact_arm_travel_m == 0.0
        if rolling is not None:
            self._relative_reference_a, self._relative_sigma_a = rolling
        elif not armed:
            # Warm-up follows the latest filtered free-motion current, but is
            # never allowed to trigger until the causal rolling fit is ready.
            self._relative_reference_a = filtered.copy()
        signed_residual = filtered - self._relative_reference_a
        residual = np.abs(signed_residual)
        # The baseline is sampled while stationary.  During the free-space
        # acceleration phase, normal inertial current can exceed a relative
        # limit derived from that stationary baseline.  Keep the absolute
        # motor-current limit active for the entire move, but arm all
        # baseline-relative decisions only near the expected panel surface.
        # Once the causal window is ready, its immediately pre-impact noise is
        # the limit source.  Retaining the older stationary/pre-motion sigma
        # would make a noisy pose permanently desensitize contact detection.
        noise = (
            self._relative_sigma_a
            if reference_ready and rolling is not None
            else self.baseline.sigma_a
        )
        hard_threshold = np.maximum(
            self.config.hard_delta_a,
            self.config.hard_noise_sigma_multiplier * noise,
        )
        threshold = np.maximum(
            self.config.contact_delta_a,
            self.config.noise_sigma_multiplier * noise,
        )
        ratio = residual / threshold
        hard_indices = list(self.config.hard_relative_joint_indices)
        hard_exceeded = residual[hard_indices] >= hard_threshold[hard_indices]
        if (
            self.config.relative_current_hard_stop_enabled
            and armed and np.any(hard_exceeded)
        ):
            hard_ratios = residual[hard_indices] / hard_threshold[hard_indices]
            trigger_joint = hard_indices[int(np.argmax(hard_ratios))]
            # Preserve the local moving-pose residual and finite contact ratio
            # in the audit.  Recomputing against the stationary baseline made
            # previous hard stops appear as ``inf`` and hid the triggering axis.
            return self._hard_stop(
                "relative_current_limit", filtered,
                residual=residual, contact_ratio=ratio,
                trigger_joint_index=trigger_joint,
            )
        force = None
        if self._force_observer is not None:
            if joints_rad is None or self._contact_normal_base is None:
                return self._hard_stop(
                    "cartesian_force_observer_input_missing", filtered,
                    residual=residual, contact_ratio=ratio,
                )
            if armed:
                # Do not integrate a force state before the geometrically safe
                # contact window.  Doing so allowed ordinary acceleration load
                # to freeze the free-motion predictor several millimetres too
                # early, then become an immediate false contact at the arm
                # boundary.  The last pre-arm samples now remain a causal model
                # of the exact trajectory load immediately before impact.
                observer_noise = np.maximum(
                    noise, self.config.contact_delta_a /
                    self.config.noise_sigma_multiplier
                )
                force = self._force_observer.update(
                    signed_current_residual_a=signed_residual,
                    current_noise_a=observer_noise,
                    joints_rad=joints_rad,
                    contact_normal_base=self._contact_normal_base,
                    now=now,
                )
                force_base, force_covariance = self._cartesian_force_observer.update(
                    joint_torque_residual_nm=force.joint_torque_residual_nm,
                    joint_torque_noise_nm=PIPER_TORQUE_PER_AMP_NM * observer_noise,
                    joints_rad=joints_rad,
                    now=now,
                )
                normal = self._contact_normal_base / np.linalg.norm(
                    self._contact_normal_base
                )
                normal_projection = float(normal @ force_base)
                tangential_vector = force_base - normal * normal_projection
                tangential_force = float(np.linalg.norm(tangential_vector))
                total_force = float(np.linalg.norm(force_base))
                if tangential_force > 1e-9:
                    tangent_direction = tangential_vector / tangential_force
                    tangential_variance = float(
                        tangent_direction @ force_covariance @ tangent_direction
                    )
                else:
                    tangential_variance = float(np.trace(force_covariance) / 2.0)
                tangential_sigma = float(np.sqrt(max(tangential_variance, 1e-12)))
                tangential_score = tangential_force / tangential_sigma
                force = NormalForceEstimate(
                    force.force_n,
                    force.sigma_n,
                    force.score,
                    force.alignment,
                    force.joint_torque_residual_nm,
                    force.joint_force_signature_m,
                    tangential_force,
                    tangential_sigma,
                    tangential_score,
                    total_force,
                )
                contact_candidate = (
                    # Contact with the panel opposes the commanded inward
                    # motion and has a negative sign under the configured
                    # outward panel normal.  Using abs() allowed an opposite-
                    # direction inertial/friction residual to stop button 9
                    # before the cap touched the surface.
                    force.force_n <= -self.config.normal_force_threshold_n
                    and force.score >= self.config.normal_force_sigma_multiplier
                    and force.alignment >= self.config.normal_force_min_alignment
                    # The scalar observer can inflate friction into normal load.
                    # Require independent 3-D force support throughout the hold.
                    and force.total_force_n >= self.config.normal_force_total_floor_n
                )
            else:
                contact_candidate = False
        else:
            contact_ratio = ratio[list(self.config.contact_joint_indices)]
            contact_candidate = np.max(contact_ratio) >= 1.0
        candidate = (
            armed
            and reference_ready
            and len(self._window) == self.config.median_window
            and contact_candidate
        )
        if candidate:
            if self._persistence_started_at is None:
                self._persistence_started_at = float(now)
            self._persistence += 1
        else:
            self._persistence = 0
            self._persistence_started_at = None
        tangential_candidate = bool(
            armed
            and reference_ready
            and force is not None
            and force.tangential_force_n >= self.config.tangential_force_threshold_n
            and force.tangential_force_score
            >= self.config.tangential_force_sigma_multiplier
        )
        if tangential_candidate:
            if self._tangential_persistence_started_at is None:
                self._tangential_persistence_started_at = float(now)
            self._tangential_persistence += 1
        else:
            self._tangential_persistence = 0
            self._tangential_persistence_started_at = None
        tangential_persistence_sec = (
            max(0.0, float(now) - self._tangential_persistence_started_at)
            if self._tangential_persistence_started_at is not None else 0.0
        )
        persistence_sec = (
            max(0.0, float(now) - self._persistence_started_at)
            if self._persistence_started_at is not None else 0.0
        )
        learn_free_motion = (
            not armed
            or not reference_ready
            or (self._force_observer is None and not candidate)
        )
        if learn_free_motion and len(self._window) == self.config.median_window:
            # The Cartesian observer freezes its model at the arming boundary:
            # a compliant button can build force too slowly to trip a
            # sample-wise suspicion gate and would otherwise be learned away.
            # The legacy joint-threshold detector retains adaptive behavior.
            self._free_space_window.append(filtered.copy())
        self._was_armed = self._was_armed or armed
        normal_contact_confirmed = (
                self._persistence >= self.config.persistence_samples
                and (
                    self.config.persistence_sec is None
                    or persistence_sec >= self.config.persistence_sec
                )
        )
        tangential_contact_confirmed = (
            self._tangential_persistence >= self.config.persistence_samples
            and tangential_persistence_sec
            >= self.config.tangential_force_persistence_sec
        )
        decision = (
            GuardedPressDecision.CONTACT
            if normal_contact_confirmed or tangential_contact_confirmed
            else GuardedPressDecision.CONTINUE
        )
        if decision is GuardedPressDecision.CONTACT:
            reason = (
                "sustained_cartesian_tangential_force_kalman"
                if tangential_contact_confirmed
                else (
                    "sustained_cartesian_normal_force_kalman"
                    if force is not None else "sustained_rolling_current_limit"
                )
            )
        elif not armed:
            reason = "contact_detector_not_yet_armed"
        elif not reference_ready:
            reason = "rolling_reference_warming_up"
        else:
            reason = "below_contact_threshold"
        trigger_joint = None
        if decision is GuardedPressDecision.CONTACT and force is None:
            selected = list(self.config.contact_joint_indices)
            trigger_joint = selected[int(np.argmax(residual[selected]))]
        return GuardedPressResult(
            decision, reason, filtered, residual, ratio, self._persistence,
            persistence_sec, trigger_joint,
            None if force is None else force.force_n,
            None if force is None else force.sigma_n,
            None if force is None else force.score,
            None if force is None else force.alignment,
            None if force is None else force.tangential_force_n,
            None if force is None else force.tangential_force_sigma_n,
            None if force is None else force.tangential_force_score,
            None if force is None else force.total_force_n,
        )

    def _hard_stop(
        self, reason: str, currents: np.ndarray, *,
        residual: np.ndarray | None = None,
        contact_ratio: np.ndarray | None = None,
        trigger_joint_index: int | None = None,
    ) -> GuardedPressResult:
        values = np.asarray(currents, dtype=float)
        safe_values = values if values.shape == (6,) else np.full(6, np.nan)
        residual_values = (
            np.asarray(residual, dtype=float)
            if residual is not None
            else np.abs(safe_values - self.baseline.median_a)
        )
        ratio_values = (
            np.asarray(contact_ratio, dtype=float)
            if contact_ratio is not None
            else np.full(6, np.inf)
        )
        return GuardedPressResult(
            GuardedPressDecision.HARD_STOP,
            reason,
            safe_values.copy(),
            residual_values.copy(),
            ratio_values.copy(),
            self._persistence,
            0.0,
            trigger_joint_index,
        )


def generate_bounded_retract_trajectory(
    current_joints: Iterable[float],
    prepress_joints: Iterable[float],
    *,
    max_velocities: Iterable[float],
    max_accelerations: Iterable[float],
    point_period_sec: float = 0.02,
) -> list[dict]:
    """Generate a zero-boundary quintic retreat along the press joint line."""

    q0 = np.asarray(current_joints, dtype=float)
    q1 = np.asarray(prepress_joints, dtype=float)
    vmax = np.asarray(max_velocities, dtype=float)
    amax = np.asarray(max_accelerations, dtype=float)
    if any(value.shape != (6,) for value in (q0, q1, vmax, amax)):
        raise ValueError("retract trajectory inputs must contain six values")
    if not all(np.isfinite(value).all() for value in (q0, q1, vmax, amax)):
        raise ValueError("retract trajectory inputs must be finite")
    if np.any(vmax <= 0) or np.any(amax <= 0) or point_period_sec <= 0:
        raise ValueError("retract limits and period must be positive")
    delta = q1 - q0
    # For s(u)=10u^3-15u^4+6u^5, max(ds/du)=1.875 and
    # max(abs(d2s/du2)) is below 5.78.
    duration = max(
        0.30,
        float(np.max(1.875 * np.abs(delta) / vmax)),
        float(np.sqrt(np.max(5.78 * np.abs(delta) / amax))),
    ) * 1.05
    count = max(2, int(np.ceil(duration / point_period_sec)) + 1)
    stamps = np.linspace(0.0, duration, count)
    u = stamps / duration
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    sd = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
    sdd = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
    return [
        {
            "time_from_start_sec": float(t),
            "positions": (q0 + fraction * delta).tolist(),
            "velocities": (velocity * delta).tolist(),
            "accelerations": (acceleration * delta).tolist(),
        }
        for t, fraction, velocity, acceleration in zip(stamps, s, sd, sdd)
    ]


class GuardedPressFailure(RuntimeError):
    """Keep authenticated bridge diagnostics separate from formatted error text."""
    def __init__(self, response):
        self.response = response
        super().__init__(f"guarded press bridge failed: {response}")


class GuardedPressUdpClient:
    def __init__(self, host: str, port: int = 5014, timeout_sec: float = 15.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)

    def _request(self, payload: dict) -> dict:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        if len(encoded) > 65507:
            raise ValueError("guarded press UDP payload is too large")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(self.timeout_sec)
            sock.sendto(encoded, (self.host, self.port))
            data, address = sock.recvfrom(65535)
        finally:
            sock.close()
        if address[0] != self.host:
            raise RuntimeError("guarded press response came from unexpected host")
        response = json.loads(data.decode())
        if response.get("task_id") != payload["task_id"]:
            raise RuntimeError("guarded press response task_id mismatch")
        if response.get("status") in ("failed", "busy"):
            raise GuardedPressFailure(response)
        return response

    def health(self, *, token: str) -> dict:
        task_id = f"guarded-health-{time.time_ns()}"
        response = self._request({
            "protocol": "piper_guarded_press_v1", "command": "health",
            "task_id": task_id, "token": str(token),
        })
        if response.get("status") != "ok" or not response.get("current_feedback_available"):
            raise RuntimeError(f"guarded press bridge is not ready: {response}")
        return response

    def execute(self, points: list[dict], *, token: str,
                max_forward_travel_m: float, current_guard: dict,
                contact_normal_base: Iterable[float] | None = None,
                contact_panel_origin_base: Iterable[float] | None = None,
                retract_target_joints: Iterable[float] | None = None,
                retreat_to_observation: list[dict] | None = None,
                dry_run: bool = False, task_id: str | None = None) -> dict:
        payload = {
            "protocol": "piper_guarded_press_v1",
            "command": "validate" if dry_run else "execute",
            "task_id": task_id or f"guarded-{time.time_ns()}",
            "token": str(token),
            "forward_points": points,
            "max_forward_travel_m": float(max_forward_travel_m),
            "current_guard": current_guard,
        }
        if contact_normal_base is not None:
            payload["contact_normal_base"] = list(contact_normal_base)
        if contact_panel_origin_base is not None:
            payload["contact_panel_origin_base"] = list(contact_panel_origin_base)
        if retract_target_joints is not None:
            payload["retract_target_joints"] = list(retract_target_joints)
        if retreat_to_observation is not None:
            payload["retreat_to_observation"] = retreat_to_observation
        response = self._request(payload)
        allowed = {"validated"} if dry_run else {
            "contact_retracted", "limit_retracted", "maximum_travel_retracted"
        }
        if response.get("status") not in allowed:
            raise RuntimeError(f"unexpected guarded press status: {response}")
        return response
