from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
import math
import threading
import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .geometry import average_transforms, transform
from .kinematics import (
    forward_kinematics, inverse_kinematics, forward_kinematics_with_jacobian,
    rotation_log_left_jacobian_inverse,
)
from .robot import PIPER_JOINT_LIMITS
from .pipeline_timing import traced


# Total joint change over the short press stroke, shared by the approach
# previews and the final press solve. Absolute limits and the 0.25-rad
# continuation-step bound remain independent.
PRESS_SEGMENT_MAX_JOINT_DELTA_RAD = 0.26


@dataclass(frozen=True)
class TimedButtonTransform:
    sequence: int
    capture_time_unix_sec: float
    base_to_button: np.ndarray
    confidence: float
    plane_rms_m: float


@dataclass(frozen=True)
class WindowEstimate:
    stable: bool
    reasons: tuple[str, ...]
    base_to_button: np.ndarray | None
    sequence_range: tuple[int, int] | None
    sample_count: int
    latest_age_sec: float | None
    span_sec: float | None
    position_rms_m: float | None
    normal_rms_deg: float | None
    max_position_jump_m: float | None


class SlidingButtonWindow:
    """Quality-gated, sequence-deduplicated window of historical-TF poses."""

    def __init__(
        self,
        *,
        size: int = 6,
        min_confidence: float = 0.70,
        max_input_age_sec: float = 0.15,
        max_latest_age_sec: float = 0.15,
        max_span_sec: float = 0.60,
        max_plane_rms_m: float = 0.006,
        max_position_rms_m: float = 0.001,
        max_normal_rms_deg: float = 1.0,
        max_position_jump_m: float = 0.002,
    ) -> None:
        if size < 2:
            raise ValueError("sliding window requires at least two samples")
        self.size = int(size)
        self.min_confidence = float(min_confidence)
        self.max_input_age_sec = float(max_input_age_sec)
        self.max_latest_age_sec = float(max_latest_age_sec)
        self.max_span_sec = float(max_span_sec)
        self.max_plane_rms_m = float(max_plane_rms_m)
        self.max_position_rms_m = float(max_position_rms_m)
        self.max_normal_rms_deg = float(max_normal_rms_deg)
        self.max_position_jump_m = float(max_position_jump_m)
        self._samples: deque[TimedButtonTransform] = deque(maxlen=self.size)
        self._last_sequence = 0

    def add(self, sample: TimedButtonTransform, *, now: float | None = None) -> bool:
        if sample.sequence <= self._last_sequence:
            return False
        matrix = np.asarray(sample.base_to_button, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("base_to_button must be a finite 4x4 transform")
        current = time.time() if now is None else float(now)
        age = current - float(sample.capture_time_unix_sec)
        if age < -0.05 or age > self.max_input_age_sec:
            return False
        if float(sample.confidence) < self.min_confidence:
            return False
        if float(sample.plane_rms_m) > self.max_plane_rms_m:
            return False
        self._samples.append(
            TimedButtonTransform(
                sequence=int(sample.sequence),
                capture_time_unix_sec=float(sample.capture_time_unix_sec),
                base_to_button=matrix.copy(),
                confidence=float(sample.confidence),
                plane_rms_m=float(sample.plane_rms_m),
            )
        )
        self._last_sequence = int(sample.sequence)
        return True

    def estimate(self, *, now: float | None = None) -> WindowEstimate:
        current = time.time() if now is None else float(now)
        if not self._samples:
            return WindowEstimate(
                False, ("empty",), None, None, 0, None, None, None, None, None
            )
        samples = tuple(self._samples)
        positions = np.asarray([value.base_to_button[:3, 3] for value in samples])
        center = positions.mean(axis=0)
        position_rms = float(
            np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1)))
        )
        jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        max_jump = float(jumps.max()) if len(jumps) else 0.0
        averaged = average_transforms([value.base_to_button for value in samples])
        mean_normal = averaged[:3, 2]
        angles = [
            np.arccos(
                np.clip(value.base_to_button[:3, 2] @ mean_normal, -1.0, 1.0)
            )
            for value in samples
        ]
        normal_rms = float(np.rad2deg(np.sqrt(np.mean(np.square(angles)))))
        latest_age = current - samples[-1].capture_time_unix_sec
        span = samples[-1].capture_time_unix_sec - samples[0].capture_time_unix_sec
        reasons: list[str] = []
        if len(samples) < self.size:
            reasons.append(f"window {len(samples)}/{self.size}")
        if latest_age > self.max_latest_age_sec or latest_age < -0.05:
            reasons.append(f"latest age {latest_age:.3f}s")
        if span > self.max_span_sec:
            reasons.append(f"window span {span:.3f}s")
        if position_rms > self.max_position_rms_m:
            reasons.append(f"position RMS {position_rms * 1000:.2f}mm")
        if normal_rms > self.max_normal_rms_deg:
            reasons.append(f"normal RMS {normal_rms:.2f}deg")
        if max_jump > self.max_position_jump_m:
            reasons.append(f"max jump {max_jump * 1000:.2f}mm")
        return WindowEstimate(
            stable=not reasons,
            reasons=tuple(reasons),
            base_to_button=averaged,
            sequence_range=(samples[0].sequence, samples[-1].sequence),
            sample_count=len(samples),
            latest_age_sec=float(latest_age),
            span_sec=float(span),
            position_rms_m=position_rms,
            normal_rms_deg=normal_rms,
            max_position_jump_m=max_jump,
        )


@dataclass(frozen=True)
class PreviewIkSolution:
    desired_base_to_tcp: np.ndarray
    joints: np.ndarray
    steps: int
    max_step_rad: float


@dataclass(frozen=True)
class FastIkResult:
    joints: np.ndarray
    desired_base_to_tcp: np.ndarray
    solve_sec: float
    position_error_m: float
    orientation_error_deg: float
    max_joint_delta_rad: float
    continuation_steps: int
    max_continuation_joint_step_rad: float
    orientation_offset_deg: tuple[float, float, float] | None = None
    projected_orientation_offset_deg: tuple[float, float, float] | None = None
    direct_candidate_count: int = 0
    continuation_attempt_count: int = 0
    downstream_preview_distances_m: tuple[float, ...] = ()
    downstream_preview_max_joint_step_rad: float = 0.0
    downstream_preview_max_segment_joint_delta_rad: float = 0.0
    downstream_preview_min_joint_limit_margin_rad: float | None = None
    downstream_preview_solutions: tuple[PreviewIkSolution, ...] = ()
    reused_solution: bool = False


class FastIkTimeoutError(RuntimeError):
    """Raised when a live IK search exceeds its real-time budget."""


class _FlexiblePoseObjective:
    """Shared residual/Jacobian evaluation for six joints and XYZ offsets."""

    def __init__(self, orientation_base, target_position, seed, projected):
        self.orientation_base = orientation_base
        self.target_position = target_position
        self.seed = seed
        self.projected = projected
        self.cached_value = None

    def evaluate(self, value):
        if self.cached_value is not None and np.array_equal(value, self.cached_value):
            return self.cached_result
        joints, offsets = value[:6], value[6:]
        actual, geometric = forward_kinematics_with_jacobian(joints)
        desired = self.orientation_base @ Rotation.from_euler("xyz", offsets).as_matrix()
        rotvec = Rotation.from_matrix(desired.T @ actual[:3, :3]).as_rotvec()
        residual = np.concatenate((
            (actual[:3, 3] - self.target_position) / .01,
            rotvec / np.deg2rad(2.), .02*(joints-self.seed),
            .002*(offsets-self.projected),
        ))
        # Extrinsic XYZ equals Rz(z) Ry(y) Rx(x). These are its angular
        # derivative axes, before applying the live panel/base orientation.
        cy, sy = np.cos(offsets[1]), np.sin(offsets[1])
        cz, sz = np.cos(offsets[2]), np.sin(offsets[2])
        offset_axes = self.orientation_base @ np.array([
            [cz*cy, -sz, 0.], [sz*cy, cz, 0.], [-sy, 0., 1.],
        ])
        jac = np.zeros((15, 9))
        jac[:3, :6] = geometric[:3] / .01
        jac[3:6] = (rotation_log_left_jacobian_inverse(rotvec) @ desired.T
                     @ np.hstack((geometric[3:], -offset_axes)) / np.deg2rad(2.))
        jac[6:12, :6] = .02*np.eye(6)
        jac[12:, 6:] = .002*np.eye(3)
        self.cached_value = value.copy()
        self.cached_result = residual, jac
        return self.cached_result

    def __call__(self, value):
        return self.evaluate(value)[0]

    def jacobian(self, value):
        return self.evaluate(value)[1]


class FixedRollFastIkPlanner:
    """One warm-started IK solve using the roll selected by coarse planning."""

    def __init__(
        self,
        *,
        roll_deg: float,
        stand_off_m: float = 0.0,
        max_joint_delta_rad: float = 0.80,
        max_perpendicular_deviation_deg: float = 0.0,
        workspace_bounds_m: Any | None = None,
        downstream_base_x_distances_m: Any = (),
        downstream_max_segment_joint_deltas_rad: Any = (),
        max_solve_sec: float = 3.0,
    ) -> None:
        self.roll_deg = float(roll_deg)
        self.stand_off_m = float(stand_off_m)
        self.max_joint_delta_rad = float(max_joint_delta_rad)
        self.max_perpendicular_deviation_deg = float(
            max_perpendicular_deviation_deg
        )
        self.workspace_bounds_m = (
            None
            if workspace_bounds_m is None
            else np.asarray(workspace_bounds_m, dtype=float)
        )
        self.downstream_base_x_distances_m = tuple(
            float(value) for value in downstream_base_x_distances_m
        )
        self.downstream_max_segment_joint_deltas_rad = tuple(
            None if value is None else float(value)
            for value in downstream_max_segment_joint_deltas_rad
        )
        self.max_solve_sec = float(max_solve_sec)
        if not 0.0 <= self.max_perpendicular_deviation_deg <= 35.0:
            raise ValueError("perpendicular deviation must be in 0..35 degrees")
        if self.workspace_bounds_m is not None:
            if self.workspace_bounds_m.shape != (3, 2):
                raise ValueError("workspace bounds must have shape 3x2")
            if (
                not np.isfinite(self.workspace_bounds_m).all()
                or np.any(
                    self.workspace_bounds_m[:, 0]
                    >= self.workspace_bounds_m[:, 1]
                )
            ):
                raise ValueError("workspace bounds must be finite min/max pairs")
        if (
            not np.isfinite(self.downstream_base_x_distances_m).all()
            or any(
                not -0.03 <= value <= 0.11
                for value in self.downstream_base_x_distances_m
            )
        ):
            raise ValueError(
                "downstream base-X distances must be finite values in -0.03..0.11 m"
            )
        if not np.isfinite(self.max_solve_sec) or self.max_solve_sec <= 0.0:
            raise ValueError("max IK solve time must be finite and positive")
        if self.downstream_max_segment_joint_deltas_rad and (
            len(self.downstream_max_segment_joint_deltas_rad)
            != len(self.downstream_base_x_distances_m)
        ):
            raise ValueError(
                "downstream segment joint-delta limits must match distances"
            )
        if any(
            value is not None and (not np.isfinite(value) or value <= 0.0)
            for value in self.downstream_max_segment_joint_deltas_rad
        ):
            raise ValueError(
                "downstream segment joint-delta limits must be positive"
            )

    @traced('ik.fixed_roll_solve', 'ik')
    def solve(self, base_to_button: Any, seed_joints: Any) -> FastIkResult:
        started = time.perf_counter()
        deadline = started + self.max_solve_sec

        def check_deadline(phase: str) -> None:
            now = time.perf_counter()
            elapsed = now - started
            if now >= deadline:
                raise FastIkTimeoutError(
                    "IK search timed out after "
                    f"{elapsed:.3f}s (limit {self.max_solve_sec:.3f}s, "
                    f"phase {phase}); target is not reachable quickly from "
                    "the current chassis/arm pose"
                )

        button = np.asarray(base_to_button, dtype=float)
        seed = np.asarray(seed_joints, dtype=float)
        if button.shape != (4, 4) or seed.shape != (6,):
            raise ValueError("expected a 4x4 button pose and six seed joints")
        base_rotation = (
            Rotation.from_euler("z", np.deg2rad(self.roll_deg)).as_matrix()
            @ Rotation.from_euler("x", np.pi).as_matrix()
        )
        limit = self.max_perpendicular_deviation_deg
        if limit == 0.0:
            values = (0.0,)
        else:
            # Prefer 0/10/20 degrees and use the requested outer limit only
            # when the smaller perpendicular deviations are unreachable.
            values = tuple(sorted({
                value
                for value in (-limit, -20.0, -10.0, 0.0, 10.0, 20.0, limit)
                if abs(value) <= limit
            }))
        exhaustive_offsets = sorted(
            product(values, repeat=3),
            key=lambda item: (sum(value * value for value in item), item),
        )
        # The mobile base can stop at a different pose on every visit. Derive
        # the first orientation from the live panel pose and current joints,
        # then project it into the allowed per-axis perpendicular cone. This
        # minimizes wrist motion without relying on any saved button/joint
        # profile. The deterministic exhaustive set remains the fallback.
        current_tcp_rotation = forward_kinematics(seed)[:3, :3]
        current_button_relative = button[:3, :3].T @ current_tcp_rotation
        current_offset_deg = Rotation.from_matrix(
            base_rotation.T @ current_button_relative
        ).as_euler("xyz", degrees=True)
        projected_offset = tuple(float(value) for value in np.clip(
            current_offset_deg, -limit, limit
        ))
        flexible_offset = None
        flexible_endpoint_joints = None
        if limit > 0.0:
            check_deadline("flexible_setup")
            # Jointly solve the endpoint joints and the three bounded
            # perpendicular-orientation offsets.  The discrete lattice below
            # remains the validated fallback, but this continuous proposal is
            # insensitive to where a valid lattice point happens to appear in
            # enumeration order when a mobile base changes X/Y/Z.
            if self.downstream_base_x_distances_m:
                # Choose the orientation against the hardest (closest/most
                # forward) endpoint, not merely the distant guard.  Otherwise
                # a comfortable guard orientation can be unable to complete
                # the later prepress/press segment.
                target_position = button[:3, 3].copy()
                target_position[0] -= min(
                    self.downstream_base_x_distances_m
                )
            else:
                target_position = (
                    button
                    @ np.asarray([0.0, 0.0, self.stand_off_m, 1.0])
                )[:3]
            joint_lower = PIPER_JOINT_LIMITS[:, 0] + 1e-5
            joint_upper = PIPER_JOINT_LIMITS[:, 1] - 1e-5
            offset_lower = np.full(3, np.deg2rad(-limit))
            offset_upper = np.full(3, np.deg2rad(limit))
            lower = np.concatenate((joint_lower, offset_lower))
            upper = np.concatenate((joint_upper, offset_upper))
            initial = np.clip(
                np.concatenate((seed, np.deg2rad(projected_offset))),
                lower,
                upper,
            )

            flexible_residual = _FlexiblePoseObjective(
                button[:3, :3] @ base_rotation, target_position, seed,
                np.deg2rad(projected_offset),
            )

            flexible_solution = least_squares(
                flexible_residual,
                initial,
                bounds=(lower, upper),
                jac=flexible_residual.jacobian,
                max_nfev=180,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
                x_scale="jac",
            )
            check_deadline("flexible_endpoint")
            flexible_offset = tuple(
                float(value) for value in np.rad2deg(flexible_solution.x[6:])
            )
            flexible_endpoint_joints = flexible_solution.x[:6].copy()
        half_projected_offset = tuple(0.5 * value for value in projected_offset)
        axis_order = tuple(
            int(axis) for axis in np.argsort(-np.abs(projected_offset))
        )
        dominant_axis_offsets = []
        for axis in axis_order:
            value = float(projected_offset[int(axis)])
            if abs(value) < 1e-6:
                continue
            candidate = [0.0, 0.0, 0.0]
            candidate[int(axis)] = math.copysign(min(10.0, limit), value)
            dominant_axis_offsets.append(tuple(candidate))
        # Before trying mixed-axis combinations, sweep one axis at a time.
        # High and low buttons use this identical geometry-driven ordering.
        # It brings a simple 20/35-degree escape orientation forward without
        # privileging any target label or using a saved pose.
        axis_sweep_offsets = []
        for magnitude in sorted({
            value for value in (10.0, 20.0, limit)
            if 0.0 < value <= limit
        }):
            for axis in axis_order:
                projected_value = float(projected_offset[axis])
                preferred_sign = 1.0 if projected_value >= 0.0 else -1.0
                for sign in (preferred_sign, -preferred_sign):
                    candidate = [0.0, 0.0, 0.0]
                    candidate[axis] = sign * magnitude
                    axis_sweep_offsets.append(tuple(candidate))
        # A reach-boundary solution often needs two axes together (for
        # example, pitch to preserve reach and yaw to keep the wrist inside
        # its limit).  Put geometry-directed two-axis combinations before the
        # 7^3 exhaustive lattice.  This remains fully live-pose based and does
        # not cache a button-specific orientation.
        pair_sweep_offsets = []
        sweep_magnitudes = tuple(sorted({
            value for value in (10.0, 20.0, limit)
            if 0.0 < value <= limit
        }))
        for primary_axis in axis_order:
            primary_value = float(projected_offset[primary_axis])
            primary_sign = 1.0 if primary_value >= 0.0 else -1.0
            for primary_magnitude in sweep_magnitudes:
                for secondary_axis in axis_order:
                    if secondary_axis == primary_axis:
                        continue
                    secondary_value = float(projected_offset[secondary_axis])
                    secondary_sign = 1.0 if secondary_value >= 0.0 else -1.0
                    for secondary_magnitude in sweep_magnitudes:
                        for sign in (secondary_sign, -secondary_sign):
                            candidate = [0.0, 0.0, 0.0]
                            candidate[primary_axis] = (
                                primary_sign * primary_magnitude
                            )
                            candidate[secondary_axis] = sign * secondary_magnitude
                            pair_sweep_offsets.append(tuple(candidate))
        projected_ten_deg = tuple(float(value) for value in np.clip(
            projected_offset, -min(10.0, limit), min(10.0, limit)
        ))
        offsets = []
        seen_offsets = set()
        for candidate in (
            projected_offset,
            flexible_offset,
            *dominant_axis_offsets,
            projected_ten_deg,
            (0.0, 0.0, 0.0),
            half_projected_offset,
            *axis_sweep_offsets,
            *pair_sweep_offsets,
            *exhaustive_offsets,
        ):
            if candidate is None:
                continue
            normalized = tuple(round(float(value), 9) for value in candidate)
            if normalized in seen_offsets:
                continue
            seen_offsets.add(normalized)
            offsets.append(tuple(float(value) for value in candidate))
        selected = None
        selected_offset = None
        direct_candidate_count = 0
        continuation_attempt_count = 0
        selected_downstream_preview = None

        def continuation_solve(
            desired: np.ndarray,
            resolution_levels: tuple[tuple[float, float], ...],
        ):
            """Follow one continuous IK branch through virtual Cartesian steps."""
            nonlocal continuation_attempt_count
            continuation_attempt_count += 1
            start_pose = forward_kinematics(seed)
            translation_m = float(
                np.linalg.norm(desired[:3, 3] - start_pose[:3, 3])
            )
            rotation_deg = float(np.rad2deg(
                Rotation.from_matrix(
                    start_pose[:3, :3].T @ desired[:3, :3]
                ).magnitude()
            ))
            rotations = Slerp(
                [0.0, 1.0],
                Rotation.from_matrix(
                    np.stack((start_pose[:3, :3], desired[:3, :3]))
                ),
            )
            selected_path = None
            # A fixed 40 mm / 10 degree discretisation can skip a narrow IK
            # basin near a workspace boundary. Retry deterministically with
            # finer Cartesian/rotational steps while preserving the same
            # joint, position and orientation acceptance limits.
            for translation_step_m, rotation_step_deg in resolution_levels:
                check_deadline("continuation_resolution")
                step_count = max(
                    2,
                    int(np.ceil(translation_m / translation_step_m)),
                    int(np.ceil(rotation_deg / rotation_step_deg)),
                )
                joints = seed.copy()
                maximum_step = 0.0
                failed = False
                for index in range(1, step_count + 1):
                    check_deadline("continuation_waypoint")
                    fraction = index / step_count
                    waypoint = np.eye(4)
                    waypoint[:3, 3] = (
                        start_pose[:3, 3] * (1.0 - fraction)
                        + desired[:3, 3] * fraction
                    )
                    waypoint[:3, :3] = rotations([fraction]).as_matrix()[0]
                    next_joints, position_error, angle_error = inverse_kinematics(
                        waypoint,
                        joints,
                        regularization=0.001,
                        max_nfev=160,
                        solver_tolerance=1e-8,
                    )
                    check_deadline("continuation_waypoint_result")
                    joint_step = float(np.max(np.abs(next_joints - joints)))
                    if (
                        position_error > 0.001
                        or np.rad2deg(angle_error) > 0.5
                        or joint_step > self.max_joint_delta_rad
                        or np.any(next_joints < PIPER_JOINT_LIMITS[:, 0])
                        or np.any(next_joints > PIPER_JOINT_LIMITS[:, 1])
                    ):
                        failed = True
                        break
                    maximum_step = max(maximum_step, joint_step)
                    joints = next_joints
                if not failed:
                    selected_path = (joints, step_count, maximum_step)
                    break
            if selected_path is None:
                return None
            joints, step_count, maximum_step = selected_path
            actual = forward_kinematics(joints)
            final_position_error = float(
                np.linalg.norm(actual[:3, 3] - desired[:3, 3])
            )
            final_orientation_error = float(np.rad2deg(
                Rotation.from_matrix(
                    actual[:3, :3].T @ desired[:3, :3]
                ).magnitude()
            ))
            if final_position_error > 0.001 or final_orientation_error > 0.5:
                return None
            return (
                joints,
                final_position_error,
                final_orientation_error,
                step_count,
                maximum_step,
            )

        fine_deferred = []

        def accept_continued(offset, desired, continued):
            if continued is None:
                return None
            (
                result_joints,
                result_position_error,
                result_orientation_error,
                result_steps,
                result_maximum_step,
            ) = continued
            return (
                result_joints, desired, result_position_error,
                result_orientation_error,
                float(np.max(np.abs(result_joints - seed))),
                result_steps, result_maximum_step,
            )

        def preview_downstream(candidate):
            """Require this live orientation to reach prepress and press too."""
            if not self.downstream_base_x_distances_m:
                return (0.0, None, (), 0.0)
            joints = np.asarray(candidate[0], dtype=float).copy()
            desired_orientation = np.asarray(candidate[1], dtype=float)[:3, :3]
            maximum_step = 0.0
            maximum_segment_delta = 0.0
            minimum_margin = float("inf")
            solutions = []
            total_steps = 0
            start_position = forward_kinematics(joints)[:3, 3]
            for segment_index, distance_m in enumerate(
                self.downstream_base_x_distances_m
            ):
                check_deadline("downstream_segment")
                segment_start_joints = joints.copy()
                segment_joint_delta_limit = (
                    self.downstream_max_segment_joint_deltas_rad[segment_index]
                    if self.downstream_max_segment_joint_deltas_rad else None
                )
                target_position = button[:3, 3].copy()
                target_position[0] -= float(distance_m)
                translation_m = float(np.linalg.norm(target_position - start_position))
                step_count = max(2, int(np.ceil(translation_m / 0.01)))
                for index in range(1, step_count + 1):
                    check_deadline("downstream_waypoint")
                    fraction = index / step_count
                    waypoint = np.eye(4)
                    waypoint[:3, :3] = desired_orientation
                    waypoint[:3, 3] = (
                        start_position * (1.0 - fraction)
                        + target_position * fraction
                    )
                    next_joints, position_error, angle_error = inverse_kinematics(
                        waypoint,
                        joints,
                        regularization=0.001,
                        max_nfev=100,
                        solver_tolerance=1e-8,
                    )
                    check_deadline("downstream_waypoint_result")
                    joint_step = float(np.max(np.abs(next_joints - joints)))
                    if (
                        position_error > 0.001
                        or np.rad2deg(angle_error) > 0.5
                        or joint_step > 0.25
                        or np.any(next_joints < PIPER_JOINT_LIMITS[:, 0])
                        or np.any(next_joints > PIPER_JOINT_LIMITS[:, 1])
                    ):
                        return None
                    segment_delta = float(np.max(np.abs(
                        next_joints - segment_start_joints
                    )))
                    if (
                        segment_joint_delta_limit is not None
                        and segment_delta > segment_joint_delta_limit
                    ):
                        return None
                    maximum_step = max(maximum_step, joint_step)
                    maximum_segment_delta = max(
                        maximum_segment_delta, segment_delta
                    )
                    minimum_margin = min(
                        minimum_margin,
                        float(np.min(next_joints - PIPER_JOINT_LIMITS[:, 0])),
                        float(np.min(PIPER_JOINT_LIMITS[:, 1] - next_joints)),
                    )
                    joints = next_joints
                start_position = target_position
                total_steps += step_count
                solutions.append(PreviewIkSolution(
                    waypoint.copy(), joints.copy(), total_steps, maximum_step,
                ))
            return (
                maximum_step, minimum_margin, tuple(solutions),
                maximum_segment_delta,
            )

        def accept_with_preview(candidate):
            nonlocal selected_downstream_preview
            if candidate is None:
                return None
            preview = preview_downstream(candidate)
            if preview is None:
                return None
            selected_downstream_preview = preview
            return candidate

        if flexible_offset is not None and flexible_endpoint_joints is not None:
            check_deadline("flexible_guard")
            flexible_button_to_tcp = transform(
                base_rotation @ Rotation.from_euler(
                    "xyz", flexible_offset, degrees=True
                ).as_matrix(),
                [0.0, 0.0, self.stand_off_m],
            )
            flexible_desired = button @ flexible_button_to_tcp
            flexible_guard_joints, _, _ = inverse_kinematics(
                flexible_desired,
                flexible_endpoint_joints,
                regularization=0.001,
                max_nfev=160,
                solver_tolerance=1e-8,
            )
            check_deadline("flexible_guard_result")
            flexible_actual = forward_kinematics(flexible_guard_joints)
            flexible_position_error = float(np.linalg.norm(
                flexible_actual[:3, 3] - flexible_desired[:3, 3]
            ))
            flexible_orientation_error = float(np.rad2deg(
                Rotation.from_matrix(
                    flexible_actual[:3, :3].T
                    @ flexible_desired[:3, :3]
                ).magnitude()
            ))
            flexible_joint_delta = float(np.max(np.abs(
                flexible_guard_joints - seed
            )))
            flexible_steps = max(
                2,
                int(np.ceil(
                    flexible_joint_delta / self.max_joint_delta_rad
                )),
            )
            flexible_step = flexible_joint_delta / flexible_steps
            if (
                flexible_position_error <= 0.001
                and flexible_orientation_error <= 0.5
                and flexible_step <= self.max_joint_delta_rad
                and np.all(flexible_guard_joints >= PIPER_JOINT_LIMITS[:, 0])
                and np.all(flexible_guard_joints <= PIPER_JOINT_LIMITS[:, 1])
            ):
                selected = accept_with_preview((
                    flexible_guard_joints,
                    flexible_desired,
                    flexible_position_error,
                    flexible_orientation_error,
                    flexible_joint_delta,
                    flexible_steps,
                    flexible_step,
                ))
                if selected is not None:
                    selected_offset = flexible_offset

        # Score a small batch with cheap endpoint IK before attempting the
        # expensive continuous-path solve.  This is deliberately derived from
        # the current button pose and current encoders only: a mobile base can
        # invalidate any target-specific joint/orientation cache at any time.
        batch_size = 12
        for batch_start in range(0, len(offsets), batch_size):
            check_deadline("candidate_batch")
            if selected is not None:
                break
            direct_records = []
            for rank, offset in enumerate(
                offsets[batch_start:batch_start + batch_size],
                start=batch_start,
            ):
                check_deadline("direct_candidate")
                button_to_tcp = transform(
                    base_rotation @ Rotation.from_euler(
                        "xyz", offset, degrees=True
                    ).as_matrix(),
                    [0.0, 0.0, self.stand_off_m],
                )
                desired = button @ button_to_tcp
                if self.workspace_bounds_m is not None:
                    if np.any(
                        desired[:3, 3] < self.workspace_bounds_m[:, 0]
                    ) or np.any(
                        desired[:3, 3] > self.workspace_bounds_m[:, 1]
                    ):
                        continue
                joints, _, _ = inverse_kinematics(
                    desired,
                    seed,
                    regularization=0.001,
                    max_nfev=120,
                    solver_tolerance=1e-8,
                )
                check_deadline("direct_candidate_result")
                direct_candidate_count += 1
                actual = forward_kinematics(joints)
                position_error = float(
                    np.linalg.norm(actual[:3, 3] - desired[:3, 3])
                )
                orientation_error = float(np.rad2deg(
                    Rotation.from_matrix(
                        actual[:3, :3].T @ desired[:3, :3]
                    ).magnitude()
                ))
                joint_delta = float(np.max(np.abs(joints - seed)))
                accurate = (
                    np.all(joints >= PIPER_JOINT_LIMITS[:, 0])
                    and np.all(joints <= PIPER_JOINT_LIMITS[:, 1])
                    and position_error <= 0.001
                    and orientation_error <= 0.5
                )
                if accurate and joint_delta <= self.max_joint_delta_rad:
                    candidate = (
                        joints, desired, position_error, orientation_error,
                        joint_delta, 1, joint_delta,
                    )
                    selected = accept_with_preview(candidate)
                    if selected is not None:
                        selected_offset = offset
                        break
                normalized_residual = (
                    position_error / 0.001 + orientation_error / 0.5
                )
                direct_records.append((
                    offset, desired, accurate, joint_delta,
                    normalized_residual, rank,
                ))
            if selected is not None:
                break

            # A valid endpoint on the nearest live joint branch is the best
            # continuation candidate.  For invalid endpoints, residual error
            # is a better reachability signal than the angular enumeration.
            direct_records.sort(key=lambda record: (
                not record[2],
                record[3] if record[2] else record[4],
                record[4],
                record[3],
                record[5],
            ))
            # Only the strongest endpoint records from each cheap batch earn
            # an immediate continuous-path solve.  All other records remain
            # available to the deterministic fallback below, so this changes
            # latency and ordering without reducing reachability coverage.
            immediate_records = direct_records[:6]
            deferred_records = direct_records[6:]
            for offset, desired, _, _, _, _ in immediate_records:
                check_deadline("coarse_continuation")
                continued = continuation_solve(desired, ((0.04, 10.0),))
                candidate = accept_with_preview(
                    accept_continued(offset, desired, continued)
                )
                if candidate is not None:
                    selected = candidate
                    selected_offset = offset
                    break
                fine_deferred.append((offset, desired))
            fine_deferred.extend(
                (offset, desired)
                for offset, desired, _, _, _, _ in deferred_records
            )
            if selected is not None:
                break
        if selected is None:
            for resolution in (((0.02, 5.0),), ((0.01, 2.5),)):
                for offset, desired in fine_deferred:
                    check_deadline("fine_continuation")
                    continued = continuation_solve(desired, resolution)
                    selected = accept_with_preview(
                        accept_continued(offset, desired, continued)
                    )
                    if selected is None:
                        continue
                    selected_offset = offset
                    break
                if selected is not None:
                    break
        elapsed = time.perf_counter() - started
        if selected is None:
            raise RuntimeError(
                "no fast IK solution within the perpendicular-orientation tolerance"
            )
        (
            joints, desired, position_error, orientation_error, joint_delta,
            continuation_steps, maximum_step,
        ) = selected
        return FastIkResult(
            joints=joints,
            desired_base_to_tcp=desired,
            solve_sec=float(elapsed),
            position_error_m=position_error,
            orientation_error_deg=orientation_error,
            max_joint_delta_rad=joint_delta,
            continuation_steps=int(continuation_steps),
            max_continuation_joint_step_rad=float(maximum_step),
            orientation_offset_deg=tuple(
                float(value) for value in selected_offset
            ),
            projected_orientation_offset_deg=projected_offset,
            direct_candidate_count=int(direct_candidate_count),
            continuation_attempt_count=int(continuation_attempt_count),
            downstream_preview_distances_m=(
                self.downstream_base_x_distances_m
            ),
            downstream_preview_max_joint_step_rad=float(
                selected_downstream_preview[0]
                if selected_downstream_preview is not None else 0.0
            ),
            downstream_preview_max_segment_joint_delta_rad=float(
                selected_downstream_preview[3]
                if selected_downstream_preview is not None else 0.0
            ),
            downstream_preview_min_joint_limit_margin_rad=(
                None
                if selected_downstream_preview is None
                or selected_downstream_preview[1] is None
                else float(selected_downstream_preview[1])
            ),
            downstream_preview_solutions=(
                selected_downstream_preview[2] if selected_downstream_preview else ()
            ),
        )


class RobotTransformHistory:
    """Thread-safe interpolation of timestamped base-to-TCP transforms."""

    def __init__(self, *, maxlen: int = 400, max_gap_sec: float = 0.12) -> None:
        self.max_gap_sec = float(max_gap_sec)
        self._samples: deque[tuple[float, np.ndarray]] = deque(maxlen=int(maxlen))
        self._condition = threading.Condition()

    def add(self, timestamp_monotonic: float, base_to_tcp: Any) -> None:
        matrix = np.asarray(base_to_tcp, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("base_to_tcp must be a finite 4x4 transform")
        stamp = float(timestamp_monotonic)
        with self._condition:
            if self._samples and stamp <= self._samples[-1][0]:
                return
            self._samples.append((stamp, matrix.copy()))
            self._condition.notify_all()

    def transform_at(
        self, timestamp_monotonic: float, *, wait_sec: float = 0.10
    ) -> np.ndarray:
        target = float(timestamp_monotonic)
        deadline = time.monotonic() + float(wait_sec)
        with self._condition:
            while (not self._samples or self._samples[-1][0] < target) and (
                time.monotonic() < deadline
            ):
                self._condition.wait(deadline - time.monotonic())
            samples = tuple(self._samples)
        if len(samples) < 2 or target < samples[0][0] or target > samples[-1][0]:
            raise RuntimeError("capture time is outside robot-state history")
        for (t0, pose0), (t1, pose1) in zip(samples, samples[1:]):
            if t0 <= target <= t1:
                gap = t1 - t0
                if gap <= 0.0 or gap > self.max_gap_sec:
                    raise RuntimeError(f"robot-state interpolation gap is {gap:.3f}s")
                ratio = (target - t0) / gap
                result = np.eye(4)
                result[:3, 3] = (
                    (1.0 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                )
                rotations = Rotation.from_matrix(
                    np.stack([pose0[:3, :3], pose1[:3, :3]])
                )
                result[:3, :3] = Slerp([t0, t1], rotations)([target]).as_matrix()[0]
                return result
        raise RuntimeError("failed to bracket capture time")
