from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .trajectory_transport import MAX_TRAJECTORY_BYTES, send_frame, receive_frame
from .kinematics import forward_kinematics, forward_kinematics_batch
from .robot import PIPER_JOINT_LIMITS


PIPER_JOINT_NAMES = [f"piper_joint{i}" for i in range(1, 7)]
PIPER_MAX_VELOCITIES = np.asarray([0.6, 1.0, 1.0, 1.0, 1.0, 2.0])
PIPER_BRIDGE_VELOCITY_CAP_SCALE = 0.75
PIPER_CARTESIAN_SPEED_CAP_M_S = 0.15
PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S = 0.25
PIPER_UDP_MAX_PAYLOAD_BYTES = 60000
PIPER_WIRE_DECIMAL_CANDIDATES = (14, 12, 10, 9)


@dataclass(frozen=True)
class TrajectoryAudit:
    duration_sec: float
    point_count: int
    cartesian_path_m: float
    cartesian_peak_speed_m_s: float
    joint_peak_velocity_rad_s: np.ndarray
    joint_velocity_fraction: np.ndarray


@dataclass(frozen=True)
class TwoZoneTrajectoryAudit(TrajectoryAudit):
    transition_fraction: float
    transition_time_sec: float
    fast_zone_peak_speed_m_s: float
    near_zone_peak_speed_m_s: float


@dataclass(frozen=True)
class WaypointTrajectoryAudit(TrajectoryAudit):
    waypoint_time_sec: float
    first_zone_peak_speed_m_s: float
    second_zone_peak_speed_m_s: float
    waypoint_joint_speed_rad_s: float


@dataclass(frozen=True)
class BlendedPressReturnAudit(TrajectoryAudit):
    press_endpoint_time_sec: float
    prepress_pass_time_sec: float
    prepress_joint_speed_rad_s: float
    prepress_cartesian_speed_m_s: float
    inverse_join_index: int
    inverse_join_time_sec: float


def joint_target_at_panel_distance(
    start_joints: Any,
    end_joints: Any,
    base_to_panel: Any,
    target_distance_m: float,
) -> tuple[np.ndarray, float, float]:
    """Find a joint-interpolated target at an exact signed panel distance."""
    start = _as_joints(start_joints, "start_joints")
    end = _as_joints(end_joints, "end_joints")
    panel = np.asarray(base_to_panel, dtype=float)
    if panel.shape != (4, 4) or not np.isfinite(panel).all():
        raise ValueError("base_to_panel must be a finite 4x4 transform")
    normal = panel[:3, 2]
    origin = panel[:3, 3]

    def distance(fraction: float) -> float:
        joints = start + fraction * (end - start)
        return float((forward_kinematics(joints)[:3, 3] - origin) @ normal)

    start_distance, end_distance = distance(0.0), distance(1.0)
    if not end_distance <= target_distance_m <= start_distance:
        raise ValueError(
            f"target distance {target_distance_m:.4f} is outside monotonic "
            f"range {end_distance:.4f}..{start_distance:.4f}"
        )
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if distance(middle) > target_distance_m:
            low = middle
        else:
            high = middle
    fraction = 0.5 * (low + high)
    result = start + fraction * (end - start)
    actual_distance = distance(fraction)
    samples = [distance(float(value)) for value in np.linspace(0.0, fraction, 101)]
    if any(current > previous + 0.002 for previous, current in zip(samples, samples[1:])):
        raise RuntimeError("joint-interpolated probe path is not monotonic toward panel")
    return result, fraction, actual_distance


def _smoothstep(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quintic position, first derivative, and second derivative on [0, 1]."""
    s = 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5
    ds = 30.0 * r**2 - 60.0 * r**3 + 30.0 * r**4
    dds = 60.0 * r - 180.0 * r**2 + 120.0 * r**3
    return s, ds, dds


def _quintic_boundary_coefficients(
    start_position: float,
    start_velocity: float,
    start_acceleration: float,
    end_position: float,
    end_velocity: float,
    end_acceleration: float,
    duration: float,
) -> np.ndarray:
    coefficients = np.zeros(6, dtype=float)
    coefficients[:3] = [
        start_position,
        start_velocity,
        0.5 * start_acceleration,
    ]
    t = float(duration)
    matrix = np.asarray(
        [
            [t**3, t**4, t**5],
            [3 * t**2, 4 * t**3, 5 * t**4],
            [6 * t, 12 * t**2, 20 * t**3],
        ]
    )
    known_position = coefficients[0] + coefficients[1] * t + coefficients[2] * t**2
    known_velocity = coefficients[1] + 2 * coefficients[2] * t
    known_acceleration = 2 * coefficients[2]
    coefficients[3:] = np.linalg.solve(
        matrix,
        np.asarray(
            [
                end_position - known_position,
                end_velocity - known_velocity,
                end_acceleration - known_acceleration,
            ]
        ),
    )
    return coefficients


def _evaluate_scalar_quintic(coefficients: np.ndarray, values: np.ndarray):
    position = sum(coefficients[index] * values**index for index in range(6))
    velocity = sum(
        index * coefficients[index] * values ** (index - 1)
        for index in range(1, 6)
    )
    acceleration = sum(
        index * (index - 1) * coefficients[index] * values ** (index - 2)
        for index in range(2, 6)
    )
    return position, velocity, acceleration


def _vector_quintic_coefficients(
    start_position: np.ndarray,
    start_velocity: np.ndarray,
    start_acceleration: np.ndarray,
    end_position: np.ndarray,
    end_velocity: np.ndarray,
    end_acceleration: np.ndarray,
    duration: float,
) -> np.ndarray:
    coefficients = np.zeros((6, 6), dtype=float)
    coefficients[0] = start_position
    coefficients[1] = start_velocity
    coefficients[2] = 0.5 * start_acceleration
    t = float(duration)
    matrix = np.asarray(
        [
            [t**3, t**4, t**5],
            [3 * t**2, 4 * t**3, 5 * t**4],
            [6 * t, 12 * t**2, 20 * t**3],
        ]
    )
    known_position = coefficients[0] + coefficients[1] * t + coefficients[2] * t**2
    known_velocity = coefficients[1] + 2 * coefficients[2] * t
    known_acceleration = 2 * coefficients[2]
    rhs = np.vstack(
        [
            end_position - known_position,
            end_velocity - known_velocity,
            end_acceleration - known_acceleration,
        ]
    )
    coefficients[3:] = np.linalg.solve(matrix, rhs)
    return coefficients


def _evaluate_vector_quintic(coefficients: np.ndarray, values: np.ndarray):
    powers = np.vstack([values**index for index in range(6)])
    positions = powers.T @ coefficients
    velocities = (
        np.vstack([values**index for index in range(5)]).T
        @ np.asarray([index * coefficients[index] for index in range(1, 6)])
    )
    accelerations = (
        np.vstack([values**index for index in range(4)]).T
        @ np.asarray(
            [index * (index - 1) * coefficients[index] for index in range(2, 6)]
        )
    )
    return positions, velocities, accelerations


def _waypoint_boundary_coefficients(
    start_position: np.ndarray,
    waypoint_position: np.ndarray,
    end_position: np.ndarray,
    end_velocity: np.ndarray,
    end_acceleration: np.ndarray,
    waypoint_time: float,
    duration: float,
) -> np.ndarray:
    """Degree-six joint polynomial with a fixed interior position waypoint."""
    powers = np.arange(7, dtype=float)

    def row(t: float, derivative: int) -> np.ndarray:
        values = np.zeros(7, dtype=float)
        for index in range(derivative, 7):
            factor = 1.0
            for value in range(derivative):
                factor *= index - value
            values[index] = factor * t ** (index - derivative)
        return values

    matrix = np.vstack(
        [
            row(0.0, 0),
            row(0.0, 1),
            row(0.0, 2),
            row(float(waypoint_time), 0),
            row(float(duration), 0),
            row(float(duration), 1),
            row(float(duration), 2),
        ]
    )
    rhs = np.vstack(
        [
            start_position,
            np.zeros(6),
            np.zeros(6),
            waypoint_position,
            end_position,
            end_velocity,
            end_acceleration,
        ]
    )
    return np.linalg.solve(matrix, rhs)


def _evaluate_vector_polynomial(coefficients: np.ndarray, values: np.ndarray):
    degree = coefficients.shape[0] - 1
    powers = np.vstack([values**index for index in range(degree + 1)])
    positions = powers.T @ coefficients
    velocities = (
        np.vstack([values**index for index in range(degree)]).T
        @ np.asarray([index * coefficients[index] for index in range(1, degree + 1)])
    )
    accelerations = (
        np.vstack([values**index for index in range(degree - 1)]).T
        @ np.asarray(
            [index * (index - 1) * coefficients[index] for index in range(2, degree + 1)]
        )
    )
    return positions, velocities, accelerations


def generate_continuous_waypoint_trajectory(
    start_joints: Any,
    waypoint_joints: Any,
    end_joints: Any,
    *,
    first_cartesian_speed_m_s: float,
    second_cartesian_speed_m_s: float,
    first_joint_velocity_scale: float = 0.65,
    second_joint_velocity_scale: float = 0.75,
    waypoint_velocity_fraction: float = 1.80,
    point_period_sec: float = 0.10,
    duration_margin: float = 1.02,
) -> tuple[list[dict[str, Any]], WaypointTrajectoryAudit]:
    """Create one C2 return trajectory through a non-stopping safe waypoint."""
    start = _as_joints(start_joints, "start_joints")
    waypoint = _as_joints(waypoint_joints, "waypoint_joints")
    end = _as_joints(end_joints, "end_joints")
    if not 0.01 <= first_cartesian_speed_m_s <= PIPER_CARTESIAN_SPEED_CAP_M_S:
        raise ValueError("first Cartesian speed must be in 0.01..0.15 m/s")
    if not 0.01 <= second_cartesian_speed_m_s <= PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S:
        raise ValueError("second Cartesian speed must be in 0.01..0.25 m/s")
    if not 0.05 <= first_joint_velocity_scale <= PIPER_BRIDGE_VELOCITY_CAP_SCALE:
        raise ValueError("first joint velocity scale must be in 0.05..0.75")
    if not 0.05 <= second_joint_velocity_scale <= PIPER_BRIDGE_VELOCITY_CAP_SCALE:
        raise ValueError("second joint velocity scale must be in 0.05..0.75")
    if not 0.20 <= waypoint_velocity_fraction <= 2.00:
        raise ValueError("waypoint_velocity_fraction must be in 0.20..2.00")
    if not 0.01 <= point_period_sec <= 0.10:
        raise ValueError("point_period_sec must be in 0.01..0.10")

    _, first_audit = generate_timed_joint_trajectory(
        start, waypoint,
        cartesian_speed_m_s=first_cartesian_speed_m_s,
        joint_velocity_scale=first_joint_velocity_scale,
        point_period_sec=point_period_sec,
        duration_margin=duration_margin,
    )
    _, second_audit = generate_timed_joint_trajectory(
        waypoint, end,
        cartesian_speed_m_s=second_cartesian_speed_m_s,
        joint_velocity_scale=second_joint_velocity_scale,
        point_period_sec=point_period_sec,
        duration_margin=duration_margin,
        cartesian_speed_cap_m_s=PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
    )
    first_duration = float(first_audit.duration_sec)
    second_duration = float(second_audit.duration_sec)
    delta_first = waypoint - start
    zero = np.zeros(6)
    allowed_first = PIPER_MAX_VELOCITIES * float(first_joint_velocity_scale)
    allowed_second = PIPER_MAX_VELOCITIES * float(second_joint_velocity_scale)

    waypoint_velocity_scale = 1.0
    for _ in range(20):
        # Continue in the safe retreat direction at the waypoint, then curve home.
        waypoint_velocity = (
            waypoint_velocity_scale
            * float(waypoint_velocity_fraction)
            * delta_first
            / first_duration
        )
        first_coeff = _vector_quintic_coefficients(
            start, zero, zero, waypoint, waypoint_velocity, zero, first_duration
        )
        second_coeff = _vector_quintic_coefficients(
            waypoint, waypoint_velocity, zero, end, zero, zero, second_duration
        )
        first_t = np.linspace(0.0, first_duration, 1201)
        second_t = np.linspace(0.0, second_duration, 2001)
        first_q, first_v, first_a = _evaluate_vector_quintic(first_coeff, first_t)
        second_q, second_v, second_a = _evaluate_vector_quintic(second_coeff, second_t)
        joint_tolerance = 1e-9
        if (
            np.any(first_q < PIPER_JOINT_LIMITS[:, 0] - joint_tolerance)
            or np.any(first_q > PIPER_JOINT_LIMITS[:, 1] + joint_tolerance)
            or np.any(second_q < PIPER_JOINT_LIMITS[:, 0] - joint_tolerance)
            or np.any(second_q > PIPER_JOINT_LIMITS[:, 1] + joint_tolerance)
        ):
            if (
                waypoint_velocity_scale > 1.0 / 64.0
                and np.linalg.norm(waypoint_velocity) > 1e-6
            ):
                waypoint_velocity_scale *= 0.5
                continue
            raise RuntimeError("continuous return overshoots a joint position limit")
        first_xyz = forward_kinematics_batch(first_q)[:, :3, 3]
        second_xyz = forward_kinematics_batch(second_q)[:, :3, 3]
        first_tcp = np.linalg.norm(np.gradient(first_xyz, first_t, axis=0), axis=1)
        second_tcp = np.linalg.norm(np.gradient(second_xyz, second_t, axis=0), axis=1)
        first_violation = max(
            float(first_tcp.max()) / float(first_cartesian_speed_m_s),
            float(np.max(np.abs(first_v) / allowed_first)),
        )
        second_violation = max(
            float(second_tcp.max()) / float(second_cartesian_speed_m_s),
            float(np.max(np.abs(second_v) / allowed_second)),
        )
        if first_violation <= 1.0 + 1e-6 and second_violation <= 1.0 + 1e-6:
            break
        if first_violation > 1.0 + 1e-6:
            first_duration *= first_violation * 1.02
        if second_violation > 1.0 + 1e-6:
            # If the inherited waypoint velocity itself violates zone two,
            # lengthen zone one to reduce that boundary velocity. Otherwise
            # only lengthen zone two, preserving its higher return speed.
            waypoint_tcp_ratio = float(second_tcp[0]) / float(
                second_cartesian_speed_m_s
            )
            waypoint_joint_ratio = float(
                np.max(np.abs(waypoint_velocity) / allowed_second)
            )
            boundary_violation = max(waypoint_tcp_ratio, waypoint_joint_ratio)
            if boundary_violation > 1.0 + 1e-6:
                first_duration *= boundary_violation * 1.02
            else:
                second_duration *= second_violation * 1.02
    else:
        raise RuntimeError("continuous return duration scaling did not converge")

    first_count = max(2, int(math.ceil(first_duration / point_period_sec)) + 1)
    second_count = max(2, int(math.ceil(second_duration / point_period_sec)) + 1)
    first_times = np.linspace(0.0, first_duration, first_count)
    second_times = np.linspace(0.0, second_duration, second_count)
    first_q, first_v, first_a = _evaluate_vector_quintic(first_coeff, first_times)
    second_q, second_v, second_a = _evaluate_vector_quintic(second_coeff, second_times)
    first_q = np.clip(
        first_q, PIPER_JOINT_LIMITS[:, 0], PIPER_JOINT_LIMITS[:, 1]
    )
    second_q = np.clip(
        second_q, PIPER_JOINT_LIMITS[:, 0], PIPER_JOINT_LIMITS[:, 1]
    )
    times = np.concatenate([first_times, first_duration + second_times[1:]])
    positions = np.vstack([first_q, second_q[1:]])
    velocities = np.vstack([first_v, second_v[1:]])
    accelerations = np.vstack([first_a, second_a[1:]])
    points = [
        {
            "time_from_start_sec": float(stamp),
            "positions": q.tolist(),
            "velocities": qd.tolist(),
            "accelerations": qdd.tolist(),
        }
        for stamp, q, qd, qdd in zip(times, positions, velocities, accelerations)
    ]
    validate_trajectory_payload(
        {
            "protocol": "piper_timed_trajectory_v1",
            "command": "validate",
            "joint_names": PIPER_JOINT_NAMES,
            "points": points,
        }
    )
    combined_xyz = np.vstack([first_xyz, second_xyz[1:]])
    path_length = float(
        np.sum(np.linalg.norm(np.diff(combined_xyz, axis=0), axis=1))
    )
    joint_peak = np.max(np.abs(np.vstack([first_v, second_v])), axis=0)
    return points, WaypointTrajectoryAudit(
        duration_sec=float(times[-1]),
        point_count=len(points),
        cartesian_path_m=path_length,
        cartesian_peak_speed_m_s=float(max(first_tcp.max(), second_tcp.max())),
        joint_peak_velocity_rad_s=joint_peak,
        joint_velocity_fraction=joint_peak / PIPER_MAX_VELOCITIES,
        waypoint_time_sec=float(first_duration),
        first_zone_peak_speed_m_s=float(first_tcp.max()),
        second_zone_peak_speed_m_s=float(second_tcp.max()),
        waypoint_joint_speed_rad_s=float(np.linalg.norm(waypoint_velocity)),
    )


def generate_two_zone_joint_trajectory(
    start_joints: Any,
    end_joints: Any,
    *,
    transition_fraction: float,
    fast_cartesian_speed_m_s: float = 0.15,
    near_cartesian_speed_m_s: float = 0.06,
    joint_velocity_scale: float = 0.65,
    point_period_sec: float = 0.05,
    duration_margin: float = 1.02,
) -> tuple[list[dict[str, Any]], TwoZoneTrajectoryAudit]:
    """Generate one C2 trajectory with a fast zone and slower terminal zone."""
    start = _as_joints(start_joints, "start_joints")
    end = _as_joints(end_joints, "end_joints")
    fraction = float(transition_fraction)
    if not 0.10 <= fraction <= 0.95:
        raise ValueError("transition_fraction must be in 0.10..0.95")
    if not 0.005 <= near_cartesian_speed_m_s <= fast_cartesian_speed_m_s <= 0.15:
        raise ValueError("require 0.005 <= near speed <= fast speed <= 0.15")
    if not 0.05 <= joint_velocity_scale <= PIPER_BRIDGE_VELOCITY_CAP_SCALE:
        raise ValueError("joint_velocity_scale must be in 0.05..0.75")
    if not 0.01 <= point_period_sec <= 0.10:
        raise ValueError("point_period_sec must be in 0.01..0.10")
    if duration_margin < 1.0:
        raise ValueError("duration_margin must be at least 1")

    delta = end - start
    path_s = np.linspace(0.0, 1.0, 2001)
    path_q = start[None, :] + path_s[:, None] * delta[None, :]
    path_xyz = forward_kinematics_batch(path_q)[:, :3, 3]
    xyz_rate_per_s = np.linalg.norm(np.gradient(path_xyz, path_s, axis=0), axis=1)
    fast_mask = path_s <= fraction
    near_mask = path_s >= fraction
    allowed_joint_velocity = PIPER_MAX_VELOCITIES * float(joint_velocity_scale)
    joint_s_rate_limit = float(
        np.min(
            np.divide(
                allowed_joint_velocity,
                np.abs(delta),
                out=np.full(6, np.inf),
                where=np.abs(delta) > 1e-12,
            )
        )
    )
    fast_s_rate = min(
        float(fast_cartesian_speed_m_s) / float(xyz_rate_per_s[fast_mask].max()),
        joint_s_rate_limit,
    )
    near_s_rate = min(
        float(near_cartesian_speed_m_s) / float(xyz_rate_per_s[near_mask].max()),
        joint_s_rate_limit,
    )
    estimated_fast_time = fraction / fast_s_rate
    estimated_near_time = (1.0 - fraction) / near_s_rate
    estimated_duration = estimated_fast_time + estimated_near_time
    transition_u = estimated_fast_time / estimated_duration
    transition_du_velocity = near_s_rate * estimated_duration

    for _ in range(12):
        first = _quintic_boundary_coefficients(
            0.0, 0.0, 0.0, fraction, transition_du_velocity, 0.0,
            transition_u,
        )
        second = _quintic_boundary_coefficients(
            fraction, transition_du_velocity, 0.0, 1.0, 0.0, 0.0,
            1.0 - transition_u,
        )
        dense_u = np.unique(
            np.concatenate([np.linspace(0.0, 1.0, 4001), [transition_u]])
        )
        first_mask = dense_u <= transition_u
        scalar = np.empty_like(dense_u)
        scalar_rate = np.empty_like(dense_u)
        scalar_acceleration = np.empty_like(dense_u)
        scalar[first_mask], scalar_rate[first_mask], scalar_acceleration[first_mask] = (
            _evaluate_scalar_quintic(first, dense_u[first_mask])
        )
        local = dense_u[~first_mask] - transition_u
        scalar[~first_mask], scalar_rate[~first_mask], scalar_acceleration[~first_mask] = (
            _evaluate_scalar_quintic(second, local)
        )
        if (
            scalar.min() >= -1e-9
            and scalar.max() <= 1.0 + 1e-9
            and scalar_rate.min() >= -1e-9
        ):
            break
        transition_du_velocity *= 0.8
    else:
        raise RuntimeError("could not construct a monotonic two-zone time law")

    dense_q = start[None, :] + scalar[:, None] * delta[None, :]
    dense_xyz = forward_kinematics_batch(dense_q)[:, :3, 3]
    xyz_rate_per_u = np.linalg.norm(np.gradient(dense_xyz, dense_u, axis=0), axis=1)
    joint_rate_per_u = np.max(np.abs(scalar_rate[:, None] * delta[None, :]), axis=0)
    dense_fast = scalar <= fraction + 1e-9
    dense_near = scalar >= fraction - 1e-9
    duration = max(
        float(xyz_rate_per_u[dense_fast].max()) / float(fast_cartesian_speed_m_s),
        float(xyz_rate_per_u[dense_near].max()) / float(near_cartesian_speed_m_s),
        float(np.max(joint_rate_per_u / allowed_joint_velocity)),
    ) * float(duration_margin)
    if duration > 15.0:
        raise RuntimeError("two-zone trajectory exceeds 15 second protocol limit")

    transition_time = transition_u * duration
    sample_times = np.unique(
        np.concatenate(
            [np.arange(0.0, duration, float(point_period_sec)), [transition_time, duration]]
        )
    )
    sample_u = sample_times / duration
    sample_first = sample_u <= transition_u
    sample_s = np.empty_like(sample_u)
    sample_ds_du = np.empty_like(sample_u)
    sample_d2s_du2 = np.empty_like(sample_u)
    sample_s[sample_first], sample_ds_du[sample_first], sample_d2s_du2[sample_first] = (
        _evaluate_scalar_quintic(first, sample_u[sample_first])
    )
    sample_local = sample_u[~sample_first] - transition_u
    sample_s[~sample_first], sample_ds_du[~sample_first], sample_d2s_du2[~sample_first] = (
        _evaluate_scalar_quintic(second, sample_local)
    )
    positions = start[None, :] + sample_s[:, None] * delta[None, :]
    velocities = sample_ds_du[:, None] * delta[None, :] / duration
    accelerations = sample_d2s_du2[:, None] * delta[None, :] / duration**2
    points = [
        {
            "time_from_start_sec": float(stamp),
            "positions": q.tolist(),
            "velocities": qd.tolist(),
            "accelerations": qdd.tolist(),
        }
        for stamp, q, qd, qdd in zip(sample_times, positions, velocities, accelerations)
    ]
    actual_xyz_rate = xyz_rate_per_u / duration
    actual_joint_peak = joint_rate_per_u / duration
    path_length = float(np.sum(np.linalg.norm(np.diff(dense_xyz, axis=0), axis=1)))
    audit = TwoZoneTrajectoryAudit(
        duration_sec=float(duration),
        point_count=len(points),
        cartesian_path_m=path_length,
        cartesian_peak_speed_m_s=float(actual_xyz_rate.max()),
        joint_peak_velocity_rad_s=actual_joint_peak,
        joint_velocity_fraction=actual_joint_peak / PIPER_MAX_VELOCITIES,
        transition_fraction=fraction,
        transition_time_sec=float(transition_time),
        fast_zone_peak_speed_m_s=float(actual_xyz_rate[dense_fast].max()),
        near_zone_peak_speed_m_s=float(actual_xyz_rate[dense_near].max()),
    )
    return points, audit


def _as_joints(value: Any, name: str) -> np.ndarray:
    joints = np.asarray(value, dtype=float)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError(f"{name} must contain six finite joint radians")
    if np.any(joints < PIPER_JOINT_LIMITS[:, 0]) or np.any(
        joints > PIPER_JOINT_LIMITS[:, 1]
    ):
        raise ValueError(f"{name} exceeds Piper joint limits")
    return joints


def generate_timed_joint_trajectory(
    start_joints: Any,
    end_joints: Any,
    *,
    cartesian_speed_m_s: float = 0.05,
    joint_velocity_scale: float = 0.25,
    point_period_sec: float = 0.05,
    minimum_duration_sec: float = 0.5,
    duration_margin: float = 1.08,
    cartesian_speed_cap_m_s: float = PIPER_CARTESIAN_SPEED_CAP_M_S,
) -> tuple[list[dict[str, Any]], TrajectoryAudit]:
    """Generate a smooth trajectory whose sampled TCP and joint speeds are bounded."""
    start = _as_joints(start_joints, "start_joints")
    end = _as_joints(end_joints, "end_joints")
    if not (
        PIPER_CARTESIAN_SPEED_CAP_M_S
        <= cartesian_speed_cap_m_s
        <= PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S
    ):
        raise ValueError("cartesian_speed_cap_m_s must be in 0.15..0.25")
    if not 0.005 <= cartesian_speed_m_s <= cartesian_speed_cap_m_s:
        raise ValueError(
            f"cartesian_speed_m_s must be in 0.005..{cartesian_speed_cap_m_s:.2f}"
        )
    if not 0.05 <= joint_velocity_scale <= PIPER_BRIDGE_VELOCITY_CAP_SCALE:
        raise ValueError("joint_velocity_scale must be in 0.05..0.75")
    if not 0.01 <= point_period_sec <= 0.20:
        raise ValueError("point_period_sec must be in 0.01..0.20")
    if duration_margin < 1.0:
        raise ValueError("duration_margin must be at least 1")

    delta = end - start
    dense_r = np.linspace(0.0, 1.0, 2001)
    dense_s, dense_ds, _ = _smoothstep(dense_r)
    dense_joints = start[None, :] + dense_s[:, None] * delta[None, :]
    tcp_positions = forward_kinematics_batch(dense_joints)[:, :3, 3]
    dr = float(dense_r[1] - dense_r[0])
    tcp_rate_per_unit_duration = np.linalg.norm(
        np.gradient(tcp_positions, dr, axis=0), axis=1
    )
    cartesian_duration = float(
        np.max(tcp_rate_per_unit_duration) / cartesian_speed_m_s
    )
    joint_rate_per_unit_duration = np.max(
        np.abs(dense_ds[:, None] * delta[None, :]), axis=0
    )
    allowed_joint_velocity = PIPER_MAX_VELOCITIES * joint_velocity_scale
    joint_duration = float(
        np.max(
            np.divide(
                joint_rate_per_unit_duration,
                allowed_joint_velocity,
                out=np.zeros(6),
                where=allowed_joint_velocity > 0,
            )
        )
    )
    duration = max(minimum_duration_sec, cartesian_duration, joint_duration)
    duration *= duration_margin
    point_count = max(2, int(math.ceil(duration / point_period_sec)) + 1)
    times = np.linspace(0.0, duration, point_count)
    r = times / duration
    s, ds, dds = _smoothstep(r)
    positions = start[None, :] + s[:, None] * delta[None, :]
    velocities = ds[:, None] * delta[None, :] / duration
    accelerations = dds[:, None] * delta[None, :] / duration**2

    actual_cartesian_peak = float(np.max(tcp_rate_per_unit_duration) / duration)
    actual_joint_peak = joint_rate_per_unit_duration / duration
    if actual_cartesian_peak > cartesian_speed_m_s + 1e-9:
        raise RuntimeError("generated trajectory exceeds Cartesian speed limit")
    if np.any(actual_joint_peak > allowed_joint_velocity + 1e-9):
        raise RuntimeError("generated trajectory exceeds scaled joint velocity limit")
    path_length = float(np.sum(np.linalg.norm(np.diff(tcp_positions, axis=0), axis=1)))
    points = [
        {
            "time_from_start_sec": float(t),
            "positions": q.tolist(),
            "velocities": qd.tolist(),
            "accelerations": qdd.tolist(),
        }
        for t, q, qd, qdd in zip(times, positions, velocities, accelerations)
    ]
    audit = TrajectoryAudit(
        duration_sec=float(duration),
        point_count=point_count,
        cartesian_path_m=path_length,
        cartesian_peak_speed_m_s=actual_cartesian_peak,
        joint_peak_velocity_rad_s=actual_joint_peak,
        joint_velocity_fraction=actual_joint_peak / PIPER_MAX_VELOCITIES,
    )
    return points, audit


def validate_trajectory_payload(payload: dict[str, Any]) -> None:
    """Shared validation used before sending and again by the OrangePi bridge."""
    if payload.get("protocol") != "piper_timed_trajectory_v1":
        raise ValueError("unsupported trajectory protocol")
    if payload.get("command") not in ("execute", "validate"):
        raise ValueError("unsupported trajectory command")
    if payload.get("joint_names") != PIPER_JOINT_NAMES:
        raise ValueError("unexpected joint names or ordering")
    points = payload.get("points")
    if not isinstance(points, list) or not 2 <= len(points) <= 500:
        raise ValueError("trajectory requires 2..500 points")
    previous_time = -1.0
    previous_positions = None
    for index, point in enumerate(points):
        timestamp = float(point["time_from_start_sec"])
        positions = _as_joints(point["positions"], f"point {index} positions")
        velocities = np.asarray(point["velocities"], dtype=float)
        accelerations = np.asarray(point["accelerations"], dtype=float)
        if velocities.shape != (6,) or accelerations.shape != (6,):
            raise ValueError(f"point {index} velocity/acceleration shape is invalid")
        if not np.isfinite(velocities).all() or not np.isfinite(accelerations).all():
            raise ValueError(f"point {index} contains non-finite derivatives")
        if not math.isfinite(timestamp) or timestamp <= previous_time:
            raise ValueError("trajectory times must be strictly increasing")
        if timestamp > 15.0:
            raise ValueError("trajectory duration exceeds 15 seconds")
        if np.any(
            np.abs(velocities)
            > PIPER_MAX_VELOCITIES * PIPER_BRIDGE_VELOCITY_CAP_SCALE + 1e-9
        ):
            raise ValueError(f"point {index} declared velocity exceeds bridge cap")
        if previous_positions is not None:
            dt = timestamp - previous_time
            average_velocity = np.abs(positions - previous_positions) / dt
            if np.any(
                average_velocity
                > PIPER_MAX_VELOCITIES * PIPER_BRIDGE_VELOCITY_CAP_SCALE + 1e-9
            ):
                raise ValueError(f"segment {index} average joint velocity exceeds bridge cap")
        previous_time = timestamp
        previous_positions = positions
    if abs(float(points[0]["time_from_start_sec"])) > 1e-9:
        raise ValueError("first trajectory point must be at time zero")


def compact_trajectory_payload(
    payload: dict[str, Any], *, decimals: int
) -> dict[str, Any]:
    """Shorten redundant float tails without changing trajectory sampling."""
    compact = dict(payload)
    compact_points = []
    for point in payload["points"]:
        value = dict(point)
        value["time_from_start_sec"] = round(
            float(point["time_from_start_sec"]), decimals
        )
        for key in ("positions", "velocities", "accelerations"):
            value[key] = [
                0.0 if (rounded := round(float(item), decimals)) == 0.0
                else rounded
                for item in point[key]
            ]
        compact_points.append(value)
    compact["points"] = compact_points
    return compact


def encode_trajectory_udp_payload(
    payload: dict[str, Any],
    *,
    max_bytes: int = PIPER_UDP_MAX_PAYLOAD_BYTES,
) -> tuple[bytes, dict[str, Any], int]:
    """Encode every original sample at the finest precision fitting one UDP packet.

    Each candidate is validated after quantisation, so joint limits, timing,
    velocity caps and segment-average velocity checks apply to the exact values
    received by the bridge. No sample is removed and no duration is changed.
    """
    validate_trajectory_payload(payload)
    last_size = None
    last_error = None
    for decimals in PIPER_WIRE_DECIMAL_CANDIDATES:
        compact = compact_trajectory_payload(payload, decimals=decimals)
        try:
            validate_trajectory_payload(compact)
        except ValueError as exc:
            last_error = exc
            continue
        encoded = json.dumps(compact, separators=(",", ":")).encode("utf-8")
        last_size = len(encoded)
        if last_size <= int(max_bytes):
            return encoded, compact, decimals
    detail = (
        f"last compact size={last_size} bytes"
        if last_size is not None
        else f"all compact forms failed validation: {last_error}"
    )
    raise ValueError(
        f"trajectory UDP payload cannot fit within {int(max_bytes)} bytes; {detail}"
    )



def encode_trajectory_transport_payload(payload, *, max_bytes=MAX_TRAJECTORY_BYTES):
    """Keep UDP compatibility; retain full precision/sampling for large TCP requests."""
    validate_trajectory_payload(payload)
    try:
        return encode_trajectory_udp_payload(payload,max_bytes=min(max_bytes,PIPER_UDP_MAX_PAYLOAD_BYTES))
    except ValueError as exc:
        if 'trajectory UDP payload cannot fit' not in str(exc):raise
    encoded=json.dumps(payload,separators=(',',':'),allow_nan=False).encode('utf-8')
    if len(encoded)>max_bytes:raise ValueError('trajectory exceeds bounded bulk transport size')
    return encoded,payload,None


def reverse_timed_trajectory(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the exact time reversal of a validated joint trajectory.

    For q_rev(t) = q(T-t), velocity changes sign while acceleration does not.
    """
    validate_trajectory_payload(
        {
            "protocol": "piper_timed_trajectory_v1",
            "command": "validate",
            "joint_names": PIPER_JOINT_NAMES,
            "points": points,
        }
    )
    duration = float(points[-1]["time_from_start_sec"])
    reversed_points = []
    for index, point in enumerate(reversed(points)):
        timestamp = duration - float(point["time_from_start_sec"])
        if index == 0:
            timestamp = 0.0
        reversed_points.append(
            {
                "time_from_start_sec": float(timestamp),
                "positions": np.asarray(point["positions"], dtype=float).tolist(),
                "velocities": (
                    -np.asarray(point["velocities"], dtype=float)
                ).tolist(),
                "accelerations": np.asarray(
                    point["accelerations"], dtype=float
                ).tolist(),
            }
        )
    validate_trajectory_payload(
        {
            "protocol": "piper_timed_trajectory_v1",
            "command": "validate",
            "joint_names": PIPER_JOINT_NAMES,
            "points": reversed_points,
        }
    )
    return reversed_points


def generate_press_cycle_trajectory(
    prepress_joints: Any,
    pressed_joints: Any,
    *,
    cartesian_speed_m_s: float = 0.15,
    joint_velocity_scale: float = 0.75,
    point_period_sec: float = 0.05,
    duration_margin: float = 1.02,
) -> tuple[list[dict[str, Any]], TrajectoryAudit]:
    """Generate one fast prepress -> pressed -> prepress command.

    The reversal is included in the same bridge command, avoiding a network or
    state-query pause while the tool is against the button.
    """
    outbound, outbound_audit = generate_timed_joint_trajectory(
        prepress_joints,
        pressed_joints,
        cartesian_speed_m_s=cartesian_speed_m_s,
        joint_velocity_scale=joint_velocity_scale,
        point_period_sec=point_period_sec,
        minimum_duration_sec=0.20,
        duration_margin=duration_margin,
    )
    inbound = reverse_timed_trajectory(outbound)
    outbound_duration = float(outbound[-1]["time_from_start_sec"])
    cycle = [dict(point) for point in outbound]
    for point in inbound[1:]:
        value = dict(point)
        value["time_from_start_sec"] = (
            outbound_duration + float(point["time_from_start_sec"])
        )
        cycle.append(value)
    validate_trajectory_payload(
        {
            "protocol": "piper_timed_trajectory_v1",
            "command": "validate",
            "joint_names": PIPER_JOINT_NAMES,
            "points": cycle,
        }
    )
    return cycle, TrajectoryAudit(
        duration_sec=2.0 * outbound_audit.duration_sec,
        point_count=len(cycle),
        cartesian_path_m=2.0 * outbound_audit.cartesian_path_m,
        cartesian_peak_speed_m_s=outbound_audit.cartesian_peak_speed_m_s,
        joint_peak_velocity_rad_s=outbound_audit.joint_peak_velocity_rad_s,
        joint_velocity_fraction=outbound_audit.joint_velocity_fraction,
    )


def generate_blended_press_inverse_return(
    prepress_joints: Any,
    pressed_joints: Any,
    inverse_return: list[dict[str, Any]],
    *,
    press_speed_m_s: float = 0.15,
    release_speed_m_s: float = 0.20,
    press_joint_velocity_scale: float = 0.75,
    return_joint_velocity_scale: float = 0.75,
    point_period_sec: float = 0.05,
    duration_margin: float = 1.02,
    _include_forward_segment: bool = True,
) -> tuple[list[dict[str, Any]], BlendedPressReturnAudit]:
    """Press, reverse once, pass prepress at speed, then rejoin inverse return.

    With ``_include_forward_segment=False``, ``pressed_joints`` is the live
    stopped pose after a guarded press. The returned trajectory starts there
    and smoothly passes through prepress into the supplied outward return.
    """
    prepress = _as_joints(prepress_joints, "prepress_joints")
    pressed = _as_joints(pressed_joints, "pressed_joints")
    validate_trajectory_payload(
        {
            "protocol": "piper_timed_trajectory_v1",
            "command": "validate",
            "joint_names": PIPER_JOINT_NAMES,
            "points": inverse_return,
        }
    )
    if not np.allclose(
        np.asarray(inverse_return[0]["positions"], dtype=float),
        prepress,
        atol=1e-8,
    ):
        raise ValueError("inverse return must start at prepress joints")
    if _include_forward_segment:
        outbound, outbound_audit = generate_timed_joint_trajectory(
            prepress,
            pressed,
            cartesian_speed_m_s=press_speed_m_s,
            joint_velocity_scale=press_joint_velocity_scale,
            point_period_sec=point_period_sec,
            minimum_duration_sec=0.20,
            duration_margin=duration_margin,
        )
        outbound_peak_speed = outbound_audit.cartesian_peak_speed_m_s
    else:
        outbound = []
        outbound_peak_speed = 0.0

    inverse_positions = np.asarray(
        [point["positions"] for point in inverse_return], dtype=float
    )
    inverse_xyz = forward_kinematics_batch(inverse_positions)[:, :3, 3]
    prepress_xyz = inverse_xyz[0]
    distances = np.linalg.norm(inverse_xyz - prepress_xyz, axis=1)
    candidates = [
        index for index in range(2, len(inverse_return) - 1)
        if 0.010 <= distances[index] <= 0.035
    ]
    if not candidates:
        candidates = list(range(2, min(len(inverse_return) - 1, 10)))
    allowed_velocity = PIPER_MAX_VELOCITIES * float(return_joint_velocity_scale)
    selected = None
    for join_index in candidates:
        join = inverse_return[join_index]
        join_position = np.asarray(join["positions"], dtype=float)
        join_velocity = np.asarray(join["velocities"], dtype=float)
        join_acceleration = np.asarray(join["accelerations"], dtype=float)
        _, release_audit = generate_timed_joint_trajectory(
            pressed,
            join_position,
            cartesian_speed_m_s=release_speed_m_s,
            cartesian_speed_cap_m_s=PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
            joint_velocity_scale=return_joint_velocity_scale,
            point_period_sec=point_period_sec,
            minimum_duration_sec=0.30,
            duration_margin=duration_margin,
        )
        pressed_xyz = forward_kinematics(pressed)[:3, 3]
        first_length = float(np.linalg.norm(pressed_xyz - prepress_xyz))
        second_length = max(float(distances[join_index]), 1e-4)
        waypoint_fraction = float(
            np.clip(first_length / (first_length + second_length), 0.45, 0.85)
        )
        duration = max(float(release_audit.duration_sec), 0.45)
        for _ in range(14):
            waypoint_time = waypoint_fraction * duration
            coefficients = _waypoint_boundary_coefficients(
                pressed,
                prepress,
                join_position,
                join_velocity,
                join_acceleration,
                waypoint_time,
                duration,
            )
            dense_time = np.linspace(0.0, duration, 2401)
            dense_q, dense_v, dense_a = _evaluate_vector_polynomial(
                coefficients, dense_time
            )
            if (
                np.any(dense_q < PIPER_JOINT_LIMITS[:, 0] - 1e-9)
                or np.any(dense_q > PIPER_JOINT_LIMITS[:, 1] + 1e-9)
            ):
                break
            dense_xyz = forward_kinematics_batch(dense_q)[:, :3, 3]
            dense_tcp_speed = np.linalg.norm(
                np.gradient(dense_xyz, dense_time, axis=0), axis=1
            )
            speed_violation = max(
                float(dense_tcp_speed.max()) / float(release_speed_m_s),
                float(np.max(np.abs(dense_v) / allowed_velocity)),
            )
            if speed_violation <= 1.0 + 1e-6:
                selected = (
                    join_index,
                    duration,
                    waypoint_time,
                    coefficients,
                    dense_tcp_speed,
                    dense_v,
                )
                break
            duration *= speed_violation * 1.03
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("could not blend press release into inverse return")

    join_index, blend_duration, waypoint_time, coefficients, blend_tcp, blend_v = selected
    blend_times = np.unique(
        np.concatenate(
            [
                np.arange(0.0, blend_duration, float(point_period_sec)),
                [waypoint_time, blend_duration],
            ]
        )
    )
    blend_q, blend_qd, blend_qdd = _evaluate_vector_polynomial(
        coefficients, blend_times
    )
    blend = [
        {
            "time_from_start_sec": float(stamp),
            "positions": q.tolist(),
            "velocities": qd.tolist(),
            "accelerations": qdd.tolist(),
        }
        for stamp, q, qd, qdd in zip(blend_times, blend_q, blend_qd, blend_qdd)
    ]
    press_duration = (
        float(outbound[-1]["time_from_start_sec"]) if outbound else 0.0
    )
    points = [dict(point) for point in outbound]
    for point in (blend[1:] if outbound else blend):
        value = dict(point)
        value["time_from_start_sec"] = press_duration + float(
            point["time_from_start_sec"]
        )
        points.append(value)
    inverse_join_time = float(inverse_return[join_index]["time_from_start_sec"])
    tail_offset = press_duration + blend_duration
    for point in inverse_return[join_index + 1 :]:
        value = dict(point)
        value["time_from_start_sec"] = tail_offset + (
            float(point["time_from_start_sec"]) - inverse_join_time
        )
        points.append(value)
    validate_trajectory_payload(
        {
            "protocol": "piper_timed_trajectory_v1",
            "command": "validate",
            "joint_names": PIPER_JOINT_NAMES,
            "points": points,
        }
    )
    combined_joints = np.asarray([point["positions"] for point in points])
    combined_times = np.asarray([point["time_from_start_sec"] for point in points])
    combined_xyz = forward_kinematics_batch(combined_joints)[:, :3, 3]
    combined_tcp = np.linalg.norm(
        np.gradient(combined_xyz, combined_times, axis=0), axis=1
    )
    combined_velocity = np.asarray([point["velocities"] for point in points])
    prepress_velocity = _evaluate_vector_polynomial(
        coefficients, np.asarray([waypoint_time])
    )[1][0]
    velocity_probe_sec = min(1e-4, waypoint_time / 10.0)
    probe_q = _evaluate_vector_polynomial(
        coefficients,
        np.asarray(
            [waypoint_time - velocity_probe_sec, waypoint_time + velocity_probe_sec]
        ),
    )[0]
    probe_xyz = forward_kinematics_batch(probe_q)[:, :3, 3]
    prepress_cartesian_speed = float(
        np.linalg.norm(probe_xyz[1] - probe_xyz[0]) / (2.0 * velocity_probe_sec)
    )
    return points, BlendedPressReturnAudit(
        duration_sec=float(points[-1]["time_from_start_sec"]),
        point_count=len(points),
        cartesian_path_m=float(
            np.sum(np.linalg.norm(np.diff(combined_xyz, axis=0), axis=1))
        ),
        cartesian_peak_speed_m_s=float(
            max(combined_tcp.max(), outbound_peak_speed)
        ),
        joint_peak_velocity_rad_s=np.max(np.abs(combined_velocity), axis=0),
        joint_velocity_fraction=(
            np.max(np.abs(combined_velocity), axis=0) / PIPER_MAX_VELOCITIES
        ),
        press_endpoint_time_sec=press_duration,
        prepress_pass_time_sec=press_duration + waypoint_time,
        prepress_joint_speed_rad_s=float(np.linalg.norm(prepress_velocity)),
        prepress_cartesian_speed_m_s=prepress_cartesian_speed,
        inverse_join_index=int(join_index),
        inverse_join_time_sec=float(tail_offset),
    )


def generate_blended_guarded_retract(
    current_joints: Any,
    prepress_joints: Any,
    retreat_to_observation: list[dict[str, Any]],
    *,
    release_speed_m_s: float = 0.25,
    return_joint_velocity_scale: float = 0.75,
    point_period_sec: float = 0.02,
    duration_margin: float = 1.02,
) -> tuple[list[dict[str, Any]], BlendedPressReturnAudit]:
    """Blend contact-triggered retreat continuously through prepress to hold.

    The guarded press cancels at a live encoder pose. This starts at that
    stopped pose, reverses through the known prepress waypoint, then rejoins
    the already planned retreat-to-observation trajectory without stopping at
    prepress or requiring a second trajectory action.
    """
    return generate_blended_press_inverse_return(
        prepress_joints,
        current_joints,
        retreat_to_observation,
        press_speed_m_s=release_speed_m_s,
        release_speed_m_s=release_speed_m_s,
        press_joint_velocity_scale=return_joint_velocity_scale,
        return_joint_velocity_scale=return_joint_velocity_scale,
        point_period_sec=point_period_sec,
        duration_margin=duration_margin,
        _include_forward_segment=False,
    )


class TimedTrajectoryUdpClient:
    """Compatible trajectory API, with bounded TCP for larger payloads."""
    def __init__(self, host: str, port: int = 5010, timeout_sec: float = 25.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)

    def execute(
        self,
        points: list[dict[str, Any]],
        *,
        token: str,
        task_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "protocol": "piper_timed_trajectory_v1",
            "command": "validate" if dry_run else "execute",
            "task_id": task_id or str(time.time_ns()),
            "token": token,
            "joint_names": PIPER_JOINT_NAMES,
            "points": points,
        }
        encoded, wire_payload, wire_decimals = encode_trajectory_transport_payload(payload)
        transport = 'udp' if len(encoded)<=PIPER_UDP_MAX_PAYLOAD_BYTES else 'tcp'
        if transport == 'tcp':
            health = self.health(token=token)
            if health.get('trajectory_tcp_v1') is not True:
                raise RuntimeError('bridge lacks bulk TCP trajectory support; no motion sent')
            try:
                with socket.create_connection((self.host,self.port),timeout=self.timeout_sec) as sock:
                    sock.settimeout(self.timeout_sec)
                    send_frame(sock,encoded)
                    data=receive_frame(sock,timeout_sec=self.timeout_sec)
            except (OSError,ConnectionError) as exc:
                message = ('TCP trajectory validation response unavailable; no motion requested' if dry_run
                           else 'TCP trajectory response unavailable; execution state unknown, do not automatically resend')
                raise RuntimeError(message) from exc
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.settimeout(self.timeout_sec)
                sock.sendto(encoded, (self.host, self.port))
                data, address = sock.recvfrom(65535)
            finally:
                sock.close()
            if address[0] != self.host:
                raise RuntimeError(f'trajectory response came from unexpected host {address[0]}')
        response = json.loads(data.decode('utf-8'))
        if response.get("task_id") != wire_payload["task_id"]:
            raise RuntimeError("trajectory response task_id mismatch")
        if response.get("status") not in ("validated", "complete"):
            raise RuntimeError(f"trajectory bridge failed: {response}")
        response["wire_payload_bytes"] = len(encoded)
        response["wire_decimal_places"] = wire_decimals
        response["wire_transport"] = transport
        response["wire_point_count"] = len(wire_payload["points"])
        return response

    def health(self, *, token: str) -> dict[str, Any]:
        """Fail fast when the single UDP bridge is alive but not servicing work."""
        task_id = f"trajectory-health-{time.time_ns()}"
        payload = {"command": "health", "task_id": task_id, "token": str(token)}
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(min(3.0, self.timeout_sec))
            sock.sendto(json.dumps(payload, separators=(",", ":")).encode(), (self.host, self.port))
            data, address = sock.recvfrom(65535)
        finally:
            sock.close()
        if address[0] != self.host:
            raise RuntimeError("trajectory health response came from an unexpected host")
        response = json.loads(data.decode())
        if response.get("task_id") != task_id or response.get("status") != "ok":
            raise RuntimeError(f"trajectory bridge health failed: {response}")
        return response
