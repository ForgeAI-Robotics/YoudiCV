"""Pure-numpy trajectory rejoin used by OrangePi bridge processes."""

from __future__ import annotations

import math

import numpy as np


JOINT_LIMITS = np.asarray(
    [
        [-2.6179, 2.6179], [0.0, 3.14], [-2.967, 0.03490658503988659],
        [-1.745, 1.745], [-1.22, 1.22], [-2.09439, 2.09439],
    ],
    dtype=float,
)


GOAL_TOLERANCE_VIOLATED = -5
NEAR_GOAL_ACCEPTANCE_RAD = 0.003


def accept_near_goal_tolerance_violation(error_code, end_error_rad):
    """Accept only the controller's tolerance code at a measured near-goal pose."""
    return (
        error_code == GOAL_TOLERANCE_VIOLATED
        and np.isfinite(end_error_rad)
        and float(end_error_rad) <= NEAR_GOAL_ACCEPTANCE_RAD
    )


def _coefficients(q0, q1, v1, a1, duration):
    coefficients = np.zeros((6, 6), dtype=float)
    coefficients[0] = q0
    matrix = np.asarray(
        [
            [duration**3, duration**4, duration**5],
            [3 * duration**2, 4 * duration**3, 5 * duration**4],
            [6 * duration, 12 * duration**2, 20 * duration**3],
        ]
    )
    coefficients[3:] = np.linalg.solve(matrix, np.vstack([q1 - q0, v1, a1]))
    return coefficients


def _evaluate(coefficients, values):
    powers = np.vstack([values**index for index in range(6)])
    q = powers.T @ coefficients
    qd = (
        np.vstack([values**index for index in range(5)]).T
        @ np.asarray([index * coefficients[index] for index in range(1, 6)])
    )
    qdd = (
        np.vstack([values**index for index in range(4)]).T
        @ np.asarray(
            [index * (index - 1) * coefficients[index] for index in range(2, 6)]
        )
    )
    return q, qd, qdd


def generate_rejoin_trajectory(
    points,
    current_joints,
    *,
    elapsed_sec=0.0,
    max_velocities,
    max_accelerations,
    point_period_sec=0.05,
    lookahead_sec=0.25,
):
    """Blend from a zero-speed hold into the unexecuted original path."""
    current = np.asarray(current_joints, dtype=float)
    positions = np.asarray([point["positions"] for point in points], dtype=float)
    times = np.asarray([point["time_from_start_sec"] for point in points], dtype=float)
    if current.shape != (6,) or positions.ndim != 2 or positions.shape[1] != 6:
        raise ValueError("trajectory rejoin requires six joints")
    eligible = np.flatnonzero(times >= max(0.0, float(elapsed_sec) - 0.10))
    if not len(eligible):
        eligible = np.asarray([len(points) - 1])
    normalized = (positions[eligible] - current) / np.asarray(max_velocities)
    nearest = int(eligible[np.argmin(np.linalg.norm(normalized, axis=1))])
    join_target_time = times[nearest] + float(lookahead_sec)
    join_index = int(np.searchsorted(times, join_target_time, side="left"))
    join_index = min(max(nearest + 1, join_index), len(points) - 1)
    join = points[join_index]
    q1 = np.asarray(join["positions"], dtype=float)
    v1 = np.asarray(join["velocities"], dtype=float)
    a1 = np.asarray(join["accelerations"], dtype=float)
    duration = max(0.25, float(times[join_index] - times[nearest]))
    dense = np.linspace(0.0, 1.0, 201)
    for _ in range(40):
        coefficients = _coefficients(current, q1, v1, a1, duration)
        q, qd, qdd = _evaluate(coefficients, dense * duration)
        if (
            np.all(q >= JOINT_LIMITS[:, 0] - 1e-9)
            and np.all(q <= JOINT_LIMITS[:, 1] + 1e-9)
            and np.all(np.abs(qd) <= np.asarray(max_velocities) + 1e-9)
            and np.all(np.abs(qdd) <= np.asarray(max_accelerations) + 1e-9)
        ):
            break
        duration *= 1.15
    else:
        raise RuntimeError("could not generate a bounded safety rejoin")
    sample_count = max(2, int(math.ceil(duration / point_period_sec)) + 1)
    connector_times = np.linspace(0.0, duration, sample_count)
    q, qd, qdd = _evaluate(coefficients, connector_times)
    resumed = [
        {
            "time_from_start_sec": float(stamp),
            "positions": position.tolist(),
            "velocities": velocity.tolist(),
            "accelerations": acceleration.tolist(),
        }
        for stamp, position, velocity, acceleration in zip(
            connector_times, q, qd, qdd
        )
    ]
    offset = duration - times[join_index]
    for point in points[join_index + 1 :]:
        value = dict(point)
        value["time_from_start_sec"] = float(point["time_from_start_sec"] + offset)
        resumed.append(value)
    return resumed, {
        "nearest_index": nearest,
        "join_index": join_index,
        "connector_duration_sec": float(duration),
        "remaining_point_count": len(resumed),
        "start_error_rad": float(np.max(np.abs(q[0] - current))),
        "join_position_error_rad": float(np.max(np.abs(q[-1] - q1))),
        "join_velocity_error_rad_s": float(np.max(np.abs(qd[-1] - v1))),
        "peak_connector_velocity_rad_s": np.max(np.abs(qd), axis=0).tolist(),
        "peak_connector_acceleration_rad_s2": np.max(np.abs(qdd), axis=0).tolist(),
    }
