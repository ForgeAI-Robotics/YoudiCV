from __future__ import annotations

from dataclasses import dataclass
import json
import math
import socket
import time
from typing import Any

import numpy as np

from .kinematics import forward_kinematics_batch
from .robot import PIPER_JOINT_LIMITS
from .pipeline_timing import traced
from .timed_trajectory import (
    PIPER_BRIDGE_VELOCITY_CAP_SCALE,
    PIPER_CARTESIAN_SPEED_CAP_M_S,
    PIPER_MAX_VELOCITIES,
    validate_trajectory_payload,
)


@dataclass(frozen=True)
class HandoffState:
    active_index: int
    active_time_sec: float
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray


@dataclass(frozen=True)
class SpliceAudit:
    duration_sec: float
    point_count: int
    cartesian_path_m: float
    cartesian_peak_speed_m_s: float
    end_zone_distance_m: float | None
    end_zone_speed_limit_m_s: float | None
    end_zone_peak_speed_m_s: float | None
    joint_peak_velocity_rad_s: np.ndarray
    start_position_error_rad: float
    start_velocity_error_rad_s: float
    start_acceleration_error_rad_s2: float
    encoded_payload_bytes: int
    cartesian_chord_m: float
    cartesian_max_overshoot_m: float
    cartesian_max_backtrack_step_m: float
    cartesian_max_cross_track_m: float
    joint_max_overshoot_rad: float


def select_future_handoff(
    active_points: list[dict[str, Any]],
    *,
    active_elapsed_sec: float,
    lead_sec: float = 0.50,
) -> HandoffState:
    if lead_sec < 0.20:
        raise ValueError("handoff lead must be at least 0.20 s")
    target_time = float(active_elapsed_sec) + float(lead_sec)
    for index, point in enumerate(active_points):
        if float(point["time_from_start_sec"]) >= target_time:
            return HandoffState(
                active_index=index,
                active_time_sec=float(point["time_from_start_sec"]),
                positions=np.asarray(point["positions"], dtype=float),
                velocities=np.asarray(point["velocities"], dtype=float),
                accelerations=np.asarray(point["accelerations"], dtype=float),
            )
    raise RuntimeError("active trajectory has no point after requested handoff")


def _quintic_coefficients(
    start_q: np.ndarray,
    start_v: np.ndarray,
    start_a: np.ndarray,
    end_q: np.ndarray,
    duration: float,
) -> np.ndarray:
    coefficients = np.zeros((6, 6), dtype=float)
    coefficients[0] = start_q
    coefficients[1] = start_v
    coefficients[2] = 0.5 * start_a
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
        [end_q - known_position, -known_velocity, -known_acceleration]
    )
    coefficients[3:] = np.linalg.solve(matrix, rhs)
    return coefficients


def _evaluate(coefficients: np.ndarray, times: np.ndarray):
    powers = np.vstack([times**index for index in range(6)])
    positions = powers.T @ coefficients
    velocity_coefficients = np.asarray(
        [index * coefficients[index] for index in range(1, 6)]
    )
    velocities = np.vstack([times**index for index in range(5)]).T @ velocity_coefficients
    acceleration_coefficients = np.asarray(
        [index * (index - 1) * coefficients[index] for index in range(2, 6)]
    )
    accelerations = (
        np.vstack([times**index for index in range(4)]).T
        @ acceleration_coefficients
    )
    return positions, velocities, accelerations


def generate_splice_trajectory(
    handoff: HandoffState,
    end_joints: Any,
    *,
    cartesian_speed_m_s: float = 0.02,
    joint_velocity_scale: float = 0.25,
    point_period_sec: float = 0.10,
    maximum_duration_sec: float = 15.0,
    end_zone_distance_m: float | None = None,
    end_zone_speed_m_s: float | None = None,
    maximum_cartesian_overshoot_m: float = 0.002,
    maximum_cartesian_backtrack_step_m: float = 0.0005,
    maximum_cartesian_cross_track_m: float = 0.025,
    maximum_joint_overshoot_rad: float = 0.020,
) -> tuple[list[dict[str, Any]], SpliceAudit]:
    end = np.asarray(end_joints, dtype=float)
    if end.shape != (6,) or not np.isfinite(end).all():
        raise ValueError("end_joints must contain six finite radians")
    if np.any(end < PIPER_JOINT_LIMITS[:, 0]) or np.any(
        end > PIPER_JOINT_LIMITS[:, 1]
    ):
        raise ValueError("end_joints exceeds position limits")
    if not 0.005 <= cartesian_speed_m_s <= PIPER_CARTESIAN_SPEED_CAP_M_S:
        raise ValueError("cartesian speed must be in 0.005..0.15 m/s")
    if not 0.05 <= joint_velocity_scale <= PIPER_BRIDGE_VELOCITY_CAP_SCALE:
        raise ValueError("joint velocity scale must be in 0.05..0.75")
    if not 0.01 <= point_period_sec <= 0.10:
        raise ValueError("point period must be in 0.01..0.10 s")
    if (end_zone_distance_m is None) != (end_zone_speed_m_s is None):
        raise ValueError("end-zone distance and speed must be specified together")
    geometric_limits = (
        maximum_cartesian_overshoot_m,
        maximum_cartesian_backtrack_step_m,
        maximum_cartesian_cross_track_m,
        maximum_joint_overshoot_rad,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in geometric_limits):
        raise ValueError("splice geometric limits must be finite and non-negative")
    if end_zone_distance_m is not None:
        if not 0.01 <= float(end_zone_distance_m) <= 0.10:
            raise ValueError("end-zone distance must be in 0.01..0.10 m")
        if not 0.005 <= float(end_zone_speed_m_s) <= cartesian_speed_m_s:
            raise ValueError(
                "end-zone speed must be in 0.005..cartesian speed m/s"
            )

    allowed_joint_velocity = PIPER_MAX_VELOCITIES * joint_velocity_scale
    duration = max(
        0.5,
        float(
            np.max(
                np.divide(
                    np.abs(end - handoff.positions),
                    allowed_joint_velocity,
                    out=np.zeros(6),
                    where=allowed_joint_velocity > 0,
                )
            )
        )
        * 2.2,
    )
    dense_count = 1201
    for _ in range(12):
        if duration > maximum_duration_sec:
            raise RuntimeError(
                "splice trajectory requires more than "
                f"{float(maximum_duration_sec):g} seconds"
            )
        coefficients = _quintic_coefficients(
            handoff.positions,
            handoff.velocities,
            handoff.accelerations,
            end,
            duration,
        )
        dense_times = np.linspace(0.0, duration, dense_count)
        dense_q, dense_v, _ = _evaluate(coefficients, dense_times)
        if np.any(dense_q < PIPER_JOINT_LIMITS[:, 0]) or np.any(
            dense_q > PIPER_JOINT_LIMITS[:, 1]
        ):
            raise RuntimeError("splice polynomial overshoots a joint position limit")
        joint_lower = np.minimum(handoff.positions, end)
        joint_upper = np.maximum(handoff.positions, end)
        joint_overshoot = float(max(
            np.max(joint_lower[None, :] - dense_q),
            np.max(dense_q - joint_upper[None, :]),
            0.0,
        ))
        if joint_overshoot > float(maximum_joint_overshoot_rad):
            raise RuntimeError(
                "splice reverses a joint beyond its endpoint corridor: "
                f"{joint_overshoot:.6f} rad"
            )
        joint_peak = np.max(np.abs(dense_v), axis=0)
        tcp_positions = forward_kinematics_batch(dense_q)[:, :3, 3]
        chord = tcp_positions[-1] - tcp_positions[0]
        chord_length = float(np.linalg.norm(chord))
        if chord_length <= 1e-6:
            raise RuntimeError("splice Cartesian endpoints are indistinguishable")
        chord_direction = chord / chord_length
        relative_tcp = tcp_positions - tcp_positions[0]
        along_chord = relative_tcp @ chord_direction
        cross_track = np.linalg.norm(
            relative_tcp - along_chord[:, None] * chord_direction[None, :], axis=1
        )
        cartesian_overshoot = float(max(
            -float(np.min(along_chord)),
            float(np.max(along_chord)) - chord_length,
            0.0,
        ))
        cartesian_backtrack = float(max(0.0, -float(np.min(np.diff(along_chord)))))
        cartesian_cross_track = float(np.max(cross_track))
        if cartesian_overshoot > float(maximum_cartesian_overshoot_m):
            raise RuntimeError(
                "splice TCP overshoots its endpoint corridor: "
                f"{cartesian_overshoot:.6f} m"
            )
        if cartesian_backtrack > float(maximum_cartesian_backtrack_step_m):
            raise RuntimeError(
                "splice TCP backtracks toward its start: "
                f"{cartesian_backtrack:.6f} m per audit step"
            )
        if cartesian_cross_track > float(maximum_cartesian_cross_track_m):
            raise RuntimeError(
                "splice TCP leaves its direct-path corridor: "
                f"{cartesian_cross_track:.6f} m"
            )
        tcp_velocity = np.linalg.norm(
            np.gradient(tcp_positions, dense_times, axis=0), axis=1
        )
        tcp_peak = float(tcp_velocity.max())
        end_zone_peak = None
        if end_zone_distance_m is not None:
            distance_to_end = np.linalg.norm(
                tcp_positions - tcp_positions[-1], axis=1
            )
            zone_mask = distance_to_end <= float(end_zone_distance_m)
            end_zone_peak = float(np.max(tcp_velocity[zone_mask]))
        scale = max(
            tcp_peak / float(cartesian_speed_m_s),
            float(np.max(joint_peak / allowed_joint_velocity)),
            (
                0.0
                if end_zone_peak is None
                else end_zone_peak / float(end_zone_speed_m_s)
            ),
        )
        if scale <= 1.0 + 1e-6:
            break
        duration *= scale * 1.05
    else:
        raise RuntimeError("splice duration scaling did not converge")

    point_count = max(2, int(math.ceil(duration / point_period_sec)) + 1)
    times = np.linspace(0.0, duration, point_count)
    positions, velocities, accelerations = _evaluate(coefficients, times)
    points = [
        {
            "time_from_start_sec": float(stamp),
            "positions": q.tolist(),
            "velocities": qd.tolist(),
            "accelerations": qdd.tolist(),
        }
        for stamp, q, qd, qdd in zip(times, positions, velocities, accelerations)
    ]
    payload = {
        "protocol": "piper_timed_trajectory_v1",
        "command": "validate",
        "task_id": "splice-size-audit",
        "token": "redacted",
        "joint_names": [f"piper_joint{index}" for index in range(1, 7)],
        "points": points,
    }
    validate_trajectory_payload(payload)
    encoded_size = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    if encoded_size > 60000:
        raise RuntimeError(f"splice payload is too large: {encoded_size} bytes")
    path_length = float(
        np.sum(np.linalg.norm(np.diff(tcp_positions, axis=0), axis=1))
    )
    audit = SpliceAudit(
        duration_sec=float(duration),
        point_count=point_count,
        cartesian_path_m=path_length,
        cartesian_peak_speed_m_s=tcp_peak,
        end_zone_distance_m=(
            None if end_zone_distance_m is None else float(end_zone_distance_m)
        ),
        end_zone_speed_limit_m_s=(
            None if end_zone_speed_m_s is None else float(end_zone_speed_m_s)
        ),
        end_zone_peak_speed_m_s=end_zone_peak,
        joint_peak_velocity_rad_s=joint_peak,
        start_position_error_rad=float(
            np.max(np.abs(positions[0] - handoff.positions))
        ),
        start_velocity_error_rad_s=float(
            np.max(np.abs(velocities[0] - handoff.velocities))
        ),
        start_acceleration_error_rad_s2=float(
            np.max(np.abs(accelerations[0] - handoff.accelerations))
        ),
        encoded_payload_bytes=encoded_size,
        cartesian_chord_m=chord_length,
        cartesian_max_overshoot_m=cartesian_overshoot,
        cartesian_max_backtrack_step_m=cartesian_backtrack,
        cartesian_max_cross_track_m=cartesian_cross_track,
        joint_max_overshoot_rad=joint_overshoot,
    )
    return points, audit


def generate_first_feasible_splice(
    active_points: list[dict[str, Any]],
    end_joints: Any,
    *,
    earliest_handoff_time_sec: float,
    **trajectory_options: Any,
) -> tuple[HandoffState, list[dict[str, Any]], SpliceAudit, list[dict[str, Any]]]:
    """Choose the earliest future active state that yields a valid C2 splice.

    A handoff near the end of a quintic can still carry substantial
    acceleration.  Preserving that acceleration while also imposing a slow
    final zone may require an invalid, very long replacement.  Search the
    already validated future states instead of treating that local numerical
    incompatibility as a motion failure.  The active trajectory's zero-speed
    guard hold provides the bounded final fallback.
    """
    earliest = float(earliest_handoff_time_sec)
    if not math.isfinite(earliest) or earliest < 0.0:
        raise ValueError("earliest handoff time must be finite and non-negative")
    failures: list[dict[str, Any]] = []
    for index, point in enumerate(active_points):
        stamp = float(point["time_from_start_sec"])
        if stamp + 1e-9 < earliest:
            continue
        handoff = HandoffState(
            active_index=index,
            active_time_sec=stamp,
            positions=np.asarray(point["positions"], dtype=float),
            velocities=np.asarray(point["velocities"], dtype=float),
            accelerations=np.asarray(point["accelerations"], dtype=float),
        )
        try:
            points, audit = generate_splice_trajectory(
                handoff, end_joints, **trajectory_options
            )
            return handoff, points, audit, failures
        except RuntimeError as exc:
            failures.append({
                "active_index": int(index),
                "active_time_sec": stamp,
                "velocity_norm_rad_s": float(np.linalg.norm(handoff.velocities)),
                "acceleration_norm_rad_s2": float(
                    np.linalg.norm(handoff.accelerations)
                ),
                "reason": str(exc),
            })
    raise RuntimeError(
        "no feasible C2 splice remains after the required planning lead; "
        f"candidate failures={failures}"
    )


class PreparedSpliceUdpClient:
    """Upload separately bounded trajectories, validate, and execute a splice."""

    def __init__(self, host: str, port: int = 5011, timeout_sec: float = 40.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)

    def _request(self, payload: dict[str, Any], timeout_sec: float | None = None):
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(encoded) > 60000:
            raise ValueError(f"prepared-splice UDP payload is too large: {len(encoded)}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(self.timeout_sec if timeout_sec is None else timeout_sec)
            sock.sendto(encoded, (self.host, self.port))
            data, address = sock.recvfrom(65535)
        finally:
            sock.close()
        if address[0] != self.host:
            raise RuntimeError(f"splice response came from unexpected host {address[0]}")
        response = json.loads(data.decode("utf-8"))
        if response.get("task_id") != payload["task_id"]:
            raise RuntimeError("splice response task_id mismatch")
        if response.get("status") == "failed":
            raise RuntimeError(f"prepared splice bridge failed: {response}")
        return response

    @traced('trajectory.prepare_splice', 'validation')
    def prepare(self, points, *, trajectory_id: str, token: str):
        validation = {
            "protocol": "piper_timed_trajectory_v1",
            "command": "validate",
            "joint_names": [f"piper_joint{index}" for index in range(1, 7)],
            "points": points,
        }
        validate_trajectory_payload(validation)
        return self._request(
            {
                "protocol": "piper_prepared_splice_v1",
                "command": "prepare",
                "task_id": f"prepare-{trajectory_id}-{time.time_ns()}",
                "trajectory_id": str(trajectory_id),
                "token": str(token),
                "joint_names": validation["joint_names"],
                "points": points,
            },
            timeout_sec=5.0,
        )

    def splice(
        self,
        *,
        active_id: str,
        replacement_id: str,
        handoff_active_time_sec: float,
        token: str,
        execute: bool = False,
    ):
        command = "execute_splice" if execute else "validate_splice"
        return self._request(
            {
                "protocol": "piper_prepared_splice_v1",
                "command": command,
                "task_id": f"{command}-{time.time_ns()}",
                "active_id": str(active_id),
                "replacement_id": str(replacement_id),
                "handoff_active_time_sec": float(handoff_active_time_sec),
                "token": str(token),
            },
            timeout_sec=self.timeout_sec if execute else 5.0,
        )

    def start_active(self, *, active_id: str, token: str):
        """Start a prepared fallback trajectory and return immediately."""
        return self._request(
            {
                "protocol": "piper_prepared_splice_v1",
                "command": "start_active",
                "task_id": f"start-active-{time.time_ns()}",
                "active_id": str(active_id),
                "token": str(token),
            },
            timeout_sec=5.0,
        )

    def schedule_replacement(
        self,
        *,
        active_id: str,
        replacement_id: str,
        handoff_active_time_sec: float,
        token: str,
    ):
        """Schedule a prepared replacement while the fallback is moving."""
        return self._request(
            {
                "protocol": "piper_prepared_splice_v1",
                "command": "schedule_replacement",
                "task_id": f"schedule-replacement-{time.time_ns()}",
                "active_id": str(active_id),
                "replacement_id": str(replacement_id),
                "handoff_active_time_sec": float(handoff_active_time_sec),
                "token": str(token),
            },
            timeout_sec=5.0,
        )

    def execution_status(self, *, token: str):
        """Return the current asynchronous execution state without blocking."""
        return self._request(
            {
                "protocol": "piper_prepared_splice_v1",
                "command": "execution_status",
                "task_id": f"execution-status-{time.time_ns()}",
                "token": str(token),
            },
            timeout_sec=5.0,
        )

    def abort_active(self, *, token: str, active_id: str | None = None):
        """Discard a stopped asynchronous trajectory before recovery unlocks."""
        payload = {
            "protocol": "piper_prepared_splice_v1",
            "command": "abort_active",
            "task_id": f"abort-active-{time.time_ns()}",
            "token": str(token),
        }
        if active_id is not None:
            payload["active_id"] = str(active_id)
        return self._request(payload, timeout_sec=5.0)
