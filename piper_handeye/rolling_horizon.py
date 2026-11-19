from __future__ import annotations

import hashlib
import json
import math
import socket
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .robot import PIPER_JOINT_LIMITS
    from .timed_trajectory import (
        PIPER_BRIDGE_VELOCITY_CAP_SCALE,
        PIPER_JOINT_NAMES,
        PIPER_MAX_VELOCITIES,
    )
except ImportError:  # Standalone copy deployed beside the OrangePi bridge.
    PIPER_JOINT_LIMITS = np.asarray(
        [
            [-2.6179, 2.6179],
            [0.0, 3.14],
            [-2.967, 0.03490658503988659],
            [-1.745, 1.745],
            [-1.22, 1.22],
            [-2.09439, 2.09439],
        ],
        dtype=float,
    )
    PIPER_JOINT_NAMES = [f"piper_joint{i}" for i in range(1, 7)]
    PIPER_MAX_VELOCITIES = np.asarray([0.6, 1.0, 1.0, 1.0, 1.0, 2.0])
    PIPER_BRIDGE_VELOCITY_CAP_SCALE = 0.75


ROLLING_HORIZON_PROTOCOL = "piper_rolling_horizon_v1"


@dataclass(frozen=True)
class RollingHorizonConfig:
    min_start_horizon_sec: float = 0.60
    low_horizon_sec: float = 0.30
    max_horizon_sec: float = 3.00
    append_timeout_sec: float = 0.50
    max_session_duration_sec: float = 15.0
    max_points_per_chunk: int = 100
    min_revision_lead_sec: float = 0.25
    position_tolerance_rad: float = 1e-6
    velocity_tolerance_rad_s: float = 1e-5
    acceleration_tolerance_rad_s2: float = 1e-4

    def __post_init__(self) -> None:
        if not 0.10 <= self.low_horizon_sec < self.min_start_horizon_sec:
            raise ValueError("low_horizon_sec must be below min_start_horizon_sec")
        if self.max_horizon_sec < self.min_start_horizon_sec:
            raise ValueError("max_horizon_sec must cover min_start_horizon_sec")
        if self.append_timeout_sec <= 0.0:
            raise ValueError("append_timeout_sec must be positive")
        if self.min_revision_lead_sec < 0.20:
            raise ValueError("min_revision_lead_sec must be at least 0.20 s")


@dataclass(frozen=True)
class AppendResult:
    sequence: int
    duplicate: bool
    accepted_new_points: int
    point_count: int
    tail_time_sec: float


@dataclass(frozen=True)
class ReviseResult:
    sequence: int
    duplicate: bool
    anchor_time_sec: float
    removed_points: int
    accepted_new_points: int
    point_count: int
    tail_time_sec: float


def _finite_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain six finite values")
    return result


def _canonical_chunk_digest(points: list[dict[str, Any]]) -> str:
    encoded = json.dumps(points, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_rolling_chunks(
    points: list[dict[str, Any]],
    *,
    initial_horizon_sec: float = 0.60,
    append_extension_sec: float = 0.20,
) -> list[list[dict[str, Any]]]:
    """Split one validated timeline into overlapping rolling chunks."""
    if len(points) < 2:
        raise ValueError("trajectory requires at least two points")
    if initial_horizon_sec <= 0.0 or append_extension_sec <= 0.0:
        raise ValueError("rolling horizons must be positive")
    times = np.asarray([float(point["time_from_start_sec"]) for point in points])
    if abs(times[0]) > 1e-9 or np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory must start at zero with increasing times")
    chunks: list[list[dict[str, Any]]] = []
    tail_index = int(np.searchsorted(times, initial_horizon_sec, side="left"))
    tail_index = min(max(1, tail_index), len(points) - 1)
    chunks.append(points[: tail_index + 1])
    while tail_index < len(points) - 1:
        target = times[tail_index] + append_extension_sec
        next_tail = int(np.searchsorted(times, target, side="left"))
        next_tail = min(max(tail_index + 1, next_tail), len(points) - 1)
        chunks.append(points[tail_index : next_tail + 1])
        tail_index = next_tail
    return chunks


class RollingHorizonBuffer:
    """Validate and buffer overlapping trajectory chunks on one absolute timeline.

    Chunk zero begins at t=0. Every later chunk repeats the prior tail as its
    first point. This overlap makes position, velocity, and acceleration
    continuity independently verifiable before any chunk can be accepted.
    """

    def __init__(
        self,
        session_id: str,
        *,
        config: RollingHorizonConfig | None = None,
        opened_monotonic: float | None = None,
    ) -> None:
        if not session_id or len(session_id) > 80:
            raise ValueError("invalid session_id")
        self.session_id = session_id
        self.config = config or RollingHorizonConfig()
        self.opened_monotonic = (
            time.monotonic() if opened_monotonic is None else float(opened_monotonic)
        )
        self.last_append_monotonic = self.opened_monotonic
        self.points: list[dict[str, Any]] = []
        self.expected_sequence = 0
        self._digests: dict[int, tuple[str, str]] = {}
        self.sealed = False
        self.aborted = False

    @property
    def tail_time_sec(self) -> float:
        return 0.0 if not self.points else float(self.points[-1]["time_from_start_sec"])

    def _validate_chunk(self, points: Any) -> list[dict[str, Any]]:
        if not isinstance(points, list) or not 2 <= len(points) <= self.config.max_points_per_chunk:
            raise ValueError(
                f"chunk requires 2..{self.config.max_points_per_chunk} points"
            )
        normalized: list[dict[str, Any]] = []
        previous_time = -math.inf
        previous_positions: np.ndarray | None = None
        velocity_cap = PIPER_MAX_VELOCITIES * PIPER_BRIDGE_VELOCITY_CAP_SCALE
        for index, point in enumerate(points):
            if not isinstance(point, dict):
                raise ValueError(f"point {index} is not an object")
            timestamp = float(point["time_from_start_sec"])
            positions = _finite_vector(point["positions"], f"point {index} positions")
            velocities = _finite_vector(point["velocities"], f"point {index} velocities")
            accelerations = _finite_vector(
                point["accelerations"], f"point {index} accelerations"
            )
            if not math.isfinite(timestamp) or timestamp < 0.0:
                raise ValueError(f"point {index} time is invalid")
            if timestamp <= previous_time:
                raise ValueError("chunk times must be strictly increasing")
            if timestamp > self.config.max_session_duration_sec:
                raise ValueError("rolling trajectory exceeds session duration")
            if np.any(positions < PIPER_JOINT_LIMITS[:, 0]) or np.any(
                positions > PIPER_JOINT_LIMITS[:, 1]
            ):
                raise ValueError(f"point {index} exceeds joint position limits")
            if np.any(np.abs(velocities) > velocity_cap + 1e-9):
                raise ValueError(f"point {index} declared velocity exceeds bridge cap")
            if previous_positions is not None:
                average_velocity = np.abs(positions - previous_positions) / (
                    timestamp - previous_time
                )
                if np.any(average_velocity > velocity_cap + 1e-9):
                    raise ValueError(
                        f"segment {index} average joint velocity exceeds bridge cap"
                    )
            normalized.append(
                {
                    "time_from_start_sec": timestamp,
                    "positions": positions.tolist(),
                    "velocities": velocities.tolist(),
                    "accelerations": accelerations.tolist(),
                }
            )
            previous_time = timestamp
            previous_positions = positions
        return normalized

    def append(
        self,
        sequence: int,
        points: Any,
        *,
        now_monotonic: float | None = None,
    ) -> AppendResult:
        if self.aborted:
            raise RuntimeError("session is aborted")
        if self.sealed:
            raise RuntimeError("session is sealed")
        sequence = int(sequence)
        normalized = self._validate_chunk(points)
        digest = _canonical_chunk_digest(normalized)
        if sequence < self.expected_sequence:
            if self._digests.get(sequence) != ("append", digest):
                raise ValueError("duplicate sequence has different content")
            return AppendResult(
                sequence, True, 0, len(self.points), self.tail_time_sec
            )
        if sequence != self.expected_sequence:
            raise ValueError(
                f"out-of-order chunk: expected {self.expected_sequence}, got {sequence}"
            )

        if not self.points:
            if abs(normalized[0]["time_from_start_sec"]) > 1e-9:
                raise ValueError("first rolling chunk must begin at time zero")
            new_points = normalized
        else:
            previous = self.points[-1]
            overlap = normalized[0]
            if abs(overlap["time_from_start_sec"] - self.tail_time_sec) > 1e-9:
                raise ValueError("chunk does not overlap the current tail time")
            checks = (
                ("positions", self.config.position_tolerance_rad),
                ("velocities", self.config.velocity_tolerance_rad_s),
                ("accelerations", self.config.acceleration_tolerance_rad_s2),
            )
            for field, tolerance in checks:
                error = float(
                    np.max(
                        np.abs(
                            np.asarray(overlap[field], dtype=float)
                            - np.asarray(previous[field], dtype=float)
                        )
                    )
                )
                if error > tolerance:
                    raise ValueError(f"chunk {field} discontinuity {error:.6g}")
            if normalized[-1]["time_from_start_sec"] - self.tail_time_sec > self.config.max_horizon_sec:
                raise ValueError("single append exceeds maximum horizon")
            new_points = normalized[1:]

        self.points.extend(new_points)
        self._digests[sequence] = ("append", digest)
        self.expected_sequence += 1
        self.last_append_monotonic = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        return AppendResult(
            sequence, False, len(new_points), len(self.points), self.tail_time_sec
        )

    def revise_tail(
        self,
        sequence: int,
        points: Any,
        *,
        playhead_sec: float,
        now_monotonic: float | None = None,
    ) -> ReviseResult:
        """Replace only the uncommitted future tail from an exact anchor point."""
        if self.aborted:
            raise RuntimeError("session is aborted")
        if self.sealed:
            raise RuntimeError("session is sealed")
        if not self.points:
            raise RuntimeError("cannot revise an empty session")
        sequence = int(sequence)
        normalized = self._validate_chunk(points)
        digest = _canonical_chunk_digest(normalized)
        anchor_time = float(normalized[0]["time_from_start_sec"])
        if sequence < self.expected_sequence:
            if self._digests.get(sequence) != ("revise", digest):
                raise ValueError("duplicate sequence has different content")
            return ReviseResult(
                sequence, True, anchor_time, 0, 0, len(self.points), self.tail_time_sec
            )
        if sequence != self.expected_sequence:
            raise ValueError(
                f"out-of-order chunk: expected {self.expected_sequence}, got {sequence}"
            )
        playhead = max(0.0, float(playhead_sec))
        if anchor_time < playhead + self.config.min_revision_lead_sec - 1e-9:
            raise ValueError("revision anchor is inside the frozen execution horizon")
        anchor_matches = [
            index for index, point in enumerate(self.points)
            if abs(float(point["time_from_start_sec"]) - anchor_time) <= 1e-9
        ]
        if len(anchor_matches) != 1:
            raise ValueError("revision anchor must match exactly one buffered point")
        anchor_index = anchor_matches[0]
        previous = self.points[anchor_index]
        overlap = normalized[0]
        checks = (
            ("positions", self.config.position_tolerance_rad),
            ("velocities", self.config.velocity_tolerance_rad_s),
            ("accelerations", self.config.acceleration_tolerance_rad_s2),
        )
        for field, tolerance in checks:
            error = float(
                np.max(
                    np.abs(
                        np.asarray(overlap[field], dtype=float)
                        - np.asarray(previous[field], dtype=float)
                    )
                )
            )
            if error > tolerance:
                raise ValueError(f"revision {field} discontinuity {error:.6g}")
        if normalized[-1]["time_from_start_sec"] - playhead > self.config.max_horizon_sec:
            raise ValueError("revised tail exceeds maximum buffered horizon")
        terminal_velocity = float(np.max(np.abs(normalized[-1]["velocities"])))
        terminal_acceleration = float(np.max(np.abs(normalized[-1]["accelerations"])))
        if terminal_velocity > self.config.velocity_tolerance_rad_s:
            raise ValueError("revised tail must end at zero velocity")
        if terminal_acceleration > self.config.acceleration_tolerance_rad_s2:
            raise ValueError("revised tail must end at zero acceleration")

        removed = len(self.points) - anchor_index - 1
        self.points[anchor_index + 1 :] = normalized[1:]
        self._digests[sequence] = ("revise", digest)
        self.expected_sequence += 1
        self.last_append_monotonic = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        return ReviseResult(
            sequence, False, anchor_time, removed, len(normalized) - 1,
            len(self.points), self.tail_time_sec,
        )

    def status(
        self,
        *,
        now_monotonic: float | None = None,
        playhead_sec: float = 0.0,
        started: bool = False,
    ) -> dict[str, Any]:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        playhead = max(0.0, float(playhead_sec))
        buffered = max(0.0, self.tail_time_sec - playhead)
        append_age = max(0.0, now - self.last_append_monotonic)
        return {
            "session_id": self.session_id,
            "expected_sequence": self.expected_sequence,
            "point_count": len(self.points),
            "tail_time_sec": self.tail_time_sec,
            "playhead_sec": playhead,
            "buffered_ahead_sec": buffered,
            "ready": bool(self.points and buffered >= self.config.min_start_horizon_sec),
            "low_horizon": bool(
                started and not self.sealed and buffered < self.config.low_horizon_sec
            ),
            "underrun": bool(
                started and not self.sealed and playhead >= self.tail_time_sec
            ),
            "append_stale": bool(
                started and not self.sealed
                and append_age > self.config.append_timeout_sec
            ),
            "append_age_sec": append_age,
            "sealed": self.sealed,
            "aborted": self.aborted,
        }

    def seal(self) -> None:
        if self.aborted:
            raise RuntimeError("session is aborted")
        if not self.points:
            raise RuntimeError("cannot seal an empty session")
        self.sealed = True

    def abort(self) -> None:
        self.aborted = True


class RollingHorizonUdpClient:
    def __init__(self, host: str, port: int = 5012, timeout_sec: float = 2.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)

    def request(self, command: str, *, token: str, **values: Any) -> dict[str, Any]:
        payload = {
            "protocol": ROLLING_HORIZON_PROTOCOL,
            "command": command,
            "task_id": str(time.time_ns()),
            "token": token,
        }
        payload.update(values)
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(encoded) > 60000:
            raise ValueError(f"rolling UDP payload is too large: {len(encoded)} bytes")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(self.timeout_sec)
            sock.sendto(encoded, (self.host, self.port))
            data, address = sock.recvfrom(65535)
        finally:
            sock.close()
        if address[0] != self.host:
            raise RuntimeError(f"response came from unexpected host {address[0]}")
        response = json.loads(data.decode("utf-8"))
        if response.get("task_id") != payload["task_id"]:
            raise RuntimeError("rolling response task_id mismatch")
        if response.get("status") == "error":
            raise RuntimeError(f"rolling bridge failed: {response}")
        return response

    def open(
        self, session_id: str, *, token: str, dry_run: bool = True
    ) -> dict[str, Any]:
        return self.request(
            "open", token=token, session_id=session_id, dry_run=bool(dry_run),
            joint_names=PIPER_JOINT_NAMES,
        )

    def append(
        self, session_id: str, sequence: int, points: list[dict[str, Any]], *, token: str
    ) -> dict[str, Any]:
        return self.request(
            "append", token=token, session_id=session_id, sequence=int(sequence),
            joint_names=PIPER_JOINT_NAMES, points=points,
        )

    def revise_tail(
        self, session_id: str, sequence: int, points: list[dict[str, Any]], *, token: str
    ) -> dict[str, Any]:
        return self.request(
            "revise_tail", token=token, session_id=session_id, sequence=int(sequence),
            joint_names=PIPER_JOINT_NAMES, points=points,
        )

    def status(self, session_id: str, *, token: str) -> dict[str, Any]:
        return self.request("status", token=token, session_id=session_id)

    def start_validation(self, session_id: str, *, token: str) -> dict[str, Any]:
        return self.request("start_validation", token=token, session_id=session_id)

    def start_execute(self, session_id: str, *, token: str) -> dict[str, Any]:
        return self.request("start_execute", token=token, session_id=session_id)

    def seal(self, session_id: str, *, token: str) -> dict[str, Any]:
        return self.request("seal", token=token, session_id=session_id)

    def abort(self, session_id: str, *, token: str) -> dict[str, Any]:
        return self.request("abort", token=token, session_id=session_id)
