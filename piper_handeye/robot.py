from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import pose_to_transform
from .pose_timing import PoseTimingTrace


PIPER_JOINT_LIMITS = np.array(
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


def extract_json(packet: bytes) -> dict[str, Any] | None:
    """Decode the UDP envelope before reading its JSON object.

    Sequence and length fields are binary: any byte can equal ``{``. Searching
    for that character can silently drop hundreds of consecutive valid states.
    Recognized envelopes must pass their length checks, without a scan fallback.
    """
    if packet.startswith(b"16"):
        if packet[2:4] not in (b"01", b"02", b"03"):
            return None
        prefix_size = 4
    elif packet.startswith(b"\x16"):
        if packet[1:2] not in (b"\x01", b"\x02", b"\x03"):
            return None
        prefix_size = 2
    else:
        prefix_size = 0

    if prefix_size:
        header_size = prefix_size + 12
        if len(packet) < header_size:
            return None
        _, topic_length, data_length = struct.unpack_from("<III", packet, prefix_size)
        data_start = header_size + topic_length
        if not topic_length or not data_length or data_start + data_length != len(packet):
            return None
        payload = packet[data_start:]
    else:
        # Preserve the existing plain-JSON and known textual-topic formats.
        # Arbitrary binary prefixes must never be searched for a JSON fragment.
        payload = packet
        for topic in (b"system_state", b"bt_result", b"bt_log"):
            if packet.startswith(topic):
                payload = packet[len(topic):]
                break
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class RobotSample:
    timestamp: float
    joints: np.ndarray
    base_to_gripper: np.ndarray
    raw: dict[str, Any]


class PiperUdpRobot:
    """Adapter for the RK3588 UDP bridge used by this Piper installation."""

    def __init__(self, config: dict[str, Any]):
        self.controller_ip = str(config["controller_ip"])
        self.controller_port = int(config.get("controller_port", 5001))
        self.local_ip = str(config["local_ip"])
        self.result_port = int(config.get("result_port", 5002))
        self.state_port = int(config.get("state_port", 5004))
        self.topic = str(config.get("topic", "bt_control"))
        self.header_mode = str(config.get("header_mode", "ascii"))
        self.task_timeout = float(config.get("task_timeout_sec", 45.0))
        self.state_timeout = float(config.get("state_timeout_sec", 3.0))
        self.limits = np.asarray(config.get("joint_limits_rad", PIPER_JOINT_LIMITS), dtype=float)
        if self.limits.shape != (6, 2):
            raise ValueError("robot.joint_limits_rad must have shape 6x2")

        self._result_socket: socket.socket | None = None
        self._state_socket: socket.socket | None = None
        self._running = False
        self._seq = 0
        # The RK3588 bridge keeps task results briefly.  Starting every client
        # at "1" can therefore collide with a result from an earlier process.
        # It still requires an integer-looking *string*, so use wall-clock
        # milliseconds as a process-unique numeric starting point.
        self._task = int(time.time_ns() // 1_000_000) % 1_000_000_000
        self._condition = threading.Condition()
        self._results: dict[tuple[str, str], int] = {}
        self._latest_state: RobotSample | None = None
        self._threads: list[threading.Thread] = []
        self._pose_timing = PoseTimingTrace()
        self._last_feedback_packet_time = None

    def __enter__(self) -> "PiperUdpRobot":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        if self._running:
            return
        self._result_socket = self._make_socket(self.result_port)
        self._state_socket = self._make_socket(self.state_port)
        self._running = True
        self._threads = [
            threading.Thread(target=self._result_loop, daemon=True),
            threading.Thread(target=self._state_loop, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def close(self) -> None:
        self._running = False
        for sock in (self._result_socket, self._state_socket):
            if sock is not None:
                sock.close()
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads = []
        self._result_socket = None
        self._state_socket = None

    def _make_socket(self, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.local_ip, port))
        sock.settimeout(0.25)
        return sock

    def _packet(self, message: dict[str, Any]) -> bytes:
        self._seq += 1
        topic = self.topic.encode("utf-8")
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self.header_mode == "ascii":
            prefix = b"16" + b"01"
        elif self.header_mode == "binary":
            prefix = b"\x16\x01"
        else:
            raise ValueError("robot.header_mode must be ascii or binary")
        return prefix + struct.pack("<III", self._seq, len(topic), len(data)) + topic + data

    def _send_tree(
        self, tree_name: str, settings: list[dict[str, str]], command: int = 1
    ) -> str:
        if self._result_socket is None:
            raise RuntimeError("robot adapter is not open")
        self._task += 1
        task_id = str(self._task)  # The installed mx_1.0.10 bridge requires a string.
        message = {
            "tree_name": tree_name,
            "command": int(command),
            "settings": settings,
            "task_id": task_id,
        }
        self._result_socket.sendto(
            self._packet(message), (self.controller_ip, self.controller_port)
        )
        return task_id

    def run_tree(self, tree_name: str, settings: list[dict[str, str]] | None = None) -> None:
        task_id = self._send_tree(tree_name, settings or [])
        result_key = (tree_name, task_id)
        deadline = time.monotonic() + self.task_timeout
        with self._condition:
            while self._running:
                result = self._results.get(result_key)
                if result == 1:
                    return
                if result in (0, 2):
                    raise RuntimeError(f"Piper task {tree_name!r} failed with result={result}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Piper task {tree_name!r} timed out")
                self._condition.wait(min(remaining, 0.5))
        raise RuntimeError("robot adapter closed while waiting for task")

    def cancel_tree(self, tree_name: str) -> None:
        """Request cancellation of a running behavior tree without starting motion."""
        self._send_tree(tree_name, [], command=0)

    def enable(self) -> None:
        self.run_tree("enable")

    def disable(self) -> None:
        self.run_tree("disable")

    def home(self) -> None:
        self.run_tree("move_joint_home")

    def move_joints(self, joints: list[float] | np.ndarray) -> None:
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("joint target must contain six finite radians")
        if np.any(joints < self.limits[:, 0]) or np.any(joints > self.limits[:, 1]):
            raise ValueError(f"joint target exceeds configured limits: {joints.tolist()}")
        settings = [
            {"key": f"piper_joint{i + 1}", "value": f"{value:.8f}"}
            for i, value in enumerate(joints)
        ]
        self.run_tree("move_joint", settings)

    def latest_state(self, max_age: float = 1.0) -> RobotSample:
        """Read the latest packet without waiting, for bounded confirmation loops."""
        with self._condition:
            if not self._running:
                raise RuntimeError("robot adapter is closed")
            value = self._latest_state
            if value is not None and 0 <= time.monotonic()-value.timestamp <= max_age:
                return value
        raise TimeoutError("latest Piper state is missing or stale")

    def state(self, max_age: float = 1.0) -> RobotSample:
        deadline = time.monotonic() + self.state_timeout
        with self._condition:
            while self._running:
                value = self._latest_state
                if value is not None and time.monotonic() - value.timestamp <= max_age:
                    return value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no fresh Piper state received on UDP {self.state_port}")
                self._condition.wait(min(remaining, 0.25))
        raise RuntimeError("robot adapter is closed")

    def _result_loop(self) -> None:
        assert self._result_socket is not None
        while self._running:
            try:
                packet, _ = self._result_socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            value = extract_json(packet)
            if not value or not {"tree_name", "task_id", "result"}.issubset(value):
                continue
            result = value["result"]
            if type(result) is int:
                with self._condition:
                    key = (str(value["tree_name"]), str(value["task_id"]))
                    self._results[key] = result
                    self._condition.notify_all()

    def _state_loop(self) -> None:
        assert self._state_socket is not None
        while self._running:
            try:
                packet, _ = self._state_socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            received = time.monotonic()
            previous = self._last_feedback_packet_time
            self._last_feedback_packet_time = received
            evidence = dict(received_monotonic=received,
                            receive_interval_sec=None if previous is None else received-previous)
            value = extract_json(packet)
            if not value or "joints_angle" not in value or "pose" not in value:
                self._pose_timing.record({**evidence, 'accepted': False,
                                          'reason': 'invalid_state_payload'}, slow=True)
                continue
            try:
                by_name = {item["key"]: float(item["value"]) for item in value["joints_angle"]}
                joints = np.array([by_name[f"piper_joint{i}"] for i in range(1, 7)])
                sample = RobotSample(time.monotonic(), joints, pose_to_transform(value["pose"]), value)
            except (KeyError, TypeError, ValueError):
                self._pose_timing.record({**evidence, 'accepted': False,
                                          'reason': 'invalid_state_values'}, slow=True)
                continue
            constructed = time.monotonic()
            with self._condition:
                acquired = time.monotonic()
                self._latest_state = sample
                self._condition.notify_all()
            self._pose_timing.record({**evidence, 'accepted': True,
                'sample_monotonic': sample.timestamp,
                'ready_monotonic': time.monotonic(),
                'decode_to_sample_stamp_sec': sample.timestamp-received,
                'decode_and_transform_sec': constructed-received,
                'store_lock_wait_sec': acquired-constructed},
                slow=previous is not None and received-previous > .1)
