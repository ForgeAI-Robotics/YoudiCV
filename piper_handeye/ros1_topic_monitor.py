"""Small read-only ROS 1 TCPROS monitors used by chassis diagnostics."""

from __future__ import annotations

import socket
import struct
import threading
import time
import xmlrpc.client
from typing import Any, Callable

from .ros1_twist import _encode_header, _read_exact, _server_proxy


def decode_twist_stamped(payload: bytes) -> dict[str, float]:
    if len(payload) < 64:
        raise ValueError("short TwistStamped payload")
    frame_size = struct.unpack_from("<I", payload, 12)[0]
    offset = 16 + frame_size
    values = struct.unpack_from("<6d", payload, offset)
    return {"linear_x": values[0], "angular_z": values[5]}


def decode_odometry(payload: bytes) -> dict[str, float]:
    if len(payload) < 12:
        raise ValueError("short Odometry payload")
    frame_size = struct.unpack_from("<I", payload, 12)[0]
    offset = 16 + frame_size
    child_size = struct.unpack_from("<I", payload, offset)[0]
    offset += 4 + child_size
    offset += 7 * 8 + 36 * 8  # pose plus pose covariance
    values = struct.unpack_from("<6d", payload, offset)
    return {"linear_x": values[0], "angular_z": values[5]}


def decode_lr_speed(payload: bytes) -> dict[str, float]:
    """Decode custom_msgs_srvs/LRSpeed (two float64 fields)."""
    if len(payload) < 16:
        raise ValueError("short LRSpeed payload")
    left, right = struct.unpack_from("<2d", payload, 0)
    return {"left_speed": left, "right_speed": right}


class Ros1TopicMonitor:
    """Continuously keep the latest decoded value from one ROS publisher."""

    def __init__(
        self,
        *,
        master_uri: str,
        topic: str,
        message_type: str,
        decoder: Callable[[bytes], dict[str, float]],
        preferred_publisher_node: str | None = None,
    ) -> None:
        self.master_uri = master_uri
        self.topic = topic
        self.message_type = message_type
        self.decoder = decoder
        self.preferred_publisher_node = preferred_publisher_node
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._latest: dict[str, Any] | None = None
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._run, name=f"ros-monitor-{topic}", daemon=True
        )
        self._thread.start()

    def _publisher_node(self) -> str:
        master = _server_proxy(self.master_uri)
        code, message, state = master.getSystemState("/dualsense_chassis_monitor")
        if int(code) != 1:
            raise ConnectionError(message)
        for candidate, nodes in state[0]:
            if candidate == self.topic and nodes:
                if self.preferred_publisher_node is not None:
                    if self.preferred_publisher_node in nodes:
                        return self.preferred_publisher_node
                    raise ConnectionError(
                        f"preferred publisher {self.preferred_publisher_node} "
                        f"is absent for {self.topic}; publishers={nodes}"
                    )
                return str(nodes[0])
        raise ConnectionError(f"no publisher for {self.topic}")

    def _connect(self) -> socket.socket:
        master = _server_proxy(self.master_uri)
        node_name = self._publisher_node()
        code, message, uri = master.lookupNode(
            "/dualsense_chassis_monitor", node_name
        )
        if int(code) != 1:
            raise ConnectionError(message)
        node = _server_proxy(str(uri))
        code, message, protocol = node.requestTopic(
            "/dualsense_chassis_monitor", self.topic, [["TCPROS"]]
        )
        if int(code) != 1 or not protocol or protocol[0] != "TCPROS":
            raise ConnectionError(message)
        sock = socket.create_connection((protocol[1], int(protocol[2])), timeout=2.0)
        sock.settimeout(1.0)
        sock.sendall(_encode_header({
            "callerid": "/dualsense_chassis_monitor",
            "topic": self.topic,
            "type": self.message_type,
            "md5sum": "*",
            "tcp_nodelay": "1",
        }))
        header_size = struct.unpack("<I", _read_exact(sock, 4))[0]
        _read_exact(sock, header_size)
        return sock

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sock = self._connect()
                self._socket = sock
                with self._lock:
                    self._error = None
                while not self._stop.is_set():
                    try:
                        size = struct.unpack("<I", _read_exact(sock, 4))[0]
                        payload = _read_exact(sock, size)
                    except socket.timeout:
                        continue
                    value = self.decoder(payload)
                    with self._lock:
                        self._latest = {
                            **value,
                            "age_sec": 0.0,
                            "received_monotonic": time.monotonic(),
                        }
            except Exception as exc:
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                if not self._stop.wait(0.5):
                    continue
            finally:
                if self._socket is not None:
                    try:
                        self._socket.close()
                    except OSError:
                        pass
                    self._socket = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._latest is None:
                return {"available": False, "error": self._error}
            value = dict(self._latest)
            value["age_sec"] = round(
                time.monotonic() - float(value.pop("received_monotonic")), 4
            )
            value["available"] = True
            value["error"] = self._error
            return value

    def close(self) -> None:
        self._stop.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self._thread.join(timeout=2.0)
