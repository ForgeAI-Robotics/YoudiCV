"""Authenticated UDP client for the OrangePi collision-checked joint bridge."""

from __future__ import annotations

import json
import socket
import time
from typing import Any


class ArmGamepadUdpClient:
    def __init__(self, host: str, port: int, token: str, timeout_sec: float = 1.0):
        self.address = (str(host), int(port))
        self.token = str(token)
        self.sequence = time.time_ns()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(float(timeout_sec))
        self.enabled = False

    def _request(
        self, command: str, *, timeout_sec: float | None = None, **values: Any
    ) -> dict[str, Any]:
        self.sequence += 1
        request = {
            "command": command,
            "token": self.token,
            "seq": self.sequence,
            **values,
        }
        previous_timeout = self.sock.gettimeout()
        if timeout_sec is not None:
            self.sock.settimeout(float(timeout_sec))
        try:
            self.sock.sendto(
                json.dumps(request, separators=(",", ":")).encode("utf-8"),
                self.address,
            )
            response, _ = self.sock.recvfrom(65535)
        finally:
            if timeout_sec is not None:
                self.sock.settimeout(previous_timeout)
        decoded = json.loads(response.decode("utf-8"))
        if not decoded.get("ok"):
            raise RuntimeError(str(decoded.get("error", "arm bridge rejected request")))
        return decoded

    def enable(self) -> dict[str, Any]:
        response = self._request("enable", timeout_sec=3.0)
        self.enabled = True
        return response

    def update_joints(self, *, velocities_rad_s: tuple[float, ...]) -> dict[str, Any]:
        return self._request(
            "update_joints",
            velocities_rad_s=[float(value) for value in velocities_rad_s],
        )

    def toggle_clamp(self, *, timeout_sec: float = 4.5) -> dict[str, Any]:
        return self._request("toggle_clamp", timeout_sec=timeout_sec)

    def close_clamp(self, *, timeout_sec: float = 4.5) -> dict[str, Any]:
        """Command a known closed state; never toggle based on cached state."""
        return self._request("close_clamp", timeout_sec=timeout_sec)

    def recover_top_chassis_boundary(self) -> dict[str, Any]:
        """Run the OrangePi's strictly bounded, full-mesh J2/J3 recovery."""
        return self._request(
            "recover_top_chassis_boundary", timeout_sec=30.0
        )

    def disable(self) -> dict[str, Any] | None:
        try:
            return self._request("disable", timeout_sec=3.0)
        finally:
            self.enabled = False

    def status(self) -> dict[str, Any]:
        return self._request("status")

    def live_joints(self, *, max_age_sec: float = 0.25) -> list[float]:
        status = self.status()["status"]
        joints = status.get("actual_joints_rad")
        age = status.get("joint_feedback_age_sec")
        if not isinstance(joints, list) or len(joints) != 6:
            raise RuntimeError("live arm encoder feedback is unavailable")
        if age is None or float(age) > float(max_age_sec):
            raise RuntimeError(f"live arm encoder feedback is stale: age={age!r}")
        return [float(value) for value in joints]

    def close(self) -> None:
        try:
            if self.enabled:
                self.disable()
        finally:
            self.sock.close()
