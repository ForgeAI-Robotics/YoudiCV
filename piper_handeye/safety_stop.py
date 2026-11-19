from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any


def safety_status_requires_stop(status: dict) -> bool:
    """Honor monitor-side debounce while remaining fail-safe for old payloads."""
    if "blocked" in status:
        return bool(status["blocked"])
    return not bool(status.get("ready"))


def safety_status_is_allowed_degraded_commit(status: dict) -> bool:
    """Accept active fallback or its single-frame confirmation transition."""
    if not status.get("ready") or status.get("blocked"):
        return False
    active = bool(
        status.get("depth_degraded_commit")
        and status.get("reason") in {
            "matrix_coverage_degraded_commit_armed",
            "matrix_coverage_insufficient_degraded_commit",
        }
    )
    confirming = bool(
        status.get("depth_degraded_commit_armed")
        and not status.get("depth_degraded_commit")
        and status.get("reason") == "matrix_prediction_coverage_transient"
        and int(status.get("matrix_unavailable_streak", 0)) == 1
    )
    return active or confirming


class SafetyStopUdpClient:
    """Authenticated client for the independent OrangePi trajectory canceller."""

    def __init__(self, host: str, port: int = 5013, timeout_sec: float = 2.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)

    def request(self, command: str, *, token: str, **values: Any) -> dict:
        task_id = f"safety-{command}-{time.time_ns()}"
        payload = {
            "protocol": "piper_visual_safety_v1",
            "command": str(command),
            "task_id": task_id,
            "token": str(token),
            **values,
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(self.timeout_sec)
            sock.sendto(encoded, (self.host, self.port))
            data, address = sock.recvfrom(65535)
        finally:
            sock.close()
        if address[0] != self.host:
            raise RuntimeError(f"safety response came from unexpected host {address[0]}")
        response = json.loads(data.decode("utf-8"))
        if response.get("task_id") != task_id:
            raise RuntimeError("safety response task_id mismatch")
        if response.get("status") == "failed":
            raise RuntimeError(f"safety stop bridge failed: {response}")
        return response

    def stop(self, *, token: str, reason: str, frame_sequence: int) -> dict:
        return self.request(
            "stop",
            token=token,
            reason=str(reason),
            frame_sequence=int(frame_sequence),
        )

    def health(self, *, token: str) -> dict:
        return self.request("health", token=token)

    def reset_latch_if_generation(self, *, token: str, generation: int, reason: str) -> dict:
        return self.request("reset_latch_if_generation", token=token,
                            expected_generation=generation, expected_reason=reason)

    def reset_latch(self, *, token: str) -> dict:
        return self.request("reset_latch", token=token)


class VisualSafetySupervisor:
    """Poll the RGB-D guard and request one independent stop when it trips."""

    def __init__(self, vision, stop_client, token: str, poll_sec: float = 0.03,
                 *, auto_resume: bool = True, event_sink=None):
        self.vision = vision
        self.stop_client = stop_client
        self.token = str(token)
        self.poll_sec = float(poll_sec)
        self.auto_resume = bool(auto_resume)
        self.event_sink = event_sink
        self.armed = threading.Event()
        self.allow_arming = threading.Event()
        self.triggered = threading.Event()
        self.stop_active = threading.Event()
        self.resumed = threading.Event()
        self.closed = threading.Event()
        self.trigger_status: dict | None = None
        self.stop_response: dict | None = None
        self.error: str | None = None
        self.events: list[dict] = []
        self.depth_pause_until = 0.0
        self.active_stop_started_monotonic: float | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _emit(self, kind: str, value: dict) -> None:
        if self.event_sink is not None:
            self.event_sink(str(kind), value)

    def preflight(self, timeout_sec: float = 2.0) -> dict:
        self.stop_client.health(token=self.token)
        deadline = time.monotonic() + float(timeout_sec)
        last = None
        while time.monotonic() < deadline:
            last = self.vision.request("safety")["visual_safety"]
            if (
                last.get("ready")
                and not last.get("blocked")
                and last.get("monitoring_available", True)
            ):
                self.stop_client.reset_latch(token=self.token)
                return last
            time.sleep(self.poll_sec)
        raise RuntimeError(f"visual safety preflight is not clear: {last}")

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def arm(self) -> None:
        if self.triggered.is_set():
            raise RuntimeError("visual safety stop is already latched")
        self.allow_arming.set()
        self.armed.set()

    def disarm(self) -> None:
        self.allow_arming.clear()
        self.armed.clear()

    def rearm_after_recovery(self) -> None:
        """Rearm after this supervisor has cleared its own transient stop."""
        if self.error:
            raise RuntimeError(f"visual safety supervisor failed: {self.error}")
        if self.stop_active.is_set() or not self.resumed.is_set():
            raise RuntimeError("visual safety recovery has not completed")
        self.triggered.clear()
        self.resumed.clear()
        self.allow_arming.set()
        self.armed.set()

    def _run(self) -> None:
        while not self.closed.is_set():
            if not self.armed.wait(timeout=self.poll_sec):
                continue
            try:
                status = self.vision.request("safety")["visual_safety"]
                # Only a short lease from fresh encoders may suppress depth
                # alarms. RPC errors still take the independent stop path.
                if time.monotonic() < self.depth_pause_until:
                    self.closed.wait(self.poll_sec)
                    continue
                degraded_commit_allowed = safety_status_is_allowed_degraded_commit(status)
                if safety_status_requires_stop(status) or (
                    not self.auto_resume
                    and not degraded_commit_allowed
                    and (not status.get("ready") or not status.get("monitoring_available", True))
                ):
                    guard_observed_unix = time.time()
                    self.trigger_status = dict(status)
                    self.stop_active.set()
                    stopped_at = time.monotonic()
                    self.active_stop_started_monotonic = stopped_at
                    stop_request_started_unix = time.time()
                    self.stop_response = self.stop_client.stop(
                        token=self.token,
                        reason=str(status.get("reason", "visual_safety")),
                        frame_sequence=int(status.get("frame_sequence", 0)),
                    )
                    stop_response_unix = time.time()
                    self.triggered.set()
                    emergency_depth_snapshot = None
                    emergency_depth_snapshot_error = None
                    try:
                        emergency_depth_snapshot = self.vision.request(
                            "snapshot",
                            reason="visual_safety_stop",
                            trigger_frame_sequence=int(
                                status.get("frame_sequence", 0)
                            ),
                        )
                    except Exception as snapshot_exc:
                        emergency_depth_snapshot_error = (
                            f"{type(snapshot_exc).__name__}: {snapshot_exc}"
                        )
                    self.armed.clear()
                    self.vision.request("safety_motion_state", paused=True)
                    self._emit("visual_safety_paused", {
                        "reason": status.get("reason"),
                        "underlying_reason": (status.get("last_obstruction") or {}).get("reason"),
                        "frame_sequence": status.get("frame_sequence"),
                        "stop_response": self.stop_response,
                    })
                    if not self.auto_resume:
                        self.events.append({
                            "stop": dict(status),
                            "stop_response": self.stop_response,
                            "emergency_depth_snapshot": emergency_depth_snapshot,
                            "emergency_depth_snapshot_error": emergency_depth_snapshot_error,
                            "policy": "latched_until_explicit_recovery",
                        })
                        continue
                    clear_status = None
                    while not self.closed.is_set():
                        clear_status = self.vision.request("safety")["visual_safety"]
                        if clear_status.get("ready") and not clear_status.get("blocked"):
                            break
                        time.sleep(self.poll_sec)
                    if self.closed.is_set():
                        continue
                    reset = self.stop_client.reset_latch(token=self.token)
                    self.vision.request("safety_motion_state", paused=False)
                    event = {
                        "stop": dict(status),
                        "clear": dict(clear_status),
                        "stop_response": self.stop_response,
                        "reset_response": reset,
                        "emergency_depth_snapshot": emergency_depth_snapshot,
                        "emergency_depth_snapshot_error": (
                            emergency_depth_snapshot_error
                        ),
                        "pause_duration_sec": time.monotonic() - stopped_at,
                        "timing": {
                            "guard_observed_unix_sec": guard_observed_unix,
                            "stop_request_started_unix_sec": stop_request_started_unix,
                            "stop_response_unix_sec": stop_response_unix,
                            "vision_to_stop_request_sec": (
                                stop_request_started_unix - guard_observed_unix
                            ),
                            "stop_request_round_trip_sec": (
                                stop_response_unix - stop_request_started_unix
                            ),
                            "first_raw_to_guard_observed_sec": (
                                guard_observed_unix
                                - float(status["matrix_first_raw_frame_unix_sec"])
                                if status.get("matrix_first_raw_frame_unix_sec")
                                is not None
                                else None
                            ),
                            # These intervals stay on the Jetson clock.  The
                            # OrangePi clock is not assumed to be synchronized,
                            # so its received_unix_sec must not be subtracted
                            # from camera timestamps.
                            "first_raw_to_stop_request_sec": (
                                stop_request_started_unix
                                - float(status["matrix_first_raw_frame_unix_sec"])
                                if status.get("matrix_first_raw_frame_unix_sec")
                                is not None
                                else None
                            ),
                            "latched_frame_to_stop_request_sec": (
                                stop_request_started_unix
                                - float(status["matrix_latched_frame_unix_sec"])
                                if status.get("matrix_latched_frame_unix_sec")
                                is not None
                                else None
                            ),
                        },
                    }
                    self.events.append(event)
                    self._emit("visual_safety_resumed", {
                        "pause_duration_sec": event["pause_duration_sec"],
                        "clear_reason": clear_status.get("reason"),
                        "frame_sequence": clear_status.get("frame_sequence"),
                    })
                    self.active_stop_started_monotonic = None
                    self.stop_active.clear()
                    self.triggered.clear()
                    self.resumed.set()
                    if self.allow_arming.is_set():
                        self.armed.set()
            except Exception as exc:
                # A blind or disconnected safety channel is itself unsafe.
                self.error = f"{type(exc).__name__}: {exc}"
                try:
                    self.stop_response = self.stop_client.stop(
                        token=self.token,
                        reason="visual_safety_monitor_failure",
                        frame_sequence=0,
                    )
                except Exception as stop_exc:
                    self.error += f"; stop_failed={type(stop_exc).__name__}: {stop_exc}"
                self.trigger_status = {"ready": False, "blocked": True, "reason": self.error}
                self.triggered.set()
                self.stop_active.set()
                if self.active_stop_started_monotonic is None:
                    self.active_stop_started_monotonic = time.monotonic()
                self.armed.clear()
                self._emit("visual_safety_failed", {"error": self.error})
            time.sleep(self.poll_sec)

    def close(self) -> None:
        self.disarm()
        self.closed.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def total_pause_sec(self) -> float:
        total = sum(float(event["pause_duration_sec"]) for event in self.events)
        if self.active_stop_started_monotonic is not None:
            total += max(
                0.0, time.monotonic() - self.active_stop_started_monotonic
            )
        return total
