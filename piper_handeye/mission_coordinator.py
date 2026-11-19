"""Transport-neutral AGX-side mission gate for multi-device elevator tasks.

This module models state and scoped green lights only. It has no ROS, UDP,
Android APK, chassis, or arm client and therefore cannot move hardware itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets
import threading
import time
from typing import Any


class MissionProtocolError(ValueError):
    pass


@dataclass
class _Permit:
    mission_id: str
    device: str
    action: str
    token: str
    expires_at: float
    ttl_sec: float
    issued_at_unix_sec: float

    def public(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "device": self.device,
            "action": self.action,
            "permit_token": self.token,
            "issued_at_unix_sec": self.issued_at_unix_sec,
            "ttl_sec": self.ttl_sec,
        }


class MissionCoordinator:
    """Single-mission FSM. Readiness facts and permits fail closed by default."""

    EXPECTED_SOURCES = {
        "nav_call_panel_arrived": "navigation",
        "base_hold_confirmed": "chassis",
        "elevator_door_open": "elevator",
        "call_button_segment_completed": "arm",
        "boarding_confirmed": "chassis",
        "target_floor_button_segment_completed": "arm",
        "origin_verified": "arm",
        "elevator_exit_confirmed": "chassis",
        "destination_arrived": "navigation",
        "safety_clear": "safety_supervisor",
    }

    def __init__(self, *, permit_ttl_sec: float = 2.0, base_hold_ttl_sec: float = 1.0):
        if not 0.25 <= float(permit_ttl_sec) <= 10.0:
            raise ValueError("permit TTL must be in 0.25..10 seconds")
        if not 0.25 <= float(base_hold_ttl_sec) <= 5.0:
            raise ValueError("base hold TTL must be in 0.25..5 seconds")
        self.permit_ttl_sec = float(permit_ttl_sec)
        self.base_hold_ttl_sec = float(base_hold_ttl_sec)
        self._lock = threading.RLock()
        self.mission_id: str | None = None
        self.current_floor: int | None = None
        self.destination_floor: int | None = None
        self.phase = "IDLE"
        self.fault: dict[str, Any] | None = None
        self.facts: set[str] = set()
        self._fact_receipt_monotonic: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []
        self._event_ids: set[str] = set()
        self._last_sequence: dict[str, int] = {}
        self._permits: dict[tuple[str, str], _Permit] = {}
        self._consumed_actions: set[str] = set()

    def start(self, mission_id: str, current_floor: int, destination_floor: int) -> dict:
        with self._lock:
            if self.phase not in {"IDLE", "COMPLETED"}:
                raise MissionProtocolError("已有任务处于活动状态")
            if not mission_id:
                raise MissionProtocolError("mission_id 不能为空")
            for name, value in (("current_floor", current_floor),
                                ("destination_floor", destination_floor)):
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16:
                    raise MissionProtocolError(f"{name} 必须是 1..16 的整数")
            if current_floor == destination_floor:
                raise MissionProtocolError("起始楼层与目标楼层相同")
            self.mission_id = str(mission_id)
            self.current_floor = current_floor
            self.destination_floor = destination_floor
            self.phase = "NAV_TO_CALL_PANEL"
            self.fault = None
            self.facts.clear()
            self._fact_receipt_monotonic.clear()
            self.events.clear()
            self._event_ids.clear()
            self._last_sequence.clear()
            self._permits.clear()
            self._consumed_actions.clear()
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "mission_id": self.mission_id,
                "current_floor": self.current_floor,
                "destination_floor": self.destination_floor,
                "phase": self.phase,
                "fault": self.fault,
                "facts": sorted(self.facts),
                "active_permits": [p.public() for p in self._permits.values()],
            }

    def ingest(self, source: str, event: dict[str, Any]) -> dict[str, Any]:
        """Accept one authenticated-device event, deduplicate it, advance FSM."""
        with self._lock:
            if self.mission_id is None:
                raise MissionProtocolError("尚未启动任务")
            if not isinstance(event, dict) or event.get("schema_version") != 1:
                raise MissionProtocolError("状态事件 schema_version 必须为 1")
            if event.get("mission_id") != self.mission_id:
                raise MissionProtocolError("mission_id 与当前任务不匹配")
            if not event.get("task_id") or not event.get("event_id"):
                raise MissionProtocolError("状态事件必须包含 task_id 和 event_id")
            event_id = str(event["event_id"])
            if event_id in self._event_ids:
                return {"accepted": False, "duplicate": True, **self.snapshot()}
            sequence = event.get("sequence")
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                raise MissionProtocolError("状态事件 sequence 必须是正整数")
            last = self._last_sequence.get(source, 0)
            if sequence <= last:
                return {"accepted": False, "stale": True, **self.snapshot()}
            phase = str(event.get("phase") or "")
            if phase in {"failed", "aborted", "abort_requested", "safety_paused"}:
                terminal = phase in {"failed", "aborted"}
                self.phase = "FAILED" if phase == "failed" else "ABORTED" if phase == "aborted" else "PAUSED"
                self.fault = {
                    "source": source,
                    "phase": phase,
                    "reason_code": event.get("reason_code"),
                    "event_id": event_id,
                }
                self._permits.clear()
                self.facts.discard("base_hold_confirmed")
                if terminal:
                    self._consumed_actions.add("__terminal__")
            else:
                self._apply_fact(source, phase, event)
            self._event_ids.add(event_id)
            self._last_sequence[source] = sequence
            self.events.append({"source": source, **event})
            self._advance()
            return {"accepted": True, **self.snapshot()}

    def _apply_fact(self, source: str, phase: str, event: dict[str, Any]) -> None:
        expected = self.EXPECTED_SOURCES.get(phase)
        if expected is not None and source != expected:
            raise MissionProtocolError(
                f"{phase} 只能由 {expected} 报告，收到来源 {source}"
            )
        if phase == "call_button_segment_completed":
            expected_call = "up" if self.destination_floor > self.current_floor else "down"
            if str(event.get("target", "")).casefold() != expected_call:
                raise MissionProtocolError("呼梯按钮方向与目标楼层不匹配")
        if phase == "target_floor_button_segment_completed":
            try:
                if isinstance(event.get("target"), bool):
                    raise ValueError
                reported_floor = int(event.get("target"))
            except (TypeError, ValueError):
                reported_floor = None
            if reported_floor != self.destination_floor:
                raise MissionProtocolError("机械臂确认的楼层按钮与任务目标不匹配")
        if phase == "destination_arrived":
            reported_floor = event.get("floor", event.get("current_floor"))
            try:
                if isinstance(reported_floor, bool):
                    raise ValueError
                reported_floor = int(reported_floor)
            except (TypeError, ValueError):
                reported_floor = None
            if reported_floor != self.destination_floor:
                raise MissionProtocolError("导航到达楼层与任务目标不匹配")
        if phase in self.EXPECTED_SOURCES:
            token = event.get("permit_token")
            required_action = {
                "nav_call_panel_arrived": "navigate_to_call_panel",
                "call_button_segment_completed": "press_call_button",
                "boarding_confirmed": "board_elevator",
                "target_floor_button_segment_completed": "press_target_button",
                "elevator_exit_confirmed": "exit_elevator",
                "destination_arrived": "navigate_to_destination",
            }.get(phase)
            if required_action:
                self._require_permit(source, required_action, token)
        if phase == "base_hold_confirmed":
            self.facts.add("base_hold_confirmed")
            self._fact_receipt_monotonic["base_hold_confirmed"] = time.monotonic()
        elif phase == "elevator_door_open":
            self.facts.add("elevator_door_open")
        elif phase == "safety_clear":
            self.facts.add("safety_clear")
        elif phase in {
            "nav_call_panel_arrived", "call_button_segment_completed",
            "boarding_confirmed", "target_floor_button_segment_completed",
            "elevator_exit_confirmed", "destination_arrived",
        }:
            self.facts.add(phase)
            self._fact_receipt_monotonic[phase] = time.monotonic()
            if phase in {
                "call_button_segment_completed",
                "boarding_confirmed",
                "target_floor_button_segment_completed",
                "elevator_exit_confirmed",
            }:
                # Door-open and base-hold observations are stage-local facts;
                # discard them at transitions that move the chassis.
                self.facts.discard("elevator_door_open")
            if phase in {"boarding_confirmed", "target_floor_button_segment_completed",
                         "elevator_exit_confirmed"}:
                self.facts.discard("base_hold_confirmed")
                self._fact_receipt_monotonic.pop("base_hold_confirmed", None)
        if phase == "call_button_segment_completed":
            evidence = event.get("evidence") or {}
            if evidence.get("returned_to_origin") is True:
                self.facts.add("arm_origin_verified")
        elif phase == "target_floor_button_segment_completed":
            evidence = event.get("evidence") or {}
            if evidence.get("returned_to_origin") is True:
                self.facts.add("arm_origin_verified")
        elif phase == "origin_verified":
            evidence = event.get("evidence") or {}
            verification = evidence.get("verification") or evidence
            if verification.get("verified") is True:
                self.facts.add("arm_origin_verified")

    def _require_permit(self, device: str, action: str, token: object) -> None:
        permit = self._permits.get((device, action))
        now = time.monotonic()
        if permit is None or permit.expires_at <= now or token != permit.token:
            raise MissionProtocolError(f"{device} 缺少有效绿灯：{action}")
        del self._permits[(device, action)]
        self._consumed_actions.add(action)

    def request_green_light(self, device: str, action: str, *, now: float | None = None) -> dict:
        """Issue/reuse a short, stage-scoped permit only when all gates pass."""
        with self._lock:
            now = time.monotonic() if now is None else float(now)
            if not self._eligible(device, action):
                raise MissionProtocolError(
                    f"当前阶段 {self.phase} 不满足 {device}/{action} 的放行条件"
                )
            key = (device, action)
            existing = self._permits.get(key)
            if existing is not None and existing.expires_at > now:
                return existing.public()
            if existing is not None and existing.expires_at <= now:
                self._permits.clear()
                self.phase = "PAUSED"
                self.fault = {
                    "source": "mission_manager",
                    "phase": "permit_heartbeat_timeout",
                    "reason_code": "GREEN_LIGHT_LEASE_EXPIRED",
                    "action": action,
                }
                raise MissionProtocolError("绿灯租约已超时，任务转安全暂停")
            permit = _Permit(
                mission_id=str(self.mission_id), device=device, action=action,
                token=secrets.token_urlsafe(24), expires_at=now + self.permit_ttl_sec,
                ttl_sec=self.permit_ttl_sec, issued_at_unix_sec=time.time(),
            )
            self._permits[key] = permit
            return permit.public()

    def _eligible(self, device: str, action: str) -> bool:
        if self.fault is not None or self.phase in {"FAILED", "ABORTED", "PAUSED", "COMPLETED"}:
            return False
        if action in self._consumed_actions:
            return False
        gates = {
            ("navigation", "navigate_to_call_panel"): self.phase == "NAV_TO_CALL_PANEL",
            ("arm", "press_call_button"): (
                "nav_call_panel_arrived" in self.facts
                and self._base_hold_is_fresh()
            ),
            ("chassis", "board_elevator"): (
                "call_button_segment_completed" in self.facts
                and "arm_origin_verified" in self.facts
                and "elevator_door_open" in self.facts
            ),
            ("arm", "press_target_button"): (
                "boarding_confirmed" in self.facts
                and self._base_hold_is_fresh()
            ),
            ("chassis", "exit_elevator"): (
                "target_floor_button_segment_completed" in self.facts
                and "arm_origin_verified" in self.facts
                and "elevator_door_open" in self.facts
                and self._base_hold_is_fresh()
            ),
            ("navigation", "navigate_to_destination"): (
                "elevator_exit_confirmed" in self.facts
                and self._base_hold_is_fresh()
            ),
        }
        return bool(gates.get((device, action), False))

    def _base_hold_is_fresh(self) -> bool:
        received = self._fact_receipt_monotonic.get("base_hold_confirmed")
        return (
            "base_hold_confirmed" in self.facts
            and received is not None
            and time.monotonic() - received <= self.base_hold_ttl_sec
        )

    def renew_green_light(
        self, device: str, action: str, mission_id: str, token: str,
        *, now: float | None = None,
    ) -> dict:
        """Refresh an active action lease; expiration is fail-closed."""
        with self._lock:
            now = time.monotonic() if now is None else float(now)
            permit = self._permits.get((device, action))
            if mission_id != self.mission_id or permit is None or token != permit.token:
                raise MissionProtocolError("绿灯续租凭据无效")
            if not self._eligible(device, action):
                self._permits.clear()
                self.phase = "PAUSED"
                self.fault = {
                    "source": "mission_manager",
                    "phase": "interlock_lost",
                    "reason_code": "GREEN_LIGHT_INTERLOCK_LOST",
                    "action": action,
                }
                raise MissionProtocolError("绿灯联锁已失效，任务转安全暂停")
            if permit.expires_at <= now:
                self._permits.clear()
                self.phase = "PAUSED"
                self.fault = {
                    "source": "mission_manager",
                    "phase": "permit_heartbeat_timeout",
                    "reason_code": "GREEN_LIGHT_LEASE_EXPIRED",
                    "action": action,
                }
                raise MissionProtocolError("绿灯租约已过期，不能续租")
            permit.expires_at = now + self.permit_ttl_sec
            permit.ttl_sec = self.permit_ttl_sec
            permit.issued_at_unix_sec = time.time()
            return permit.public()

    def _advance(self) -> None:
        if self.phase in {"FAILED", "ABORTED", "PAUSED", "COMPLETED"}:
            return
        if "nav_call_panel_arrived" in self.facts and "base_hold_confirmed" in self.facts:
            self.phase = "CALL_PANEL_READY"
        if "call_button_segment_completed" in self.facts:
            self.phase = "WAITING_FOR_DOOR_AND_BOARDING"
        if "boarding_confirmed" in self.facts:
            self.phase = "INSIDE_ELEVATOR"
        if "target_floor_button_segment_completed" in self.facts:
            self.phase = "WAITING_TO_EXIT"
        if "elevator_exit_confirmed" in self.facts:
            self.phase = "NAV_TO_DESTINATION"
        if "destination_arrived" in self.facts:
            self.phase = "COMPLETED"
