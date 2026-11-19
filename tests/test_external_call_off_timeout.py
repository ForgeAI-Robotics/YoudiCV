"""Virtual time and fake hardware only: no camera, ROS or robot connections."""
from collections import deque
from itertools import repeat

import pytest

from piper_handeye.button_light_flow import (
    ButtonLightFlow, ButtonFlowConfig, CALL_OFF_TIMEOUT_FLOW, production_flow_config,
)
from piper_handeye.arm_report import terminal_event
from piper_handeye.mission_status import normalize_flow_event
from piper_handeye.external_call_mission import ExternalCallMission
from run_button_light_closed_loop import flow_status_exit_code
from tools.navigation_arm_worker import arm_result


class Clock:
    now = 0.0
    def monotonic(self): return self.now


class Hardware:
    waiting_position_is_origin = False
    def __init__(self, clock, after="UNCERTAIN", initial="OFF", fail=None):
        self.clock = clock
        self.initial = deque([initial]*3)
        self.after = repeat(after) if isinstance(after, str) else iter(after)
        self.pressed = 0
        self.returned = False
        self.fail = fail
    def observe_light(self, position):
        if position in ("near", "near_confirmation"): return "ON"
        if self.initial: return self.initial.popleft()
        if self.fail == "observation": raise RuntimeError("robot observation failed")
        return next(self.after)
    def wait_for_next_observation(self): self.clock.now += 1
    def refresh_button_target(self): pass
    def press_button(self, attempt): self.pressed += 1
    def move_to_near_observation(self, distance): return "near"
    def move_to_waiting_observation(self):
        self.clock.now += 5
    def move_to_origin(self):
        if self.fail == "origin": raise RuntimeError("origin failed")
        self.returned = True
        return dict(verified=True, hardware_ok=1)


def simulate(monkeypatch, *, label="down", after="UNCERTAIN", fail=None):
    clock = Clock()
    monkeypatch.setattr("piper_handeye.button_light_flow.time.monotonic", clock.monotonic)
    h = Hardware(clock, after=after, fail=fail)
    marks = []
    cfg = production_flow_config(label, {"external_call_wait": {"post_press_off_timeout_sec": 30}})
    result = ButtonLightFlow(h, cfg, event_sink=lambda k, v: marks.append((clock.now, dict(v)))).run()
    return result, h, marks


@pytest.mark.parametrize("label", ["up", "down"])
@pytest.mark.parametrize("after", ["UNCERTAIN", "ON", ["OFF", "UNCERTAIN"]*30])
def test_timeout_after_verified_press_not_fake_off(monkeypatch, label, after):
    result, h, marks = simulate(monkeypatch, label=label, after=after)
    assert result.status == CALL_OFF_TIMEOUT_FLOW and h.returned and h.pressed == 1
    on_time = next(t for t, e in marks if e["event"] == "press_verified_on")
    timeout_time, timeout = next((t, e) for t, e in marks if e["event"] == "external_call_off_wait_timeout")
    assert timeout_time-on_time == 30  # includes the five-second retreat
    assert timeout["off_confirmed"] is False and timeout["target_label"] == label
    assert not any(e["event"] == "button_became_off" for e in result.events)
    assert flow_status_exit_code(result.status) == 0


def test_reliable_off_before_deadline_uses_original_flow(monkeypatch):
    result, h, _ = simulate(monkeypatch, after=["UNCERTAIN"]*4+["OFF"]*3)
    assert result.status == "arrived_after_verified_press" and h.returned
    assert not any(e["event"] == "external_call_off_wait_timeout" for e in result.events)


def test_numeric_wait_is_not_shortened(monkeypatch):
    result, _, _ = simulate(monkeypatch, label="2", after=["UNCERTAIN"]*35+["OFF"]*3)
    assert result.status == "arrived_after_verified_press"
    assert not any(e["event"] == "external_call_off_wait_timeout" for e in result.events)


def test_initially_on_without_this_tasks_press_does_not_timeout(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("piper_handeye.button_light_flow.time.monotonic", clock.monotonic)
    h = Hardware(clock, initial="ON", after=["UNCERTAIN"]*35+["OFF"]*3)
    cfg = production_flow_config("down", {"external_call_wait": {"post_press_off_timeout_sec": 30}})
    result = ButtonLightFlow(h, cfg).run()
    assert result.status == "arrived_after_initial_on" and h.pressed == 0 and clock.now > 30


@pytest.mark.parametrize("failure", ["observation", "origin"])
def test_timeout_does_not_suppress_hardware_or_return_failure(monkeypatch, failure):
    with pytest.raises(RuntimeError): simulate(monkeypatch, fail=failure)


def test_timeout_result_survives_report_worker_and_only_call_gate(monkeypatch):
    result, _, _ = simulate(monkeypatch)
    timeout = next(e for e in result.events if e["event"] == "external_call_off_wait_timeout")
    report = dict(flow_status=result.status, process_exit_code=0, target_label="down",
                  returned_to_origin=True, origin_verification=dict(verified=True, hardware_ok=1),
                  press_execution_started=True, press_attempts=1, external_call_off_wait_timeout=timeout)
    records = [dict(normalize_flow_event(e), producer="arm_pressing") for e in result.events]
    records.append(terminal_event(report))
    r = arm_result(records, 0)
    assert r["on_confirmed"] and not r["off_confirmed"] and r["off_wait_timed_out"]
    assert r["external_call_off_wait_timeout"] == timeout
    assert next(e for e in records if e["phase"] == "external_call_timeout_release")["outcome"] == "degraded"
    m = ExternalCallMission(include_floor_panel=True)
    mid = m.begin(3, 2, 0); m.phase = "arm_call"
    assert m.arm_finished(mid, r) == "check_origin"
    assert m.call_off_wait_timed_out
    assert m.origin_checked(mid, dict(ok=True, at_origin=True, hardware_ok=1,
                                     joint_state_source="orange_pi_ros_joint_states")) == "navigate_floor"
    for phase, changes in [
        ("arm_floor", {}), ("arm_call", {"on_confirmed": False}),
        ("arm_call", {"origin_verified": False}), ("arm_call", {"returned_to_origin": False}),
        ("arm_call", {"return_code": 1}), ("arm_call", {"press_execution_started": False}),
        ("arm_call", {"off_wait_timed_out": False}),
        ("arm_call", {"external_call_off_wait_timeout": dict(timeout, target_label="up")}),
        ("arm_call", {"external_call_off_wait_timeout": dict(timeout, elapsed_sec=29)}),
    ]:
        m = ExternalCallMission(); mid = m.begin(3, 2, 0); m.phase = phase
        assert m.arm_finished(mid, dict(r, **changes)) == "failed"


def test_timeout_disabled_by_default_and_invalid_scopes_rejected():
    assert production_flow_config("down", {}).external_call_off_wait_timeout_sec is None
    assert production_flow_config("4", {"external_call_wait": {"post_press_off_timeout_sec": 30}}).external_call_off_wait_timeout_sec is None
    for label, seconds in [("4", 30), ("down", -1), ("up", float("nan")), ("up", True)]:
        with pytest.raises(ValueError):
            ButtonFlowConfig(target_label=label, external_call_off_wait_timeout_sec=seconds)
