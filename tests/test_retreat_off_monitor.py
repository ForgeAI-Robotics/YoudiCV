"""No hardware: frame validation and concurrent return-motion handoff."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

from piper_handeye.retreat_off_monitor import RetreatOffStreak, confirmed_retreat_off
from piper_handeye.button_light_hardware import assemble_button_light_runtime
from piper_handeye.button_light_flow import ButtonLightFlow, ButtonFlowConfig


def entry(seq, captured, **changes):
    return dict(frame_sequence=seq, capture_time_unix_sec=captured,
                target_detection=dict(label="down", frame_sequence=seq, confidence=.95),
                visibility=dict(visible=True), image="/tmp/source.jpg", **changes)


def test_three_distinct_fresh_frames_required():
    s = RetreatOffStreak("down", 10)
    assert not s.add("OFF", entry(1, 10.1), 10.2)
    assert not s.add("OFF", entry(2, 10.2), 10.3)
    assert s.add("OFF", entry(3, 10.3), 10.4)


@pytest.mark.parametrize("bad", [
    "uncertain", "on", "duplicate", "old", "before_start", "mismatch", "wrong_label",
    "weak", "clipped", "gap", "missing",
])
def test_invalid_observation_breaks_off_streak(bad):
    s = RetreatOffStreak("down", 10)
    s.add("OFF", entry(1, 10.1), 10.2)
    s.add("OFF", entry(2, 10.2), 10.3)
    e = entry(3, 10.3); state = "OFF"; now = 10.4
    if bad == "uncertain": state = "UNCERTAIN"
    if bad == "on": state = "ON"
    if bad == "duplicate": e["frame_sequence"] = e["target_detection"]["frame_sequence"] = 2
    if bad == "old": now = 11
    if bad == "before_start": e["capture_time_unix_sec"] = 9
    if bad == "mismatch": e["target_detection"]["frame_sequence"] = 4
    if bad == "wrong_label": e["target_detection"]["label"] = "open"
    if bad == "weak": e["target_detection"]["confidence"] = .69
    if bad == "clipped": e["visibility"]["visible"] = False
    if bad == "gap": e["capture_time_unix_sec"] = 10.9; now = 11
    if bad == "missing": e = {}
    assert not s.add(state, e, now)
    assert len(s.frames) <= 1


class Observer:
    target_label = "down"
    paired_light = None
    event_sink = None
    def __init__(self, directory, states):
        self.audit = []
        self.frames = iter(states)
        self.directory = directory
        self.ready = threading.Event()
        self.sample_count = 0
        self.fixed_wait_calls = 0
    def observe(self, position, **kwargs):
        if position == "waiting_retreat":
            self.sample_count += 1
            seq = self.sample_count
            p = self.directory / ("frame-%s.jpg" % seq)
            p.write_bytes(b"original-%d" % seq)
            state = next(self.frames, "UNCERTAIN")
            self.audit.append(dict(entry(seq, time.time()), image=str(p)))
            if seq >= 3: self.ready.set()
            return state
        if position == "waiting":
            self.fixed_wait_calls += 1
            return "OFF"
        return "ON"
    def wait_for_next_frame(self): pass


def run_scene(tmp_path, states, *, return_failure=False):
    observer = Observer(tmp_path, states)
    actions = []
    def return_motion():
        actions.append("return_started")
        assert observer.ready.wait(2)
        if return_failure: raise RuntimeError("independent stop latched")
        actions.append("return_endpoint_verified")
        return dict(encoder_error_rad=.001)
    executor = SimpleNamespace(load_plan=lambda p: None, approach_and_press=lambda: None,
                               retreat_to_observation=lambda d: None, return_to_waiting=return_motion)
    runtime = assemble_button_light_runtime(observer, executor, lambda: None,
        lambda: actions.append("origin_verified"), waiting_position_is_origin=False,
        retreat_off_evidence_directory=tmp_path / "saved")
    flow = ButtonLightFlow(runtime, ButtonFlowConfig(target_label="down"))
    return flow, observer, actions


def test_early_off_skips_fixed_wait_but_never_skips_motion_or_origin(tmp_path):
    flow, observer, actions = run_scene(tmp_path, ["OFF"]*3)
    result = flow.run()
    assert result.status == "arrived_after_verified_press"
    assert observer.fixed_wait_calls == 3  # initial OFF only; no endpoint reread
    assert actions == ["return_started", "return_endpoint_verified", "origin_verified"]
    event = next(e for e in result.events if e["event"] == "button_became_off")
    assert event["position"] == "waiting_retreat"
    evidence = event["evidence"]
    assert confirmed_retreat_off(evidence, "down")
    assert not confirmed_retreat_off(evidence, "up")
    for frame in evidence["frames"]:
        assert Path(frame["image"]).read_bytes().startswith(b"original-")
    bad = deepcopy(evidence); bad["frames"][2]["frame_sequence"] = 1
    assert not confirmed_retreat_off(bad, "down")


def test_interrupted_off_streak_falls_back_to_original_fixed_observation(tmp_path):
    flow, observer, actions = run_scene(tmp_path, ["OFF", "UNCERTAIN", "OFF"])
    result = flow.run()
    assert result.status == "arrived_after_verified_press" and observer.fixed_wait_calls == 6
    event = next(e for e in result.events if e["event"] == "button_became_off")
    assert event["position"] == "waiting"


def test_motion_failure_overrides_any_early_visual_off(tmp_path):
    flow, observer, actions = run_scene(tmp_path, ["OFF"]*3, return_failure=True)
    with pytest.raises(RuntimeError, match="independent stop latched"):
        flow.run()
    assert "origin_verified" not in actions
    assert not any(e["event"] == "button_became_off" for e in flow.events)


def test_moving_wait_observer_uses_original_absolute_light_not_stationary_pairing():
    from tests.test_button_light_hardware import FakeVision
    from piper_handeye.button_light_hardware import FreshLightObserver
    class Pairing:
        def classify(self, *a, **kw):
            raise AssertionError("moving arrival must not compare against a different camera pose")
    observer = FreshLightObserver(FakeVision(), SimpleNamespace(
        observe=lambda *a, **kw: {"state": "OFF"}), "15", paired_light=Pairing())
    assert observer.observe("waiting_retreat", not_before_unix=time.time()-.01) == "OFF"
    assert observer.audit[-1]["visibility"]["visible"] is True


def test_persistence_failure_does_not_produce_off_confirmation(tmp_path, monkeypatch):
    from piper_handeye import retreat_off_monitor
    def fail(*a, **kw): raise OSError("evidence disk unavailable")
    monkeypatch.setattr(retreat_off_monitor.shutil, "copy2", fail)
    flow, observer, _ = run_scene(tmp_path, ["OFF"]*3)
    result = flow.run()
    assert result.status == "arrived_after_verified_press" and observer.fixed_wait_calls == 6
    assert next(e for e in result.events if e["event"] == "button_became_off")["position"] == "waiting"
