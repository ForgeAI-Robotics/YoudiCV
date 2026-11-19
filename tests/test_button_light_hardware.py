from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from piper_handeye.button_light_hardware import (
    DryValidatedSegmentExecutor,
    FreshLightObserver,
    assemble_button_light_runtime,
    target_is_safely_visible,
    VisualSegmentMotionSafety,
)
from piper_handeye.button_light_flow import ButtonLightFlow


class FakeVision:
    def __init__(self):
        self.sequence = 3
    def request(self, command, after_sequence, target_label):
        assert command == "light_snapshot"
        assert target_label == "15"
        self.sequence += 1
        return {
            "fresh": True,
            "frame_sequence": self.sequence,
            "path": f"/tmp/frame-{self.sequence}.jpg",
            "capture_time_unix_sec": time.time(),
            "image_size": [640,360],
            "target_detection": {"bbox_xyxy": [20,20,50,50], "confidence":.95,"frame_sequence":self.sequence},
        }


class FakeRecognizer:
    def __init__(self): self.paths = []
    def observe(self, path, target, bbox_xyxy=None):
        self.paths.append((str(path), target))
        return {"state": "ON", "inference_sec": 1.0}


def test_light_observer_requires_and_records_fresh_frames():
    recognizer = FakeRecognizer()
    observer = FreshLightObserver(FakeVision(), recognizer, "15")
    assert observer.observe("near") == "ON"
    assert observer.observe("waiting") == "ON"
    assert [row["frame_sequence"] for row in observer.audit] == [4, 5]
    assert recognizer.paths == [("/tmp/frame-4.jpg", "15"), ("/tmp/frame-5.jpg", "15")]


class FakeBridge:
    def __init__(self): self.calls = []
    def execute(self, points, **options):
        self.calls.append((points, options))
        return {"status": "validated" if options.get("dry_run") else "complete"}


class FakeGuardedBridge(FakeBridge):
    def execute(self, points, **options):
        self.calls.append((points, options))
        return {
            "status": "validated" if options.get("dry_run") else "contact_retracted",
            "feedback_contact_confirmed": not options.get("dry_run", False),
            "reason": None if options.get("dry_run") else "sustained_rolling_current_limit",
        }


def _guarded_config():
    return {
        "max_forward_travel_m": 0.022,
        "current_guard": {"contact_delta_a": [0.2] * 6},
    }


class FakeRobot:
    def __init__(self): self.target = np.zeros(6)
    def state(self, max_age):
        return SimpleNamespace(joints=self.target.copy(), raw={"hardware_ok": 1}, timestamp=time.monotonic())
    def latest_state(self, max_age):
        return self.state(max_age=max_age)


def _segment(start, end):
    return [
        {"positions": list(start), "velocities": [0] * 6, "accelerations": [0] * 6, "time_from_start_sec": 0.0},
        {"positions": list(end), "velocities": [0] * 6, "accelerations": [0] * 6, "time_from_start_sec": 1.0},
    ]


def _plan():
    waiting = np.array([0.1] * 6)
    prepress = np.array([0.15] * 6)
    observation = np.array([0.15] * 6)
    from piper_handeye.segmented_button_motion import SegmentedPressPlan
    return SegmentedPressPlan(
        observation, 0.15,
        _segment(waiting, observation), _segment(observation, prepress),
        _segment(prepress, prepress),
        _segment(prepress, observation), _segment(observation, waiting),
        None, None, None, None, None,
    )


def test_executor_validates_every_segment_and_defaults_to_no_motion():
    bridge, robot = FakeBridge(), FakeRobot()
    executor = DryValidatedSegmentExecutor(bridge, robot, "token")
    executor.load_plan(_plan())
    assert len(bridge.calls) == 5
    assert all(call[1]["dry_run"] for call in bridge.calls)
    with pytest.raises(RuntimeError, match="disabled"):
        executor.approach_and_press()


def test_guarded_press_can_blend_directly_to_observation_without_second_motion():
    plan = _plan()
    observation = np.zeros(6)
    observation[:] = 0.10
    prepress = np.full(6, 0.15)
    plan = replace(
        plan,
        observation_joints=observation,
        observation_distance_m=0.05,
        observation_to_prepress=_segment(observation, prepress),
        press_cycle=_segment(prepress, prepress),
        retreat_to_observation=_segment(prepress, observation),
    )
    bridge = FakeBridge()
    robot = FakeRobot()
    robot.target[:] = prepress

    class BlendedGuardedBridge(FakeGuardedBridge):
        def execute(self, points, **options):
            self.calls.append((points, options))
            if options.get("dry_run"):
                return {"status": "validated"}
            robot.target[:] = observation
            return {
                "status": "contact_retracted",
                "feedback_contact_confirmed": True,
                "retract_destination": "observation",
                "retract_mode": "blended_to_observation",
            }

    class Safety:
        def __init__(self): self.disabled = False
        def set_target(self, _target): pass
        def before_segment(self, _name): pass
        def after_segment(self, _name): pass
        def disable_for_return(self): self.disabled = True

    guarded = BlendedGuardedBridge()
    safety = Safety()
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=safety,
        guarded_bridge=guarded, guarded_press_config=_guarded_config(),
    )
    executor.load_plan(plan)
    executor.press_from_prepress()
    trajectory_calls_before_observation = len(bridge.calls)

    result = executor.retreat_to_observation(0.05)

    assert result["result"]["status"] == "already_at_target"
    assert len(bridge.calls) == trajectory_calls_before_observation
    assert safety.disabled


def test_enabled_executor_checks_encoder_after_each_segment():
    bridge, guarded, robot = FakeBridge(), FakeGuardedBridge(), FakeRobot()
    class Safety:
        def __init__(self): self.events = []
        def set_target(self, target): pass
        def before_segment(self, name): self.events.append(("before", name))
        def after_segment(self, name): self.events.append(("after", name))
        def disable_for_return(self): self.events.append(("disabled", "return"))
    safety = Safety()
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=safety,
        guarded_bridge=guarded, guarded_press_config=_guarded_config(),
    )
    plan = _plan()
    executor.load_plan(plan)
    robot.target[:] = 0.15
    executor.approach_and_press()
    robot.target[:] = 0.15
    executor.retreat_to_observation(0.15)
    robot.target[:] = 0.1
    executor.return_to_waiting()
    assert [event["segment"] for event in executor.executions] == [
        "approach_to_observation", "observation_to_prepress", "guarded_press",
        "retreat_to_observation", "observation_to_waiting"
    ]
    assert safety.events[-1] == ("disabled", "return")
    assert len(safety.events) == 7


def test_observation_only_uses_validated_approach_without_press():
    bridge, robot = FakeBridge(), FakeRobot()
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
        def disable_for_return(self): pass
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=Safety()
    )
    executor.load_plan(_plan())
    robot.target[:] = 0.15
    executor.approach_to_observation()
    robot.target[:] = 0.1
    executor.return_to_waiting()
    assert [event["segment"] for event in executor.executions] == [
        "approach_to_observation", "observation_to_waiting"
    ]


@pytest.mark.parametrize("already_at_prepress",[False,True])
def test_press_from_prepress_sends_no_reaching_segment(already_at_prepress):
    bridge, guarded, robot = FakeBridge(), FakeGuardedBridge(), FakeRobot()
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=Safety(),
        guarded_bridge=guarded, guarded_press_config=_guarded_config(),
    )
    robot.target[:] = 0.15
    executor.load_plan(_plan(),already_at_prepress=already_at_prepress)
    executor.press_from_prepress()
    assert [event["segment"] for event in executor.executions] == ["guarded_press"]
    executed = [options for _, options in guarded.calls if not options.get("dry_run")]
    assert len(executed) == 1


def test_physical_press_refuses_fixed_trajectory_without_feedback_bridge():
    bridge, robot = FakeBridge(), FakeRobot()
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=Safety()
    )
    executor.load_plan(_plan())
    robot.target[:] = 0.15
    with pytest.raises(RuntimeError, match="current-feedback guarded bridge"):
        executor.press_from_prepress()
    assert not [options for _, options in bridge.calls if not options.get("dry_run")]


def test_hard_limit_retraction_is_not_reported_as_completed_press():
    class LimitBridge(FakeGuardedBridge):
        def execute(self, points, **options):
            result = super().execute(points, **options)
            if not options.get("dry_run"):
                result.update(
                    status="limit_retracted", reason="relative_current_limit",
                    feedback_contact_confirmed=False,
                )
            return result
    bridge, guarded, robot = FakeBridge(), LimitBridge(), FakeRobot()
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=Safety(),
        guarded_bridge=guarded, guarded_press_config=_guarded_config(),
    )
    executor.load_plan(_plan())
    robot.target[:] = 0.15
    with pytest.raises(RuntimeError, match="without feedback-confirmed contact"):
        executor.press_from_prepress()
    assert executor.executions[-1]["result"]["status"] == "limit_retracted"


def test_maximum_travel_retraction_continues_to_light_check_for_retry():
    class MaximumTravelBridge(FakeGuardedBridge):
        def execute(self, points, **options):
            result = super().execute(points, **options)
            if not options.get("dry_run"):
                result.update(
                    status="maximum_travel_retracted",
                    reason="below_contact_threshold",
                    feedback_contact_confirmed=False,
                )
            return result
    bridge, guarded, robot = FakeBridge(), MaximumTravelBridge(), FakeRobot()
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
        def disable_for_return(self): pass
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=Safety(),
        guarded_bridge=guarded, guarded_press_config=_guarded_config(),
    )
    executor.load_plan(_plan())
    robot.target[:] = 0.15
    event = executor.press_from_prepress()
    assert event["result"]["status"] == "maximum_travel_retracted"
    assert event["result"]["feedback_contact_confirmed"] is False


def test_stale_feedback_retraction_continues_to_light_check_for_retry():
    class StaleBridge(FakeGuardedBridge):
        def execute(self, points, **options):
            result = super().execute(points, **options)
            if not options.get("dry_run"):
                result.update(
                    status="limit_retracted", reason="stale_current_sample",
                    feedback_contact_confirmed=False,
                )
            return result
    bridge, guarded, robot = FakeBridge(), StaleBridge(), FakeRobot()
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=Safety(),
        guarded_bridge=guarded, guarded_press_config=_guarded_config(),
    )
    executor.load_plan(_plan())
    robot.target[:] = 0.15
    event = executor.press_from_prepress()
    assert event["result"]["status"] == "limit_retracted"
    assert event["result"]["reason"] == "stale_current_sample"


def test_execution_cannot_be_enabled_without_visual_safety():
    bridge, robot = FakeBridge(), FakeRobot()
    executor = DryValidatedSegmentExecutor(bridge, robot, "token", execute_enabled=True)
    executor.load_plan(_plan())
    with pytest.raises(RuntimeError, match="requires motion safety"):
        executor.approach_and_press()


def test_executor_preserves_bridge_error_when_safety_cleanup_also_fails():
    class FailedBridge(FakeBridge):
        def execute(self, points, **options):
            if not options.get("dry_run"):
                raise TimeoutError("bridge terminal response timed out")
            return super().execute(points, **options)
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): raise RuntimeError("visual safety stop is active")
    executor = DryValidatedSegmentExecutor(
        FailedBridge(), FakeRobot(), "token", execute_enabled=True, motion_safety=Safety()
    )
    executor.load_plan(_plan())
    with pytest.raises(TimeoutError, match="terminal response timed out"):
        executor.approach_and_press()


def test_encoder_failure_closes_started_journal_with_failed_event():
    events = []
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
    executor = DryValidatedSegmentExecutor(
        FakeBridge(), FakeRobot(), "token", execute_enabled=True,
        motion_safety=Safety(), event_sink=lambda kind, value: events.append((kind, value)),
    )
    executor.load_plan(_plan())
    with pytest.raises(RuntimeError, match="encoder target error"):
        executor.approach_to_observation()
    assert events[-1][0] == "motion_segment_failed"
    assert events[-1][1]["segment"] == "approach_to_observation"
    assert events[-1][1]["phase"] == "terminal_encoder_verification"
    assert events[-1][1]["bridge_result"]["status"] == "complete"


def test_safety_cleanup_failure_is_explicitly_journaled():
    events = []
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): raise RuntimeError("cleanup latch")
    executor = DryValidatedSegmentExecutor(
        FakeBridge(), FakeRobot(), "token", execute_enabled=True,
        motion_safety=Safety(), event_sink=lambda kind, value: events.append((kind, value)),
    )
    executor.load_plan(_plan())
    with pytest.raises(RuntimeError, match="cleanup latch"):
        executor.approach_to_observation()
    assert events[-2][0] == "motion_segment_cleanup_failed"
    assert events[-1][0] == "motion_segment_failed"
    assert events[-1][1]["segment"] == "approach_to_observation"
    assert events[-1][1]["phase"] == "safety_cleanup"
    assert events[-1][1]["error"] == "RuntimeError: cleanup latch"
    assert events[-1][1]["task_id"].startswith(
        "button-light-approach_to_observation-execute-"
    )


def test_return_mode_disarms_visual_safety_and_bypasses_segment_hooks():
    bridge, robot = FakeBridge(), FakeRobot()
    class Safety:
        def __init__(self): self.events = []
        def set_target(self, target): pass
        def before_segment(self, name): self.events.append(("before", name))
        def after_segment(self, name): self.events.append(("after", name))
        def disable_for_return(self): self.events.append(("disabled", "return"))
    safety = Safety()
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=safety
    )
    executor.load_plan(_plan())
    executor.disable_visual_safety_for_return()
    robot.target[:] = 0.1
    executor.return_to_waiting()
    assert safety.events == [("disabled", "return")]
    assert executor.executions[-1]["segment"] == "observation_to_waiting"


def test_return_from_prepress_builds_and_validates_current_state_trajectory():
    bridge = FakeBridge()
    prepress = np.asarray([0.10, 0.30, -0.30, 0.10, 0.30, 0.10])
    waiting = np.asarray([0.04, 0.106, 0.0, 0.05, 0.204, 0.0])
    class Robot:
        def __init__(self): self.calls = 0
        def state(self, max_age):
            self.calls += 1
            joints = prepress if self.calls == 1 else waiting
            return SimpleNamespace(joints=joints, raw={"hardware_ok": 1})
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
        def disable_for_return(self): pass
    executor = DryValidatedSegmentExecutor(
        bridge, Robot(), "token", execute_enabled=True, motion_safety=Safety()
    )
    plan = _plan()
    plan.observation_joints[:] = [0.08, 0.25, -0.25, 0.08, 0.25, 0.08]
    plan.observation_to_waiting[-1]["positions"] = waiting.tolist()
    executor.load_plan(plan)
    executor.return_to_waiting()
    assert executor.executions[-1]["segment"] == "current_to_waiting"
    dynamic_calls = [
        options for _points, options in bridge.calls
        if "current_to_waiting" in options.get("task_id", "")
    ]
    assert [options.get("dry_run", False) for options in dynamic_calls] == [True, False]


def test_return_from_prepress_reuses_prevalidated_direct_trajectory():
    bridge = FakeBridge()
    prepress = np.asarray([0.10, 0.30, -0.30, 0.10, 0.30, 0.10])
    waiting = np.asarray([0.04, 0.106, 0.0, 0.05, 0.204, 0.0])

    class Robot:
        def __init__(self): self.calls = 0
        def state(self, max_age):
            self.calls += 1
            joints = prepress if self.calls == 1 else waiting
            return SimpleNamespace(joints=joints, raw={"hardware_ok": 1})

    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
        def disable_for_return(self): pass

    executor = DryValidatedSegmentExecutor(
        bridge, Robot(), "token", execute_enabled=True, motion_safety=Safety()
    )
    base = _plan()
    plan = replace(
        base,
        observation_joints=np.asarray([0.08, 0.25, -0.25, 0.08, 0.25, 0.08]),
        observation_to_prepress=_segment(
            [0.08, 0.25, -0.25, 0.08, 0.25, 0.08], prepress
        ),
        observation_to_waiting=_segment(
            [0.08, 0.25, -0.25, 0.08, 0.25, 0.08], waiting
        ),
        prepress_to_waiting=_segment(prepress, waiting),
    )
    executor.load_plan(plan)
    executor.return_to_waiting()
    assert executor.executions[-1]["segment"] == "prepress_to_waiting"
    direct_calls = [
        options for _points, options in bridge.calls
        if "prepress_to_waiting" in options.get("task_id", "")
    ]
    assert [options.get("dry_run", False) for options in direct_calls] == [True, False]
    assert not [
        options for _points, options in bridge.calls
        if "current_to_waiting" in options.get("task_id", "")
    ]


def test_light_monitor_hook_runs_only_at_guarded_press_boundary():
    bridge, guarded, robot = FakeBridge(), FakeGuardedBridge(), FakeRobot()
    calls = []

    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): calls.append(("safety", name))
        def after_segment(self, name): pass

    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=Safety(),
        guarded_bridge=guarded, guarded_press_config=_guarded_config(),
    )
    executor.set_before_guarded_press_callback(
        lambda: calls.append(("monitor", "started"))
    )
    executor.load_plan(_plan())
    assert calls == []
    robot.target[:] = 0.15
    executor.press_from_prepress()
    assert calls[:2] == [("safety", "press_cycle"), ("monitor", "started")]


def test_new_press_attempt_restores_outbound_safety_after_return_mode():
    bridge, robot = FakeBridge(), FakeRobot()
    class Safety:
        def __init__(self): self.events = []
        def set_target(self, target): pass
        def before_segment(self, name): self.events.append(("before", name))
        def after_segment(self, name): self.events.append(("after", name))
        def disable_for_return(self): self.events.append(("disabled", "return"))
        def begin_outbound_cycle(self): self.events.append(("begin", "outbound"))
    safety = Safety()
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=safety,
        guarded_bridge=FakeGuardedBridge(), guarded_press_config=_guarded_config(),
    )
    executor.load_plan(_plan())
    executor.disable_visual_safety_for_return()
    executor.begin_press_attempt()
    robot.target[:] = 0.15
    executor.press_from_observation()
    assert ("begin", "outbound") in safety.events
    assert ("before", "observation_to_prepress") in safety.events


def test_retry_from_observation_skips_far_approach_segment():
    bridge, guarded, robot = FakeBridge(), FakeGuardedBridge(), FakeRobot()
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
        def begin_outbound_cycle(self): pass
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=Safety(),
        guarded_bridge=guarded, guarded_press_config=_guarded_config(),
    )
    executor.load_plan(_plan())
    executor.begin_press_attempt()
    robot.target[:] = 0.15
    executor.press_from_observation()
    assert [event["segment"] for event in executor.executions] == [
        "observation_to_prepress", "guarded_press"
    ]


def test_retry_retreat_accepts_refreshed_near_hold_distance():
    plan = replace(_plan(), observation_distance_m=0.158)
    bridge, robot = FakeBridge(), FakeRobot()
    class Safety:
        def set_target(self, target): pass
        def before_segment(self, name): pass
        def after_segment(self, name): pass
        def disable_for_return(self): pass
    executor = DryValidatedSegmentExecutor(
        bridge, robot, "token", execute_enabled=True, motion_safety=Safety()
    )
    executor.load_plan(plan)
    robot.target[:] = 0.15
    executor.retreat_to_observation(0.15)
    assert executor.executions[-1]["segment"] == "retreat_to_observation"


def test_visual_safety_return_mode_disarms_and_clears_near_exceptions():
    calls = []
    safety = VisualSegmentMotionSafety.__new__(VisualSegmentMotionSafety)
    safety.supervisor = SimpleNamespace(
        depth_pause_until=123.0,
        disarm=lambda: calls.append("disarm"),
    )
    safety.vision = SimpleNamespace(
        request=lambda command, **kwargs: calls.append((command, kwargs))
    )
    safety.near_pause = SimpleNamespace(finished=False, active=True)
    safety.depth_degraded_commit_active = True
    safety._deactivate_depth_degraded_commit = lambda reason: calls.append(reason)
    safety.disable_for_return()
    assert safety.supervisor.depth_pause_until == 0.0
    assert safety.near_pause.finished is True
    assert safety.near_pause.active is False
    assert calls == [
        "disarm",
        ("safety_motion_state", {"paused": True}),
        "return_visual_safety_disabled",
    ]


def test_segment_safety_accepts_auto_resume_policy():
    supervisor = SimpleNamespace(auto_resume=True)
    safety = VisualSegmentMotionSafety(supervisor, object(), object(), np.eye(4))
    assert safety.supervisor.auto_resume is True


def test_segment_safety_accepts_explicit_ten_cm_coverage_entry_distance():
    supervisor = SimpleNamespace(auto_resume=False)
    safety = VisualSegmentMotionSafety(
        supervisor, object(), object(), np.eye(4),
        pause_depth_near_button=True, depth_degraded_entry_distance_m=0.10,
    )
    assert safety.near_pause.entry_distance_m == pytest.approx(0.10)


def test_segment_safety_accepts_twelve_cm_coverage_entry_distance():
    supervisor = SimpleNamespace(auto_resume=False)
    safety = VisualSegmentMotionSafety(
        supervisor, object(), object(), np.eye(4),
        pause_depth_near_button=True, depth_degraded_entry_distance_m=0.12,
    )
    assert safety.near_pause.entry_distance_m == pytest.approx(0.12)


def test_segment_safety_stays_armed_until_near_retreat_finishes():
    class Event:
        def __init__(self): self.value = False
        def is_set(self): return self.value
    class Supervisor:
        auto_resume = False
        error = None
        depth_pause_until = 0.0
        triggered = Event()
        stop_active = Event()
        armed = Event()
        def arm(self): self.armed.value = True
        def disarm(self): self.armed.value = False
    class Vision:
        def __init__(self): self.paused = []
        def request(self, command, **values):
            if command == "safety_motion_state": self.paused.append(values["paused"])
            return {"ok": True}
    supervisor, vision = Supervisor(), Vision()
    safety = VisualSegmentMotionSafety(supervisor, vision, object(), np.eye(4))
    for name in ("approach", "press_cycle", "retreat_to_observation"):
        safety.before_segment(name)
        safety.after_segment(name)
        if name != "retreat_to_observation": assert supervisor.armed.is_set()
    assert not supervisor.armed.is_set()
    assert vision.paused == [False, False, False, True]


def test_near_commit_is_disabled_only_after_retreat_is_disarmed_and_paused():
    class Event:
        def __init__(self, value=False): self.value = value
        def is_set(self): return self.value
    class Supervisor:
        auto_resume = False
        error = None
        depth_pause_until = 1.0
        triggered = Event()
        stop_active = Event()
        armed = Event(True)
        def arm(self): self.armed.value = True
        def disarm(self): self.armed.value = False
    class Vision:
        def __init__(self, supervisor): self.supervisor = supervisor; self.calls = []
        def request(self, command, **values):
            self.calls.append((command, dict(values), self.supervisor.armed.is_set()))
            return {"ok": True}
    supervisor = Supervisor()
    vision = Vision(supervisor)
    safety = VisualSegmentMotionSafety(supervisor, vision, object(), np.eye(4))
    safety.depth_degraded_commit_active = True
    safety.after_segment("retreat_to_observation")
    assert vision.calls == [
        ("safety_motion_state", {"paused": True}, False),
        ("safety_set_depth_degraded_commit", {"enabled": False}, False),
    ]
    assert not safety.depth_degraded_commit_active
    assert safety.near_pause.finished
    assert not safety.near_pause.active


def test_assembled_runtime_refreshes_and_validates_each_retry():
    class Observer:
        def __init__(self): self.waiting = iter(["OFF"]*6); self.near = iter(["OFF", "OFF", "ON", "ON"])
        def observe(self, position): return next(self.waiting if position == "waiting" else self.near)
        def wait_for_next_frame(self): pass
    class Executor:
        def __init__(self): self.loads = 0; self.attempts = 0; self.near = 0; self.waiting = 0
        def load_plan(self, plan): self.loads += 1
        def approach_and_press(self): self.attempts += 1
        def retreat_to_observation(self, distance): assert distance == 0.05; self.near += 1
        def return_to_waiting(self): self.waiting += 1
    observer, executor = Observer(), Executor()
    origins = []
    runtime = assemble_button_light_runtime(
        observer, executor, lambda: object(), lambda: origins.append(True),
        waiting_position_is_origin=False,
    )
    result = ButtonLightFlow(runtime).run()
    assert result.status == "arrived_after_verified_press"
    assert result.press_attempts == 2
    # The retry starts at the 5 cm holding pose; only verified ON returns to
    # the raised waiting observation pose.
    assert (executor.loads, executor.attempts, executor.near, executor.waiting) == (2, 2, 2, 1)
    assert origins == [True]


def test_assembled_runtime_can_delegate_press_to_receding_executor():
    class Observer:
        def __init__(self):
            self.waiting = iter(["OFF"]*6)
            self.near = iter(["ON", "ON"])
        def observe(self, position):
            return next(self.waiting if position == "waiting" else self.near)
        def wait_for_next_frame(self): pass
    class Executor:
        def __init__(self): self.loads = 0; self.near = 0; self.waiting = 0
        def load_plan(self, plan): self.loads += 1
        def approach_and_press(self): raise AssertionError("static reaching path used")
        def retreat_to_observation(self, distance): self.near += 1
        def return_to_waiting(self): self.waiting += 1
    observer, executor = Observer(), Executor()
    receding_attempts = []
    runtime = assemble_button_light_runtime(
        observer, executor, lambda: object(), lambda: None,
        waiting_position_is_origin=False,
        press_attempt=lambda attempt: receding_attempts.append(attempt),
    )
    result = ButtonLightFlow(runtime).run()
    assert result.status == "arrived_after_verified_press"
    assert receding_attempts == [1]
    assert executor.loads == 1


def test_assembled_runtime_allows_first_online_approach_to_defer_static_plan():
    class Observer:
        def __init__(self):
            self.waiting = iter(["OFF"]*6)
            self.near = iter(["ON", "ON"])
        def observe(self, position):
            return next(self.waiting if position == "waiting" else self.near)
        def wait_for_next_frame(self): pass
    class Executor:
        def __init__(self): self.loads = 0
        def load_plan(self, plan): self.loads += 1
        def retreat_to_observation(self, distance): pass
        def return_to_waiting(self): pass
    executor = Executor()
    runtime = assemble_button_light_runtime(
        Observer(), executor, lambda: None, lambda: None,
        waiting_position_is_origin=False,
        press_attempt=lambda _attempt: None,
    )
    result = ButtonLightFlow(runtime).run()
    assert result.status == "arrived_after_verified_press"
    assert executor.loads == 0


def test_target_visibility_requires_fresh_uncropped_recent_detection():
    base = {
        "fresh": True,
        "frame_sequence": 100,
        "image_size": [640, 360],
        "target_detection": {
            "frame_sequence": 100,
            "confidence": 0.91,
            "bbox_xyxy": [280, 140, 320, 180],
        },
    }
    visible, audit = target_is_safely_visible(base)
    assert visible
    assert all(audit["checks"].values())

    cropped = {**base, "target_detection": {
        **base["target_detection"], "bbox_xyxy": [1, 140, 41, 180]
    }}
    assert target_is_safely_visible(cropped)[0] is False
    stale = {**base, "target_detection": {
        **base["target_detection"], "frame_sequence": 99
    }}
    assert target_is_safely_visible(stale)[0] is False

    # The button-9 hardware run ended 3.85 px above the image edge. It is a
    # complete usable crop and must not be rejected by a nominal 4 px margin.
    edge_tolerant = {**base, "target_detection": {
        **base["target_detection"], "bbox_xyxy": [258, 271, 356, 356.15]
    }}
    assert target_is_safely_visible(edge_tolerant)[0] is True


def test_early_two_frame_on_monitor_skips_full_near_retreat():
    confirmed = threading.Event()

    class Observer:
        def __init__(self):
            self.waiting = iter(["OFF"]*6)
            self.audit = []
            self.samples = 0
        def observe(self, position):
            return next(self.waiting)
        def observe_during_motion(self):
            self.samples += 1
            self.audit.append({
                "frame_sequence": 100 + self.samples,
                "capture_time_unix_sec": float(self.samples),
                "state": "ON",
                "image": f"/tmp/on-{self.samples}.jpg",
            })
            if self.samples >= 2:
                confirmed.set()
            return "ON"
        def wait_for_next_frame(self): pass

    class Executor:
        def __init__(self): self.near = 0; self.waiting = 0
        def load_plan(self, _plan): pass
        def approach_and_press(self):
            assert confirmed.wait(timeout=1.0)
        def retreat_to_observation(self, _distance): self.near += 1
        def return_to_waiting(self): self.waiting += 1

    observer, executor = Observer(), Executor()
    runtime = assemble_button_light_runtime(
        observer, executor, lambda: object(), lambda: None,
        waiting_position_is_origin=True,
        hold_at_prepress_if_visible=lambda: False,
    )
    result = ButtonLightFlow(runtime).run()
    assert result.status == "arrived_after_verified_press"
    assert executor.near == 0
    assert executor.waiting == 1
    assert any(
        event["event"] == "post_press_on_captured_early"
        for event in result.events
    )


def test_visible_prepress_skips_near_retreat_then_returns_to_waiting():
    class Observer:
        def __init__(self):
            self.waiting = iter(["OFF"]*6)
            self.prepress = iter(["ON", "ON"])
        def observe(self, position):
            return next(self.waiting if position == "waiting" else self.prepress)
        def wait_for_next_frame(self): pass
    class Executor:
        def __init__(self): self.near = 0; self.waiting = 0
        def load_plan(self, _plan): pass
        def approach_and_press(self): pass
        def retreat_to_observation(self, _distance): self.near += 1
        def return_to_waiting(self): self.waiting += 1
    executor = Executor()
    runtime = assemble_button_light_runtime(
        Observer(), executor, lambda: object(), lambda: None,
        waiting_position_is_origin=True,
        hold_at_prepress_if_visible=lambda: True,
    )
    result = ButtonLightFlow(runtime).run()
    assert result.status == "arrived_after_verified_press"
    assert executor.near == 0
    assert executor.waiting == 1
    assert any(
        event["event"] == "at_post_press_observation"
        and event["position"] == "prepress"
        and event["requested_panel_distance_m"] is None
        for event in result.events
    )


def test_prepress_off_moves_to_near_before_retry():
    class Observer:
        def __init__(self):
            self.states = {
                "waiting": iter(["OFF"]*6),
                "prepress": iter(["OFF", "OFF", "ON", "ON"]),
            }
        def observe(self, position):
            return next(self.states["waiting" if position == "waiting" else "prepress"])
        def wait_for_next_frame(self): pass
    class Executor:
        def __init__(self): self.near = 0
        def load_plan(self, _plan): pass
        def approach_and_press(self): pass
        def retreat_to_observation(self, _distance): self.near += 1
        def return_to_waiting(self): pass
    executor = Executor()
    runtime = assemble_button_light_runtime(
        Observer(), executor, lambda: object(), lambda: None,
        waiting_position_is_origin=True,
        hold_at_prepress_if_visible=lambda: True,
    )
    result = ButtonLightFlow(runtime).run()
    assert result.status == "arrived_after_verified_press"
    assert result.press_attempts == 2
    assert executor.near == 1


def test_observer_reacquires_new_target_frame_instead_of_using_missing_box():
    class DropoutVision:
        def __init__(self): self.calls=[]
        def request(self, command, after_sequence, target_label):
            self.calls.append(after_sequence)
            n=len(self.calls)
            return dict(fresh=True,frame_sequence=n,path=f"/tmp/frame-{n}.jpg",
                        image_size=[640,360],capture_time_unix_sec=time.time(),
                        target_detection=None if n==1 else {"bbox_xyxy":[20,20,50,50],"confidence":.95,"frame_sequence":n})
    vision=DropoutVision();recognizer=FakeRecognizer()
    observer=FreshLightObserver(vision,recognizer,"15")
    assert observer.observe("near")=="ON"
    assert vision.calls==[-1,1]
    assert observer.audit[0]['frame_sequence']==2
    assert recognizer.paths==[("/tmp/frame-2.jpg","15")]


def test_prepress_scope_skips_unused_reaching_but_keeps_press_and_retreat_checks():
    plan=_plan();bridge=FakeBridge();guarded=FakeGuardedBridge();robot=FakeRobot()
    robot.target=np.asarray(plan.observation_to_prepress[-1]['positions'])
    events=[]
    executor=DryValidatedSegmentExecutor(bridge,robot,'token',guarded_bridge=guarded,
        guarded_press_config=_guarded_config(),event_sink=lambda kind,value:events.append((kind,value)))
    executor.load_plan(plan,already_at_prepress=True)
    assert set(executor.validations)=={'press_cycle','retreat_to_observation','observation_to_waiting','guarded_press_forward'}
    assert len(bridge.calls)==3 and len(guarded.calls)==1
    assert all(opts['dry_run'] for _,opts in bridge.calls+guarded.calls)
    assert events[-1][1]['skipped_unused_segments']==['approach_to_observation','observation_to_prepress']
    assert executor._require_plan() is plan
    with pytest.raises(RuntimeError,match='not been validated'):executor.approach_to_observation()
    assert len(bridge.calls)==3


def test_prepress_scope_requires_actual_endpoint_and_invalidates_old_plan():
    robot=FakeRobot();executor=DryValidatedSegmentExecutor(FakeBridge(),robot,'token')
    executor.load_plan(_plan())
    with pytest.raises(RuntimeError,match='not at prepress'):executor.load_plan(_plan(),already_at_prepress=True)
    with pytest.raises(RuntimeError,match='all dry validations'):executor._require_plan()


def test_failed_post_approach_collision_check_cannot_leave_old_plan_ready():
    plan=_plan();robot=FakeRobot();robot.target=np.asarray(plan.observation_to_prepress[-1]['positions'])
    bridge=FakeBridge();executor=DryValidatedSegmentExecutor(bridge,robot,'token')
    executor.load_plan(plan)
    bridge.execute=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('collision'))
    with pytest.raises(RuntimeError,match='collision'):executor.load_plan(plan,already_at_prepress=True)
    with pytest.raises(RuntimeError,match='all dry validations'):executor._require_plan()
