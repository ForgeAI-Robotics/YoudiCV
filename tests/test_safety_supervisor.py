from collections import deque
import time

from piper_handeye.safety_stop import (
    VisualSafetySupervisor,
    safety_status_is_allowed_degraded_commit,
    safety_status_requires_stop,
)


class FakeVision:
    def __init__(self, statuses):
        self.statuses = deque(statuses)
        self.last = statuses[-1]

    def request(self, command, **values):
        if command == "safety_motion_state":
            return {"paused": values["paused"]}
        if command == "snapshot":
            return {
                "ok": True,
                "path": "/tmp/rgb.jpg",
                "depth_mm_path": "/tmp/depth_mm.png",
                "depth_color_path": "/tmp/depth_color.png",
            }
        assert command == "safety"
        if self.statuses:
            self.last = self.statuses.popleft()
        return {"visual_safety": self.last}


class FakeStop:
    def __init__(self):
        self.stops = []
        self.resets = 0

    def health(self, **_):
        return {"status": "ok"}

    def reset_latch(self, **_):
        self.resets += 1
        return {"status": "ok"}

    def stop(self, **values):
        self.stops.append(values)
        return {"status": "stopped"}


def clear(sequence=1):
    return {"ready": True, "blocked": False, "reason": "clear", "frame_sequence": sequence}


def test_preflight_accepts_clear_guard():
    stop = FakeStop()
    supervisor = VisualSafetySupervisor(FakeVision([clear()]), stop, "token")
    assert supervisor.preflight(0.1)["reason"] == "clear"
    assert not stop.stops


def test_stale_pose_reason_is_forwarded_without_changing_stop_protection():
    import threading
    event = threading.Event(); rows = []
    status = {"ready": True, "blocked": True, "reason": "matrix_monitor_failure_latched",
              "last_obstruction": {"reason": "matrix_camera_pose_stale"}, "frame_sequence": 12}
    def emitted(kind, value):
        rows.append((kind, value)); event.set()
    stop = FakeStop()
    guard = VisualSafetySupervisor(FakeVision([status]), stop, 'token', poll_sec=.001,
                                   auto_resume=False, event_sink=emitted)
    try:
        guard.start(); guard.arm()
        assert event.wait(1)
        assert rows[0][0] == 'visual_safety_paused'
        assert rows[0][1]['underlying_reason'] == 'matrix_camera_pose_stale'
        assert guard.stop_active.is_set() and len(stop.stops) == 1 and stop.resets == 0
    finally:
        guard.close()


def test_recovery_stays_latched_after_obstacle_clears():
    vision = FakeVision([{"ready": False, "blocked": True}, clear()])
    stop = FakeStop()
    guard = VisualSafetySupervisor(vision, stop, "token", poll_sec=.001, auto_resume=False)
    try:
        guard.start()
        guard.arm()
        assert guard.triggered.wait(1)
        time.sleep(.03)
        assert guard.stop_active.is_set()
        assert not guard.resumed.is_set()
        assert stop.resets == 0
        assert len(stop.stops) == 1
    finally:
        guard.close()


def test_recovery_stops_on_unavailable_monitor_even_without_blocked_flag():
    stop = FakeStop()
    guard = VisualSafetySupervisor(
        FakeVision([{**clear(), "monitoring_available": False}]),
        stop, "token", poll_sec=.001, auto_resume=False,
    )
    try:
        guard.start()
        guard.arm()
        assert guard.triggered.wait(1)
        assert stop.resets == 0
        assert guard.stop_active.is_set()
    finally:
        guard.close()


def test_transient_unready_frame_honors_monitor_debounce():
    transient = {
        "ready": False,
        "blocked": False,
        "reason": "matrix_camera_pose_stale",
        "blocked_streak": 1,
    }
    latched = {**transient, "blocked": True, "blocked_streak": 3}
    assert not safety_status_requires_stop(transient)
    assert safety_status_requires_stop(latched)
    assert safety_status_requires_stop({"ready": False})


def test_only_explicit_clear_matrix_coverage_commit_is_allowed():
    allowed = {
        "ready": True,
        "blocked": False,
        "monitoring_available": False,
        "depth_degraded_commit": True,
        "reason": "matrix_coverage_insufficient_degraded_commit",
    }
    assert safety_status_is_allowed_degraded_commit(allowed)
    assert not safety_status_is_allowed_degraded_commit({**allowed, "blocked": True})
    assert not safety_status_is_allowed_degraded_commit(
        {**allowed, "reason": "residual_latched_during_matrix_coverage_loss"}
    )


def test_armed_commit_allows_only_first_coverage_confirmation_frame():
    confirming = {
        "ready": True,
        "blocked": False,
        "monitoring_available": False,
        "depth_degraded_commit_armed": True,
        "depth_degraded_commit": False,
        "reason": "matrix_prediction_coverage_transient",
        "matrix_unavailable_streak": 1,
    }
    assert safety_status_is_allowed_degraded_commit(confirming)
    assert not safety_status_is_allowed_degraded_commit(
        {**confirming, "matrix_unavailable_streak": 2}
    )
    assert not safety_status_is_allowed_degraded_commit(
        {**confirming, "reason": "rgb_intrusion_during_matrix_coverage_loss"}
    )


def test_latched_supervisor_accepts_armed_confirmation_frame():
    status = {
        "ready": True,
        "blocked": False,
        "monitoring_available": False,
        "depth_degraded_commit_armed": True,
        "depth_degraded_commit": False,
        "reason": "matrix_prediction_coverage_transient",
        "matrix_unavailable_streak": 1,
        "frame_sequence": 9,
    }
    stop = FakeStop()
    guard = VisualSafetySupervisor(
        FakeVision([status]), stop, "token", poll_sec=.001, auto_resume=False,
    )
    try:
        guard.start()
        guard.arm()
        time.sleep(.03)
        assert not guard.triggered.is_set()
        assert not stop.stops
    finally:
        guard.close()


def test_latched_supervisor_accepts_explicit_degraded_coverage_commit():
    status = {
        "ready": True,
        "blocked": False,
        "monitoring_available": False,
        "depth_degraded_commit": True,
        "reason": "matrix_coverage_insufficient_degraded_commit",
        "frame_sequence": 9,
    }
    stop = FakeStop()
    guard = VisualSafetySupervisor(
        FakeVision([status]), stop, "token", poll_sec=.001, auto_resume=False,
    )
    try:
        guard.start()
        guard.arm()
        time.sleep(.03)
        assert not guard.triggered.is_set()
        assert not stop.stops
    finally:
        guard.close()


def test_preflight_rejects_blind_monitoring_even_if_cached_state_is_clear():
    stop = FakeStop()
    blind = {
        "ready": True,
        "blocked": False,
        "monitoring_available": False,
        "reason": "target_temporarily_not_visible",
        "frame_sequence": 4,
    }
    supervisor = VisualSafetySupervisor(
        FakeVision([blind]), stop, "token", poll_sec=0.001
    )
    try:
        supervisor.preflight(0.01)
    except RuntimeError as exc:
        assert "not clear" in str(exc)
    else:
        raise AssertionError("blind visual monitoring must not pass preflight")


def test_armed_supervisor_stops_once_on_obstacle():
    stop = FakeStop()
    emitted = []
    blocked = {"ready": True, "blocked": True, "reason": "unexpected_foreground", "frame_sequence": 9}
    supervisor = VisualSafetySupervisor(
        FakeVision([clear(), blocked, clear(10)]), stop, "token", poll_sec=0.001,
        event_sink=lambda kind, value: emitted.append((kind, value)),
    )
    supervisor.preflight(0.1)
    supervisor.start()
    supervisor.arm()
    assert supervisor.triggered.wait(0.2)
    assert supervisor.resumed.wait(0.2)
    time.sleep(0.01)
    supervisor.close()
    assert len(stop.stops) == 1
    assert stop.stops[0]["frame_sequence"] == 9
    assert len(supervisor.events) == 1
    capture = supervisor.events[0]["emergency_depth_snapshot"]
    assert capture["depth_mm_path"].endswith("depth_mm.png")
    assert supervisor.events[0]["emergency_depth_snapshot_error"] is None
    assert [kind for kind, _value in emitted] == [
        "visual_safety_paused", "visual_safety_resumed"
    ]


def test_supervisor_can_be_explicitly_rearmed_after_its_recovery():
    stop = FakeStop()
    blocked = {
        "ready": True, "blocked": True,
        "reason": "unexpected_foreground", "frame_sequence": 9,
    }
    supervisor = VisualSafetySupervisor(
        FakeVision([clear(), blocked, clear(10)]), stop, "token", poll_sec=0.001
    )
    try:
        supervisor.preflight(0.1)
        supervisor.start()
        supervisor.arm()
        assert supervisor.triggered.wait(0.2)
        supervisor.disarm()
        assert supervisor.resumed.wait(0.2)
        supervisor.rearm_after_recovery()
        assert supervisor.armed.is_set()
        assert not supervisor.triggered.is_set()
        assert not supervisor.resumed.is_set()
    finally:
        supervisor.close()


def test_supervisor_rearms_after_clear_and_handles_second_obstacle():
    stop = FakeStop()
    blocked_1 = {
        "ready": True, "blocked": True,
        "reason": "unexpected_foreground", "frame_sequence": 9,
    }
    blocked_2 = {
        "ready": True, "blocked": True,
        "reason": "unexpected_foreground", "frame_sequence": 19,
    }
    supervisor = VisualSafetySupervisor(
        FakeVision([clear(), blocked_1, clear(10), blocked_2, clear(20)]),
        stop,
        "token",
        poll_sec=0.001,
    )
    supervisor.preflight(0.1)
    supervisor.start()
    supervisor.arm()
    deadline = time.monotonic() + 0.2
    while len(supervisor.events) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    supervisor.close()
    assert [value["frame_sequence"] for value in stop.stops] == [9, 19]
    assert len(supervisor.events) == 2
