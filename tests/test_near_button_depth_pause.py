import time
import numpy as np
import pytest

from piper_handeye.near_button_depth_pause import NearButtonDepthPause
from piper_handeye.safety_stop import VisualSafetySupervisor
from test_safety_supervisor import FakeStop, FakeVision


def pose(distance, lateral=0):
    p = np.eye(4)
    p[0, 3] = -distance
    p[1, 3] = lateral
    return p


def test_only_explicit_fresh_aligned_near_target_is_paused():
    gate = NearButtonDepthPause()
    def update(d, **kw):
        args = dict(sample_time=10, now=10.1, enabled=True, hardware_ok=True)
        args.update(kw)
        return gate.update(pose(d), np.eye(4), **args)
    assert update(.06) == 0
    assert update(.04, enabled=False) == 0
    assert update(.04, hardware_ok=False) == 0
    assert update(.04, sample_time=9) == 0
    assert update(.04) == pytest.approx(10.25)
    assert update(.051) == 0
    assert gate.finished
    assert update(.04) == 0  # Cannot re-enter repeatedly after the return.


def test_panel_normal_gate_does_not_wait_for_lateral_convergence():
    gate = NearButtonDepthPause()
    assert gate.update(
        pose(.04, .03), np.eye(4), 10, 10,
        enabled=True, hardware_ok=True,
    ) > 0


def test_overtravel_and_invalid_pose_reenable_depth():
    for tcp in [pose(-.02), np.full((4, 4), np.nan)]:
        gate = NearButtonDepthPause()
        assert gate.update(tcp, np.eye(4), 10, 10, enabled=True, hardware_ok=True) == 0


def test_pause_expires_after_twenty_five_seconds_despite_fresh_encoders():
    gate = NearButtonDepthPause()
    assert gate.update(pose(.04), np.eye(4), 10, 10, enabled=True, hardware_ok=True) > 0
    assert gate.update(pose(.04), np.eye(4), 25, 25, enabled=True, hardware_ok=True) > 0
    assert gate.update(pose(.04), np.eye(4), 34.9, 34.9, enabled=True, hardware_ok=True) == pytest.approx(35.0)
    assert gate.started == 10  # Fresh encoders do not restart the overall clock.
    assert gate.update(pose(.04), np.eye(4), 35, 35, enabled=True, hardware_ok=True) == 0
    assert gate.finished


def test_new_press_cycle_can_reenter_after_previous_retreat():
    gate = NearButtonDepthPause(entry_distance_m=.15)
    assert gate.update(pose(.10), np.eye(4), 10, 10,
                       enabled=True, hardware_ok=True) > 0
    assert gate.update(pose(.16), np.eye(4), 11, 11,
                       enabled=True, hardware_ok=True) == 0
    assert gate.finished
    gate.begin_cycle()
    assert not gate.finished and gate.started is None and not gate.active
    assert gate.update(pose(.10), np.eye(4), 12, 12,
                       enabled=True, hardware_ok=True) > 0


def test_depth_alarm_is_suppressed_only_until_lease_expires():
    stop = FakeStop()
    supervisor = VisualSafetySupervisor(FakeVision([{
        'ready': True, 'blocked': True, 'reason': 'absolute_near_field_object',
    }]), stop, 'test', poll_sec=.001, auto_resume=False)
    try:
        supervisor.depth_pause_until = time.monotonic() + .08
        supervisor.start()
        supervisor.arm()
        time.sleep(.02)
        assert not stop.stops
        assert supervisor.triggered.wait(1)
        assert len(stop.stops) == 1
        assert stop.resets == 0
    finally:
        supervisor.close()


def test_vision_rpc_failure_still_stops_during_pause():
    class BrokenVision:
        def request(self, *_a, **_kw):
            raise RuntimeError('connection lost')
    stop = FakeStop()
    supervisor = VisualSafetySupervisor(BrokenVision(), stop, 'test', poll_sec=.001)
    try:
        supervisor.depth_pause_until = time.monotonic() + .25
        supervisor.start()
        supervisor.arm()
        assert supervisor.triggered.wait(1)
        assert stop.stops
    finally:
        supervisor.close()
