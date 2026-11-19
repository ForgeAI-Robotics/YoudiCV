"""Offline timing evidence, without opening robot or camera connections."""
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from piper_handeye.pose_timing import PoseTimingTrace, collect_pose_timing
from piper_handeye.button_light_hardware import VisualSegmentMotionSafety
from piper_handeye.depth_safety import DepthPanelSafetyMonitor, DepthSafetyConfig
from run_button_vision_server import VisionService


def test_long_wait_buffers_are_bounded_and_preserve_startup():
    trace = PoseTimingTrace(first_limit=2, recent_limit=2, slow_limit=2)
    for i in range(10000):
        trace.record(dict(start_monotonic=float(i), rpc_sec=i/10000), slow=True)
    s = trace.snapshot()
    assert s['count'] == 10000
    assert [r['start_monotonic'] for r in s['first']] == [0., 1.]
    assert [r['start_monotonic'] for r in s['recent']] == [9998., 9999.]
    assert len(s['slow']) == 2
    assert s['metrics']['rpc_sec']['maximum'] == pytest.approx(.9999)


def test_pose_update_records_real_lock_wait_without_replacing_old_stamps():
    monitor = DepthPanelSafetyMonitor(None, lambda: None, DepthSafetyConfig())
    stamp = time.monotonic()
    entered = threading.Event()
    result = []
    def worker():
        entered.set()
        result.append(monitor.update_camera_pose(stamp, np.eye(4)))
    with monitor._lock:
        t = threading.Thread(target=worker)
        t.start()
        assert entered.wait(.5)
        time.sleep(.03)
    t.join(1.)
    assert not t.is_alive()
    assert result[0]['lock_wait_sec'] >= .02
    original = monitor._latest_camera_pose
    rejected = monitor.update_camera_pose(stamp-1., np.eye(4)*2.)
    assert rejected['accepted'] is False
    assert monitor._latest_camera_pose is original
    assert monitor.config.camera_pose_max_age_sec == .35


def test_publisher_records_state_read_rpc_and_original_sample_time():
    stamp = time.monotonic()
    calls = []
    supervisor = SimpleNamespace(depth_pause_until=0.)
    sample = SimpleNamespace(timestamp=stamp, base_to_gripper=np.eye(4))
    def read(max_age):
        assert max_age == .25
        time.sleep(.01)
        return sample
    def request(command, **kwargs):
        calls.append((command, kwargs))
        time.sleep(.02)
        safety.closed.set()
        return dict(ok=True, pose_timing={'lock_wait_sec': .006},
                    server_request_timing={'elapsed_sec': .012})
    safety = VisualSegmentMotionSafety(supervisor, SimpleNamespace(request=request),
                                      SimpleNamespace(state=read), np.eye(4))
    safety._publish_pose()
    assert [c[0] for c in calls] == ['safety_camera_pose']
    assert calls[0][1]['timestamp_monotonic'] == stamp
    row = safety._pose_timing.snapshot()['first'][0]
    assert row['state_read_sec'] >= .009
    assert row['rpc_sec'] >= .019
    assert row['feedback_sample_monotonic'] == stamp
    assert row['service']['lock_wait_sec'] == .006
    assert safety.publisher_error is None


def test_publisher_error_still_sets_original_failure_state():
    supervisor = SimpleNamespace(depth_pause_until=2.)
    def read(max_age):
        safety.closed.set()
        raise TimeoutError('simulated missing encoder feedback')
    safety = VisualSegmentMotionSafety(supervisor, object(), SimpleNamespace(state=read), np.eye(4))
    safety._publish_pose()
    assert safety.publisher_error.startswith('TimeoutError:')
    assert supervisor.depth_pause_until == 0.
    assert safety._pose_timing.snapshot()['slow'][0]['error_type'] == 'TimeoutError'


def test_collection_is_read_only_separate_timeout_and_failure_is_optional(monkeypatch):
    calls = []
    class Probe:
        def __init__(self, host, port, token, timeout):
            calls.append(('connection', timeout))
        def request(self, command):
            calls.append(('command', command))
            raise TimeoutError('simulated diagnostic failure')
    monkeypatch.setattr('piper_handeye.vision_rpc.VisionRpcClient', Probe)
    vision = SimpleNamespace(host='127.0.0.1', port=1, token='offline', timeout=12.)
    safety = SimpleNamespace(vision=vision, _pose_timing=PoseTimingTrace())
    evidence = collect_pose_timing(safety, SimpleNamespace(_pose_timing=PoseTimingTrace()))
    assert evidence['service']['diagnostic_error'] == 'TimeoutError'
    assert calls == [('connection', .5), ('command', 'safety_pose_timing')]
    assert vision.timeout == 12.
    assert collect_pose_timing(None, None)['service']['reason'] == 'safety_not_initialized'


def test_timing_rpc_does_not_change_safety_or_activate_perception():
    service = VisionService.__new__(VisionService)
    service.args = SimpleNamespace(token='offline')
    monitor = DepthPanelSafetyMonitor(None, lambda: None, DepthSafetyConfig())
    service.safety = monitor
    service.activity = SimpleNamespace(touch=lambda: pytest.fail('diagnostic must not renew active lease'))
    before = monitor.status()
    r = service.dispatch({'command': 'safety_pose_timing', 'token': 'offline'})
    assert r['ok']
    assert r['evidence']['pose_frame_max_difference_sec'] == .35
    assert monitor.status() == before
    with pytest.raises(PermissionError):
        service.dispatch({'command': 'safety_pose_timing', 'token': 'invalid'})
