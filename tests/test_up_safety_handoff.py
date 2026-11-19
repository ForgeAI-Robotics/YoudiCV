"""Exercise reference handoff against the real depth monitor's latch logic."""
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from piper_handeye.button_light_hardware import VisualSegmentMotionSafety
from tests.test_depth_safety import matrix_monitor, matrix_frame


def setup(monkeypatch):
    metrics={'matrix_monitoring_available':False,'blocked_raw':False,
             'matrix_residual_detected_raw':False,'matrix_largest_component_pixels':0,
             'coverage_fraction':.048}
    monitor=matrix_monitor(monkeypatch,metrics)
    monitor.set_depth_degraded_commit(True)
    calls=[]
    class Vision:
        n=100
        freeze=False
        def request(self,command,**kw):
            calls.append((command,kw))
            if command=='safety_camera_pose':
                monitor.update_camera_pose(kw['timestamp_monotonic'],np.array(kw['base_to_camera']))
            elif command=='safety_motion_state': monitor.set_motion_paused(kw['paused'])
            elif command=='safety_set_depth_degraded_commit': return monitor.set_depth_degraded_commit(kw['enabled'])
            elif command=='safety':
                if not self.freeze:
                    self.n+=1;monitor._process(matrix_frame(self.n))
                return {'visual_safety':monitor.status()}
            else:raise AssertionError(command)
            return {'ok':True}
    vision=Vision()
    class Supervisor:
        error=None
        depth_pause_until=0
        def __init__(self):
            self.armed=threading.Event();self.triggered=threading.Event();self.stop_active=threading.Event();self.arm_calls=0
        def disarm(self): self.armed.clear()
        def arm(self):
            assert not monitor.status()['blocked']
            self.armed.set();self.arm_calls+=1
    supervisor=Supervisor()
    pose=np.eye(4);target=np.eye(4);target[0,3]=.05
    state={'age':0,'health':1}
    robot=SimpleNamespace(state=lambda **kw:SimpleNamespace(base_to_gripper=pose,
        timestamp=time.monotonic()-state['age'],raw={'hardware_ok':state['health']}))
    safety=VisualSegmentMotionSafety(supervisor,vision,robot,np.eye(4),pause_depth_near_button=True,
                                    depth_degraded_entry_distance_m=.15,stationary_reference_handoff=True)
    safety.set_target(target);safety.depth_degraded_commit_active=True;safety.near_pause.started=time.monotonic()-2
    normal_wait=safety._wait_reference_status
    safety._wait_reference_status=lambda **kw:normal_wait(**kw,timeout_sec=.08)
    vision.request('safety')
    return safety,vision,monitor,supervisor,metrics,calls,state


def test_old_retreat_policy_reproduces_coverage_latch_without_any_obstacle(monkeypatch):
    s,v,m,guard,metrics,calls,state=setup(monkeypatch)
    s.disable_for_return()
    for _ in range(4):v.request('safety')
    assert m.status()['blocked']
    assert m.status()['reason']=='matrix_prediction_coverage_insufficient'
    assert not metrics['blocked_raw']


@pytest.mark.parametrize('reference_frames',[4,45,60])
def test_up_reference_hold_does_not_latch_depth_blind_zone_or_depend_on_timing(monkeypatch,reference_frames):
    s,v,m,guard,metrics,calls,state=setup(monkeypatch)
    original_start=s.near_pause.started
    with s.stationary_reference_transition():
        s.disable_for_return()  # actual validated prepress->observation retreat
        for _ in range(reference_frames):
            assert not v.request('safety')['visual_safety']['blocked']
        s.begin_outbound_cycle()
        s.before_segment('observation_to_prepress')
        assert guard.arm_calls==1 and m.status()['depth_monitoring_suppressed']
        assert s.near_pause.started==original_start  # no lease extension
    assert not s._reference_hold
    assert not any(cmd=='safety_set_depth_degraded_commit' and not data['enabled'] for cmd,data in calls)
    assert s._reference_observation_hold
    s.finish_near_observation()
    s.disable_for_return()  # final return explicitly ends the near observation policy
    assert not s.depth_degraded_commit_active and s.near_pause.finished
    assert not guard.armed.is_set()


@pytest.mark.parametrize('fault',['stale','health','far','expired','stop','residual','frozen'])
def test_bad_handoff_never_arms_and_never_clears_latches(monkeypatch,fault):
    s,v,m,guard,metrics,calls,state=setup(monkeypatch)
    if fault=='stale':state['age']=1
    if fault=='health':state['health']=0
    if fault=='far':s.target[0,3]=.3
    if fault=='expired':s.near_pause.started=time.monotonic()-26
    if fault=='stop':guard.stop_active.set()
    if fault=='residual':m._latched_blocked=True;m._matrix_latch_reason='residual'
    if fault=='frozen':v.freeze=True
    with pytest.raises(RuntimeError):
        with s.stationary_reference_transition():s.before_segment('observation_to_prepress')
    assert guard.arm_calls==0 and not s._reference_hold
    assert all('reset' not in cmd for cmd,_ in calls)
    if fault=='stop':assert guard.stop_active.is_set()
    if fault=='residual':assert m._latched_blocked


def test_rpc_failure_releases_context_flag_without_submitting_motion(monkeypatch):
    s,v,m,guard,metrics,calls,state=setup(monkeypatch)
    request=v.request
    def failing(command,**kw):
        if command=='safety_set_depth_degraded_commit':raise RuntimeError('RPC failed')
        return request(command,**kw)
    v.request=failing
    with pytest.raises(RuntimeError,match='RPC failed'):
        with s.stationary_reference_transition():s.before_segment('observation_to_prepress')
    assert not s._reference_hold and guard.arm_calls==0


def test_no_ack_of_near_policy_prevents_inward_arm(monkeypatch):
    s,v,m,guard,metrics,calls,state=setup(monkeypatch)
    request=v.request
    def missing_ack(command,**kw):
        result=request(command,**kw)
        if command=='safety':result['visual_safety'].pop('depth_monitoring_suppressed',None)
        return result
    v.request=missing_ack
    with pytest.raises(RuntimeError,match='handoff not ready'):
        with s.stationary_reference_transition():s.before_segment('observation_to_prepress')
    assert guard.arm_calls==0 and not s._reference_hold


def test_executor_wraps_actual_retreat_capture_and_reentry_in_handoff(monkeypatch):
    from tests.test_up_stationary_light_reference import executor_fixture
    safety,vision,monitor,guard,metrics,calls,state=setup(monkeypatch)
    executor,trace,_=executor_fixture()
    executor.motion_safety=safety
    retreat=executor.retreat_to_observation
    def guarded_retreat(distance):
        safety.disable_for_return();retreat(distance)
        for _ in range(45):vision.request('safety')
    executor.retreat_to_observation=guarded_retreat
    executor.begin_press_attempt=safety.begin_outbound_cycle
    def inward():
        safety.before_segment('observation_to_prepress')
        trace.append('press');return []
    executor.press_from_observation=inward
    def capture():
        assert not monitor.status()['blocked']
        assert safety._reference_hold
        trace.append('capture');return {}
    executor.press_with_stationary_reference(capture,already_at_observation=False)
    assert trace==['validate','retreat','capture','evidence','press']
    assert guard.arm_calls==1 and not safety._reference_hold


def test_expiration_during_reference_capture_never_arms(monkeypatch):
    safety,vision,monitor,guard,metrics,calls,state=setup(monkeypatch)
    with pytest.raises(RuntimeError,match='lease expired'):
        with safety.stationary_reference_transition():
            safety.disable_for_return()
            safety.near_pause.started=time.monotonic()-26
            safety.begin_outbound_cycle()
            safety.before_segment('observation_to_prepress')
    assert not safety._reference_hold and guard.arm_calls==0


@pytest.mark.parametrize('elapsed', [16, 24])
def test_reference_handoff_uses_extended_lease_without_restarting_timer(monkeypatch, elapsed):
    safety, vision, monitor, guard, metrics, calls, state = setup(monkeypatch)
    original_start = time.monotonic() - elapsed
    safety.near_pause.started = original_start
    with safety.stationary_reference_transition():
        safety.begin_outbound_cycle()
        safety.before_segment('observation_to_prepress')
    assert guard.arm_calls == 1
    assert safety.near_pause.started == original_start
    assert not safety.near_pause.finished


@pytest.mark.parametrize('elapsed, expires', [(16, False), (24.9, False), (25, True), (26, True)])
def test_pose_publisher_commit_uses_same_twenty_five_second_deadline(monkeypatch, elapsed, expires):
    import piper_handeye.button_light_hardware as hardware
    safety, vision, monitor, guard, metrics, calls, state = setup(monkeypatch)
    now = 100.0
    safety.near_pause.started = now - elapsed
    safety.robot.state = lambda **kw: SimpleNamespace(
        base_to_gripper=np.eye(4), timestamp=now, raw={'hardware_ok': 1})
    guard.armed.set()
    monkeypatch.setattr(hardware, 'time', SimpleNamespace(monotonic=lambda: now))
    safety.closed.wait = lambda _timeout: safety.closed.set()
    safety._publish_pose()
    assert safety.publisher_error is None
    assert safety.near_pause.finished is expires
    assert safety.depth_degraded_commit_active is not expires
    expiry_events = [e for e in safety.depth_degraded_commit_events if e.get('reason') == 'lease_expired']
    assert bool(expiry_events) is expires
