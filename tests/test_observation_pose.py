"""No motion: photo-pose tolerance, distinct feedback, drift and expiry gates."""
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock
import time
import numpy as np
import pytest

from piper_handeye.observation_pose import (
    OBSERVATION_JOINT_TOLERANCE_RAD, OBSERVATION_STABLE_SEC,
    OBSERVATION_CONFIRM_TIMEOUT_SEC, OBSERVATION_STABILITY_DELTA_RAD,
    ObservationPoseError, wait_for_stable_observation_pose)


class Clock:
    def __init__(self): self.now=10.
    def __call__(self): return self.now
    def sleep(self,seconds): self.now+=seconds


def confirm(value, *, safety=lambda:None):
    clock=Clock()
    def read():
        q,stamp,hardware=value(clock)
        return NS(joints=q,timestamp=stamp,raw={'hardware_ok':hardware})
    return wait_for_stable_observation_pose([0.]*6,read,safety,clock=clock,sleep=clock.sleep)


@pytest.mark.parametrize('offset',[0.,.0053,.006,.008])
def test_new_tolerance_accepts_distinct_stable_samples(offset):
    _,audit=confirm(lambda c:([offset]*6,c(),1))
    assert audit['outcome']=='stable' and audit['distinct_samples']>=2
    assert .08<=audit['elapsed_sec']<.12
    assert audit['tolerance_rad']==.008
    assert audit['stable_window_spread_rad']==0


def test_limits_remain_explicit_and_shared():
    assert OBSERVATION_JOINT_TOLERANCE_RAD==.008
    assert OBSERVATION_STABLE_SEC==.08
    assert OBSERVATION_CONFIRM_TIMEOUT_SEC==1.0
    assert OBSERVATION_STABILITY_DELTA_RAD==.002


def test_short_out_of_tolerance_settling_is_observed_without_resending():
    _,audit=confirm(lambda c:([.009 if c()<10.07 else .0053]*6,c(),1))
    assert audit['final_error_rad']==.0053
    assert .15<=audit['elapsed_sec']<.22
    assert any(r['max_error_rad']>.008 for r in audit['samples'])


def test_persistent_out_of_tolerance_does_not_pass():
    with pytest.raises(ObservationPoseError,match='timeout') as failure:
        confirm(lambda c:([.0081]*6,c(),1))
    assert 1.<=failure.value.audit['elapsed_sec']<1.02


def test_timestamp_reuse_never_counts_as_distinct_or_stable():
    # Even a perfectly positioned fresh cache cannot fake two encoder packets.
    clock=Clock();stamp=clock()
    def read():return NS(joints=[0.]*6,timestamp=stamp,raw={'hardware_ok':1})
    with pytest.raises(ObservationPoseError):
        wait_for_stable_observation_pose([0.]*6,read,lambda:None,clock=clock,sleep=clock.sleep)


@pytest.mark.parametrize('fault',['stale','future','bad_timestamp','nan','hardware','shape'])
def test_invalid_feedback_is_rejected(fault):
    def value(c):
        stamp=c()-.251 if fault=='stale' else c()+.01 if fault=='future' else float('nan') if fault=='bad_timestamp' else c()
        q=[float('nan')]*6 if fault=='nan' else [0.]*5 if fault=='shape' else [0.]*6
        return q,stamp,0 if fault=='hardware' else 1
    with pytest.raises((ObservationPoseError,ValueError)):confirm(value)


def test_reverse_feedback_timestamp_is_rejected():
    i=[0]
    def value(c):
        i[0]+=1
        return [0.]*6,c() if i[0]==1 else c()-.02,1
    with pytest.raises(ObservationPoseError,match='reversed'):confirm(value)


def test_motion_within_target_tolerance_still_needs_stationary_window():
    i=[0]
    def value(c):
        i[0]+=1
        return [.007 if i[0]%2 else -.007]*6,c(),1
    with pytest.raises(ObservationPoseError,match='timeout') as failure:confirm(value)
    assert failure.value.audit['motion_resets']>0


def test_safety_stop_during_confirmation_propagates_without_motion():
    calls=[0]
    def safety():
        calls[0]+=1
        if calls[0]==4:raise RuntimeError('new safety stop')
    with pytest.raises(RuntimeError,match='new safety stop'):confirm(lambda c:([.0053]*6,c(),1),safety=safety)


def test_real_robot_nonblocking_snapshot_does_not_wait_or_send():
    from piper_handeye.robot import PiperUdpRobot,RobotSample
    robot=PiperUdpRobot.__new__(PiperUdpRobot)
    robot._condition=threading.Condition();robot._running=True;robot._latest_state=None
    robot._send_tree=Mock()
    with pytest.raises(TimeoutError):robot.latest_state(max_age=.25)
    robot._latest_state=RobotSample(time.monotonic(),np.zeros(6),np.eye(4),{'hardware_ok':1})
    assert robot.latest_state(max_age=.25) is robot._latest_state
    robot._running=False
    with pytest.raises(RuntimeError):robot.latest_state(max_age=.25)
    robot._send_tree.assert_not_called()


def test_load_plan_and_stationary_check_accept_stable_0053_pose():
    from piper_handeye.button_light_hardware import DryValidatedSegmentExecutor
    from tests.test_button_light_hardware import FakeRobot,FakeBridge,_plan
    robot=FakeRobot();plan=_plan();robot.target=plan.observation_joints.copy();robot.target[4]+=.0053
    executor=DryValidatedSegmentExecutor(FakeBridge(),robot,'test')
    executor.load_plan(plan,already_at_observation=True)
    assert executor.stationary_observation_verified()
    robot.target[4]+=.003
    assert not executor.stationary_observation_verified()


def test_stationary_capture_accepts_offset_but_rejects_drift_inside_target_bound():
    from tests.test_up_stationary_light_reference import executor_fixture
    e,trace,joints=executor_fixture();e.validations['observation_to_prepress']={}
    joints[:]=.0053
    e.press_with_stationary_reference(lambda:{},already_at_observation=True)
    assert 'press' in trace
    e,trace,joints=executor_fixture();e.validations['observation_to_prepress']={}
    def capture():joints[:]=.007;return {}
    with pytest.raises(RuntimeError,match='arm moved'):
        e.press_with_stationary_reference(capture,already_at_observation=True)
    assert 'press' not in trace and 'rearm' not in trace



def test_read_delay_cannot_authorize_after_confirmation_deadline():
    clock=Clock()
    def read():
        clock.sleep(.6)
        return NS(joints=[0.]*6,timestamp=clock(),raw={'hardware_ok':1})
    with pytest.raises(ObservationPoseError,match='timeout'):
        wait_for_stable_observation_pose([0.]*6,read,lambda:None,clock=clock,sleep=clock.sleep)


def test_pre_confirmation_cached_packet_is_not_endpoint_evidence():
    clock=Clock()
    def read():
        stamp=9.99 if clock()<10.1 else clock()
        return NS(joints=[.0053]*6,timestamp=stamp,raw={'hardware_ok':1})
    _,audit=wait_for_stable_observation_pose([0.]*6,read,lambda:None,clock=clock,sleep=clock.sleep)
    assert audit['elapsed_sec']>=.18
    assert all(row['timestamp']>=10.1 for row in audit['samples'])
