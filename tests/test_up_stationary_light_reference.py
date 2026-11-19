import time
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from piper_handeye.button_light_hardware import DryValidatedSegmentExecutor, FreshLightObserver
from piper_handeye.paired_button_light import PairedButtonLight
from tests.test_paired_button_light import patch_image, snapshot


def executor_fixture(*, fail_validation=False, wrong_pose=False):
    trace=[];joints=np.zeros(6)
    plan=SimpleNamespace(observation_joints=np.zeros(6),observation_distance_m=.05,
                         observation_to_prepress=[{'positions':[.1]*6}])
    e=DryValidatedSegmentExecutor.__new__(DryValidatedSegmentExecutor)
    e._require_plan=lambda:plan;e.validations={};e.token='test';e.event_sink=lambda name, value: trace.append('evidence') if name=='stationary_press_reference_confirmed' else None
    def bridge(*a,**k):
        assert k['dry_run'];trace.append('validate')
        if fail_validation:raise RuntimeError('collision')
        return {'status':'validated'}
    e.bridge=SimpleNamespace(execute=bridge)
    e.robot=SimpleNamespace(state=lambda **k:SimpleNamespace(joints=joints.copy(),raw={'hardware_ok':1},timestamp=time.monotonic()))
    e.robot.latest_state=e.robot.state
    def retreat(distance):
        assert distance==.05;trace.append('retreat')
        joints[:]=.02 if wrong_pose else 0
    e.retreat_to_observation=retreat
    e.begin_press_attempt=lambda:trace.append('rearm')
    e.press_from_observation=lambda:trace.append('press') or []
    return e,trace,joints


def test_first_press_validates_then_observes_same_retract_pose_before_rearming():
    e,trace,_=executor_fixture()
    e.press_with_stationary_reference(lambda:trace.append('capture') or {},already_at_observation=False)
    assert trace==['validate','retreat','capture','evidence','rearm','press']


def test_retry_uses_current_observation_without_an_extra_retreat():
    e,trace,_=executor_fixture();e.validations['observation_to_prepress']={}
    e.press_with_stationary_reference(lambda:trace.append('capture') or {},already_at_observation=True)
    assert trace==['capture','evidence','rearm','press']


@pytest.mark.parametrize('failure',['collision','position','capture','drift'])
def test_unusable_reference_never_starts_press(failure):
    e,trace,joints=executor_fixture(fail_validation=failure=='collision',wrong_pose=failure=='position')
    def capture():
        trace.append('capture')
        if failure=='capture':raise RuntimeError('cannot confirm OFF')
        if failure=='drift':joints[:]=.02
        return {}
    with pytest.raises(RuntimeError):e.press_with_stationary_reference(capture,already_at_observation=False)
    assert 'press' not in trace and 'rearm' not in trace


def test_fixed_mode_does_not_allow_random_approach_reference_to_authorize_press():
    paired=PairedButtonLight(require_stationary_reference=True)
    paired.references.append(patch_image())
    with pytest.raises(RuntimeError,match='stationary'):paired.freeze()
    paired.stationary_reference_ready=True;paired.freeze()
    paired.begin_attempt()
    assert not paired.stationary_reference_ready and not paired.references


def test_reference_capture_excludes_premove_frame_and_confirms_two_new_off(tmp_path):
    class Vision:
        n=0
        def request(self,*a,**k):
            self.n+=1
            frame=patch_image(self.n)
            if self.n==1:frame.captured-=10
            return snapshot(tmp_path,frame)
    vision=Vision();paired=PairedButtonLight(tmp_path/'pairs',require_stationary_reference=True);paired.begin_attempt()
    obs=FreshLightObserver(vision,SimpleNamespace(observe=lambda *a,**k:{'state':'OFF'}),'up',paired_light=paired)
    result=obs.capture_stationary_off_reference(timeout_sec=1)
    assert result['sequences']==[2,3]
    assert paired.stationary_reference_ready
    assert min(p.captured for p in paired.references)>result['capture_barrier_unix']
    paired.freeze()


def test_no_off_reference_does_not_become_ready(tmp_path):
    class Vision:
        n=0
        def request(self,*a,**k):
            self.n+=1
            return snapshot(tmp_path,patch_image(self.n,increase=80))
    paired=PairedButtonLight(require_stationary_reference=True);paired.begin_attempt()
    obs=FreshLightObserver(Vision(),SimpleNamespace(observe=lambda *a,**k:{'state':'ON'}),'up',paired_light=paired)
    with pytest.raises(RuntimeError,match='press withheld'):obs.capture_stationary_off_reference(timeout_sec=.05)
    assert not paired.stationary_reference_ready
