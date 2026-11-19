"""OFF-only change of photo pose, with no hardware or network access."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from piper_handeye.button_light_flow import ButtonLightFlow,production_flow_config,ButtonFlowConfig
from piper_handeye.button_light_hardware import DryValidatedSegmentExecutor
from piper_handeye.retry_observation import (
    plan_retry_observation_transition,checked_retry_distance,RETRY_PREPARATION_DISTANCE_M,
)
from piper_handeye.kinematics import forward_kinematics_batch
from tests.test_button_light_flow import FakeHardware


def case():
    p=Path(__file__).parent/'fixtures/retry_panel_localization/historical4.json'
    c=json.loads(p.read_text())['cases'][1]
    panel=np.array(c['initial_plane']);panel[:3,2]=[-1,0,0]
    return c,panel


@pytest.mark.parametrize('button',['4','6','up','down'])
def test_first_press_stays_5cm_and_success_never_requests_8cm(button):
    h=FakeHardware(['OFF','OFF'],['ON'])
    cfg=production_flow_config(button,{})
    assert cfg.near_observation_distance_m==.05 and cfg.retry_observation_distance_m==.08
    result=ButtonLightFlow(h,cfg).run()
    assert result.press_attempts==1
    assert [a for a in h.actions if a[0]=='near']==[('near',.05)]


@pytest.mark.parametrize('button',['4','up','down'])
def test_retry_observation_requests_8cm_after_off_only(button):
    h=FakeHardware(['OFF','OFF'],['OFF','OFF','ON'])
    result=ButtonLightFlow(h,production_flow_config(button,{})).run()
    assert result.press_attempts==2
    assert [a for a in h.actions if a[0]=='near']==[('near',.05),('near',.08)]
    retry=next(e for e in result.events if e['event']=='retry_required')
    assert retry['next_retry_panel_distance_m']==.08
    observation=[e for e in result.events if e['event']=='at_post_press_observation']
    assert [x['requested_panel_distance_m'] for x in observation]==[.05,.08]


def test_three_attempt_limit_and_legacy_injected_config_preserved():
    h=FakeHardware(['OFF'],['OFF']*6)
    r=ButtonLightFlow(h,production_flow_config('4',{})).run()
    assert r.press_attempts==3 and r.status=='button_not_lit_after_max_attempts'
    assert [a for a in h.actions if a[0]=='near']==[('near',.05),('near',.08),('near',.08)]
    assert ButtonFlowConfig().retry_observation_distance_m is None


def test_uncertain_does_not_trigger_retry_distance_change():
    h=FakeHardware(['OFF'],['UNCERTAIN']*3)
    r=ButtonLightFlow(h,production_flow_config('4',{})).run()
    assert r.press_attempts==1
    assert [a for a in h.actions if a[0]=='near']==[('near',.05)]


def test_historical_pose_can_plan_monotonic_8cm_retreat_and_aim_camera():
    c,panel=case();q=np.array(c['observation_joints'])
    points,target,audit=plan_retry_observation_transition(q,panel,c['handeye'])
    distances=(forward_kinematics_batch(np.array([p['positions'] for p in points]))[:,:3,3]-panel[:3,3])@panel[:3,2]
    assert abs(distances[-1]-.08)<.001 and np.min(np.diff(distances))>=-.002
    assert audit['camera_depth_m']>.21 and audit['camera_lateral_error_m']<.002
    assert checked_retry_distance(target,panel)==pytest.approx(.08,abs=.001)
    # The actual distance can differ by millimetres after a fresh target update;
    # retain the exact validated joint endpoint instead of solving another IK.
    shifted=panel.copy();shifted[0,3]+=.003
    assert checked_retry_distance(target,shifted)==pytest.approx(.083,abs=.001)
    shifted[0,3]+=.03
    with pytest.raises(RuntimeError,match='outside'):checked_retry_distance(target,shifted)


def executor_fixture(*,fail_at=0,drift=False,wrong_pose=False):
    c,panel=case();joints=np.array(c['observation_joints']);trace=[];validations=[0]
    q=joints.copy();q[0]+=.02 if wrong_pose else 0
    plan=NS(observation_joints=q,observation_distance_m=.05,base_to_button=panel,
            observation_to_waiting=[dict(positions=joints.tolist())])
    e=DryValidatedSegmentExecutor.__new__(DryValidatedSegmentExecutor)
    e._require_plan=lambda:plan;e.validations={};e.token='test';e.execute_enabled=True
    e.visual_safety_disabled_for_return=True
    def bridge(points,**kw):
        assert kw['dry_run'];trace.append('validate');validations[0]+=1
        if fail_at==validations[0]:raise RuntimeError('collision rejected')
        return dict(status='validated')
    def read(**kw):
        if drift and validations[0]>=2:joints[0]+=.004
        return NS(joints=joints.copy(),raw=dict(hardware_ok=1))
    e.robot=NS(state=read);e.bridge=NS(execute=bridge)
    e.motion_safety=NS(_assert_clear=lambda:None,disable_for_return=lambda:trace.append('outward_policy'))
    def execute(name,points,target,**kwargs):
        assert name in e.validations and trace[:2]==['validate','validate']
        trace.append('execute');joints[:]=target
        return dict(segment=name,result=dict(status='completed'))
    e._execute=execute
    return e,c,plan,joints,trace


def test_transition_and_return_both_validated_before_any_motion():
    e,c,plan,joints,trace=executor_fixture()
    out=e.prepare_retry_observation(c['handeye'])
    assert trace==['validate','validate','outward_policy','execute']
    assert out['requested_panel_distance_m']==.08 and out['panel_distance_m']==pytest.approx(.08,abs=.001)
    assert np.allclose(joints,out['observation_joints'])
    # After the new full press plan is loaded, attempt3 retains these exact
    # joints even if its measured panel distance is 7.7cm rather than exactly8.
    plan.observation_joints=np.array(out['observation_joints']);plan.observation_distance_m=.077
    second=e.prepare_retry_observation(c['handeye'])
    assert second['mode']=='retry_reused_8cm' and trace.count('execute')==1


@pytest.mark.parametrize('failure',[1,2])
def test_any_transition_or_return_collision_prevents_motion(failure):
    e,c,_,_,trace=executor_fixture(fail_at=failure)
    with pytest.raises(RuntimeError,match='collision'):e.prepare_retry_observation(c['handeye'])
    assert 'execute' not in trace and 'outward_policy' not in trace


@pytest.mark.parametrize('kw,reason',[(dict(drift=True),'moved during'),(dict(wrong_pose=True),'verified photo pose')])
def test_bad_start_or_validation_drift_prevents_motion(kw,reason):
    e,c,_,_,trace=executor_fixture(**kw)
    with pytest.raises(RuntimeError,match=reason):e.prepare_retry_observation(c['handeye'])
    assert 'execute' not in trace


def test_failed_8cm_localization_does_not_start_another_press():
    class Failure(FakeHardware):
        count=0
        def refresh_button_target(self):
            self.count+=1
            if self.count==2:raise RuntimeError('new 8cm picture not qualified')
            super().refresh_button_target()
    h=Failure(['OFF'],['OFF','OFF'])
    with pytest.raises(RuntimeError,match='8cm picture'):ButtonLightFlow(h,production_flow_config('4',{})).run()
    assert [a for a in h.actions if a[0]=='press']==[('press',1)] and h.actions[-1]==('origin',)


def test_post_retry_reference_is_recaptured_by_original_stationary_boundary():
    # Production still calls the same stationary capture boundary at each
    # attempt. That boundary uses plan.observation_joints for capture and retract.
    import run_button_light_closed_loop as runner
    tree=ast.parse(Path(runner.__file__).read_text())
    fn=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='press_attempt')
    calls=[x for x in ast.walk(fn) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)
           and x.func.attr=='press_with_stationary_reference']
    assert len(calls)==2
    assert all(any(k.arg=='already_at_observation' and isinstance(k.value,ast.Constant)
                   and k.value.value is True for k in x.keywords) for x in calls)
