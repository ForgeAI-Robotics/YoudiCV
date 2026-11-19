"""Real near-mode monitor with fake frames; return/communication use no hardware."""
import ast
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from tests.test_up_safety_handoff import setup
from piper_handeye.button_light_hardware import DryValidatedSegmentExecutor
from tools.button_cycle_worker import Readiness, SessionDispatcher
from run_direct_return_to_origin import _run_orangepi_command, ReturnSshCommandError, prepare_return_mode


def test_near_mode_is_enabled_before_requesting_fresh_coverage_ack(monkeypatch):
    s,v,m,g,metrics,calls,state=setup(monkeypatch)
    s.depth_degraded_commit_active=False
    m.set_depth_degraded_commit(False)
    s.near_pause.begin_cycle()
    calls.clear()
    with s.stationary_reference_transition():
        s.before_segment('observation_to_prepress')
    assert g.arm_calls==1 and m.status()['depth_monitoring_suppressed']
    enable=next(i for i,(n,k) in enumerate(calls) if n=='safety_set_depth_degraded_commit' and k['enabled'])
    assert sum(n=='safety' for n,k in calls[:enable])==0  # no raw-depth processing gap
    assert s._reference_observation_hold
    assert not any('reset' in n for n,k in calls)


def test_postpress_photo_and_three_retries_keep_near_mode_without_observation_renewal(monkeypatch):
    s,v,m,g,metrics,calls,state=setup(monkeypatch)
    old=s.near_pause.started
    for attempt in range(1,4):
        if attempt>1:
            s.begin_outbound_cycle()
            assert s.near_pause.started>=old
        start=s.near_pause.started
        with s.stationary_reference_transition():
            s.before_segment('observation_to_prepress')
        assert s._reference_observation_hold and not s._reference_hold
        s.disable_for_return()  # 5 cm photo retreat, not the final outward return
        for _ in range(40):
            assert not v.request('safety')['visual_safety']['blocked']
        assert s.near_pause.started==start  # photos/polling cannot extend 25 s
    assert g.arm_calls==3
    assert not any(n=='safety_set_depth_degraded_commit' and not k['enabled'] for n,k in calls)
    s.finish_near_observation();s.disable_for_return()
    assert not s._reference_observation_hold and not s.depth_degraded_commit_active


@pytest.mark.parametrize('fault',['stale','hardware','far','expired','stop','residual','frozen'])
def test_near_hold_does_not_remove_original_rejection_conditions(monkeypatch,fault):
    s,v,m,g,metrics,calls,state=setup(monkeypatch)
    with s.stationary_reference_transition():pass
    before=g.arm_calls
    if fault=='stale':state['age']=1
    if fault=='hardware':state['health']=0
    if fault=='far':s.target[0,3]=.3
    if fault=='expired':s.near_pause.started=time.monotonic()-26
    if fault=='stop':g.stop_active.set()
    if fault=='residual':m._latched_blocked=True;m._matrix_latch_reason='residual'
    if fault=='frozen':v.freeze=True
    with pytest.raises(RuntimeError):
        s.begin_outbound_cycle()
        with s.stationary_reference_transition():s.before_segment('observation_to_prepress')
    assert g.arm_calls==before
    assert not any('reset' in n for n,k in calls)


def test_existing_coverage_latch_is_not_cleared_by_new_handoff(monkeypatch):
    s,v,m,g,metrics,calls,state=setup(monkeypatch)
    s.disable_for_return()
    for _ in range(4):v.request('safety')
    assert m.status()['blocked']
    s.near_pause.begin_cycle()
    with pytest.raises(RuntimeError):
        with s.stationary_reference_transition():s.before_segment('observation_to_prepress')
    assert m._latched_blocked and g.arm_calls==0


def test_final_return_ends_hold_even_if_executor_flag_was_set_for_photo(monkeypatch):
    s,v,m,g,metrics,calls,state=setup(monkeypatch)
    with s.stationary_reference_transition():pass
    e=DryValidatedSegmentExecutor.__new__(DryValidatedSegmentExecutor)
    e.motion_safety=s;e.visual_safety_disabled_for_return=True
    e.prepare_origin_return()
    assert e.visual_safety_disabled_for_return and not s._reference_observation_hold
    assert not s.depth_degraded_commit_active and s.near_pause.finished
    count=len(calls);e.prepare_origin_return();assert len(calls)==count


def test_exception_return_switch_is_after_path_validation_before_send():
    source=(Path(__file__).resolve().parents[1]/'run_button_light_closed_loop.py').read_text()
    tree=ast.parse(source)
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='move_origin')
    text=ast.unparse(fn)
    assert text.index('dry_run=True') < text.index('executor.prepare_origin_return()') < text.index("'origin_return_started'")


def test_cycle_origin_return_delegates_mode_switch_to_existing_bridge(tmp_path,monkeypatch):
    monkeypatch.setenv('YOUXIAOMEI_ROBOT_MOTION_LOCK',str(tmp_path/'lock'))
    monkeypatch.setenv('YOUXIAOMEI_MISSION_STATUS_OUTBOX',str(tmp_path/'outbox'))
    d=SessionDispatcher(execute=True,config_path=tmp_path/'cfg',hold_report=tmp_path/'hold')
    launch=Mock(return_value=NS(pid=123,poll=lambda:None))
    monkeypatch.setattr('run_voice_command_gateway.subprocess.Popen',launch)
    try:
        d.dispatch_origin_operation('return',required_unlatched_generation=7)
        argv=launch.call_args.args[0]
        assert '--bridge-mode-handoff' in argv and '--require-unlatched-generation' in argv
        assert launch.call_args.kwargs['stdin']==subprocess.DEVNULL
    finally:d.active=None;d.release_session()
    monkeypatch.setattr('run_direct_return_to_origin._run_orangepi_command',Mock(side_effect=AssertionError('no SSH')))
    mode=prepare_return_mode(Mock(),dict(status='ok',busy=False,safety_handoff_v2=True),bridge_handoff=True)
    assert mode['delegated_to_bridge'] and not mode['local_ssh_mode_switch']


@pytest.mark.parametrize('error,kind',[
    (subprocess.CalledProcessError(255,['ssh'],stderr='Connection reset by peer',output=''), 'ssh_command_exit'),
    (subprocess.CalledProcessError(255,['ssh'],stderr='Permission denied (publickey).',output=''), 'ssh_command_exit'),
    (subprocess.TimeoutExpired(['ssh'],4,stderr=b'connection timeout'), 'ssh_timeout'),
    (FileNotFoundError('ssh missing'), 'ssh_launch_error'),
])
def test_ssh_failures_preserve_real_error_text_and_operation(monkeypatch,error,kind):
    monkeypatch.setattr('run_direct_return_to_origin.subprocess.run',Mock(side_effect=error))
    with pytest.raises(ReturnSshCommandError) as caught:_run_orangepi_command('cansend can0 test')
    detail=caught.value.diagnostic
    assert detail['failure_kind']==kind and detail['stderr'] and detail['operation']=='can_mode_change'
    assert detail['stderr'] in str(caught.value)


def test_link_epoch_changes_only_on_observed_transport_transition():
    r=Readiness.__new__(Readiness)
    assert r._communication_request('bridge',lambda:'ok')=='ok'
    assert getattr(r,'_recovery_communication_epoch',0)==0
    for _ in range(2):
        with pytest.raises(TimeoutError):r._communication_request('bridge',Mock(side_effect=TimeoutError('offline')))
    assert r._recovery_communication_epoch==1
    r._communication_request('bridge',lambda:'ok');assert r._recovery_communication_epoch==2
    for _ in range(5):r._communication_request('bridge',lambda:'ok')
    assert r._recovery_communication_epoch==2
    with pytest.raises(RuntimeError):r._communication_request('bridge',Mock(side_effect=RuntimeError('busy or controller fault')))
    assert r._recovery_communication_epoch==2


def test_three_failures_recover_after_observed_network_restoration_with_full_recheck(tmp_path):
    from test_cycle_fault_recovery import running,fail,advance_to_return
    r,d,store,clock,ready=running(tmp_path);fail(r,d,clock)
    tracker=Readiness.__new__(Readiness);offline=[False];base=ready.auto_recovery_probe
    def probe():
        tracker._communication_request('bridge',lambda:(_ for _ in ()).throw(TimeoutError('offline')) if offline[0] else 'ok')
        return dict(base(),communication_epoch=getattr(tracker,'_recovery_communication_epoch',0),
                    communication_observer_id=getattr(tracker,'_recovery_communication_observer_id',None))
    ready.auto_recovery_probe=probe
    for attempt in range(3):
        if attempt:clock.advance(10.01)
        advance_to_return(r,clock)
        d.events.append(dict(event='origin_operation_completed',operation='return',return_code=1));r.tick()
    clock.advance(10.01);r.tick();assert ready.authorizations==3
    clock.advance(10.01);r.tick();assert ready.authorizations==3  # healthy alone does not reset strikes
    offline[0]=True;clock.advance(10.01);r.tick();assert ready.authorizations==3
    offline[0]=False;clock.advance(10.01);r.tick();assert r.state=='fault_stabilizing'
    clock.advance(.49);r.tick();assert ready.authorizations==3
    clock.advance(.02);r.tick();assert r.state=='fault_planning'
    r.tick();assert ready.authorizations==4 and r.state=='fault_returning'
    assert store.cursor==1 and not store.rows[0]['full_cycle_success']


@pytest.mark.parametrize('independent_stop',[False,True])
def test_real_exception_return_closure_commits_outward_policy_before_execute(independent_stop):
    import run_button_light_closed_loop as runtime
    from piper_handeye.button_light_hardware import DryValidatedSegmentExecutor
    source=(Path(runtime.__file__)).read_text()
    fn=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='move_origin')
    events=[];origin=np.full(6,.1)
    robot=NS(joints=np.zeros(6))
    robot.state=lambda **kw:NS(joints=robot.joints.copy(),raw={'hardware_ok':1})
    safety=NS(finish_near_observation=lambda:True,
        disable_for_return=lambda:events.append('disable'),
        before_segment=Mock(side_effect=AssertionError('outward return must not re-arm visual guard')),
        after_segment=Mock())
    executor=DryValidatedSegmentExecutor.__new__(DryValidatedSegmentExecutor)
    executor.motion_safety=safety;executor.visual_safety_disabled_for_return=False
    def execute(points,**kw):
        if kw.get('dry_run'):
            events.append('validate');return {'status':'validated'}
        assert executor.visual_safety_disabled_for_return
        events.append('execute')
        if independent_stop:raise RuntimeError('independent stop remains latched')
        robot.joints=origin.copy();return {'status':'complete'}
    namespace=dict(robot=robot,origin=origin,np=np,time=time,
        origin_verification=runtime.origin_verification,
        checkpoint=lambda *a,**k:None,
        generate_timed_joint_trajectory=lambda *a,**k:([{'positions':origin.tolist()}],None),
        bridge=NS(execute=execute),trajectory_cfg={'token':'test-only'},executor=executor,safety=safety)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(runtime.__file__),'exec'),namespace)
    if independent_stop:
        with pytest.raises(RuntimeError,match='independent stop'):namespace['move_origin']()
    else:
        assert namespace['move_origin']()['verified']
    assert events==['validate','disable','execute']
    safety.before_segment.assert_not_called()


def test_real_probe_requires_bridge_mode_handoff_capability(monkeypatch):
    from test_button_cycle_worker import ready_probe
    from piper_handeye.guarded_press import GuardedPressUdpClient
    from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient
    p=ready_probe();p.dispatcher.origin_return_bridge_handoff=True
    p.emergency=Mock();p.emergency.snapshot.return_value=dict(engaged=False,available=True,age_sec=.1,error=None)
    p.config={k:dict(host='test',port=1,token='test') for k in ['guarded_press','timed_trajectory']}
    p.stopper.health.return_value=dict(status='ok',latched=False,generation=7)
    monkeypatch.setattr(GuardedPressUdpClient,'health',lambda *a,**k:dict(status='ok',busy=False))
    monkeypatch.setattr(TimedTrajectoryUdpClient,'health',lambda *a,**k:dict(status='ok',busy=False))
    with pytest.raises(RuntimeError,match='模式交接能力'):p.auto_recovery_probe()
    monkeypatch.setattr(TimedTrajectoryUdpClient,'health',lambda *a,**k:dict(status='ok',busy=False,safety_handoff_v2=True))
    assert p.auto_recovery_probe()['stop_identity']['generation']==7


def test_restarting_communication_observer_does_not_clear_three_failure_limit(tmp_path):
    from test_cycle_fault_recovery import running,fail,advance_to_return
    r,d,store,clock,ready=running(tmp_path);fail(r,d,clock)
    observer=['old'];epoch=[2];base=ready.auto_recovery_probe
    ready.auto_recovery_probe=lambda:dict(base(),communication_observer_id=observer[0],communication_epoch=epoch[0])
    for attempt in range(3):
        if attempt:clock.advance(10.01)
        advance_to_return(r,clock)
        d.events.append(dict(event='origin_operation_completed',operation='return',return_code=1));r.tick()
    clock.advance(10.01);r.tick();assert ready.authorizations==3
    observer[0]='new';epoch[0]=0;clock.advance(10.01);r.tick()
    assert ready.authorizations==3 and r.state=='fault_wait'
    epoch[0]=2;clock.advance(10.01);r.tick();clock.advance(.6);r.tick();r.tick()
    assert ready.authorizations==4 and r.state=='fault_returning'


@pytest.mark.parametrize('wrapped',[False,True])
def test_vision_eof_and_wrapped_socket_loss_are_recorded_as_transport_faults(wrapped):
    tracker=Readiness.__new__(Readiness)
    error=RuntimeError('vision service closed without a response')
    if wrapped:
        error=RuntimeError('transport wrapper');error.__cause__=TimeoutError('timeout')
    with pytest.raises(RuntimeError):tracker._communication_request('vision',Mock(side_effect=error))
    assert tracker._recovery_communication_epoch==1
    tracker._communication_request('vision',lambda:{'ok':True})
    assert tracker._recovery_communication_epoch==2
