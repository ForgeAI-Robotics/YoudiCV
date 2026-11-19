"""No-motion fault-wait, bounded recovery and conditional stop-reset tests."""
import ast
import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest

from piper_handeye.button_cycle import CycleSettings, CycleStore, summarize
from piper_handeye.button_cycle_runner import CycleRunner
from piper_handeye.cycle_fault_recovery import (
    require_emergency_released, require_reset_identity, recovery_stop_identity,
    require_fresh_clear_visual, RECOVERY_STABLE_SEC, RETURN_REASSESS_SEC)
from test_button_cycle import setup, begin, finish_task


class Ready:
    def __init__(self):
        self.home=False;self.error=None;self.seq=0;self.generation=7
        self.plan_error=None;self.future=None;self.authorizations=0;self.plans=0
    def __call__(self):
        if self.error:raise RuntimeError(self.error)
        if not self.home:raise RuntimeError('not home')
        return {'at_origin':True}
    def verify_return(self):return self()
    def auto_recovery_probe(self):
        if self.error:raise RuntimeError(self.error)
        self.seq+=1
        return dict(at_origin=self.home,joints=[0.]*6,camera_sequence=self.seq,
                    stop_identity=dict(generation=self.generation,latched=False,reason=None))
    def begin_return_preflight(self, probe):
        self.plans+=1;self.future=Future()
        if self.plan_error:self.future.set_exception(RuntimeError(self.plan_error))
        else:self.future.set_result(dict(stop_identity=probe['stop_identity'],start_joints=probe['joints']))
        return self.future
    def authorize_auto_return(self, proof):
        self.authorizations+=1
        if self.error:raise RuntimeError(self.error)
        if proof['stop_identity']['generation']!=self.generation:raise RuntimeError('new stop')
        return self.generation


def running(tmp_path, *, total=2):
    ready=Ready();ready.home=True
    r,d,s,c,e=setup(tmp_path,CycleSettings('内呼',('1','2'),max(1,total//2),failure_policy='recover_wait'),ready)
    def dispatch(op,**kw):
        assert op=='return' and kw==dict(required_unlatched_generation=ready.generation)
        assert d.active is None and d.session_reserved
        d.active=object();return dict(event='origin_return_started')
    d.dispatch_origin_operation=Mock(side_effect=dispatch)
    begin(r,c);ready.home=False
    return r,d,s,c,ready


def fail(r,d,c):
    finish_task(r,d,c,rc=2,flow='post_press_confirmation_uncertain',origin=False,on=False,off=False)
    assert r.state in {'fault_wait','fault_stabilizing'} and d.active is None


def advance_to_return(r,c):
    r.tick();assert r.state=='fault_stabilizing'
    c.advance(.49);r.tick();assert r.state=='fault_stabilizing'
    c.advance(.02);r.tick();assert r.state=='fault_planning'
    r.tick();assert r.state=='fault_returning'


def test_exact_recovery_intervals():
    assert RECOVERY_STABLE_SEC==.5 and RETURN_REASSESS_SEC==10.


def test_failed_item_preserved_then_return_and_continue_only_remaining(tmp_path):
    r,d,s,c,ready=running(tmp_path);fail(r,d,c)
    assert s.cursor==1 and not s.rows[0]['full_cycle_success']
    advance_to_return(r,c)
    ready.home=True
    d.events.append(dict(event='origin_operation_completed',operation='return',return_code=0));r.tick()
    assert d.commands==[1,2] and s.cursor==1
    finish_task(r,d,c)
    assert r.state=='completed' and s.cursor==2
    assert summarize(s.rows)['full_cycle_success']==1
    assert summarize(s.rows)['failed_or_uncertain']==1
    assert not (s.directory/'pending_recovery.json').exists()


def test_already_home_failure_can_continue_without_redundant_return(tmp_path):
    r,d,s,c,ready=running(tmp_path);fail(r,d,c);ready.home=True
    r.tick();c.advance(.6);r.tick();r.tick()
    assert d.commands==[1,2] and ready.plans==0


def test_transient_disconnection_waits_then_recovers_without_operator_resume(tmp_path):
    r,d,s,c,ready=running(tmp_path);fail(r,d,c);ready.error='network offline'
    for _ in range(5):r.tick();c.advance(1)
    assert r.state=='fault_wait' and ready.authorizations==0 and d.commands==[1]
    ready.error=None;r.tick();c.advance(.6);r.tick();r.tick()
    assert r.state=='fault_returning' and ready.authorizations==1


def test_failed_return_rechecks_after_ten_seconds_not_immediately(tmp_path):
    r,d,s,c,ready=running(tmp_path);fail(r,d,c);advance_to_return(r,c)
    d.events.append(dict(event='origin_operation_completed',operation='return',return_code=1));r.tick()
    assert r.state=='fault_wait' and ready.plans==1
    c.advance(9.99);r.tick();assert ready.plans==1 and r.state=='fault_wait'
    c.advance(.02);r.tick();assert r.state=='fault_stabilizing'


def test_collision_preflight_failure_never_authorizes_or_moves(tmp_path):
    r,d,s,c,ready=running(tmp_path);fail(r,d,c);ready.plan_error='MoveIt collision'
    r.tick();c.advance(.6);r.tick();r.tick()
    assert r.state=='fault_wait' and ready.authorizations==0
    d.dispatch_origin_operation.assert_not_called()
    c.advance(9.9);r.tick();assert ready.plans==1


def test_new_stop_during_preflight_prevents_return(tmp_path):
    r,d,s,c,ready=running(tmp_path);fail(r,d,c)
    r.tick();c.advance(.6);r.tick();ready.generation+=1;r.tick()
    assert r.state=='fault_wait';d.dispatch_origin_operation.assert_not_called()


@pytest.mark.parametrize('action',['pause','finish','close'])
def test_user_controls_cancel_pending_recovery_and_late_proof(tmp_path,action):
    r,d,s,c,ready=running(tmp_path);fail(r,d,c);r.tick();c.advance(.6);r.tick()
    if action=='pause':r.pause()
    elif action=='finish':r.finish()
    else:r.stop(closing=True)
    c.advance(30);r.tick()
    assert ready.authorizations==0 and d.commands==[1]


def test_only_same_group_explicit_resume_loads_recovery_checkpoint(tmp_path):
    r,d,s,c,ready=running(tmp_path);fail(r,d,c)
    r.pause();d.release_session()
    r2=CycleRunner(CycleStore(s.directory),d,ready,lambda e:None,clock=c)
    assert r2.state=='paused'
    r2.tick();assert ready.plans==0
    r2.resume();assert r2.state=='fault_wait'


def test_feedback_stop_is_recorded_as_failure_not_success_or_removed(tmp_path):
    r,d,s,c,ready=running(tmp_path)
    r.interrupt_for_recovery('feedback unavailable',stop_reason='cycle_feedback_unavailable')
    assert r.state=='fault_wait' and s.cursor==1
    assert s.rows[0]['counted'] and not s.rows[0]['full_cycle_success']
    assert s.rows[0]['result']=='失败'


def test_three_same_condition_return_failures_stop_repeated_motion_but_keep_waiting(tmp_path):
    r,d,s,c,ready=running(tmp_path);fail(r,d,c)
    for i in range(3):
        if i:c.advance(10.01)
        advance_to_return(r,c)
        d.events.append(dict(event='origin_operation_completed',operation='return',return_code=1));r.tick()
    assert ready.authorizations==3
    c.advance(10.01);r.tick();c.advance(10.01);r.tick()
    assert ready.authorizations==3 and r.state=='fault_wait'
    ready.generation+=1;c.advance(10.01);r.tick();c.advance(.6);r.tick();r.tick()
    assert ready.authorizations==4


@pytest.mark.parametrize('change',[dict(engaged=True),dict(available=False),dict(age_sec=1.6),dict(age_sec=float('nan')),dict(error='lost')])
def test_real_emergency_gate_does_not_use_timeout_as_release(change):
    state=dict(engaged=False,available=True,age_sec=.1,error=None);state.update(change)
    with pytest.raises(RuntimeError):require_emergency_released(state)


@pytest.mark.parametrize('payload,last_stop',[({'expected_generation':6,'expected_reason':'cycle_physical_estop'},{'reason':'cycle_physical_estop'}),
 ({'expected_generation':True,'expected_reason':'cycle_physical_estop'},{'reason':'cycle_physical_estop'}),
 ({'expected_generation':7,'expected_reason':'gui_abort_and_return'},{'reason':'gui_abort_and_return'}),
 ({'expected_generation':7,'expected_reason':'cycle_physical_estop'},{'reason':'cycle_test_stopped'})])
def test_conditional_reset_never_clears_newer_or_operator_stop(payload,last_stop):
    with pytest.raises(RuntimeError):require_reset_identity(payload,7,last_stop)


def test_clear_visual_gate_requires_live_frame_not_just_old_clear_flag():
    good=dict(ready=True,blocked=False,monitoring_available=True,pose_timing=dict(last_frame=dict(frame_received_monotonic=10.)))
    require_fresh_clear_visual(good,10.2)
    with pytest.raises(RuntimeError):require_fresh_clear_visual(good,10.36)
    with pytest.raises(RuntimeError):require_fresh_clear_visual(dict(ready=True,blocked=False,monitoring_available=True),10.2)


def test_real_stop_bridge_compares_generation_before_resetting(monkeypatch):
    # Compile actual production dispatch method with fake ROS/socket dependencies.
    path=Path(__file__).resolve().parents[1]/'tools/orangepi_visual_safety_stop_bridge.py'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SafetyStopBridge')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='spin')
    import socket
    for expected,success in [(7,True),(6,False)]:
        ros=Mock();ros.is_shutdown.side_effect=[False,True]
        ns=dict(rospy=ros,socket=socket,json=json,require_reset_identity=require_reset_identity)
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
        bridge=NS(args=NS(allowed_client='host',token='test',bind_ip='host',port=1),latched=True,stop_generation=7,last_stop={'reason':'cycle_physical_estop'},sock=Mock(),response=Mock())
        payload=dict(protocol='piper_visual_safety_v1',token='test',command='reset_latch_if_generation',task_id='test',expected_generation=expected,expected_reason='cycle_physical_estop')
        bridge.sock.recvfrom.return_value=(json.dumps(payload).encode(),('host',1))
        ns['spin'](bridge)
        assert bridge.latched is (not success)
        if not success:ros.set_param.assert_not_called()


def test_last_failed_item_still_has_explicit_resumable_return_checkpoint(tmp_path):
    r,d,s,c,ready=running(tmp_path)
    ready.home=True;finish_task(r,d,c)
    ready.home=False;fail(r,d,c)
    assert s.cursor==s.settings.total
    r.pause();d.release_session()
    r2=CycleRunner(CycleStore(s.directory),d,ready,lambda e:None,clock=c)
    r2.resume()
    assert r2.state=='fault_wait' and r2.auto_recovery is not None


def test_gui_return_restarts_missing_backend_without_starting_test(tmp_path):
    from test_button_cycle_gui import panel
    p=panel();p.backend=None;p.store=NS(directory=tmp_path,cursor=1)
    p.launch=Mock()
    p.abort_return()
    p.launch.assert_called_once_with(auto_resume=False,initial_command='abort_return')


def test_gui_ready_sends_only_pending_return_command_not_resume():
    from test_button_cycle_gui import panel
    p=panel();p.auto_resume=False;p.initial_worker_command='abort_return'
    p.events.put(dict(channel='button_cycle_worker',event='cycle_ready',rows=[]))
    p.poll()
    commands=[c.args[0]['command'] for c in p.backend.send.call_args_list]
    assert commands==['heartbeat','abort_return']


def test_return_verification_is_independent_of_yolo_and_group_code_hash():
    from test_button_cycle_worker import ready_probe
    p=ready_probe()
    p.vision.request.side_effect=RuntimeError('camera offline')
    p.store.plan['metadata']={'sha256':{'/nonexistent/historical/file':'old'}}
    assert p.verify_return()['at_origin']
    p.vision.request.assert_not_called()
    p.stopper.reset_latch.assert_not_called()


def test_auto_probe_rechecks_emergency_and_idle_bridges_without_reset(monkeypatch):
    from test_button_cycle_worker import ready_probe
    from piper_handeye.guarded_press import GuardedPressUdpClient
    from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient
    p=ready_probe();p.emergency=Mock();p.emergency.snapshot.return_value=dict(engaged=False,available=True,age_sec=.1,error=None)
    p.config={k:dict(host='test',port=1,token='test') for k in ['guarded_press','timed_trajectory']}
    p.stopper.health.return_value=dict(status='ok',latched=False,generation=7)
    monkeypatch.setattr(GuardedPressUdpClient,'health',lambda *a,**kw:dict(status='ok',busy=False))
    monkeypatch.setattr(TimedTrajectoryUdpClient,'health',lambda *a,**kw:dict(status='ok',busy=False))
    assert p.auto_recovery_probe()['stop_identity']['generation']==7
    p.emergency.snapshot.return_value['engaged']=True
    with pytest.raises(RuntimeError,match='急停'):p.auto_recovery_probe()
    p.stopper.reset_latch.assert_not_called();p.stopper.reset_latch_if_generation.assert_not_called()


def test_real_authorization_rejects_cancelled_epoch_before_clear():
    from test_button_cycle_worker import ready_probe
    p=ready_probe();p.cancel_epoch=[2]
    p.auto_recovery_probe=Mock()
    with pytest.raises(RuntimeError,match='取消'):
        p.authorize_auto_return(dict(control_cancel_epoch=1))
    p.auto_recovery_probe.assert_not_called()
    p.stopper.reset_latch_if_generation.assert_not_called()


def test_actual_preflight_only_dry_runs_and_never_enables_or_resets(monkeypatch):
    from test_button_cycle_worker import ready_probe
    import run_direct_return_to_origin as origin
    from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient
    p=ready_probe();p.config=dict(timed_trajectory=dict(host='test',port=1,token='test'))
    monkeypatch.setattr(origin,'build_transportable_origin_return_trajectory',lambda *a,**kw:('points',NS(duration_sec=1),'selection'))
    def selection(start,end,prepared,validate,**kw):
        return 'points',NS(duration_sec=1),'selection',validate('points')
    monkeypatch.setattr(origin,'select_collision_checked_origin_return',selection)
    execute=Mock(return_value={'ok':True})
    monkeypatch.setattr(TimedTrajectoryUdpClient,'execute',execute)
    proof=p._preflight_return_path(dict(joints=[0.]*6,stop_identity=dict(generation=7),control_cancel_epoch=3))
    assert proof['control_cancel_epoch']==3
    assert all(c.kwargs['dry_run'] is True for c in execute.call_args_list)
    p.arm.enable.assert_not_called();p.stopper.reset_latch_if_generation.assert_not_called()


def test_thousand_virtual_tasks_with_fifty_faults_keep_order_and_failure_statistics(tmp_path):
    ready=Ready();ready.home=True
    r,d,s,c,events=setup(tmp_path,CycleSettings('内呼',('1','2','3','4','5'),200,failure_policy='recover_wait'),ready)
    def dispatch(op,**kw):
        assert op=='return' and kw==dict(required_unlatched_generation=7)
        assert d.active is None
        d.active=object();return dict(event='origin_return_started')
    d.dispatch_origin_operation=Mock(side_effect=dispatch)
    begin(r,c)
    for i in range(1000):
        if i%20==19:
            ready.home=False
            finish_task(r,d,c,rc=2,flow='post_press_confirmation_uncertain',origin=False,on=False,off=False)
            c.advance(.6);r.tick();r.tick()
            assert r.state=='fault_returning'
            ready.home=True
            d.events.append(dict(event='origin_operation_completed',operation='return',return_code=0));r.tick()
        else:finish_task(r,d,c)
    assert r.state=='completed' and s.cursor==1000
    assert d.commands==[1,2,3,4,5]*200
    summary=summarize(CycleStore(s.directory).rows)
    assert summary['completed']==1000 and summary['failed_or_uncertain']==50
    assert summary['full_cycle_success']==950 and d.dispatch_origin_operation.call_count==50


@pytest.mark.parametrize('enabled,fault,kept',[(False,True,False),(True,False,False),(True,True,True)])
def test_original_safety_reference_retained_only_for_opt_in_fault_recovery(monkeypatch,enabled,fault,kept):
    import threading
    import numpy as np
    from piper_handeye.button_light_hardware import VisualSegmentMotionSafety
    if enabled:monkeypatch.setenv('YOUXIAOMEI_CYCLE_AUTO_RECOVERY','1')
    else:monkeypatch.delenv('YOUXIAOMEI_CYCLE_AUTO_RECOVERY',raising=False)
    guard=Mock(error=None,triggered=threading.Event(),stop_active=threading.Event())
    if fault:guard.triggered.set()
    vision=Mock();s=VisualSegmentMotionSafety(guard,vision,Mock(),np.eye(4))
    s.close()
    commands=[c.args[0] for c in vision.request.call_args_list]
    assert ('safety_disable_matrix_reference' not in commands) is kept
    assert 'safety_capture_matrix_reference' not in commands


def test_recovery_feedback_only_reads_and_updates_pose_then_releases_ports(monkeypatch):
    import time
    import numpy as np
    from test_button_cycle_worker import ready_probe
    import run_online_replan_no_contact as online
    import piper_handeye.robot as robot_module
    p=ready_probe();p.dispatcher=NS(config_path=Path('/tmp/config'))
    p.config=dict(robot={},calibration=dict(handeye_yaml='calib'))
    p._recovery_feedback_thread=None;p._recovery_feedback_stop=None;p._recovery_feedback=None;p._recovery_feedback_error=None
    robot=Mock();robot.state.return_value=NS(raw={'hardware_ok':1},timestamp=time.monotonic(),base_to_gripper=np.eye(4))
    monkeypatch.setattr(robot_module,'PiperUdpRobot',lambda c:robot)
    monkeypatch.setattr(online,'resolve_path',lambda *a:Path('/tmp/calib'))
    monkeypatch.setattr(online,'load_handeye',lambda p:(np.eye(4),None,None))
    p._ensure_recovery_feedback()
    deadline=time.monotonic()+1
    while not p.vision.request.called and time.monotonic()<deadline:time.sleep(.01)
    p.end_recovery()
    assert robot.open.called and robot.close.called
    robot.enable.assert_not_called();robot._send_tree.assert_not_called()
    assert p._recovery_feedback is None
    assert all(c.args[0]=='safety_camera_pose' for c in p.vision.request.call_args_list)
