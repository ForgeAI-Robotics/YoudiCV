"""No-motion execution of the real scheduler and origin entrypoint."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import struct

import pytest
import yaml

from piper_handeye.cycle_recovery import require_unlatched_generation, recoverable_retract_failure
from test_button_cycle import setup, begin, terminal


def failure(outcome='timeout'):
    return dict(failure_code='GUARDED_RETRACT_CONVERGENCE_FAILED', retract_convergence=dict(
        outcome=outcome,safety_generation=7,final_error_rad=.08,distinct_samples=20))


@pytest.mark.parametrize('change',[dict(latched=True),dict(generation=8),dict(generation=6),
    dict(generation=True),dict(status='failed'),dict(generation=None)])
def test_no_automatic_clearing_or_ignoring_stop_generation(change):
    value=dict(status='ok',latched=False,generation=7);value.update(change)
    with pytest.raises(RuntimeError):require_unlatched_generation(value,7)


@pytest.mark.parametrize('outcome',['feedback_invalid_or_stale','safety_or_callback_failure','feedback_unavailable'])
def test_only_encoder_convergence_timeout_is_recoverable(outcome):
    assert recoverable_retract_failure(failure(outcome)) is None
    assert recoverable_retract_failure(failure())['safety_generation']==7


class Ready:
    def __init__(self): self.probes=[]; self.error=None
    def __call__(self): return dict(at_origin=True)
    def recovery_probe(self, generation):
        self.probes.append(generation)
        if self.error: raise RuntimeError(self.error)
        return dict(status='ok',latched=False,generation=generation)


def start_failure(tmp_path, *, veto=False, rc=1, outcome='timeout', origin=False, retry=False):
    ready=Ready()
    r,d,s,c,emitted=setup(tmp_path, readiness=ready)
    def dispatch(op,**kw):
        assert op=='return' and kw==dict(required_unlatched_generation=7)
        assert d.active is None and d.session_reserved
        d.active=object();d.returning=True
        return dict(event='origin_return_started')
    d.dispatch_origin_operation=Mock(side_effect=dispatch)
    begin(r,c)
    # Evidence may arrive before process completion and must survive polling.
    rows=[dict(task_id='task1',producer='arm_pressing',raw_event='motion_segment_failed' if retry else 'guarded_press_failed',
               phase='motion_segment_failed' if retry else 'press_failed',
               evidence=dict(segment='guarded_press',bridge_failure=failure(outcome)))]
    if veto: rows.append(dict(task_id='task1',producer='dispatcher',phase='abort_requested'))
    d.events.append(dict(event='mission_progress',mission_status_events=rows));r.tick()
    assert d.active is not None
    d.events.append(terminal(rc=rc,origin=origin,on=False,off=False,flow='failed'));r.tick()
    return r,d,s,c,ready


def test_failed_task_exits_then_stationary_hold_then_formal_return_and_pause(tmp_path):
    r,d,s,c,ready=start_failure(tmp_path)
    assert r.state=='recovery_checking' and d.active is None
    d.dispatch_origin_operation.assert_not_called()
    assert s.rows[0]['counted'] and not s.rows[0]['full_cycle_success']
    c.advance(.6);r.tick()
    assert r.state=='recovering' and d.returning and ready.probes==[7,7]
    d.events.append(dict(event='origin_operation_completed',operation='return',return_code=0));r.tick()
    assert r.state=='paused' and d.commands==[1] and not s.rows[0]['returned_to_origin']
    for _ in range(5):c.advance();r.tick()
    assert d.commands==[1]


@pytest.mark.parametrize('options',[dict(veto=True),dict(rc=-15),dict(outcome='feedback_invalid_or_stale'),dict(origin=True)])
def test_no_recovery_for_user_stop_signal_stale_or_already_home(tmp_path,options):
    r,d,s,c,ready=start_failure(tmp_path,**options)
    assert r.state=='paused'
    c.advance(2);r.tick();d.dispatch_origin_operation.assert_not_called()


def test_pending_automatic_return_cancelled_by_window_close(tmp_path):
    r,d,s,c,ready=start_failure(tmp_path)
    r.stop(closing=True)
    c.advance();r.tick()
    assert r.state=='closed' and not d.session_reserved
    d.dispatch_origin_operation.assert_not_called()


def test_stop_latch_change_fails_gate_without_trying_again(tmp_path):
    r,d,s,c,ready=start_failure(tmp_path)
    ready.error='停止锁变化'
    c.advance(.2);r.tick()
    assert r.state=='paused' and '停止锁变化' in r.reason
    ready.error=None;c.advance(1);r.tick();d.dispatch_origin_operation.assert_not_called()


def test_failed_formal_return_does_not_trigger_second_return(tmp_path):
    r,d,s,c,ready=start_failure(tmp_path)
    c.advance(.6);r.tick()
    d.events.append(dict(event='origin_operation_completed',operation='return',return_code=1));r.tick()
    assert r.state=='paused' and '未通过' in r.reason
    c.advance(5);r.tick();assert d.dispatch_origin_operation.call_count==1


def test_retry_failure_uses_same_structured_recovery_as_first_attempt(tmp_path):
    r,d,s,c,ready=start_failure(tmp_path,retry=True)
    assert r.state=='recovery_checking'
    c.advance(.6);r.tick()
    assert r.state=='recovering' and d.dispatch_origin_operation.call_count==1


def test_only_cycle_child_defers_recognized_timeout_to_parent():
    from piper_handeye.cycle_recovery import cycle_parent_recovery_required
    from piper_handeye.guarded_press import GuardedPressFailure
    exc=GuardedPressFailure(failure())
    assert cycle_parent_recovery_required(exc,True)
    assert not cycle_parent_recovery_required(exc,False)
    assert not cycle_parent_recovery_required(RuntimeError('timeout'),True)
    assert not cycle_parent_recovery_required(GuardedPressFailure(failure('feedback_invalid_or_stale')),True)


@pytest.mark.parametrize('condition',['ready','guard_busy','stop_changed'])
def test_real_recovery_probe_requires_idle_bridges_and_unchanged_unlatched_stop(monkeypatch,condition):
    from test_button_cycle_worker import ready_probe
    from piper_handeye.guarded_press import GuardedPressUdpClient
    from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient
    p=ready_probe()
    p.arm.status.return_value['status']['actual_joints_rad']=[.2]*6
    p.config={k:dict(host='test',port=1,token='x') for k in ['guarded_press','timed_trajectory']}
    p.stopper.health.return_value=dict(status='ok',latched=False,generation=8 if condition=='stop_changed' else 7)
    monkeypatch.setattr(GuardedPressUdpClient,'health',lambda *a,**kw:dict(status='ok',busy=condition=='guard_busy'))
    monkeypatch.setattr(TimedTrajectoryUdpClient,'health',lambda *a,**kw:dict(status='ok',busy=False))
    if condition=='ready': assert p.recovery_probe(7)['generation']==7
    else:
        with pytest.raises(RuntimeError):p.recovery_probe(7)
    p.stopper.reset_latch.assert_not_called()
    p.arm.enable.assert_not_called()


def test_real_dispatcher_passes_generation_only_for_protected_return(tmp_path,monkeypatch):
    from run_voice_command_gateway import CommandDispatcher
    monkeypatch.setenv('YOUXIAOMEI_ROBOT_MOTION_LOCK',str(tmp_path/'lock'))
    monkeypatch.setenv('YOUXIAOMEI_MISSION_STATUS_OUTBOX',str(tmp_path/'status'))
    d=CommandDispatcher(execute=True,config_path=tmp_path/'cfg',hold_report=tmp_path/'hold')
    launch=Mock(return_value=SimpleNamespace(pid=10,poll=lambda:None))
    monkeypatch.setattr('run_voice_command_gateway.subprocess.Popen',launch)
    try:
        d.dispatch_origin_operation('return',required_unlatched_generation=7)
        args=launch.call_args.args[0]
        assert args[-2:]==['--require-unlatched-generation','7']
    finally:d.active=None;d._release_motion_lock()


def test_odometry_decoder_keeps_header_and_pose_for_independent_motion_checks():
    from tools.button_cycle_worker import decode_cycle_odometry
    prefix=struct.pack('<4I',42,100,123000000,0)+struct.pack('<I',0)
    pose=struct.pack('<7d',1.,2.,0.,0.,0.,0.,1.)+bytes(36*8)
    result=decode_cycle_odometry(prefix+pose+struct.pack('<6d',*[0.]*6))
    assert result['sequence']==42 and result['source_stamp_sec']==100.123
    assert result['pose_xy_yaw']==[1.,2.,0.]


@pytest.mark.parametrize('changed_when',[None, 'initial', 'before_execution'])
def test_actual_return_main_never_resets_latch_in_automatic_mode(tmp_path,monkeypatch,changed_when):
    import run_direct_return_to_origin as m
    cfg=tmp_path/'cfg.yaml'; gp=tmp_path/'gp.yaml'; hold=tmp_path/'hold.yaml'
    cfg.write_text(yaml.safe_dump(dict(robot={},timed_trajectory=dict(host='test',port=1,token='x'),
        safety_stop=dict(host='test',port=2,token='x'))))
    gp.write_text(yaml.safe_dump(dict(reserved_ports=dict(arm_output=dict(host='test',port=3,token='x')))))
    origin=[.1,.1,0.,.1,.1,.1]
    hold.write_text(yaml.safe_dump(dict(stage_1_states=[dict(joints_rad=origin)])))
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr('sys.argv',['return','--config',str(cfg),'--gamepad-config',str(gp),'--hold-report',str(hold),
        '--execute','--acknowledge-clear-workspace','--require-unlatched-generation','7'])
    stopper=Mock();calls=[]
    def health(**_):
        calls.append(1)
        gen=8 if changed_when=='initial' or (changed_when=='before_execution' and len(calls)>=5) else 7
        return dict(status='ok',latched=gen!=7,generation=gen)
    stopper.health.side_effect=health
    monkeypatch.setattr(m,'SafetyStopUdpClient',lambda *a,**kw:stopper)
    robot=Mock();robot.state.return_value=SimpleNamespace(raw={'hardware_ok':1})
    live=Mock();live.live_joints.return_value=origin;live.status.return_value={'status':{}}
    timed=Mock();timed.health.return_value={'status':'ok'}
    timed.execute.return_value={'status':'completed'}
    monkeypatch.setattr(m,'PiperUdpRobot',lambda *_:robot)
    monkeypatch.setattr(m,'ArmGamepadUdpClient',lambda *a,**kw:live)
    monkeypatch.setattr(m,'TimedTrajectoryUdpClient',lambda *a,**kw:timed)
    monkeypatch.setattr(m,'prepare_active_return_state',lambda *a:(robot.state(),{},None,m.np.asarray(origin)))
    monkeypatch.setattr(m,'ensure_origin_clamp_closed',lambda *a,**kw:{})
    monkeypatch.setattr(m,'prepare_return_mode',lambda *a,**kw:{})
    if changed_when is None:
        rc=m.main()
    else:
        with pytest.raises(RuntimeError,match='停止'):
            m.main()
        rc=1
    stopper.reset_latch.assert_not_called()
    if changed_when is None:
        assert rc==0
        assert any(not c.kwargs.get('dry_run') for c in timed.execute.call_args_list)
    else:
        assert rc==1
        assert all(c.kwargs.get('dry_run') for c in timed.execute.call_args_list)
    if changed_when=='initial':robot.open.assert_not_called()
