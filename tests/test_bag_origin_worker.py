from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import itertools
import time
import pytest
import numpy as np
from tools.bag_origin_worker import OriginTask


Q=[-.03,.3,-.33,0,.12,.1]


def setup_task():
    dispatcher=MagicMock();dispatcher.active=None
    emit=MagicMock();t=OriginTask(emit,gui_heartbeat=False,dispatcher=dispatcher)
    def launch(*_):dispatcher.active=MagicMock()
    def refresh():
        dispatcher.active=None
        return {'event':'origin_operation_completed','return_code':0}
    def stop(**_):
        dispatcher.active=None
        return {'event':'stop'}
    dispatcher.dispatch_origin_operation.side_effect=launch
    dispatcher.refresh.side_effect=refresh
    dispatcher.stop.side_effect=stop
    return t,dispatcher,emit


def mocks():
    live=MagicMock();live.status.return_value={'status':{'active':False,'other_motion_active':False,
        'actual_joints_rad':Q,'joint_feedback_age_sec':.01,'clamp_open':False,'gripper_position':.01}}
    live.live_joints.return_value=Q
    timed=MagicMock();timed.health.return_value={'busy':False}
    guarded=MagicMock();guarded.health.return_value={'busy':False}
    odom=MagicMock();odom.snapshot.return_value={'available':True,'age_sec':0,'velocity':[0]*6}
    return live,timed,guarded,odom


def run_with(t,resources,reader):
    live,timed,guarded,odom=resources
    ticks=itertools.count(0,.05)
    robot=MagicMock();robot.state.return_value=SimpleNamespace(joints=np.array(Q),raw={'hardware_ok':1})
    with patch('tools.bag_origin_worker.ArmGamepadUdpClient',return_value=live), \
         patch('tools.bag_origin_worker.TimedTrajectoryUdpClient',return_value=timed), \
         patch('tools.bag_origin_worker.GuardedPressUdpClient',return_value=guarded), \
         patch('tools.bag_origin_worker.Ros1TopicMonitor',return_value=odom), \
         patch('tools.bag_origin_worker.PiperUdpRobot',return_value=robot), \
         patch('tools.bag_origin_worker.ensure_origin_clamp_closed',return_value={'command_sent':False,'status':{'clamp_open':False,'gripper_position':.01}}) as clamp, \
         patch('tools.bag_origin_worker.time.monotonic',side_effect=lambda:next(ticks)):
        if hasattr(t,'clamp_side_effect'):clamp.side_effect=t.clamp_side_effect
        t.cancel.wait=MagicMock()
        t.run(origin_reader=reader)


def test_return_reuses_existing_dispatcher_then_verifies_fresh_origin():
    t,d,emit=setup_task()
    reader=MagicMock(return_value={'ok':True,'at_origin':True,'hardware_ok':1})
    resources=mocks()
    run_with(t,resources,reader)
    d.dispatch_origin_operation.assert_called_once_with('return')
    reader.assert_called_once_with(d)
    assert emit.call_args.args[0]=='complete'
    assert emit.call_args.kwargs['report']['returned_to_origin'] is True
    d.stop.assert_not_called()
    resources[0].live_joints.assert_not_called()


def test_another_busy_bridge_prevents_return_without_stopping_that_owner():
    t,d,emit=setup_task();r=mocks();r[1].health.return_value={'busy':True}
    with pytest.raises(RuntimeError,match='旧运动'):run_with(t,r,MagicMock())
    d.dispatch_origin_operation.assert_not_called();d.stop.assert_not_called()


def test_moving_base_prevents_return():
    t,d,emit=setup_task();r=mocks();r[3].snapshot.return_value={'available':True,'age_sec':0,'velocity':[.1,0,0,0,0,0]}
    with pytest.raises(RuntimeError):run_with(t,r,MagicMock())
    d.dispatch_origin_operation.assert_not_called()


def test_cancel_during_return_stops_own_child_and_does_not_report_success():
    t,d,emit=setup_task()
    def launch(*_):d.active=MagicMock();t.stop('operator stop')
    d.dispatch_origin_operation.side_effect=launch
    reader=MagicMock()
    with pytest.raises(RuntimeError,match='operator stop'):run_with(t,mocks(),reader)
    d.stop.assert_called_once();reader.assert_not_called()
    assert all(c.args[0]!='complete' for c in emit.call_args_list)


def test_return_process_failure_and_bad_origin_evidence_are_not_success():
    for child_failure in [True,False]:
        t,d,emit=setup_task()
        if child_failure:
            def failure():
                d.active=None
                return {'event':'origin_operation_completed','return_code':2}
            d.refresh.side_effect=failure
        reader=MagicMock(return_value={'ok':True,'at_origin':False,'hardware_ok':1})
        with pytest.raises(RuntimeError):run_with(t,mocks(),reader)
        assert all(c.args[0]!='complete' for c in emit.call_args_list)


def test_no_owned_child_means_no_global_stop():
    t,d,emit=setup_task();t.safe_stop();d.stop.assert_not_called()


def test_return_children_cannot_consume_gui_heartbeat_pipe():
    import subprocess
    import sys
    script='''
import subprocess,sys
from tools.bag_origin_worker import isolate_control_stdin
control=isolate_control_stdin()
child=subprocess.run([sys.executable,'-c','import sys; print(repr(sys.stdin.read()))'],capture_output=True,text=True)
assert child.returncode==0
assert child.stdout.strip()=="''"
assert control.readline()=='heartbeat\\n'
'''
    result=subprocess.run([sys.executable,'-c',script],input='heartbeat\n',capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr


def test_clamp_return_is_confirmed_before_dispatching_arm_return():
    t,d,emit=setup_task();events=[]
    def clamp(*args,**kwargs):
        assert kwargs['clamp_timeout_sec']==6.5
        assert d.active is None
        events.append('clamp_closed')
        return {'command_sent':True,'status':{'clamp_open':False,'gripper_position':.01}}
    t.clamp_side_effect=clamp
    def launch(*_):
        events.append('arm_return');d.active=MagicMock()
    d.dispatch_origin_operation.side_effect=launch
    run_with(t,mocks(),MagicMock(return_value={'ok':True,'at_origin':True,'hardware_ok':1}))
    assert events==['clamp_closed','arm_return']
    assert emit.call_args.kwargs['report']['pre_return_clamp']['command_sent'] is True


@pytest.mark.parametrize('case',['timeout','still_open','nan','stale_final_feedback','cancel','moving_base'])
def test_failed_clamp_or_cancel_never_starts_arm_return(case):
    t,d,emit=setup_task();r=mocks();calls=[]
    def clamp(*args,**kwargs):
        calls.append(1)
        if case=='timeout':raise TimeoutError('clamp result unknown')
        if case=='cancel':t.stop('operator stop')
        if case=='moving_base':r[3].snapshot.return_value={'available':True,'age_sec':0,'velocity':[.1,0,0,0,0,0]}
        if case=='stale_final_feedback':r[0].status.return_value['status']['joint_feedback_age_sec']=.4
        return {'status':{'clamp_open':case=='still_open','gripper_position':float('nan') if case=='nan' else 1.3 if case=='still_open' else .01}}
    t.clamp_side_effect=clamp
    with pytest.raises(RuntimeError):run_with(t,r,MagicMock())
    assert calls==[1]
    d.dispatch_origin_operation.assert_not_called()
    assert all(c.args[0]!='complete' for c in emit.call_args_list)
