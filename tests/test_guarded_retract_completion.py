"""Run actual bridge methods with fake ROS, encoders and action transport."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from piper_handeye.joint_convergence import wait_for_joint_convergence, JointConvergenceError
from tools.piper_action_watchdog import send_goal_with_safety, require_safety_generation
from test_joint_convergence import Clock

SOURCE = Path('tools/orangepi_guarded_press_bridge.py')


def method(name, namespace):
    tree=ast.parse(SOURCE.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GuardedBridge')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace[name]


@pytest.mark.parametrize('converges',[True,False])
def test_actual_retract_waits_on_cached_measured_feedback_and_sends_only_one_goal(converges):
    c=Clock();bridge=Mock()
    bridge.current_joints.return_value=np.full(6,.2)
    bridge.client.get_result.return_value=SimpleNamespace(error_code=0,SUCCESSFUL=0)
    bridge.joint_feedback.snapshot.side_effect=lambda:(round(c.t*100),c.t,
        [.0256 if converges and c.t>=100.12 else .1225]*6)
    points=[dict(positions=[.2]*6,time_from_start_sec=0),dict(positions=[0.]*6,time_from_start_sec=1.)]
    ns=dict(np=np,rospy=SimpleNamespace(get_param=lambda key,default:7 if key.endswith('generation') else False,Duration=lambda x:x),
            generate_bounded_retract_trajectory=lambda *a,**kw:points, validate=lambda *_:None,
            MAX_VELOCITIES=[1.]*6,MAX_ACCELERATIONS=[1.]*6,JOINT_NAMES=['j']*6,
            wait_for_action_result=lambda *a,**kw:None,send_goal_with_safety=send_goal_with_safety,
            require_safety_generation=require_safety_generation,JointConvergenceError=JointConvergenceError,
            wait_for_joint_convergence=lambda *a,**kw:wait_for_joint_convergence(*a,**kw,clock=c,sleep=c.sleep))
    fn=method('retract',ns)
    if converges:
        result=fn(bridge,[0.]*6,safety_generation=7)
        assert result['end_error_rad']==.0256 and result['convergence']['first_error_rad']==.1225
    else:
        with pytest.raises(JointConvergenceError) as caught:fn(bridge,[0.]*6,safety_generation=7)
        assert caught.value.audit['safety_generation']==7
    bridge.client.send_goal.assert_called_once()
    bridge.current_joints.assert_called_once()


def test_actual_wrapper_keeps_failed_press_current_trace_and_convergence_error(tmp_path):
    error=JointConvergenceError('timeout',dict(final_error_rad=.1225))
    def failed(*args):
        args[-1].append(dict(raw_current_a=[.1]*6,measured_joints_rad=[.2]*6))
        raise error
    bridge=Mock();bridge._execute.side_effect=failed
    log=Mock();fn=method('execute',dict(os=os,json=json,rospy=SimpleNamespace(logerr=log),GUARDED_PRESS_TRACE_DIRECTORY=str(tmp_path)))
    with pytest.raises(JointConvergenceError) as caught:fn(bridge,[],None,None,'button2')
    assert caught.value is error
    rows=[json.loads(line) for line in (tmp_path/'button2.jsonl').read_text().splitlines()]
    assert rows[0]['raw_current_a']==[.1]*6
    audit=json.loads((tmp_path/'button2.jsonl.retract.json').read_text())
    assert audit['retract_convergence']['outcome']=='timeout'
    assert audit['full_feedback_trace_samples']==1
    log.assert_not_called()


def test_trace_write_failure_does_not_replace_original_safety_exception(tmp_path):
    original=RuntimeError('user stop')
    bridge=Mock();bridge._execute.side_effect=original
    fakeos=SimpleNamespace(path=os.path,makedirs=Mock(side_effect=OSError('disk full')))
    log=Mock();fn=method('execute',dict(os=fakeos,json=json,rospy=SimpleNamespace(logerr=log),GUARDED_PRESS_TRACE_DIRECTORY=str(tmp_path)))
    with pytest.raises(RuntimeError,match='user stop') as caught:fn(bridge,[],None,None,'task')
    assert caught.value is original and log.called
