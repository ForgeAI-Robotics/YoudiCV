"""Exercise the production orchestration method without ROS or hardware."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace
import pytest
from tools.orangepi_arm_mode import EscapeModeDriftError
from tools.top_chassis_recovery import geometry_score,require_expected_contacts,validate_escape_transition_pose

START=[0.,.07,.03,0.,.2,0.]
SETTLED=[0.,.07,0.,0.,.2,0.]

def setup_gate(*,repeated_failure=False,moving=False,other_collision=False):
    clock=[0.]
    def sleep(dt):clock[0]+=dt
    code=ast.parse(Path('tools/orangepi_gamepad_servo_bridge.py').read_text())
    cls=next(n for n in code.body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='recover_top_chassis_boundary')
    env=dict(EscapeModeDriftError=EscapeModeDriftError,geometry_score=geometry_score,
        require_expected_contacts=require_expected_contacts,validate_escape_transition_pose=validate_escape_transition_pose,
        time=SimpleNamespace(monotonic=lambda:clock[0]),rospy=SimpleNamespace(sleep=sleep))
    exec(compile(ast.Module(body=[method],type_ignores=[]),'production_replan','exec'),env)
    calls=[]
    audit=dict(held_joints_rad=START,ctrl_mode_before=[2]*3,ctrl_mode_after=[1]*3,max_hold_drift_rad=.005)
    def once():
        calls.append('attempt')
        if len(calls)==1 or repeated_failure:raise EscapeModeDriftError(audit)
        return dict(status='recovered')
    def current():
        q=list(SETTLED)
        if moving:q[2]=-.005*clock[0]
        return q
    gate=SimpleNamespace(lock=threading.RLock(),_recover_top_chassis_boundary_once=once,
        actual_joints=SETTLED, last_boundary_recovery={'failed_stage':'confirming_can_actuation_mode','error':'drift'},
        _current_joints=current,_read_can_modes=lambda:[1]*3,
        collision=SimpleNamespace(check_state=lambda q:dict(valid=not other_collision,contact_pairs=[['piper_link3','top_chassis_link']])),
        escape_geometry=SimpleNamespace(measure=lambda q:dict(overlap=True,upward_exit_m=.007,clearance_m=0.)))
    return lambda:env['recover_top_chassis_boundary'](gate),calls,clock

def test_one_automatic_replan_after_stable_confirmed_mode():
    run,calls,clock=setup_gate()
    result=run()
    assert len(calls)==2
    assert clock[0]>=.3
    assert result['activation_replan']['count']==1
    assert result['activation_replan']['stable_start_rad']==SETTLED

def test_never_third_attempt():
    run,calls,_=setup_gate(repeated_failure=True)
    with pytest.raises(EscapeModeDriftError):run()
    assert len(calls)==2

def test_no_replan_while_moving():
    run,calls,_=setup_gate(moving=True)
    with pytest.raises(RuntimeError,match='stabilize'):run()
    assert len(calls)==1

def test_new_contact_blocks_replan():
    run,calls,_=setup_gate(other_collision=True)
    with pytest.raises(RuntimeError,match='unexpected contacts'):run()
    assert len(calls)==1

def test_wrong_direction_or_fixed_joint_motion_rejected():
    for index,delta in [(0,.02),(1,-.003),(2,.003)]:
        q=list(START);q[index]+=delta
        with pytest.raises(RuntimeError):validate_escape_transition_pose(START,q)
