"""Candidate observer tests with numeric inputs only; never accesses a robot."""
from dataclasses import replace
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from piper_handeye.contact_prevention import CandidateGuardedPressMonitor, JointPoseCurrentReference
from piper_handeye.guarded_press import (GuardedPressConfig, GuardedPressMonitor,
    RobustCurrentBaseline, MotorCurrentSample, piper_tcp_linear_jacobian,
    PIPER_TORQUE_PER_AMP_NM)


Q = np.array([-.0241, 1.7019, -.9849, .0784, -.3871, .0249])
NORMAL = np.array([-1., 0., 0.])


def config():
    return GuardedPressConfig(contact_delta_a=np.full(6, .05), hard_delta_a=np.full(6, 2.),
        absolute_current_limit_a=np.full(6, 5.),
        contact_estimator='cartesian_normal_force_kalman', contact_arm_travel_m=.008,
        max_forward_travel_m=.022, max_press_time_sec=5., persistence_sec=.4,
        relative_current_hard_stop_enabled=False, absolute_current_hard_stop_enabled=False)


def monitor(**kwargs):
    return CandidateGuardedPressMonitor(RobustCurrentBaseline(np.zeros(6), np.full(6, .01), 60),
        config(), contact_normal_base=NORMAL, measured_progress=False, motion_reference=False, **kwargs)


def warmup(m):
    for i in range(200):
        t=i*.005
        m.update(MotorCurrentSample(t, np.zeros(6)), now=t, elapsed_sec=t,
                 forward_travel_m=.007, joints_rad=Q, encoder_age_sec=.001)


def update(m, t, values, travel=.009, **kwargs):
    return m.update(MotorCurrentSample(t, values), now=t, elapsed_sec=t,
        forward_travel_m=travel, joints_rad=Q, encoder_age_sec=.001, **kwargs)


def test_long_gap_does_not_complete_four_hundred_ms_contact():
    m=monitor(); warmup(m)
    current=-3.*(piper_tcp_linear_jacobian(Q).T@NORMAL)/PIPER_TORQUE_PER_AMP_NM
    for i in range(18):
        result=update(m, 1.+i*.005, current)
    assert result.persistence_sec < .1
    result=update(m, 1.411, current, travel=.013)
    assert result.decision.value == 'CONTINUE' and result.persistence_sec == 0.
    assert m.continuity_resets == 1
    for i in range(1,100):
        result=update(m, 1.411+i*.005, current, travel=.013)
        if result.decision.value == 'CONTACT':break
    assert result.decision.value == 'CONTACT' and result.persistence_sec >= .4


def test_normal_contact_still_triggers_with_continuous_full_rate_input():
    m=monitor();warmup(m)
    current=-3.*(piper_tcp_linear_jacobian(Q).T@NORMAL)/PIPER_TORQUE_PER_AMP_NM
    for i in range(120):
        r=update(m,1.+i*.005,current)
        if r.decision.value!='CONTINUE':break
    assert r.decision.value=='CONTACT' and r.persistence_sec>=.4
    assert i*.005<.5 and m.continuity_resets==0


def test_tangential_evidence_also_resets_across_gap():
    m=monitor();warmup(m)
    current=3.*(piper_tcp_linear_jacobian(Q).T@np.array([0.,1.,0.]))/PIPER_TORQUE_PER_AMP_NM
    for i in range(13):r=update(m,1.+i*.005,current)
    assert r.decision.value=='CONTINUE'
    r=update(m,1.4,current)
    assert r.decision.value=='CONTINUE' and m._tangential_persistence==1


@pytest.mark.parametrize('stamp,now',[(1.,1.),(.99,1.01),(1.02,1.)])
def test_duplicate_reversed_or_future_time_never_accumulates_contact(stamp,now):
    m=monitor();warmup(m);update(m,1.,np.zeros(6))
    r=m.update(MotorCurrentSample(stamp,np.zeros(6)),now=now,elapsed_sec=1.,
        forward_travel_m=.009,joints_rad=Q,encoder_age_sec=.001)
    assert r.decision.value=='HARD_STOP'


def test_raw_current_spike_does_not_become_contact():
    m=monitor();warmup(m)
    assert update(m,1.,np.full(6,3.)).decision.value=='CONTINUE'
    for i in range(1,100):
        assert update(m,1.+i*.005,np.zeros(6)).decision.value=='CONTINUE'


def test_measured_progress_can_keep_contact_gate_closed_when_plan_is_ahead():
    m=CandidateGuardedPressMonitor(RobustCurrentBaseline(np.zeros(6),np.full(6,.01),60),
        config(),contact_normal_base=NORMAL,motion_reference=False)
    for i in range(240):
        t=i*.005
        r=update(m,t,np.zeros(6),travel=.015)
    assert r.reason=='contact_detector_not_yet_armed'
    assert m.candidate_audit['measured_travel_m']==0.
    assert m.candidate_audit['planned_travel_m']==.015


@pytest.mark.parametrize('age',[None,.101,float('nan')])
def test_measured_encoder_freshness_is_required(age):
    m=CandidateGuardedPressMonitor(RobustCurrentBaseline(np.zeros(6),np.full(6,.01),60),
        config(),contact_normal_base=NORMAL)
    r=m.update(MotorCurrentSample(0.,np.zeros(6)),now=0.,elapsed_sec=0.,
        forward_travel_m=0.,joints_rad=Q,encoder_age_sec=age)
    assert r.decision.value=='HARD_STOP' and 'feedback' in r.reason


def test_original_planned_displacement_limit_remains_active():
    m=CandidateGuardedPressMonitor(RobustCurrentBaseline(np.zeros(6),np.full(6,.01),60),
        config(),contact_normal_base=NORMAL)
    r=update(m,0.,np.zeros(6),travel=.02201)
    assert r.reason=='maximum_forward_travel' and r.decision.value=='HARD_STOP'


def reference():
    m=JointPoseCurrentReference(40,np.ones(6))
    for x in np.linspace(0,.03,40):
        q=np.zeros(6);q[1]=x
        m.observe(q,np.full(6,.2+2*x))
    m.freeze()
    return m


def test_reference_follows_measured_pose_after_learning_is_frozen():
    m=reference();q=np.zeros(6);q[1]=.04
    predicted,_=m.predict(q,(np.full(6,.26),np.full(6,.005)))
    np.testing.assert_allclose(predicted,.28,atol=1e-10)
    before=m.model['coefficients'].copy()
    m.observe(q,np.full(6,10.))
    np.testing.assert_array_equal(before,m.model['coefficients'])


def test_model_cannot_extrapolate_to_unseen_path_or_unlimited_current():
    m=reference();q=np.zeros(6);q[1]=1.
    assert m.predict(q,(np.zeros(6),np.full(6,.005))) is None
    q[1]=.04;m.limit[:]=.01
    pred,_=m.predict(q,(np.zeros(6),np.full(6,.005)))
    assert np.max(np.abs(pred))<=.01
    q[2]=.1
    assert m.predict(q,(np.zeros(6),np.full(6,.005))) is None


def test_stationary_joint_packets_cannot_identify_a_pose_model():
    m=JointPoseCurrentReference(40,np.ones(6))
    for _ in range(40):m.observe(Q,np.ones(6))
    m.freeze()
    assert m.model is None


def test_sparse_logging_is_not_misrepresented_as_continuous_sensor_input():
    m=monitor()
    for i in range(40):
        r=update(m,i*.075,np.zeros(6),travel=.007 if i<24 else .009)
        if r.decision.value!='CONTINUE':break
    assert m.continuity_resets>0 and r.reason=='precontact_motion_reference_not_ready'


def test_disabled_candidate_features_match_original_observer():
    old=GuardedPressMonitor(RobustCurrentBaseline(np.zeros(6),np.full(6,.01),60),
        config(),contact_normal_base=NORMAL)
    new=monitor(continuous_evidence=False)
    for i in range(500):
        t=i*.005;travel=.007 if t<1. else .009
        values=np.zeros(6) if t<1. else -3.*(piper_tcp_linear_jacobian(Q).T@NORMAL)/PIPER_TORQUE_PER_AMP_NM
        a=old.update(MotorCurrentSample(t,values),now=t,elapsed_sec=t,forward_travel_m=travel,joints_rad=Q)
        b=update(new,t,values,travel=travel)
        assert (a.decision,a.reason,a.persistence_sec)==(b.decision,b.reason,b.persistence_sec)
        np.testing.assert_allclose(a.filtered_current_a,b.filtered_current_a)
        if a.decision.value!='CONTINUE':break


@pytest.mark.parametrize('mode',[None,'legacy','candidate'])
def test_actual_bridge_defaults_to_candidate_and_keeps_explicit_legacy_override(mode):
    from tools.piper_action_watchdog import require_safety_generation
    source=Path('tools/orangepi_guarded_press_bridge.py')
    cls=next(n for n in ast.parse(source.read_text()).body
             if isinstance(n,ast.ClassDef) and n.name=='GuardedBridge')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_execute')
    original,candidate=Mock(),Mock()
    ros=SimpleNamespace(get_param=lambda key,default:1 if key.endswith('generation') else False)
    ns=dict(np=np,rospy=ros,require_safety_generation=require_safety_generation,
            GuardedPressMonitor=original,CandidateGuardedPressMonitor=candidate)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    bridge=Mock();bridge.args=SimpleNamespace() if mode is None else SimpleNamespace(contact_observer=mode)
    bridge.current_joints.return_value=np.zeros(6)
    bridge.baseline.return_value=RobustCurrentBaseline(np.zeros(6),np.full(6,.01),60)
    bridge.numeric_contact_trace=None
    bridge.goal.side_effect=RuntimeError('inspection before goal send')
    with pytest.raises(RuntimeError,match='inspection before goal send'):
        ns['_execute'](bridge,[dict(positions=[0.]*6)],config(),[-1.,0.,0.],
                       'virtual',None,None,None,[])
    assert candidate.called is (mode!='legacy')
    assert original.called is (mode=='legacy')
    bridge.client.send_goal.assert_not_called()


@pytest.mark.parametrize('override',[[],['--contact-observer','legacy']])
def test_bridge_command_line_uses_candidate_by_default_without_sending_motion(override,monkeypatch):
    import argparse
    import sys
    source=Path('tools/orangepi_guarded_press_bridge.py')
    main=next(n for n in ast.parse(source.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='main')
    body=[]
    for statement in main.body:
        body.append(statement)
        if (isinstance(statement,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id=='args' for t in statement.targets)):
            break
    namespace=dict(argparse=argparse)
    monkeypatch.setattr(sys,'argv',['bridge','--bind-ip','127.0.0.1',
        '--allowed-client','127.0.0.1','--token','synthetic-test-only']+override)
    # Compile only the parser; exclude ROS initialization and bridge startup.
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),namespace)
    assert namespace['args'].contact_observer==('legacy' if override else 'candidate')
    assert namespace['args'].full_rate_trace is True


@pytest.mark.parametrize('observer_class',[GuardedPressMonitor,CandidateGuardedPressMonitor])
@pytest.mark.parametrize('direction,required_sec,reason',[
    (np.array([3.,0.,0.]),.8,'sustained_cartesian_normal_force_kalman'),
    (np.array([0.,0.,.9]),.4,'sustained_cartesian_tangential_force_kalman')])
def test_800ms_normal_and_400ms_side_confirmation_remain_independent(observer_class,direction,required_sec,reason):
    cfg=replace(config(),persistence_sec=.8,tangential_force_persistence_sec=.4)
    kwargs=dict(contact_normal_base=NORMAL)
    if observer_class is CandidateGuardedPressMonitor:
        kwargs.update(measured_progress=False,motion_reference=False)
    m=observer_class(RobustCurrentBaseline(np.zeros(6),np.full(6,.01),60),cfg,**kwargs)
    def step(t,current,travel):
        extra=dict(encoder_age_sec=.001) if observer_class is CandidateGuardedPressMonitor else {}
        return m.update(MotorCurrentSample(t,current),now=t,elapsed_sec=t,
            forward_travel_m=travel,joints_rad=Q,**extra)
    for i in range(200):
        step(i*.005,np.zeros(6),.007)
    current=(piper_tcp_linear_jacobian(Q).T@direction)/PIPER_TORQUE_PER_AMP_NM
    for i in range(220):
        elapsed=i*.005
        result=step(1.+elapsed,current,.009)
        if elapsed<required_sec:
            assert result.decision.value=='CONTINUE'
        if result.decision.value!='CONTINUE':
            break
    assert result.decision.value=='CONTACT' and result.reason==reason
    assert required_sec<=elapsed<required_sec+.06
