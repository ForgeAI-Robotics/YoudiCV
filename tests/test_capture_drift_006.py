"""Separate capture-span drift from pre-capture stability, no robot movement."""
import ast
from pathlib import Path
import numpy as np
import pytest
from piper_handeye.observation_pose import (
    OBSERVATION_CAPTURE_DRIFT_RAD,OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD,
    OBSERVATION_STABILITY_DELTA_RAD,OBSERVATION_JOINT_TOLERANCE_RAD,
    OBSERVATION_FEEDBACK_MAX_AGE_SEC,capture_drift_exceeds_limit)
from tests.test_reference_failure_audit import harness,rejected
from tests.test_observation_pose import confirm,ObservationPoseError


@pytest.mark.parametrize('drift',[.002,.003,.005,.006])
def test_new_capture_span_allows_change_up_to_six_milliradians(monkeypatch,drift):
    after=np.zeros(6);after[4]=drift
    e,trace,events,capture,_=harness(monkeypatch,after=after)
    e.press_with_stationary_reference(capture,already_at_observation=True)
    assert 'rearm' in trace and 'press' in trace
    evidence=next(x['value'] for x in events if x['name']=='stationary_press_reference_confirmed')
    assert evidence['capture_drift_rad']==drift
    assert evidence['capture_drift_tolerance_rad']==.006
    assert evidence['capture_drift_numeric_epsilon_rad']==1e-12
    assert not any(x['value'].get('stage')=='after_capture' for x in events)


@pytest.mark.parametrize('drift',[.00600001,.007])
def test_real_excess_still_rejects_and_logs_new_threshold(monkeypatch,drift):
    e,trace,events,capture,_=harness(monkeypatch,after=[drift,0,0,0,0,0])
    audit,_=rejected(e,capture,trace,events)
    assert audit['rejection_reasons']==['capture_drift_exceeded']
    assert audit['thresholds']['capture_drift_rad']==.006
    assert audit['thresholds']['stability_window_delta_rad']==.002
    assert audit['thresholds']['capture_drift_numeric_epsilon_rad']==1e-12


def test_six_milliradian_float_representation_boundary_is_not_false_failure(monkeypatch):
    before=np.full(6,.071);after=before.copy();after[5]=.077
    assert after[5]-before[5]>.006
    e,trace,events,capture,_=harness(monkeypatch,before=before,after=after)
    e._require_plan().observation_joints[:]=before
    e.press_with_stationary_reference(capture,already_at_observation=True)
    assert 'press' in trace
    assert not capture_drift_exceeds_limit(after[5]-before[5])
    assert capture_drift_exceeds_limit(.006+2e-12)


def test_target_error_is_still_independent_and_not_relaxed(monkeypatch):
    before=np.zeros(6);before[2]=.004;after=before.copy();after[2]=.009
    e,trace,events,capture,_=harness(monkeypatch,before=before,after=after)
    a,_=rejected(e,capture,trace,events)
    assert a['capture_drift_rad']<.006 and a['rejection_reasons']==['target_error_exceeded']
    assert a['thresholds']['target_error_rad']==.008


def test_pre_capture_stability_window_is_not_silently_widened():
    calls=[0]
    def reading(c):
        calls[0]+=1
        return ([0. if calls[0]%2 else .003]*6,c(),1)
    with pytest.raises(ObservationPoseError,match='timeout'):
        confirm(reading)
    assert OBSERVATION_STABILITY_DELTA_RAD==.002
    assert OBSERVATION_CAPTURE_DRIFT_RAD==.006
    assert OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD==1e-12
    assert OBSERVATION_JOINT_TOLERANCE_RAD==.008
    assert OBSERVATION_FEEDBACK_MAX_AGE_SEC==.25


def test_check_and_execution_report_policies_both_expose_separate_limits():
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'run_button_light_closed_loop.py').read_text())
    policies=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Dict):continue
        for key,value in zip(node.keys,node.values):
            if isinstance(key,ast.Constant) and key.value=='observation_pose_policy':
                policies.append({k.value:ast.unparse(v) for k,v in zip(value.keys,value.values)})
    assert len(policies)==2
    for p in policies:
        assert p['stability_delta_rad']=='OBSERVATION_STABILITY_DELTA_RAD'
        assert p['capture_drift_tolerance_rad']=='OBSERVATION_CAPTURE_DRIFT_RAD'
        assert p['capture_drift_numeric_epsilon_rad']=='OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD'
