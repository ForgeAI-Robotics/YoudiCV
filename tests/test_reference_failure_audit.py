"""Failure-only producer -> fsynced journal -> final report, no robot movement."""
from types import SimpleNamespace as NS
import json
import time
import traceback
from unittest.mock import Mock
import numpy as np
import pytest
import yaml
from piper_handeye import button_light_hardware as hardware
from piper_handeye.arm_report import capture_failure,write_report_atomic
from piper_handeye.observation_pose import (
    OBSERVATION_JOINT_TOLERANCE_RAD,OBSERVATION_STABILITY_DELTA_RAD)
from tests.test_up_stationary_light_reference import executor_fixture


def harness(monkeypatch,before=None,after=None,hardware_after=1,read_failure=None):
    e,trace,joints=executor_fixture();e.validations['observation_to_prepress']={}
    if before is not None:joints[:]=before
    start=joints.copy();done=[False];events=[]
    def read(**kwargs):
        assert kwargs==dict(max_age=.25)
        if done[0] and read_failure:raise read_failure
        return NS(joints=(np.asarray(after,dtype=float).copy() if done[0] and after is not None else joints.copy()),
                  timestamp=time.monotonic()-.01,
                  raw={'hardware_ok':hardware_after if done[0] else 1,'not_for_journal':'SECRET'})
    e.robot.latest_state=read
    e.event_sink=lambda name,value:events.append(dict(name=name,value=value))
    monkeypatch.setattr(hardware,'wait_for_stable_observation_pose',
        lambda target,read_state,safety:(read_state(),dict(outcome='stable')))
    def capture():
        done[0]=True
        return dict(mode='stationary_5cm_off_reference',reference_count=2,sequences=[101,102],capture_barrier_unix=time.time())
    return e,trace,events,capture,start


def rejected(e,capture,trace,events,exception_class=RuntimeError):
    with pytest.raises(exception_class) as err:e.press_with_stationary_reference(capture,already_at_observation=True)
    assert 'press' not in trace and 'rearm' not in trace
    failures=[x['value'] for x in events if x['name']=='stationary_reference_pose_confirmation' and x['value'].get('stage')=='after_capture']
    assert len(failures)==1 and failures[0]['status']=='failed'
    audit=err.value.stationary_reference_pose_audit
    assert failures[0]['audit']==audit
    assert 'SECRET' not in json.dumps(audit,allow_nan=False)
    return audit,err.value


def test_drift_only_logs_joint_vectors_timestamp_freshness_and_exact_threshold(monkeypatch):
    q=np.zeros(6);q[4]=.007
    e,trace,events,capture,start=harness(monkeypatch,after=q)
    audit,exc=rejected(e,capture,trace,events)
    assert str(exc)=='arm moved while collecting the stationary reference'
    assert audit['rejection_reasons']==['capture_drift_exceeded']
    assert audit['max_drift_joint_1based']==5
    assert audit['joint_drift_signed_rad']==q.tolist()
    assert audit['joint_drift_abs_rad']==q.tolist()
    assert audit['before']['joints_rad']==start.tolist() and audit['after']['joints_rad']==q.tolist()
    assert audit['after']['hardware_ok']==1 and 0<=audit['after']['age_sec']<=.25
    assert audit['after']['timestamp_monotonic_sec']>audit['before']['timestamp_monotonic_sec']
    assert audit['thresholds']['capture_drift_rad']==.006
    assert audit['thresholds']['target_error_rad']==.008
    assert audit['reference']['sequences']==[101,102]


def test_target_error_only_is_not_misreported_as_capture_motion(monkeypatch):
    before=np.zeros(6);before[1]=.0079;after=before.copy();after[1]=.0082
    e,trace,events,capture,_=harness(monkeypatch,before=before,after=after)
    a,_=rejected(e,capture,trace,events)
    assert a['rejection_reasons']==['target_error_exceeded']
    assert a['max_target_error_joint_1based']==2 and a['final_error_rad']==.0082
    assert a['capture_drift_rad']==pytest.approx(.0003)


@pytest.mark.parametrize('flag',[0,None,'1'])
def test_hardware_flag_and_original_type_are_preserved(monkeypatch,flag):
    e,trace,events,capture,_=harness(monkeypatch,hardware_after=flag)
    a,_=rejected(e,capture,trace,events)
    assert a['rejection_reasons']==['hardware_not_ok']
    assert a['after']['hardware_ok']==flag and a['after']['hardware_ok_type']==type(flag).__name__


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_readings_are_explicit_and_strict_json_safe(monkeypatch,bad):
    q=np.zeros(6);q[2]=bad
    e,trace,events,capture,_=harness(monkeypatch,after=q)
    a,_=rejected(e,capture,trace,events)
    assert a['rejection_reasons']==['target_error_nonfinite','capture_drift_nonfinite']
    assert a['after']['nonfinite_or_invalid_joint_indices_1based']==[3]
    assert a['after']['joints_rad'][2] is None and a['capture_drift_rad'] is None
    assert a['max_drift_joint_1based'] is None and not a['capture_drift_is_finite']


def test_all_actual_guard_conditions_are_recorded_not_just_first(monkeypatch):
    e,trace,events,capture,_=harness(monkeypatch,after=[.009,0,0,0,0,0],hardware_after=0)
    a,_=rejected(e,capture,trace,events)
    assert a['rejection_reasons']==['hardware_not_ok','target_error_exceeded','capture_drift_exceeded']


def test_larger_drift_is_rejected_and_visible_at_new_limit(monkeypatch):
    before=np.full(6,.071);after=before.copy();after[5]=.078
    e,trace,events,capture,_=harness(monkeypatch,before=before,after=after)
    e._require_plan().observation_joints[:]=before
    a,_=rejected(e,capture,trace,events)
    assert a['capture_drift_rad']==.078-.071>.006
    assert a['max_drift_joint_1based']==6 and a['rejection_reasons']==['capture_drift_exceeded']


def test_unavailable_or_stale_feedback_keeps_timeout_and_does_not_invent_after_angles(monkeypatch):
    failure=TimeoutError('latest Piper state is missing or stale')
    e,trace,events,capture,_=harness(monkeypatch,read_failure=failure)
    a,exc=rejected(e,capture,trace,events,TimeoutError)
    assert exc is failure and a['after'] is None and a['joint_drift_signed_rad'] is None
    assert a['rejection_reasons']==['feedback_read_failed'] and not a['guard_evaluated']


def test_shape_calculation_error_keeps_original_exception(monkeypatch):
    e,trace,events,capture,_=harness(monkeypatch,after=[0.]*5)
    a,_=rejected(e,capture,trace,events,ValueError)
    assert a['rejection_reasons']==['joint_error_calculation_failed']
    assert a['after']['joint_count']==5 and a['joint_target_error_signed_rad'] is None


def test_safety_exception_is_never_relabeled_as_pose_drift(monkeypatch):
    e,trace,events,capture,_=harness(monkeypatch)
    failure=RuntimeError('independent stop latched')
    e.motion_safety=NS(_assert_clear=Mock(side_effect=failure))
    a,exc=rejected(e,capture,trace,events)
    assert exc is failure and a['rejection_reasons']==['safety_check_failed']


def test_real_journal_and_atomic_report_keep_failure_audit_through_return_phase(tmp_path,monkeypatch):
    from run_button_light_closed_loop import append_journal_checkpoint,persist_journal_row
    e,trace,events,capture,_=harness(monkeypatch,after=[.007,0,0,0,0,0])
    report={'active_stage':'stationary_reference_pose_confirmation'};p=tmp_path/'journal.jsonl'
    def sink(kind,value):
        events.append(dict(name=kind,value=value))
        row=append_journal_checkpoint(report,kind,value);persist_journal_row(p,row)
    e.event_sink=sink
    a,exc=rejected(e,capture,trace,events)
    capture_failure(report,exc,'original traceback')
    report['phase']='origin_return_completed'
    write_report_atomic(tmp_path/'closed_loop_report.yaml',report)
    saved=yaml.safe_load((tmp_path/'closed_loop_report.yaml').read_text())
    assert saved['stationary_reference_pose_failure']==a
    assert saved['failure']['message']==str(exc) and saved['failure']['traceback']=='original traceback'
    lines=[json.loads(l) for l in p.read_text().splitlines()]
    assert next(x['value']['audit'] for x in lines if x['value'].get('stage')=='after_capture')==a


def test_journal_failure_does_not_replace_rejection_or_allow_motion(monkeypatch):
    e,trace,events,capture,_=harness(monkeypatch,after=[.007,0,0,0,0,0])
    def sink(name,value):
        if value.get('stage')=='after_capture':raise OSError('disk unavailable')
    e.event_sink=sink
    with pytest.raises(RuntimeError,match='arm moved') as err:
        e.press_with_stationary_reference(capture,already_at_observation=True)
    report={};capture_failure(report,err.value,'trace')
    assert report['stationary_reference_pose_failure']['capture_drift_rad']==.007
    assert report['stationary_reference_pose_diagnostics_error']=='OSError: disk unavailable'
    assert 'press' not in trace and 'rearm' not in trace


def test_success_has_no_extra_failure_record_or_failure_builder_call(monkeypatch):
    e,trace,events,capture,_=harness(monkeypatch,after=[.001,0,0,0,0,0])
    forbidden=Mock(side_effect=AssertionError('failure audit on success'))
    monkeypatch.setattr(hardware,'stationary_reference_failure_audit',forbidden)
    e.press_with_stationary_reference(capture,already_at_observation=True)
    assert 'press' in trace and 'rearm' in trace
    assert not any(x['value'].get('stage')=='after_capture' for x in events)
    forbidden.assert_not_called()
    assert OBSERVATION_JOINT_TOLERANCE_RAD==.008 and OBSERVATION_STABILITY_DELTA_RAD==.002


def test_reference_capture_error_is_not_mislabeled_as_post_capture_pose_failure(monkeypatch):
    e,trace,events,_,_=harness(monkeypatch)
    def capture():raise RuntimeError('cannot confirm OFF')
    with pytest.raises(RuntimeError,match='cannot confirm OFF'):
        e.press_with_stationary_reference(capture,already_at_observation=True)
    assert not any(x['value'].get('stage')=='after_capture' for x in events)
    assert 'press' not in trace
