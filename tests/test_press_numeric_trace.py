import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from piper_handeye.press_numeric_trace import NumericPressTrace, load_numeric_trace
from piper_handeye.contact_prevention import CandidateGuardedPressMonitor
from piper_handeye.guarded_press import (GuardedPressMonitor, RobustCurrentBaseline,
    MotorCurrentSample,piper_tcp_linear_jacobian,PIPER_TORQUE_PER_AMP_NM)
from test_contact_prevention import config,Q,NORMAL


def capture(path,capacity=2000):
    baseline=RobustCurrentBaseline(np.zeros(6),np.full(6,.01),60)
    cfg=config();monitor=GuardedPressMonitor(baseline,cfg,contact_normal_base=NORMAL)
    trace=NumericPressTrace(capacity);trace.configure(baseline,cfg,NORMAL,'original')
    current=-3.*(piper_tcp_linear_jacobian(Q).T@NORMAL)/PIPER_TORQUE_PER_AMP_NM
    for i in range(600):
        t=i*.005;values=np.zeros(6) if t<1. else current
        sample=MotorCurrentSample(t,values);travel=.007 if t<1. else .009
        result=monitor.update(sample,now=t+.0001,elapsed_sec=t,
            forward_travel_m=travel,joints_rad=Q)
        trace.record(sample,t+.0001,t,travel,Q,.001,Q,monitor,result)
        if result.decision.value!='CONTINUE':break
    outcome=trace.save(path,task_id='synthetic_no_motion',historical_reason=result.reason)
    return trace,result,outcome


def test_full_numeric_capture_replays_original_decision_sequence_exactly(tmp_path):
    from tools.replay_guarded_press import replay
    path=tmp_path/'capture.npz';trace,original,_=capture(path)
    metadata,rows=load_numeric_trace(path)
    sample=dict(metadata,travel_mm=1000*rows[-1]['forward_travel_m'])
    result=replay(sample,rows,'original_sparse_input')
    assert result['captured_decision_sequence_matches'] is True
    assert result['decision']==original.decision.value=='CONTACT'
    assert result['reason']==original.reason and result['stop_difference_mm']==0.
    assert len(rows)==trace.used and metadata['complete'] is True
    assert rows[0]['processing_monotonic_sec']==.0001


def test_overflow_is_explicit_and_cannot_be_replayed_as_complete(tmp_path):
    path=tmp_path/'overflow.npz';trace,_,outcome=capture(path,capacity=20)
    assert trace.dropped>0 and outcome['complete'] is False
    with pytest.raises(ValueError,match='incomplete'):
        load_numeric_trace(path)


def test_cli_replay_of_complete_bundle_has_no_network_or_motion(tmp_path):
    source=Path('tools/replay_guarded_press.py');path=tmp_path/'capture.npz'
    capture(path);out=tmp_path/'replay'
    result=subprocess.run([sys.executable,str(source),'--bundle',str(path),
        '--output',str(out),'--variants','original_full_rate'],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    summary=json.loads((out/'summary.json').read_text())
    assert summary['full_rate_numeric_input'] is True
    assert summary['physical_motion_sent'] is False and summary['network_prohibited'] is True
    row=json.loads((out/'replay_results.jsonl').read_text().splitlines()[0])
    assert row['captured_decision_sequence_matches'] is True


def test_unconfigured_baseline_failure_does_not_create_a_fake_trace(tmp_path):
    trace=NumericPressTrace()
    assert trace.save(tmp_path/'no_baseline.npz') is None
    assert not (tmp_path/'no_baseline.npz').exists()


def capture_candidate(path, scenario='contact', **options):
    baseline=RobustCurrentBaseline(np.zeros(6),np.full(6,.01),60)
    cfg=config()
    monitor=CandidateGuardedPressMonitor(baseline,cfg,contact_normal_base=NORMAL,**options)
    trace=NumericPressTrace();trace.configure(baseline,cfg,NORMAL,'candidate',monitor=monitor)
    # Numeric encoder path only: no IK service, robot client or motion.
    jacobian=piper_tcp_linear_jacobian(Q)
    direction=jacobian[0]/(jacobian[0]@jacobian[0])
    for i in range(600):
        t=i*.005
        if scenario=='sampling_gap' and 210<=i<218:
            continue
        planned=t*.01;q=Q+direction*planned
        values=np.zeros(6)
        if scenario!='maximum_travel' and t>=1.:
            values=-3.*(piper_tcp_linear_jacobian(q).T@NORMAL)/PIPER_TORQUE_PER_AMP_NM
        age=.11 if scenario=='stale_encoder' and t>=1.1 else .001
        sample=MotorCurrentSample(t,values)
        result=monitor.update(sample,now=t+.0001,elapsed_sec=t,
            forward_travel_m=planned,joints_rad=q,encoder_age_sec=age)
        trace.record(sample,t+.0001,t,planned,q,age,q,monitor,result)
        if result.decision.value!='CONTINUE':
            break
    trace.save(path,task_id='candidate_numeric_only',historical_reason=result.reason)
    return result,monitor


@pytest.mark.parametrize('scenario', ['contact','stale_encoder','sampling_gap','maximum_travel'])
def test_full_candidate_capture_replays_its_actual_decisions(tmp_path,scenario):
    from tools.replay_guarded_press import replay
    path=tmp_path/'candidate.npz';original,monitor=capture_candidate(path,scenario)
    metadata,rows=load_numeric_trace(path)
    assert metadata['producer']=='candidate'
    assert metadata['candidate_observer_options']==dict(continuous_evidence=True,
        measured_progress=True,motion_reference=True,encoder_max_age_sec=.1)
    sample=dict(metadata,travel_mm=1000*rows[-1]['forward_travel_m'])
    outcome=replay(sample,rows,'candidate_full_rate')
    assert outcome['matching_captured_observer'] is True
    assert outcome['captured_decision_sequence_matches'] is True
    assert (outcome['decision'],outcome['reason'])==(original.decision.value,original.reason)
    if scenario in ('contact','sampling_gap'):
        assert original.decision.value=='CONTACT'
        if scenario=='sampling_gap':assert monitor.continuity_resets==1
    else:
        assert original.decision.value=='HARD_STOP'
    comparison=replay(sample,rows,'original_full_rate')
    assert comparison['matching_captured_observer'] is False
    assert comparison['captured_decision_sequence_matches'] is None


def test_full_candidate_replay_preserves_recorded_feature_options(tmp_path):
    from tools.replay_guarded_press import replay
    path=tmp_path/'candidate.npz'
    capture_candidate(path,measured_progress=False,motion_reference=False,encoder_max_age_sec=.05)
    metadata,rows=load_numeric_trace(path)
    sample=dict(metadata,travel_mm=1000*rows[-1]['forward_travel_m'])
    assert replay(sample,rows,'candidate_full_rate')['captured_decision_sequence_matches'] is True
    assert replay(sample,rows,'candidate_strict_continuity')['captured_decision_sequence_matches'] is None


@pytest.mark.parametrize('field', ['original_observer_sha256','candidate_observer_sha256',
                                 'candidate_observer_options'])
def test_candidate_capture_with_mismatched_source_or_missing_settings_is_rejected(tmp_path,field):
    from tools.replay_guarded_press import replay
    path=tmp_path/'candidate.npz';capture_candidate(path)
    metadata,rows=load_numeric_trace(path);metadata.pop(field)
    with pytest.raises(ValueError,match='source differs|settings are missing'):
        replay(dict(metadata,travel_mm=0.),rows,'candidate_full_rate')


def test_candidate_cli_replays_without_network_or_motion(tmp_path):
    path=tmp_path/'candidate.npz';capture_candidate(path);out=tmp_path/'replay'
    result=subprocess.run([sys.executable,'tools/replay_guarded_press.py','--bundle',str(path),
        '--output',str(out),'--variants','candidate_full_rate','original_full_rate'],
        capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    summary=json.loads((out/'summary.json').read_text())
    assert summary['network_prohibited'] and not summary['physical_motion_sent']
    rows=[json.loads(line) for line in (out/'replay_results.jsonl').read_text().splitlines()]
    assert rows[0]['captured_decision_sequence_matches'] is True
    assert rows[1]['captured_decision_sequence_matches'] is None
