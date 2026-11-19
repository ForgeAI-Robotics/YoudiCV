import pytest

from piper_handeye.exception_reporting import exception_report
from piper_handeye.arm_report import terminal_event


@pytest.mark.parametrize('report', [
    {'failure_code':'BUTTON_LOCALIZATION_TIMEOUT'},
    {'failure':{'type':'TimeoutError','message':'IK timed out'}},
    {'failure_stage':'pressing','origin_verification':{'verified':False}},
    {'status':'failed'},
])
def test_ordinary_or_unknown_failure_not_invented_as_level_one(report):
    assert exception_report(report) == {}


def test_failed_origin_reports_manual_attention_without_commands():
    r = exception_report({'failure_recovery_error':'return failed'})
    assert r['exception_status_code']=='MANUAL_INTERVENTION_REQUIRED'
    assert r['exception_level']==1 and r['manual_intervention_required'] is True
    assert r['report_only'] is True


def test_cycle_parent_retains_original_recovery_authority():
    r = exception_report({'failure_code':'ARM_COMMUNICATION_LOST'},cycle_test=True)
    assert r['exception_status_code']=='TEST_RECOVERY_PENDING'
    assert r['manual_intervention_required'] is False
    assert r['production_exception_status_code']=='MANUAL_INTERVENTION_REQUIRED'


def test_terminal_preserves_existing_reason_outcome_and_attaches_attention(monkeypatch):
    monkeypatch.delenv('YOUXIAOMEI_CYCLE_PARENT_RECOVERY',raising=False)
    report={'process_exit_code':1,'failure_code':'ARM_ORIGIN_RETURN_FAILED',
            'failure':{'type':'RuntimeError','message':'cannot return'},'task_id':'a'}
    event=terminal_event(report)
    assert event['reason_code']=='ARM_ORIGIN_RETURN_FAILED'
    assert event['phase']=='failed' and event['outcome']=='failed'
    assert event['exception_status_code']=='MANUAL_INTERVENTION_REQUIRED'
    assert event['evidence']['exception_report']['report_only'] is True


def test_already_lit_reports_triggered_without_completing_or_changing_flow():
    from piper_handeye.mission_status import normalize_flow_event
    event=normalize_flow_event({'event':'button_already_on','press_performed':False})
    assert event['reason_code']=='BUTTON_ALREADY_TRIGGERED'
    assert event['phase']=='button_on_confirmed' and event['outcome']=='active'
    assert event['evidence']['press_performed'] is False
