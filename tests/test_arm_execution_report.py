"""Failure reporting must preserve physical evidence without permitting motion."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock
import json

import numpy as np
import pytest
import yaml

from piper_handeye.arm_report import (
    arm_failure_message, capture_failure, execution_evidence, finalize_report,
    plain_data, terminal_event, update_report_checkpoint, write_report_atomic,
)
from piper_handeye.assistant_gui_events import AssistantEventPresenter, completion_message
from piper_handeye.external_call_mission import ExternalCallMission
from piper_handeye.mission_status import append_status, describe_phase_zh, read_events
from run_button_light_closed_loop import append_journal_checkpoint
from run_voice_command_gateway import CommandDispatcher
from tools.navigation_arm_worker import arm_result


SCENE = json.loads((Path(__file__).parent / 'fixtures/arm_report/down_failure_20261019.json').read_text())
EXPECTED = '任务失败：按压前定位采样超时；未执行按压；机械臂已归位'


def failure_report():
    report = dict(schema_version=2, task_id=SCENE['task_id'], mission_id='test-mission',
                  target_label='down', current_floor=8, destination_floor=None,
                  segment_id='arm_task', status='running', lifecycle_status='running',
                  phase=SCENE['failure_stage'], active_stage=SCENE['failure_stage'],
                  watch_position=dict(predicted_min_edge_margin_px=np.float64(SCENE['numpy_margin'])))
    capture_failure(report, RuntimeError(SCENE['failure_message']), 'original traceback')
    value = deepcopy(SCENE['origin_return'])
    append_journal_checkpoint(report, 'origin_return_completed', value)
    update_report_checkpoint(report, 'origin_return_completed', value, 'origin_return_completed')
    report['phase'] = 'failed:RuntimeError'
    return report


def test_numpy_measurements_keep_numeric_types_and_nested_structure(tmp_path):
    data = dict(a=np.float32(1.25), b=np.float64(2.5), yes=np.bool_(True),
                count=np.int64(3), array=np.array([[1., 2.]]), nested=[np.float32(4)], path=tmp_path)
    converted = plain_data(data)
    assert type(converted['a']) is float and type(converted['yes']) is bool
    assert type(converted['count']) is int and converted['array'] == [[1., 2.]]
    path = tmp_path / 'report.yaml'
    write_report_atomic(path, data)
    assert yaml.safe_load(path.read_text()) == converted


def test_recorded_down_failure_now_saves_and_publishes_consistent_facts(tmp_path):
    report = failure_report()
    with pytest.raises(yaml.representer.RepresenterError):
        yaml.safe_dump(report)
    events = []
    path = tmp_path / 'report.yaml'
    assert finalize_report(report, path, events.append) == []
    saved = yaml.safe_load(path.read_text())
    evidence = events[-1]['evidence']
    assert saved['status'] == 'failed' and saved['lifecycle_status'] == 'failed'
    assert saved['failure']['message'] == SCENE['failure_message']
    assert saved['failure_stage'] == SCENE['failure_stage']
    assert type(saved['watch_position']['predicted_min_edge_margin_px']) is float
    assert saved['origin_verification']['verified'] is True
    assert saved['returned_to_origin'] is True
    assert saved['press_attempts'] == 0 and saved['press_execution_started'] is False
    assert evidence['origin_verification'] == saved['origin_verification']
    assert evidence['failure_reason'] == saved['failure']['message']
    assert events[-1]['phase'] == 'failed' and events[-1]['reason_code'] == 'BUTTON_LOCALIZATION_TIMEOUT'
    assert arm_failure_message(evidence) == EXPECTED


def test_atomic_replace_failure_leaves_old_report_intact_and_removes_temp(tmp_path, monkeypatch):
    path = tmp_path / 'report.yaml';path.write_text('old: intact\n')
    def fail(*args):
        raise OSError('simulated disk failure')
    monkeypatch.setattr('piper_handeye.arm_report.os.replace', fail)
    with pytest.raises(OSError):
        write_report_atomic(path, {'new': True})
    assert path.read_text() == 'old: intact\n'
    assert list(tmp_path.iterdir()) == [path]


def test_bad_optional_diagnostic_uses_minimal_report_without_masking_primary_exception(tmp_path):
    report = failure_report();report['optional_diagnostic'] = object()
    events = []
    with pytest.raises(RuntimeError, match=SCENE['failure_message']):
        try:
            raise RuntimeError(SCENE['failure_message'])
        except RuntimeError:
            errors = finalize_report(report, tmp_path / 'report.yaml', events.append)
            assert len(errors) == 1
            raise
    saved = yaml.safe_load((tmp_path / 'report.yaml').read_text())
    assert saved['report_incomplete'] is True
    assert saved['failure']['message'] == SCENE['failure_message']
    assert saved['process_exit_code'] == 1
    assert events[-1]['evidence']['failure_reason'] == SCENE['failure_message']
    assert arm_failure_message(events[-1]['evidence']) == EXPECTED


def test_disk_failures_do_not_replace_original_error_or_prevent_status_attempt(tmp_path, monkeypatch):
    report = failure_report();events = []
    def fail(*args):
        raise OSError('read-only disk')
    monkeypatch.setattr('piper_handeye.arm_report.write_report_atomic', fail)
    errors = finalize_report(report, tmp_path / 'report.yaml', events.append)
    assert len(errors) == 2 and len(events) == 1
    assert report['failure']['message'] == SCENE['failure_message']
    assert events[-1]['reason_code'] == 'BUTTON_LOCALIZATION_TIMEOUT'


def test_reporting_failure_cannot_publish_success_for_an_otherwise_successful_task(tmp_path):
    report = failure_report()
    report.update(status='arrived_after_verified_press', lifecycle_status='completed', process_exit_code=0)
    report.pop('failure');report.pop('failure_code');report.pop('failure_stage')
    report['optional_diagnostic'] = object();events = []
    assert finalize_report(report, tmp_path / 'report.yaml', events.append)
    assert events[-1]['phase'] == 'failed'
    assert events[-1]['reason_code'] == 'REPORTING_FAILED'
    assert yaml.safe_load((tmp_path / 'report.yaml').read_text())['process_exit_code'] == 1


def test_status_publish_failure_is_secondary_to_existing_motion_failure(tmp_path):
    report = failure_report()
    def fail(event):
        raise OSError('status unavailable')
    errors = finalize_report(report, tmp_path / 'report.yaml', fail)
    assert errors[-1]['component'] == 'terminal_status'
    assert report['failure']['message'] == SCENE['failure_message']
    assert yaml.safe_load((tmp_path / 'report.yaml').read_text())['reporting_errors']


def test_bad_core_evidence_never_raises_over_the_original_failure(tmp_path):
    report = failure_report()
    report['origin_verification']['unexpected_field'] = object()
    events = []
    errors = finalize_report(report, tmp_path / 'report.yaml', events.append)
    assert errors and report['failure']['message'] == SCENE['failure_message']
    assert not events  # Unserializable evidence cannot be advertised as verified.


@pytest.mark.parametrize('segment,phase', [
    ('call', 'call_button_segment_completed'),
    ('target_floor', 'target_floor_button_segment_completed'),
    ('arm_task', 'arm_button_task_completed'),
])
@pytest.mark.parametrize('flow', ['arrived_after_verified_press', 'arrived_after_initial_on'])
def test_normal_success_keeps_existing_terminal_and_navigation_requirements(tmp_path, segment, phase, flow):
    report = failure_report()
    for key in ('failure', 'failure_code', 'failure_stage'):
        report.pop(key)
    report.update(segment_id=segment, status=flow, flow_status=flow,
                  lifecycle_status='completed', process_exit_code=0)
    events = []
    assert finalize_report(report, tmp_path / 'report.yaml', events.append) == []
    assert events[-1]['phase'] == phase and events[-1]['outcome'] == 'completed'
    assert completion_message(dict(return_code=0, arm_report=events[-1]['evidence'])) == '机械臂按钮操作已完成'
    result = arm_result(events, 0)
    assert result['origin_verified']
    assert not result['on_confirmed'] and not result['off_confirmed']
    # Exit code and origin alone still cannot authorize navigation advancement.
    mission = ExternalCallMission();mid = mission.begin(8, 2, 0);mission.phase = 'arm_call'
    assert mission.arm_finished(mid, result) == 'failed'


def test_uncertain_flow_keeps_existing_nonzero_result_and_uncertainty_message(tmp_path):
    report = failure_report()
    for key in ('failure', 'failure_code', 'failure_stage'):
        report.pop(key)
    report.update(status='post_press_confirmation_uncertain', flow_status='post_press_confirmation_uncertain',
                  lifecycle_status='completed', process_exit_code=2)
    events = []
    assert finalize_report(report, tmp_path / 'report.yaml', events.append) == []
    assert events[-1]['phase'] == 'failed' and events[-1]['reason_code'] == 'ARM_TASK_FAILED'
    assert '不确定' in completion_message(dict(return_code=2, mission_status_events=events))


def test_aborted_task_remains_aborted_and_does_not_become_motion_success(tmp_path):
    report = failure_report()
    report['failure']['type'] = 'TerminationRequested'
    report.pop('failure_code')
    events = []
    assert finalize_report(report, tmp_path / 'report.yaml', events.append) == []
    assert events[-1]['phase'] == 'aborted' and events[-1]['reason_code'] == 'TASK_ABORTED'
    assert events[-1]['evidence']['process_exit_code'] == 1


@pytest.mark.parametrize('kind', ['origin_return_completed', 'origin_return_verified', 'final_origin_verification'])
def test_origin_encoder_evidence_survives_recovery_phase_changes(kind):
    report = failure_report()
    original_stage = report['failure_stage']
    value = dict(verified=np.bool_(True), hardware_ok=np.int64(1), encoder_error_rad=np.float64(.002))
    update_report_checkpoint(report, kind, value, kind)
    evidence = execution_evidence(report)
    assert evidence['origin_verification']['verified'] is True
    assert evidence['failure_stage'] == original_stage


@pytest.mark.parametrize('origin', [None, {'verified': False, 'hardware_ok': 1}, {'verified': True, 'hardware_ok': 0}])
def test_missing_or_failed_origin_verification_is_never_promoted_by_return_flag(origin):
    report = failure_report();report['returned_to_origin'] = True
    report['origin_verification'] = origin
    evidence = execution_evidence(report)
    assert '机械臂已归位' not in arm_failure_message(evidence)
    if origin is not None:
        assert evidence['returned_to_origin'] is False


def test_guarded_stage_started_is_reported_without_claiming_light_success():
    report = failure_report()
    value = dict(segment='guarded_press', task_id='press')
    append_journal_checkpoint(report, 'motion_segment_started', value)
    update_report_checkpoint(report, 'motion_segment_started', value, 'guarded_press')
    evidence = execution_evidence(report)
    assert evidence['press_execution_started'] and evidence['press_attempts'] == 1
    assert evidence['flow_status'] is None


def test_healthy_origin_does_not_turn_failed_navigation_arm_task_into_success(tmp_path):
    report = failure_report();events = []
    finalize_report(report, tmp_path / 'report.yaml', events.append)
    result = arm_result(events, 1)
    assert result['origin_verified'] and result['returned_to_origin']
    assert not result['on_confirmed'] and not result['off_confirmed']
    mission = ExternalCallMission();mid = mission.begin(8, 2, 0);mission.phase = 'arm_call'
    assert mission.arm_finished(mid, result) == 'failed'
    assert mission.reason == EXPECTED and mission.phase == 'failed'


def test_navigation_gui_displays_failure_facts_and_does_not_queue_next_goal(tmp_path):
    from unittest.mock import patch
    from tests.test_navigation_link_gui import gui, InlineThread
    report = failure_report();events = []
    finalize_report(report, tmp_path / 'report.yaml', events.append)
    g = gui();g.mission.begin(8, 2, 0);g.mission.phase = 'arm_call'
    with patch('tools.navigation_task_gui.threading.Thread', InlineThread):
        g.handle_arm_worker(dict(event='completed', operation='press', request_id='test-press',
                                mission_id=g.mission.mission_id, result=arm_result(events, 1)))
    assert g.mission.phase == 'failed' and g.mission.reason == EXPECTED
    assert EXPECTED in g.status.set.call_args.args[0]
    assert not any(command == 1 for command, _, _ in g.sent)


def test_voice_and_navigation_read_the_same_terminal_failure_without_generic_overwrite(tmp_path):
    report = failure_report();events = []
    finalize_report(report, tmp_path / 'report.yaml', events.append)
    terminal = events[-1]
    assert describe_phase_zh(terminal) == EXPECTED
    event = dict(event='motion_completed', task_id=SCENE['task_id'], return_code=1,
                 mission_status_events=events)
    assert completion_message(event) == EXPECTED
    presenter = AssistantEventPresenter();presenter.consume({'event': 'motion_started'})
    messages = presenter.consume(dict(event='mission_progress', mission_status_events=events))
    assert messages[-1].text == EXPECTED
    assert presenter.consume(event) == []  # Do not append a less informative duplicate.


def test_dispatcher_retains_terminal_evidence_consumed_in_earlier_poll(tmp_path):
    dispatcher = CommandDispatcher(execute=False, config_path=tmp_path / 'config.yaml', hold_report=tmp_path / 'hold.yaml')
    dispatcher._status_outbox = tmp_path / 'status.jsonl'
    dispatcher._active_task_id = SCENE['task_id'];dispatcher.mission_id = 'test-mission'
    dispatcher.active = MagicMock();dispatcher.active.poll.side_effect = [None, 1]
    event = terminal_event(failure_report());append_status(dispatcher._status_outbox, event)
    progress = dispatcher.refresh()
    assert progress['event'] == 'mission_progress'
    done = dispatcher.refresh()
    assert done['mission_status_events'] == []
    assert completion_message(done) == EXPECTED
    assert done['arm_report']['origin_verification']['verified'] is True


def test_dispatcher_does_not_reuse_last_task_report_for_a_different_task(tmp_path):
    dispatcher = CommandDispatcher(execute=False, config_path=tmp_path / 'config.yaml', hold_report=tmp_path / 'hold.yaml')
    dispatcher._status_outbox = tmp_path / 'status.jsonl'
    dispatcher._active_task_id = 'new-task';dispatcher.active = MagicMock();dispatcher.active.poll.return_value = 1
    dispatcher._arm_terminal_record = terminal_event(failure_report())
    done = dispatcher.refresh()
    assert 'arm_report' not in done and completion_message(done) == '任务失败'


def test_unknown_other_task_terminal_record_does_not_change_voice_failure_text():
    event = dict(event='motion_completed', task_id='new-task', return_code=1,
                 mission_status_events=[terminal_event(failure_report())])
    assert completion_message(event) == '任务失败'
