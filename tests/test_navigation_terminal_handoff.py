"""Exercise arm completion handoff without subprocesses or robot connections."""
from unittest.mock import MagicMock

import pytest

from piper_handeye.external_call_mission import ExternalCallMission
from piper_handeye.mission_status import append_status
from run_voice_command_gateway import CommandDispatcher
from tools.navigation_arm_worker import Worker


@pytest.mark.parametrize('return_code', [0, 2])
@pytest.mark.parametrize('publish_at_exit', [True, False])
def test_terminal_events_written_during_exit_are_delivered_before_dock_gate(
    tmp_path, monkeypatch, return_code, publish_at_exit,
):
    outbox = tmp_path / 'mission.jsonl'
    monkeypatch.setenv('YOUXIAOMEI_MISSION_STATUS_OUTBOX', str(outbox))
    monkeypatch.setenv('YOUXIAOMEI_ROBOT_MOTION_LOCK', str(tmp_path / 'motion.lock'))
    dispatcher = CommandDispatcher(
        execute=True, config_path=tmp_path / 'robot.yaml',
        hold_report=tmp_path / 'hold.yaml',
    )
    process = MagicMock(pid=123)
    monkeypatch.setattr('run_voice_command_gateway.subprocess.Popen', lambda *a, **k: process)
    emitted = []
    worker = Worker(dispatcher, emitted.append)
    request = dict(operation='press', request_id='press-1', mission_id='mission-1',
                   current_floor=16, target_floor=8, direction='down')
    worker.handle(request)
    assert emitted[-1]['event'] == 'started'
    task_id = dispatcher._active_task_id

    def publish_completion():
        for phase in ('button_on_confirmed', 'floor_arrival_observed'):
            append_status(outbox, dict(task_id=task_id, phase=phase, producer='arm_pressing'))
        append_status(outbox, dict(
            task_id=task_id, producer='arm_pressing', raw_event='task_terminal',
            evidence=dict(flow_status='arrived_after_verified_press', returned_to_origin=True,
                          origin_verification=dict(verified=True, hardware_ok=1)),
        ))
        # A different task's failure must not replace this task's terminal evidence.
        append_status(outbox, dict(task_id='other-task', producer='arm_pressing',
                                  raw_event='task_terminal', evidence={}))

    if publish_at_exit:
        # refresh() has already read the outbox when it checks process.poll().
        def finish():
            publish_completion()
            return return_code
        process.poll.side_effect = finish
    else:
        publish_completion()
        process.poll.return_value = return_code

    worker.refresh()
    result = emitted[-1]['result']
    assert emitted[-1]['event'] == 'completed'
    assert result['on_confirmed'] is True
    assert result['off_confirmed'] is True
    assert result['origin_verified'] is True
    assert result['returned_to_origin'] is True
    assert worker.press is None
    assert dispatcher.active is None
    assert dispatcher._motion_lock_stream is None

    mission = ExternalCallMission()
    mid = mission.begin(16, 8, 0)
    mission.phase = 'arm_call'
    expected = 'check_origin' if return_code == 0 else 'failed'
    assert mission.arm_finished(mid, result) == expected


def test_running_process_does_not_consume_terminal_handoff_or_release_lock(tmp_path, monkeypatch):
    outbox = tmp_path / 'mission.jsonl'
    monkeypatch.setenv('YOUXIAOMEI_MISSION_STATUS_OUTBOX', str(outbox))
    monkeypatch.setenv('YOUXIAOMEI_ROBOT_MOTION_LOCK', str(tmp_path / 'motion.lock'))
    dispatcher = CommandDispatcher(
        execute=True, config_path=tmp_path / 'robot.yaml', hold_report=tmp_path / 'hold.yaml',
    )
    process = MagicMock(pid=123)
    process.poll.return_value = None
    monkeypatch.setattr('run_voice_command_gateway.subprocess.Popen', lambda *a, **k: process)
    emitted = []
    worker = Worker(dispatcher, emitted.append)
    request = dict(operation='press', request_id='press-1', mission_id='mission-1',
                   current_floor=16, target_floor=8, direction='down')
    worker.handle(request)
    try:
        append_status(outbox, dict(task_id=dispatcher._active_task_id,
                                  producer='arm_pressing', phase='button_on_confirmed'))
        worker.refresh()
        assert emitted[-1]['event'] == 'progress'
        assert worker.press == request
        assert dispatcher.active is process
        assert dispatcher._motion_lock_stream is not None
        worker.refresh()
        assert emitted[-1]['event'] == 'progress'
        assert not any(row['event'] == 'completed' for row in emitted)
    finally:
        dispatcher._release_motion_lock()
