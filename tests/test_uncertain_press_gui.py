"""An incomplete visual confirmation must remain an incomplete robot task."""
import queue
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock

import pytest

from piper_handeye.assistant_gui_events import AssistantEventPresenter, completion_message
from piper_handeye.external_call_mission import ExternalCallMission
from piper_handeye.mission_status import PRESS_RESULT_UNCERTAIN_MESSAGE as MESSAGE, append_status, press_result_is_uncertain
from piper_handeye.voice_command import VoiceCommand, VoiceIntent
from run_voice_command_gateway import CommandDispatcher


def uncertain_record(task='mine'):
    return dict(task_id=task, producer='arm_pressing', phase='failed', raw_event='task_terminal',
                evidence=dict(flow_status='post_press_confirmation_uncertain', process_exit_code=2,
                              returned_to_origin=True, origin_verification={'verified': True, 'hardware_ok': 1}))


def texts(values):
    return [x.text for x in values]


def test_uncertainty_survives_later_origin_events_and_is_reported_once():
    p = AssistantEventPresenter(); p.consume({'event': 'motion_started'})
    event = dict(event='mission_progress', task_id='mine', mission_status_events=[
        dict(task_id='mine', phase='press_result_uncertain'), dict(task_id='mine', phase='origin_verified')])
    assert texts(p.consume(event)) == [MESSAGE]
    assert texts(p.consume(event)) == []
    assert texts(p.consume(dict(event='motion_completed', return_code=2))) == []
    assert not p.task_active
    assert completion_message({'return_code': 2}, uncertainty_seen=p.press_result_uncertain) == MESSAGE


def test_terminal_only_uncertainty_is_not_reported_as_success_or_generic_failure():
    p = AssistantEventPresenter(); p.consume({'event': 'motion_started'})
    event = dict(event='motion_completed', return_code=2, task_id='mine', mission_status_events=[uncertain_record()])
    assert texts(p.consume(event)) == [MESSAGE]
    assert not p.task_active


def test_other_task_and_next_task_do_not_inherit_uncertainty():
    event = dict(task_id='mine', mission_status_events=[uncertain_record('other')])
    assert not press_result_is_uncertain(event)
    p = AssistantEventPresenter(); p.consume({'event': 'motion_started'})
    p.consume(dict(event='motion_completed', return_code=2, mission_status_events=[uncertain_record()]))
    p.consume({'event': 'motion_started'})
    assert texts(p.consume({'event': 'motion_completed', 'return_code': 2})) == ['任务失败']


def test_later_hard_return_failure_is_still_reported():
    p = AssistantEventPresenter(); p.consume({'event': 'motion_started'})
    p.consume(dict(event='mission_progress', mission_status_events=[dict(phase='press_result_uncertain')]))
    assert texts(p.consume(dict(event='motion_completed', return_code=1))) == ['任务失败']


def test_real_dispatcher_drains_uncertainty_written_during_exit_poll(tmp_path, monkeypatch):
    outbox = tmp_path/'outbox.jsonl'
    monkeypatch.setenv('YOUXIAOMEI_MISSION_STATUS_OUTBOX', str(outbox))
    monkeypatch.setenv('YOUXIAOMEI_ROBOT_MOTION_LOCK', str(tmp_path/'lock'))
    dispatcher = CommandDispatcher(execute=True, config_path=tmp_path/'c.yaml', hold_report=tmp_path/'h.yaml')
    process = MagicMock(pid=123)
    monkeypatch.setattr('run_voice_command_gateway.subprocess.Popen', lambda *a, **kw: process)
    started = dispatcher.dispatch_motion(VoiceCommand(VoiceIntent.PRESS_BUTTON, '4', '4', button=4))
    tid = started['task_id']
    def exit_with_terminal():
        append_status(outbox, uncertain_record(tid))
        append_status(outbox, uncertain_record('other'))
        return 2
    process.poll.side_effect = exit_with_terminal
    event = dispatcher.refresh()
    assert event['return_code'] == 2
    assert {r['task_id'] for r in event['mission_status_events']} == {tid}
    assert completion_message(event) == MESSAGE
    assert dispatcher.active is None and dispatcher._motion_lock_stream is None


@pytest.mark.parametrize('phase', ['arm_call', 'arm_floor'])
def test_navigation_cannot_advance_or_return_to_dock_on_uncertainty(phase):
    m = ExternalCallMission(include_floor_panel=True)
    mid = m.begin(3, 8, 0); m.phase = phase
    result = dict(return_code=2, flow_status='post_press_confirmation_uncertain',
                  returned_to_origin=True, origin_verified=True, on_confirmed=False, off_confirmed=False)
    assert m.arm_finished(mid, result) == 'failed'
    assert m.reason == MESSAGE
    assert m.phase == 'failed' and not m.charged()


def test_real_gui_poll_shows_explicit_message_in_chat_and_final_status():
    from run_voice_assistant_gui import VoiceAssistantGui
    g = VoiceAssistantGui.__new__(VoiceAssistantGui)
    g.events = queue.Queue()
    g.events.put(dict(event='motion_started'))
    g.events.put(dict(event='mission_progress', task_id='mine', message=MESSAGE,
                      mission_status_events=[uncertain_record()]))
    g.events.put(dict(event='motion_completed', task_id='mine', return_code=2,
                      mission_status_events=[uncertain_record()]))
    g.floor_status = Mock(); g.floor_estimate = Mock(); g.status = Mock()
    g.presenter = AssistantEventPresenter(); g.args = NS(assistant_name='优小妹')
    g._refresh_controls = Mock(); g.add_message = Mock(); g.root = Mock()
    g.voice_enabled = False; g.closing = False
    g.poll_events()
    messages = [call.args[1] for call in g.add_message.call_args_list]
    assert MESSAGE in messages and '任务失败' not in messages
    assert not any('已完成' in x for x in messages)
    assert g.status.set.call_args.args[0] == MESSAGE
