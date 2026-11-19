"""Bounded post-press tolerance must not weaken OFF references or visual guards."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from piper_handeye.paired_button_light import compare_patches, extract_patch, _PANEL_RING, _X, _Y
from piper_handeye.mission_status import (
    describe_phase_zh, normalize_flow_event, normalize_light_observation,
    press_result_is_uncertain,
)
from tests.test_panel_background_reference import bezel_change


FIXTURES = Path(__file__).parent / 'fixtures/relaxed_light_residuals'


def real_pair():
    frames = json.loads((FIXTURES / 'manifest.json').read_text())['frames']
    return [extract_patch(dict(
        path=str(FIXTURES / f['file']), frame_sequence=f['sequence'],
        capture_time_unix_sec=f['time'],
        target_detection=dict(frame_sequence=f['sequence'], bbox_xyxy=f['box'])))
        for f in (frames['reference'], frames['current'])]


def test_real_lit_numeric_button_recovers_with_audited_15_35_panel_limits():
    result = compare_patches(*real_pair(), 'ON')
    assert result['state'] == 'ON'
    assert result['background_region'] == 'outer_panel_ring_v1'
    assert result['background_tolerance_profile'] == 'post_press_relaxed_v1'
    assert 10 < result['background_error'] <= 15
    assert 25 < result['background_error_p90'] <= 35
    assert result['background_max_mean_error'] == 15
    assert result['background_max_p90_error'] == 35
    assert result['panel_background_strict_attempt']['normalization_rejection'] == 'nonuniform_background_change'
    assert result['background_inlier_fraction'] >= .75
    assert result['background_spatial_error'] <= 12


@pytest.mark.parametrize('light', [False, True])
def test_mild_panel_residual_needs_independent_lamp_increase(light):
    before, after = bezel_change(light=light)
    variation = ((_X + _Y) % 2 * 2 - 1) * 11
    after.gray[_PANEL_RING] += variation[_PANEL_RING]
    result = compare_patches(before, after, 'ON')
    assert (result['state'] == 'ON') is light
    assert result['background_tolerance_profile'] == 'post_press_relaxed_v1'


def test_relaxation_does_not_collect_off_reference_or_override_absolute_off():
    result = compare_patches(*real_pair(), 'OFF')
    assert result['state'] == 'UNCERTAIN'
    assert result['background_max_mean_error'] == 10
    assert result['background_max_p90_error'] == 25
    assert 'background_tolerance_profile' not in result
    assert 'panel_background_attempt' not in result


@pytest.mark.parametrize('value', [
    dict(event='light_observed', state='UNCERTAIN', position='near'),
    dict(event='light_observed', state='UNCERTAIN', position='waiting'),
    dict(event='waiting_observation_uncertain', position='waiting'),
    dict(event='waiting_button_still_on', observation_uncertain=True),
])
def test_pending_observation_is_not_presented_as_off_or_arrival(value):
    event = normalize_flow_event(value)
    assert event['phase'] == 'observation_reacquiring'
    assert '等待重新观测' in describe_phase_zh(event)
    assert event['evidence'] == {k: v for k, v in value.items() if k != 'event'}
    assert not press_result_is_uncertain(event)  # Not a terminal failure either.


@pytest.mark.parametrize('state', ['ON', 'OFF'])
def test_confirmed_states_and_waiting_heartbeat_keep_their_meaning(state):
    event = normalize_flow_event(dict(event='light_observed', state=state))
    assert event['phase'] == 'observing_button'
    assert event['evidence']['state'] == state
    assert normalize_flow_event(dict(event='waiting_button_still_on'))['phase'] == 'waiting_for_floor'


def test_invalid_image_evidence_remains_available_in_live_status():
    value = dict(state='UNCERTAIN', frame_sequence=3, recognition=dict(
        recognizer='skipped_unsafe_target_crop', visibility={'visible': False}))
    event = normalize_light_observation(value)
    assert event['phase'] == 'observation_reacquiring'
    assert event['evidence'] == value
    assert normalize_light_observation(dict(state='ON'))['phase'] == 'observing_button'


def test_terminal_unconfirmed_press_remains_an_explicit_incomplete_result():
    event = normalize_flow_event(dict(event='post_press_confirmation_uncertain'))
    assert event['phase'] == 'press_result_uncertain'
    assert press_result_is_uncertain(event)
    assert '已执行按压，但亮灯确认不确定' in describe_phase_zh(event)


@pytest.mark.parametrize('operation,phase', [('press', 'arm_call'), ('press_floor', 'arm_floor')])
def test_navigation_gui_displays_pending_observation_without_advancing(operation, phase):
    from tests.test_navigation_link_gui import gui
    g = gui(); g.mission.begin(3, 4, 0); g.mission.phase = phase
    event = normalize_flow_event(dict(event='light_observed', state='UNCERTAIN'))
    g.handle_arm_worker(dict(event='progress', operation=operation, request_id='p',
        mission_id=g.mission.mission_id, result=dict(mission_status_events=[event],
        message=describe_phase_zh(event))))
    assert '等待重新观测' in g.status.set.call_args.args[0]
    assert g.mission.phase == phase
    assert not g._arm_client.requests and not g.sent


def test_voice_gui_displays_pending_observation_without_false_success():
    import queue
    from run_voice_assistant_gui import VoiceAssistantGui
    from piper_handeye.assistant_gui_events import AssistantEventPresenter
    g = VoiceAssistantGui.__new__(VoiceAssistantGui)
    g.events = queue.Queue()
    event = normalize_flow_event(dict(event='light_observed', state='UNCERTAIN'))
    g.events.put(dict(event='mission_progress', message=describe_phase_zh(event),
                     mission_status_events=[event]))
    g.floor_status = Mock(); g.floor_estimate = Mock(); g.status = Mock()
    g.presenter = AssistantEventPresenter(); g.args = NS(assistant_name='优小妹')
    g._refresh_controls = Mock(); g.add_message = Mock(); g.root = Mock()
    g.voice_enabled = False; g.closing = False
    g.poll_events()
    assert '等待重新观测' in g.status.set.call_args.args[0]
    assert g.task_active
    assert not any('已完成' in str(c) for c in g.add_message.call_args_list)
