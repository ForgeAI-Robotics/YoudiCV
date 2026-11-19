"""No-motion recovery scenarios, including the actual 10:13 missing DOWN cue."""
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time
from unittest.mock import patch

import numpy as np
import pytest

from piper_handeye.button_light_flow import ButtonLightFlow, LightState
from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.initial_light_guard import InitialLightGuard
from piper_handeye.initial_observation_recovery import (
    InitialObservationRecovery, InitialRecoveryPolicy,
    choose_initial_call_search_joints, recheck_initial_call_view,
)
from piper_handeye.mission_status import normalize_flow_event, describe_phase_zh
from test_button_light_flow import FakeHardware


FIXTURE = Path(__file__).parent/'fixtures/initial_visibility_recovery'
RECORDED = json.loads((FIXTURE/'manifest.json').read_text())


class Scene:
    def __init__(self, state):
        self.t = 0.
        self.state = state
        self.samples = 0
        self.looks = []
        self.events = []
        self.sleeps = []
        self.stopped = False

    def observe(self):
        self.t += .25
        self.samples += 1
        return self.state(self)

    def check(self):
        if self.stopped:
            raise RuntimeError('operator stopped')

    def look(self, attempt):
        self.looks.append((attempt, self.t))
        return True

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds

    def record(self, event, **values):
        self.events.append(dict(event=event, time=self.t, **values))

    def recovery(self, **kwargs):
        return InitialObservationRecovery(
            self.observe, self.check, self.look, lambda: 'missing_target_box',
            clock=lambda: self.t, sleep=self.sleep, **kwargs)


def test_good_initial_state_keeps_fast_path_and_three_fresh_votes():
    s = Scene(lambda _: 'OFF')
    assert s.recovery().run(s.record) is LightState.OFF
    assert s.samples == 3 and s.t == .75 and not s.looks and not s.sleeps


def test_three_missing_frames_no_longer_terminate_or_force_a_wrist_move():
    s = Scene(lambda s: 'UNCERTAIN' if s.samples <= 7 else 'OFF')
    assert s.recovery().run(s.record) is LightState.OFF
    assert not s.looks
    assert any(e['event'] == 'initial_observation_reacquiring' for e in s.events)
    assert not any(e['event'] == 'initial_observation_waiting' for e in s.events)


def test_two_six_second_views_then_one_hz_wait_and_recovery():
    s = Scene(lambda s: 'UNCERTAIN' if s.t < 35 else 'OFF')
    assert s.recovery().run(s.record) is LightState.OFF
    assert [a for a, _ in s.looks] == [1, 2]
    assert s.looks[0][1] == 6 and s.looks[1][1] == 12
    waits = [e for e in s.events if e['event'] == 'initial_observation_waiting']
    assert waits[0]['time'] == 18 and len(waits) >= 2
    assert s.sleeps and set(s.sleeps) == {1.0}
    assert any(e['event'] == 'initial_observation_recovered' for e in s.events)


def test_unavailable_search_reference_consumes_attempts_without_infinite_motion():
    s = Scene(lambda s: 'UNCERTAIN' if s.t < 25 else 'ON')
    def unavailable(attempt):
        s.looks.append((attempt, s.t)); return False
    recovery = s.recovery(); recovery.recheck_view = unavailable
    assert recovery.run(s.record) is LightState.ON
    assert len(s.looks) == 2


def test_uncertain_and_opposite_states_reset_confirmation():
    states = deque(['ON', 'ON', 'UNCERTAIN', 'ON', 'ON', 'OFF', 'OFF', 'OFF'])
    s = Scene(lambda _: states.popleft())
    assert s.recovery().run(s.record) is LightState.OFF and s.samples == 8


def test_confirmation_cannot_cross_a_view_change():
    # At six seconds a two-frame ON streak must be discarded before moving.
    s = Scene(lambda s: 'UNCERTAIN' if s.samples < 23 else 'ON')
    assert s.recovery().run(s.record) is LightState.ON
    assert len(s.looks) == 1 and s.samples == 27


def test_post_view_depth_reacquisition_requires_fresh_lamp_confirmation_again():
    s = Scene(lambda _: 'OFF')
    confirmations = iter(['resample', True])
    recovery = s.recovery(confirm_target=lambda: next(confirmations))
    assert recovery.run(s.record) is LightState.OFF and s.samples == 6


def test_missing_post_view_depth_cannot_authorize_success_or_unbounded_search():
    s = Scene(lambda _: 'ON')
    def stop_after_wait(seconds):
        s.sleep(seconds)
        s.stopped = True
    recovery = s.recovery(confirm_target=lambda: False)
    # A valid lamp alone without geometry must eventually enter bounded hold.
    # Let this test inject a stop from the health callback after 25 seconds.
    original_check = recovery.check_hold
    def check():
        if s.t > 25: s.stopped = True
        original_check()
    recovery.check_hold = check
    with pytest.raises(RuntimeError, match='operator stopped'):
        recovery.run(s.record)
    assert len(s.looks) == 2
    assert any(e['event'] == 'initial_observation_waiting' for e in s.events)
    assert not any(e['event'] == 'initial_state_confirmed' for e in s.events)


@pytest.mark.parametrize('fault', ['stop', 'communication', 'collision'])
def test_failures_are_not_swallowed_as_visual_uncertainty(fault):
    s = Scene(lambda _: 'UNCERTAIN')
    recovery = s.recovery()
    def fail(*args): raise RuntimeError(fault)
    if fault == 'stop': recovery.check_hold = fail
    elif fault == 'communication': recovery.observe = fail
    else: recovery.recheck_view = fail
    with pytest.raises(RuntimeError, match=fault):
        recovery.run(s.record)


def test_operator_can_abort_indefinite_hold_and_flow_finishes_initial_mode():
    hardware = FakeHardware([], raw_frames=True)
    s = Scene(lambda _: 'UNCERTAIN')
    def stop_on_sleep(seconds):
        s.sleep(seconds); s.stopped = True
    recovery = s.recovery(); recovery.sleep = stop_on_sleep
    hardware.initial_observation = recovery.run
    finished = []
    hardware.finish_initial_observation = lambda: finished.append(True)
    with pytest.raises(RuntimeError, match='operator stopped'):
        ButtonLightFlow(hardware).run()
    assert finished == [True] and len(s.looks) == 2
    assert not any(a[0] in ('press', 'refresh') for a in hardware.actions)


def test_recovered_off_uses_existing_press_retry_and_arrival_flow():
    hardware = FakeHardware(['OFF'], ['OFF', 'OFF', 'ON', 'ON'])
    s = Scene(lambda s: 'UNCERTAIN' if s.t < 30 else 'OFF')
    hardware.initial_observation = s.recovery().run
    result = ButtonLightFlow(hardware).run()
    assert result.status == 'arrived_after_verified_press' and result.press_attempts == 2
    assert hardware.actions[-1] == ('origin',)


def test_recovered_initial_on_still_waits_for_three_off_and_does_not_press():
    hardware = FakeHardware(['ON', 'UNCERTAIN', 'OFF'])
    s = Scene(lambda s: 'UNCERTAIN' if s.t < 25 else 'ON')
    hardware.initial_observation = s.recovery().run
    result = ButtonLightFlow(hardware).run()
    assert result.status == 'arrived_after_initial_on' and result.press_attempts == 0
    assert not any(a[0] == 'press' for a in hardware.actions)


def response():
    return dict(frame_sequence=101, capture_time_unix_sec=time.time(),
                image_size=RECORDED['image_size'], detections=copy.deepcopy(RECORDED['detections']))


def view_fixture(response_value=None, failure=None):
    r = response() if response_value is None else response_value
    actions = []
    current = np.array([0., .1, -.1, 0., 0., 0.])
    plane = np.eye(4); plane[:3, 3] = [0., -.16, .5]
    def request(command, **options):
        actions.append((command, options)); return r
    def collect(*args, **kwargs):
        actions.append(('collect', args[3], kwargs))
        if failure is not None: raise failure
        return NS(base_to_button=plane), [dict(bbox_xyxy=[190, 0, 225, 32])]*4
    def execute(goal, origin):
        actions.append(('execute', goal.copy()))
        return dict(joints_rad=goal.tolist())
    def commit(execution): actions.append(('commit', execution))
    cfg = dict(detector={'target_confidence': .7}, depth={'capture_size': [640, 360]},
               calibration={'rgb_calibration_size': [640, 360]})
    kwargs = dict(vision=NS(request=request), robot=NS(state=lambda **kw:NS(joints=current.copy())),
        handeye=np.eye(4), config=cfg, matrix=np.array([[500,0,320],[0,500,180],[0,0,1]]),
        executor=NS(recheck_observation_view=execute), safety=NS(set_target=lambda p:None),
        origin=current.copy(), original_watch=current.copy(), target_label='down',
        reason='missing_target_box', attempt=1, collect_target=collect,
        wait_clear=lambda *a, **kw: {'ready': True}, commit_view=commit,
        check_hold=lambda: actions.append(('hold_check',)),
        checkpoint=lambda *a:actions.append(('checkpoint', *a)))
    return kwargs, actions


def fake_fk(q):
    c, s = np.cos(q[4]), np.sin(q[4])
    t = np.eye(4); t[:3, :3] = [[1,0,0],[0,c,-s],[0,s,c]]
    return t


def test_original_failed_frame_can_seed_search_only_from_up_not_open():
    for name, expected in RECORDED['sha256'].items():
        assert hashlib.sha256((FIXTURE/name).read_bytes()).hexdigest() == expected
    kwargs, actions = view_fixture()
    with patch('piper_handeye.initial_observation_recovery.forward_kinematics', fake_fk):
        assert recheck_initial_call_view(**kwargs)
    captured = next(a for a in actions if a[0] == 'collect')
    assert captured[1] == 'up' and 'frame_count' not in captured[2]  # default four
    restore = next(i for i,a in enumerate(actions) if a[0] == 'set_perception_target')
    motion = next(i for i,a in enumerate(actions) if a[0] == 'execute')
    commit = next(i for i,a in enumerate(actions) if a[0] == 'commit')
    assert restore < motion < commit
    assert actions[restore][1]['target_label'] == 'down'


@pytest.mark.parametrize('kind', ['open_only', 'weak', 'ambiguous', 'stale', 'nan'])
def test_bad_or_wrong_class_anchor_never_produces_motion(kind):
    r = response()
    if kind == 'open_only': r['detections'][0]['label'] = 'open'
    elif kind == 'weak': r['detections'][0]['confidence'] = .69
    elif kind == 'ambiguous': r['detections'] *= 2
    elif kind == 'stale': r['capture_time_unix_sec'] -= 1
    else: r['detections'][0]['confidence'] = float('nan')
    kwargs, actions = view_fixture(r)
    assert not recheck_initial_call_view(**kwargs)
    assert not any(a[0] in ('collect', 'execute', 'commit') for a in actions)


def test_missing_depth_waits_without_motion_and_restores_requested_label():
    failure = RuntimeError('no stable target')
    failure.stationary_target_failure = {'samples': []}
    kwargs, actions = view_fixture(failure=failure)
    assert not recheck_initial_call_view(**kwargs)
    assert any(a[0] == 'set_perception_target' and a[1]['target_label'] == 'down' for a in actions)
    assert not any(a[0] in ('execute', 'commit') for a in actions)


def test_reference_communication_failure_is_not_relabelled_as_missing_depth():
    kwargs, actions = view_fixture(failure=RuntimeError('UDP timeout'))
    with pytest.raises(RuntimeError, match='UDP timeout'):
        recheck_initial_call_view(**kwargs)
    assert any(a[0] == 'set_perception_target' for a in actions)
    assert not any(a[0] == 'execute' for a in actions)


def test_collision_failure_propagates_without_committing_a_new_watch_pose():
    kwargs, actions = view_fixture()
    def blocked(*a): raise RuntimeError('MoveIt collision')
    kwargs['executor'].recheck_observation_view = blocked
    with patch('piper_handeye.initial_observation_recovery.forward_kinematics', fake_fk):
        with pytest.raises(RuntimeError, match='MoveIt collision'):
            recheck_initial_call_view(**kwargs)
    assert not any(a[0] == 'commit' for a in actions)


def test_independent_stop_during_reference_capture_prevents_new_motion():
    kwargs, actions = view_fixture()
    checks = 0
    def check():
        nonlocal checks
        checks += 1
        if checks > 1: raise RuntimeError('stop latched during sampling')
    kwargs['check_hold'] = check
    with patch('piper_handeye.initial_observation_recovery.forward_kinematics', fake_fk):
        with pytest.raises(RuntimeError, match='stop latched'):
            recheck_initial_call_view(**kwargs)
    assert not any(a[0] in ('execute', 'commit') for a in actions)


def test_navigation_gui_wait_progress_keeps_arm_ownership_and_does_not_dispatch_goals():
    from tests.test_navigation_link_gui import gui
    g = gui()
    g.mission.mission_id = 'waiting-mission'
    g.mission.phase = 'arm_call'
    g._arm_ownership = True
    event = normalize_flow_event({'event':'initial_observation_waiting', 'reason':'missing_target_box'})
    message = describe_phase_zh(event)
    g.handle_arm_worker(dict(operation='press', event='progress', mission_id='waiting-mission',
        result=dict(message=message, mission_status_events=[event])))
    assert g.mission.phase == 'arm_call' and g._arm_ownership
    assert not g.sent and not g._arm_client.requests
    g.status.set.assert_called_with(message)


def test_search_geometry_keeps_two_small_monotonic_steps_and_rejects_third():
    q = np.array([0., .1, -.1, 0., 0., 0.]); original = q.copy()
    plane = np.eye(4); plane[:3, 3] = [0., -.16, .5]
    matrix = np.array([[500,0,320],[0,500,180],[0,0,1]])
    with patch('piper_handeye.initial_observation_recovery.forward_kinematics', fake_fk):
        for step in range(2):
            goal, audit = choose_initial_call_search_joints(q, original, np.eye(4), plane,
                matrix, [640,360], [640,360], target_label='down', anchor_label='up')
            assert goal is not None and abs(goal[4]-q[4]) <= .025001
            assert np.array_equal(np.delete(goal,4), np.delete(q,4))
            assert abs(audit['total_j5_offset_rad']) <= .050001
            q = goal
        goal, _ = choose_initial_call_search_joints(q, original, np.eye(4), plane,
            matrix, [640,360], [640,360], target_label='down', anchor_label='up')
        assert goal is None


@pytest.mark.parametrize('fault', ['near', 'other_joint', 'bad_normal', 'invalid_label'])
def test_search_geometry_rejects_unsafe_and_unrelated_motion(fault):
    q = np.array([0., .1, -.1, 0., 0., 0.]); original = q.copy()
    plane = np.eye(4); plane[:3, 3] = [0., -.16, .5]
    label = 'down'
    if fault == 'near': plane[2,3] = .10
    if fault == 'other_joint': q[1] += .01
    if fault == 'bad_normal': plane[:3,2] = 0
    if fault == 'invalid_label': label = 'open'
    with patch('piper_handeye.initial_observation_recovery.forward_kinematics', fake_fk):
        try:
            goal, _ = choose_initial_call_search_joints(q, original, np.eye(4), plane,
                np.array([[500,0,320],[0,500,180],[0,0,1]]), [640,360],[640,360],
                target_label=label, anchor_label='up')
        except (ValueError, RuntimeError): goal = None
        assert goal is None


def test_extended_wait_keeps_image_evidence_bounded_without_failing_at_frame_37(tmp_path):
    source = tmp_path/'camera.jpg'; source.write_bytes(b'original frame')
    guard = InitialLightGuard('down', tmp_path/'evidence')
    guard.extended_observation = True
    for sequence in range(250):
        guard.save({'path': str(source), 'frame_sequence': sequence}, {'state': 'OFF'})
    assert guard.initial_frames == 250
    assert len(list((tmp_path/'evidence').glob('*.jpg'))) == 72
    assert len(list((tmp_path/'evidence').glob('*.json'))) == 72
    guard.finish()
    for sequence in range(10):
        guard.save({'path': str(source), 'frame_sequence': sequence}, {'state': 'OFF'})
    assert len(list((tmp_path/'evidence').glob('*.jpg'))) == 75


def test_extended_observer_audit_preserves_start_and_recent_tail():
    observer = FreshLightObserver(None, None, 'down', initial_recovery_active=True)
    for i in range(1000): observer._append_observation_audit({'frame_sequence': i})
    assert len(observer.audit) == 192
    assert [r['frame_sequence'] for r in observer.audit[:36]] == list(range(36))
    assert observer.audit[-1]['frame_sequence'] == 999


def test_waiting_status_is_degraded_and_never_reports_completion_or_floor_arrival():
    event = normalize_flow_event({'event':'initial_observation_waiting', 'reason':'missing_target_box'})
    assert event['phase'] == 'initial_observation_waiting' and event['outcome'] == 'degraded'
    assert '保持静止' in describe_phase_zh(event)


@pytest.mark.parametrize('reason,word', [
    ('missing_target_box', '未识别到按钮'),
    ('light_state_unconfirmed', '灯态尚未确认'),
    ('target_localization_pending', '定位尚未稳定'),
])
def test_waiting_status_distinguishes_detection_light_and_depth(reason, word):
    event = normalize_flow_event({'event':'initial_observation_waiting', 'reason':reason})
    assert word in describe_phase_zh(event)


def test_real_observer_missing_boxes_recover_without_calling_lamp_worker_on_open(tmp_path):
    clock = Scene(lambda _: None)
    image = FIXTURE/'failed_847831.jpg'
    calls = []
    class Vision:
        sequence = 0
        def request(self, command, **options):
            assert command == 'light_snapshot'
            self.sequence += 1; clock.t += .06
            return dict(fresh=True, frame_sequence=self.sequence, path=str(image),
                image_size=[640,360], capture_time_unix_sec=1000+clock.t,
                target_detection=(dict(label='down', confidence=.95,
                    frame_sequence=self.sequence, bbox_xyxy=[194,38,227,73])
                    if clock.t > 4 else None))
    def recognize(*args, **kwargs):
        calls.append(kwargs['bbox_xyxy']); return {'state': 'OFF', 'legacy_state': 'OFF'}
    observer = FreshLightObserver(Vision(), NS(observe=recognize), 'down',
                                  initial_recovery_active=True)
    recovery = InitialObservationRecovery(
        lambda: observer.observe('waiting', emit_event=False), clock.check, clock.look,
        observer.initial_observation_reason, clock=lambda:clock.t, sleep=clock.sleep)
    with patch('piper_handeye.button_light_hardware.time.monotonic', lambda:clock.t), \
         patch('piper_handeye.button_light_hardware.time.time', lambda:1000+clock.t), \
         patch('piper_handeye.button_light_hardware.time.sleep', lambda t:setattr(clock,'t',clock.t+t)), \
         patch('piper_handeye.button_light_hardware.save_missing_button_view', return_value={'image':'saved.jpg'}):
        assert recovery.run(clock.record) is LightState.OFF
    assert len(calls) == 3 and not clock.looks
    assert any(a['state'] == 'UNCERTAIN' for a in observer.audit)
    assert all(a['recognition']['recognizer'] == 'skipped_unsafe_target_crop'
               for a in observer.audit if a['state'] == 'UNCERTAIN')
