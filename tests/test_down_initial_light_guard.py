from pathlib import Path
from types import SimpleNamespace
import json
import time
from unittest.mock import patch

import numpy as np
import pytest

from piper_handeye.initial_light_guard import InitialLightGuard
from piper_handeye.initial_light_recheck import choose_initial_recheck_joints
from piper_handeye.button_light_flow import ButtonLightFlow
from piper_handeye.button_light_hardware import FreshLightObserver, DryValidatedSegmentExecutor
from test_button_light_flow import FakeHardware


@pytest.mark.parametrize('legacy', ['OFF', 'UNCERTAIN', None])
def test_local_only_on_cannot_authorize_skip_or_press(tmp_path, legacy):
    guard = InitialLightGuard('down', tmp_path)
    raw = dict(state='ON', legacy_state=legacy, local_regions=[dict(area_px=13, contrast=80)])
    result = guard.filter(raw)
    assert result['state'] == 'UNCERTAIN' and guard.conflict_seen
    assert raw['state'] == 'ON'


@pytest.mark.parametrize('target', ['up', '1', '8', '16'])
def test_other_targets_keep_their_existing_classifier(tmp_path, target):
    guard = InitialLightGuard(target, tmp_path)
    raw = dict(state='ON', legacy_state='OFF')
    assert guard.filter(raw) is raw
    assert guard.save({}, raw) is None


@pytest.mark.parametrize('state', ['ON', 'OFF', 'UNCERTAIN'])
def test_agreement_is_unchanged_and_post_initial_policy_is_unchanged(tmp_path, state):
    guard = InitialLightGuard('down', tmp_path)
    raw = dict(state=state, legacy_state=state)
    assert guard.filter(raw) is raw
    guard.finish()
    raw = dict(state='ON', legacy_state='OFF')
    assert guard.filter(raw) is raw


class RecheckScene(FakeHardware):
    def __init__(self, frames, near=(), callbacks=(True, True)):
        super().__init__(frames, near, raw_frames=True)
        self.callbacks = iter(callbacks)
        self.finished = False
    def recheck_initial_observation(self):
        self.actions.append(('recheck',))
        value = next(self.callbacks)
        if isinstance(value, Exception):
            raise value
        return value
    def finish_initial_observation(self):
        self.finished = True


def test_persistent_glare_exhausts_two_rechecks_and_never_completes_or_presses():
    scene = RecheckScene(['UNCERTAIN'] * 9)
    result = ButtonLightFlow(scene).run()
    assert result.status == 'initial_observation_uncertain'
    assert result.press_attempts == 0 and scene.finished
    assert scene.actions.count(('recheck',)) == 2
    assert scene.actions[-1] == ('origin',)
    assert not any(a[0] in ('press', 'refresh') for a in scene.actions)


def test_rechecked_true_on_waits_for_off_without_press():
    scene = RecheckScene(['UNCERTAIN'] * 3 + ['ON'] * 3 + ['UNCERTAIN'] * 6 + ['ON', 'OFF', 'OFF', 'OFF'])
    result = ButtonLightFlow(scene).run()
    assert result.status == 'arrived_after_initial_on' and result.press_attempts == 0
    assert scene.actions.count(('recheck',)) == 1 and scene.finished


def test_rechecked_off_enters_regular_press_and_wait_for_off():
    scene = RecheckScene(['UNCERTAIN'] * 3 + ['OFF'] * 3 + ['ON'] * 3 + ['OFF'] * 3,
                         near=['ON', 'ON'])
    result = ButtonLightFlow(scene).run()
    assert result.status == 'arrived_after_verified_press' and result.press_attempts == 1
    assert scene.actions.count(('recheck',)) == 1


def test_confirmation_streak_cannot_cross_view_change():
    frames = ['ON', 'ON', 'UNCERTAIN'] * 4 + ['ON', 'OFF', 'OFF', 'OFF'] + ['OFF'] * 3
    scene = RecheckScene(frames, near=['ON', 'ON'])
    result = ButtonLightFlow(scene).run()
    assert result.press_attempts == 1


@pytest.mark.parametrize('callback', [False, RuntimeError('independent stop latched')])
def test_unavailable_view_and_safety_error_never_authorize_contact(callback):
    scene = RecheckScene(['UNCERTAIN'] * 3, callbacks=[callback])
    if isinstance(callback, Exception):
        with pytest.raises(RuntimeError, match='stop latched'):
            ButtonLightFlow(scene).run()
    else:
        assert ButtonLightFlow(scene).run().status == 'initial_observation_uncertain'
    assert scene.finished and not any(a[0] == 'press' for a in scene.actions)


def test_saved_initial_and_terminal_frames_survive_cache_deletion_and_are_bounded(tmp_path):
    image = tmp_path / 'cache.jpg'
    image.write_bytes(b'image bytes')
    guard = InitialLightGuard('down', tmp_path / 'evidence')
    for sequence in range(36):
        guard.save(dict(path=str(image), frame_sequence=sequence), dict(state='UNCERTAIN'))
    with pytest.raises(RuntimeError, match='budget'):
        guard.save(dict(path=str(image), frame_sequence=37), dict(state='UNCERTAIN'))
    guard.finish()
    for sequence in range(100):
        guard.save(dict(path=str(image), frame_sequence=sequence), dict(state='OFF'))
    image.unlink()
    assert len(list((tmp_path / 'evidence').glob('*.jpg'))) == 39
    assert sorted(json.loads(p.read_text())['snapshot']['frame_sequence']
                  for p in (tmp_path / 'evidence').glob('waiting*.json')) == [97, 98, 99]


def test_observer_filters_before_logging_and_finishes_before_post_press(tmp_path):
    path = tmp_path / 'frame.jpg'; path.write_bytes(b'frame')
    class Vision:
        seq = 0
        def request(self, *args, **kwargs):
            self.seq += 1
            return dict(fresh=True, frame_sequence=self.seq, path=str(path),
                image_size=[640, 360], capture_time_unix_sec=time.time(),
                target_detection=dict(frame_sequence=self.seq, bbox_xyxy=[100, 50, 140, 90], confidence=.95))
    recognizer = SimpleNamespace(observe=lambda *a, **kw: dict(state='ON', legacy_state='OFF'))
    guard = InitialLightGuard('down', tmp_path / 'evidence')
    observer = FreshLightObserver(Vision(), recognizer, 'down', initial_light_guard=guard)
    assert observer.observe('waiting') == 'UNCERTAIN'
    assert Path(observer.audit[-1]['image']).exists()
    assert observer.observe('near') == 'ON'
    assert observer.observe('before_press') == 'ON'
    observer.finish_initial_observation()
    assert observer.observe('waiting') == 'ON'


def test_recheck_does_not_reuse_premovement_images(tmp_path):
    guard = InitialLightGuard('down', tmp_path)
    guard.conflict_seen = True
    observer = FreshLightObserver(None, None, 'down', initial_light_guard=guard,
                                  initial_view_recheck=lambda attempt: True)
    assert observer.recheck_initial_observation()
    assert guard.rechecks == 1 and guard.not_before_unix is not None and not guard.conflict_seen
    assert not observer.recheck_initial_observation()
    guard.conflict_seen = True
    assert observer.recheck_initial_observation()
    guard.conflict_seen = True
    assert not observer.recheck_initial_observation()


def fake_fk(q):
    # Camera rotates about y, with a stationary TCP. Real FK is tested against
    # the saved robot pose separately; this isolates path/FOV boundary logic.
    c, s = np.cos(q[4]), np.sin(q[4])
    t = np.eye(4); t[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return t


def geometry(current=None, original=None, distance=.5, bbox=None):
    q = np.array([0, .1, -.1, 0, 0, 0.]) if current is None else current
    origin = q.copy() if original is None else original
    panel = np.eye(4); panel[2, 3] = distance
    matrix = np.array([[500., 0, 320], [0, 500, 180], [0, 0, 1]])
    return choose_initial_recheck_joints(q, origin, np.eye(4), panel, matrix,
        [640, 360], [640, 360], [280, 160, 320, 200] if bbox is None else bbox)


def test_two_small_views_follow_limits_and_third_cannot_extend_budget():
    with patch('piper_handeye.initial_light_recheck.forward_kinematics', fake_fk):
        origin = np.array([0, .1, -.1, 0, 0, 0.])
        first, audit = geometry(origin, origin)
        assert first is not None and abs(audit['delta_j5_rad']) == .025
        second, audit = geometry(first, origin)
        assert second is not None and abs(audit['total_j5_offset_rad']) == .05
        third, audit = geometry(second, origin)
        assert third is None


@pytest.mark.parametrize('kwargs', [dict(distance=.10), dict(bbox=[0, 0, 40, 40]),
                                    dict(bbox=[1, 2, float('nan'), 40])])
def test_unsafe_geometry_cannot_create_observation_motion(kwargs):
    with patch('piper_handeye.initial_light_recheck.forward_kinematics', fake_fk):
        try:
            candidate, _ = geometry(**kwargs)
        except ValueError:
            candidate = None
        assert candidate is None


def executor_fixture(failure=None):
    state = np.array([0, .1, -.1, 0, 0, 0.])
    robot = SimpleNamespace(state=lambda **kw: SimpleNamespace(joints=state.copy(), raw={'hardware_ok': 1}))
    calls = []
    class Bridge:
        def execute(self, points, **options):
            calls.append(options)
            if failure == len(calls):
                raise RuntimeError('collision validation rejected')
            if not options.get('dry_run'):
                state[:] = points[-1]['positions']
            return {'status': 'validated' if options.get('dry_run') else 'complete'}
    safety = SimpleNamespace(before_segment=lambda n: calls.append({'armed': n}),
                             after_segment=lambda n: None)
    executor = DryValidatedSegmentExecutor(Bridge(), robot, 'test', execute_enabled=True, motion_safety=safety)
    target = state.copy(); target[4] += .025
    return executor, target, state.copy(), calls


@pytest.mark.parametrize('failure', [1, 2])
def test_both_outbound_and_origin_collision_checks_required(failure):
    executor, target, origin, calls = executor_fixture(failure)
    with pytest.raises(RuntimeError, match='collision'):
        executor.recheck_observation_view(target, origin)
    assert all(c.get('dry_run') for c in calls)


def test_observation_execution_keeps_safety_and_confirms_encoders():
    executor, target, origin, calls = executor_fixture()
    result = executor.recheck_observation_view(target, origin)
    assert calls[0]['dry_run'] and calls[1]['dry_run']
    assert 'armed' in calls[2] and not calls[3].get('dry_run')
    assert np.allclose(result['joints_rad'], target)


def test_encoder_quantization_does_not_command_other_joints_or_increase_step():
    executor, target, origin, calls = executor_fixture()
    target[1] += .001
    target[4] += .001
    result = executor.recheck_observation_view(target, origin)
    assert result['commanded_joints_rad'][1] == origin[1]
    assert abs(result['commanded_joints_rad'][4] - origin[4]) == .025


def test_stationary_motor_cannot_be_logged_as_completed_recheck():
    executor, target, origin, calls = executor_fixture()
    original_execute = executor.bridge.execute
    def no_motion(points, **options):
        if options.get('dry_run'):
            return original_execute(points, **options)
        return {'status': 'complete'}
    executor.bridge.execute = no_motion
    events = []
    executor.event_sink = lambda kind, value: events.append(kind)
    with pytest.raises(RuntimeError, match='encoder target error'):
        executor.recheck_observation_view(target, origin)
    assert 'motion_segment_completed' not in events
    assert 'motion_segment_failed' in events


def test_state_drift_during_dry_validation_cancels_before_execution():
    executor, target, origin, calls = executor_fixture()
    original_state = executor.robot.state
    reads = 0
    def drifting(**kwargs):
        nonlocal reads
        reads += 1
        state = original_state(**kwargs)
        if reads > 1:
            state.joints[1] += .004
        return state
    executor.robot.state = drifting
    with pytest.raises(RuntimeError, match='moved during'):
        executor.recheck_observation_view(target, origin)
    assert all(c.get('dry_run') for c in calls)


def test_safety_stop_at_arming_prevents_bridge_execution():
    executor, target, origin, calls = executor_fixture()
    def stop(name):
        raise RuntimeError('visual safety stop is active')
    executor.motion_safety.before_segment = stop
    with pytest.raises(RuntimeError, match='safety stop'):
        executor.recheck_observation_view(target, origin)
    assert all(c.get('dry_run') for c in calls)


def test_recheck_filters_old_frames_until_after_settle_time(tmp_path):
    image = tmp_path / 'frame.jpg'; image.write_bytes(b'image')
    guard = InitialLightGuard('down', tmp_path / 'evidence')
    guard.not_before_unix = time.time()
    class Vision:
        seq = 0
        def request(self, *a, **kw):
            self.seq += 1
            return dict(fresh=True, frame_sequence=self.seq, path=str(image), image_size=[640,360],
                capture_time_unix_sec=guard.not_before_unix - .1 if self.seq == 1 else time.time(),
                target_detection=dict(frame_sequence=self.seq, bbox_xyxy=[100,50,140,90], confidence=.95))
    recognizer = SimpleNamespace(observe=lambda *a, **kw: dict(state='ON', legacy_state='ON'))
    observer = FreshLightObserver(Vision(), recognizer, 'down', initial_light_guard=guard)
    assert observer.observe('waiting') == 'ON'
    assert observer.audit[-1]['frame_sequence'] == 2


@pytest.mark.parametrize('kind', ['too_far', 'other_joint', 'safety_disabled', 'not_enabled'])
def test_observation_executor_cannot_be_used_as_generic_motion(kind):
    executor, target, origin, calls = executor_fixture()
    if kind == 'too_far': target[4] += .1
    if kind == 'other_joint': target[0] += .01
    if kind == 'safety_disabled': executor.visual_safety_disabled_for_return = True
    if kind == 'not_enabled': executor.execute_enabled = False
    with pytest.raises(RuntimeError):
        executor.recheck_observation_view(target, origin)
    assert not calls
