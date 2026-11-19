"""Original17:37 failure, stationary-only authorization and adverse views."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import time

import numpy as np
import pytest

from piper_handeye.button_light_hardware import FreshLightObserver, DryValidatedSegmentExecutor
from piper_handeye.paired_button_light import (
    PairedButtonLight, compare_patches, extract_patch, _RING, _X, _Y,
)
from tests.test_paired_button_light import patch_image, snapshot
from tools.opencv_light_worker import RelativeLuminance


FIXTURES = Path(__file__).parent / 'fixtures/clipped_background_70_80'
FRAMES = json.loads((FIXTURES / 'manifest.json').read_text())['frames']


def recorded_snapshot(frame, *, sequence=None, captured=None):
    seq = frame['sequence'] if sequence is None else sequence
    detection = dict(frame['target_detection'] or {})
    detection['frame_sequence'] = seq
    return dict(path=str(FIXTURES / frame['file']), frame_sequence=seq,
                capture_time_unix_sec=frame['captured'] if captured is None else captured,
                target_detection=detection, fresh=True, image_size=[640, 360])


def recorded_pair():
    return [extract_patch(recorded_snapshot(f)) for f in FRAMES[:2]]


def synthetic_pair(valid_pixels=480, *, light=False, dark=False, concentrated=False):
    before = patch_image()
    after = patch_image(2, increase=120 if light else 0)
    invalid_count = int(_RING.sum()) - valid_pixels
    mask = np.zeros_like(_RING)
    if concentrated:
        indices = np.argwhere(_RING)
        mask[tuple(indices[:invalid_count].T)] = True
    else:
        quadrants = [_RING & horizontal & vertical
                     for horizontal in (_X < 32, _X >= 32)
                     for vertical in (_Y < 32, _Y >= 32)]
        for index, quadrant in enumerate(quadrants):
            count = invalid_count // 4 + (index < invalid_count % 4)
            points = np.argwhere(quadrant)[:count]
            mask[tuple(points.T)] = True
    before.gray[mask] = after.gray[mask] = 0 if dark else 255
    return before, after


def test_original_first_pair_requires_stationary_authorization_and_passes_other_checks():
    before, after = recorded_pair()
    assert compare_patches(before, after, 'OFF')['normalization_rejection'] == 'clipped_background'
    result = compare_patches(before, after, 'OFF', stationary=True)
    assert result['state'] == 'OFF'
    assert result['background_tolerance_profile'] == 'clipped_background_70_80_v1'
    assert result['background_valid_fraction'] == pytest.approx(.7875)
    assert result['strict_background_attempt']['normalization_rejection'] == 'clipped_background'
    assert all(result['clipped_background_view_checks'].values())
    assert min(result['background_quadrant_valid_fraction']) >= .5
    assert result['background_error'] <= 10 and result['background_error_p90'] <= 25
    assert result['center_error'] <= 8 and result['changed_fraction'] < .02


def test_recorded_sequence_keeps_invalid_frame_breaks_and_nonuniform_rejections():
    pending = None
    results = []
    for frame in FRAMES:
        if frame['state'] != 'OFF':
            pending = None
            continue
        patch = extract_patch(recorded_snapshot(frame))
        if pending is not None:
            results.append(compare_patches(pending, patch, 'OFF', stationary=True))
        pending = patch
    assert len(results) == 7
    assert sum(r['state'] == 'OFF' for r in results) == 5
    rejected = [r for r in results if r['state'] != 'OFF']
    assert len(rejected) == 2
    assert all(r['normalization_rejection'] == 'nonuniform_background_change' for r in rejected)


@pytest.mark.parametrize('valid_pixels,expected', [(447, 'UNCERTAIN'), (448, 'OFF'),
                                                 (480, 'OFF'), (511, 'OFF'), (512, 'OFF')])
def test_support_boundaries_keep_hard_floor_and_original_eighty_percent_path(valid_pixels, expected):
    pair = synthetic_pair(valid_pixels)
    guard = Mock(return_value=True)
    result = compare_patches(*pair, 'OFF', stationary=guard)
    assert result['state'] == expected
    if valid_pixels >= 512:
        guard.assert_not_called()
        assert 'background_tolerance_profile' not in result
    elif valid_pixels < 448:
        assert result['normalization_rejection'] == 'clipped_background'
    else:
        assert result['background_tolerance_profile'] == 'clipped_background_70_80_v1'


@pytest.mark.parametrize('fault', ['dark', 'concentrated', 'textureless', 'shadow'])
def test_less_clipping_does_not_rescue_bad_or_uneven_background(fault):
    before, after = synthetic_pair(dark=fault == 'dark', concentrated=fault == 'concentrated')
    if fault == 'textureless':
        before.gray[(before.gray > 5) & (before.gray < 250)] = 40
        after.gray[(after.gray > 5) & (after.gray < 250)] = 40
    elif fault == 'shadow':
        after.gray[:, :32] = np.minimum(after.gray[:, :32] + 50, 255)
    result = compare_patches(before, after, 'OFF', stationary=True)
    assert result['state'] == 'UNCERTAIN'
    assert result.get('normalization_rejection')


@pytest.mark.parametrize('fault', ['small', 'size', 'center', 'time', 'sequence', 'pose'])
def test_changed_view_or_unverified_pose_cannot_use_seventy_percent_fallback(fault):
    before, after = synthetic_pair()
    guard = Mock(return_value=fault != 'pose')
    if fault == 'small':
        before = replace(before, box=np.array([30, 30, 54, 54]))
        after = replace(after, box=before.box.copy())
    elif fault == 'size':
        after = replace(after, box=after.box + np.array([0, 0, 5, 5]))
    elif fault == 'center':
        after = replace(after, box=after.box + 5)
    elif fault == 'time':
        after = replace(after, captured=before.captured + 31)
    elif fault == 'sequence':
        after = replace(after, sequence=before.sequence)
    result = compare_patches(before, after, 'OFF', stationary=guard)
    assert result['state'] == 'UNCERTAIN'
    assert not all(result['clipped_background_view_checks'].values())


def test_stable_glare_does_not_turn_unchanged_symbol_into_new_light():
    before, after = synthetic_pair()
    before.gray[25:39, 25:39] += 100
    after.gray[25:39, 25:39] += 100
    result = compare_patches(before, after, 'ON', stationary=True)
    assert result['state'] == 'UNCERTAIN'
    assert result['changed_fraction'] == 0


@pytest.mark.parametrize('stationary_mode,stationary_phase,pose_ok', [
    (False, True, True), (True, False, True), (True, True, False),
])
def test_off_bank_needs_stationary_mode_phase_and_live_pose(stationary_mode, stationary_phase, pose_ok):
    guard = Mock(return_value=pose_ok)
    bank = PairedButtonLight(require_stationary_reference=stationary_mode, stationary_check=guard)
    bank.begin_attempt()
    for frame in FRAMES[:2]:
        bank.collect(recorded_snapshot(frame), 'OFF', stationary=stationary_phase)
    assert not bank.references
    if not stationary_mode or not stationary_phase:
        guard.assert_not_called()


def test_real_fixed_reference_collector_recovers_without_motion_and_logs_quality(tmp_path):
    sequence = 0
    def request(*args, **kwargs):
        nonlocal sequence
        frame = FRAMES[min(sequence, 2)]
        sequence += 1
        return recorded_snapshot(frame, sequence=sequence, captured=time.time())
    guard = Mock(return_value=True)
    bank = PairedButtonLight(tmp_path, require_stationary_reference=True, stationary_check=guard)
    bank.begin_attempt()
    observer = FreshLightObserver(SimpleNamespace(request=request),
        SimpleNamespace(observe=RelativeLuminance().predict), '2', paired_light=bank)
    result = observer.capture_stationary_off_reference(timeout_sec=1)
    assert result['sequences'] == [1, 2] and bank.stationary_reference_ready
    assert not observer.stationary_reference_capture_active
    assert observer.audit[-1]['recognition']['off_reference_comparison']['state'] == 'OFF'
    assert result['reference_comparison']['background_tolerance_profile'] == 'clipped_background_70_80_v1'
    bank.freeze()
    # The same low-support comparison is permitted at the verified fixed pose,
    # and remains unavailable in the moving monitor.
    assert observer.observe('press_monitor') == 'UNCERTAIN'
    assert observer.observe('near') == 'OFF'
    assert observer.audit[-1]['recognition']['paired_comparison']['background_tolerance_profile'] == 'clipped_background_70_80_v1'


def test_stationary_collection_flag_clears_after_failure(tmp_path):
    def request(*args, **kwargs):
        raise RuntimeError('capture failed')
    bank = PairedButtonLight(require_stationary_reference=True)
    observer = FreshLightObserver(SimpleNamespace(request=request), None, '2', paired_light=bank)
    with pytest.raises(RuntimeError, match='capture failed'):
        observer.capture_stationary_off_reference()
    assert not observer.stationary_reference_capture_active


def test_lit_center_still_needs_change_against_frozen_off_reference(tmp_path):
    before, after = synthetic_pair(light=True)
    bank = PairedButtonLight(require_stationary_reference=True, stationary_check=lambda: True)
    bank.begin_attempt()
    for index in (1, 2):
        bank.collect(snapshot(tmp_path, replace(before, sequence=index, captured=time.time())),
                     'OFF', stationary=True)
    assert len(bank.references) == 2
    bank.stationary_reference_ready = True
    bank.freeze()
    for index in (3, 4):
        image = snapshot(tmp_path, replace(after, sequence=index, captured=time.time()))
        classified = RelativeLuminance().predict(image['path'], '2', image['target_detection']['bbox_xyxy'])
        assert classified['state'] == 'ON'
        result = bank.classify(image, classified['state'], stationary=True)
        assert result['state'] == 'ON' and result['reason'] == 'localized_rise_from_confirmed_off'
        assert result['background_tolerance_profile'] == 'clipped_background_70_80_v1'


@pytest.mark.parametrize('error,hardware_ok,expected', [(0, 1, True), (.004, 1, True),
                                                     (.006, 1, True), (.008, 1, True), (.0081, 1, False), (0, 0, False)])
def test_live_pose_check_is_read_only_and_requires_fixed_observation(error, hardware_ok, expected):
    executor = DryValidatedSegmentExecutor.__new__(DryValidatedSegmentExecutor)
    executor._require_plan = Mock(return_value=SimpleNamespace(observation_joints=np.zeros(6)))
    executor.robot = SimpleNamespace(state=Mock(return_value=SimpleNamespace(
        joints=np.full(6, error), raw={'hardware_ok': hardware_ok})))
    assert executor.stationary_observation_verified() is expected
    executor.robot.state.assert_called_once_with(max_age=.25)
