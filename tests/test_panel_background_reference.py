"""Real lit frames and adverse lighting for the outer-panel fallback."""
import json
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pytest

from piper_handeye.paired_button_light import (
    PairedButtonLight, compare_patches, extract_patch, _RING, _PANEL_RING,
)
from tests.test_paired_button_light import patch_image


FIXTURES = Path(__file__).parent / 'fixtures/panel_background_reference'
SCENES = json.loads((FIXTURES / 'manifest.json').read_text())['scenes']
GLARE = json.loads((FIXTURES / 'manifest.json').read_text())['unlit_glare_pairs']


def snapshot(frame):
    return dict(path=str(FIXTURES / frame['file']), frame_sequence=frame['sequence'],
                capture_time_unix_sec=frame['time'],
                target_detection=dict(frame_sequence=frame['sequence'], bbox_xyxy=frame['box']))


@pytest.mark.parametrize('scene', SCENES, ids=lambda s: s['label'])
def test_real_lit_button_uses_unchanged_limits_on_panel_outside_the_bezel(scene):
    before = extract_patch(snapshot(scene['before'][0]))
    for frame in scene['after']:
        result = compare_patches(before, extract_patch(snapshot(frame)), 'ON')
        assert result['state'] == 'ON'
        assert result['background_region'] == 'outer_panel_ring_v1'
        assert result['background_error'] <= 10
        assert result['background_error_p90'] <= 25
        assert result['background_inlier_fraction'] >= .75
        assert result['background_spatial_error'] <= 12
        assert result['button_ring_normalization']['normalization_rejection'] == 'nonuniform_background_change'


@pytest.mark.parametrize('scene', SCENES, ids=lambda s: s['label'])
def test_frozen_off_bank_confirms_two_new_lit_frames_with_actual_reference_selection(scene):
    pair = PairedButtonLight(require_stationary_reference=True)
    pair.begin_attempt()
    for index, frame in enumerate(scene['before']):
        image = snapshot(frame)
        image['capture_time_unix_sec'] = time.time() - .3 + index * .1
        pair.collect(image, 'OFF')
    assert len(pair.references) == 2
    pair.stationary_reference_ready = True
    pair.freeze()
    for frame in scene['after']:
        image = snapshot(frame)
        image['capture_time_unix_sec'] = time.time()
        assert pair.classify(image, 'ON')['state'] == 'ON'


def bezel_change(*, light=False):
    before = patch_image(increase=50)
    after = patch_image(2, increase=120 if light else 50)
    after.gray[_RING & ~_PANEL_RING] += 50
    return before, after


@pytest.mark.parametrize('light', [False, True])
def test_bezel_reflection_alone_does_not_become_new_light(light):
    before, after = bezel_change(light=light)
    result = compare_patches(before, after, 'ON')
    assert (result['state'] == 'ON') is light


@pytest.mark.parametrize('fault', ['shadow', 'clipped', 'textureless', 'covered_center'])
def test_bad_external_panel_cannot_rescue_a_failed_ring(fault):
    before, after = bezel_change(light=True)
    if fault == 'shadow':
        after.gray[:, :32] += 60
    elif fault == 'clipped':
        after.gray[_PANEL_RING] = 255
    elif fault == 'textureless':
        before.gray[_PANEL_RING] = 40
        after.gray[_PANEL_RING] = 70
    else:
        from piper_handeye.paired_button_light import _CENTER
        after.gray[_CENTER] = 210
    assert compare_patches(before, after, 'ON')['state'] == 'UNCERTAIN'


@pytest.mark.parametrize('fault', ['small', 'scale', 'shift'])
def test_fallback_requires_close_and_sufficiently_resolved_views(fault):
    before, after = bezel_change(light=True)
    if fault == 'small':
        before.box = after.box = np.array([30, 30, 54, 54])
    elif fault == 'scale':
        after.box = before.box.copy()
        after.box[2:] += 5
    else:
        after.box = before.box + np.array([5, 0, 5, 0])
    assert compare_patches(before, after, 'ON')['state'] == 'UNCERTAIN'


def test_off_reference_collection_keeps_original_full_ring_check():
    before, after = bezel_change()
    result = compare_patches(before, after, 'OFF')
    assert result['state'] == 'UNCERTAIN'
    assert 'background_region' not in result


@pytest.mark.parametrize('pair', GLARE)
@pytest.mark.parametrize('size_factor', [1, 2])
def test_recorded_unlit_glare_never_passes_even_if_resolution_gate_is_bypassed(pair, size_factor):
    before, after = (extract_patch(snapshot(pair[k])) for k in ('before', 'after'))
    before = replace(before, box=before.box * size_factor)
    after = replace(after, box=after.box * size_factor)
    # Deliberately supply an erroneous absolute ON. The comparator must reject
    # both a small moving glint and a broad non-lamp appearance change itself.
    assert compare_patches(before, after, 'ON')['state'] == 'UNCERTAIN'
