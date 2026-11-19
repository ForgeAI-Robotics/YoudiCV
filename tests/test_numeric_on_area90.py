"""Recorded lit 2, 99% candidate boundary and negative controls; no robot or network."""
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock
import time

import numpy as np
import pytest

from piper_handeye.paired_button_light import (
    PairedButtonLight, compare_patches, extract_patch, _CENTER, _RING, _PANEL_RING,
)
from tests.test_paired_button_light import patch_image
from tests.test_panel_background_reference import GLARE, snapshot as glare_snapshot

FIXTURE = Path(__file__).parent/'fixtures/numeric_on_area90'
FRAMES = json.loads((FIXTURE/'manifest.json').read_text())['frames']


def snapshot(frame):
    path = FIXTURE/frame['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == frame['sha256']
    return dict(path=str(path), fresh=True, frame_sequence=frame['sequence'],
        capture_time_unix_sec=frame['captured'],
        target_detection=dict(label='2', frame_sequence=frame['sequence'],
                              bbox_xyxy=frame['box'], confidence=.95))


def make_pair(monkeypatch):
    pair = PairedButtonLight(require_stationary_reference=True,
                             stationary_check=Mock(return_value=True))
    pair.references.extend(extract_patch(snapshot(f)) for f in FRAMES[:2])
    pair.stationary_reference_ready = True
    pair.frozen = True
    pair.press_started = FRAMES[1]['captured']+.001
    pair.last_reference_sequence = FRAMES[1]['sequence']
    monkeypatch.setattr('piper_handeye.paired_button_light.time.time',
                        lambda: FRAMES[2]['captured']+.01)
    return pair, snapshot(FRAMES[2])


def test_recorded_lit_two_now_passes_only_qualified_stationary_numeric_path(monkeypatch):
    pair, image = make_pair(monkeypatch)
    before, after = extract_patch(snapshot(FRAMES[1])), extract_patch(image)
    original = compare_patches(before, after, 'ON', stationary=True, strong_on=True)
    assert original['state'] == 'UNCERTAIN'
    assert original['reason'] == 'nonlocal_rise_for_panel_reference'
    assert .70 < original['changed_fraction'] <= .90
    new = pair.classify(image, 'ON', stationary=True, strong_on=True)
    assert new['state'] == 'ON'
    assert new['panel_max_changed_fraction'] == .99
    assert new['changed_area_relaxation']['stationary_observation_verified'] is True
    assert new['background_error'] <= new['background_max_mean_error']
    assert new['absolute_state'] == 'ON'


@pytest.mark.parametrize('fault', ['moving', 'bad_pose', 'no_strong_on', 'arrow',
    'not_fresh', 'stale', 'unconfirmed_reference'])
def test_recorded_two_cannot_use_area90_without_every_extra_guard(monkeypatch, fault):
    pair, image = make_pair(monkeypatch)
    if fault == 'bad_pose': pair.stationary_check.return_value = False
    if fault == 'arrow': image['target_detection']['label'] = 'down'
    if fault == 'not_fresh': image['fresh'] = False
    if fault == 'unconfirmed_reference': pair.stationary_reference_ready = False
    if fault == 'stale':
        monkeypatch.setattr('piper_handeye.paired_button_light.time.time',
                            lambda: FRAMES[2]['captured']+.36)
    result = pair.classify(image, 'ON', stationary=fault != 'moving',
                           strong_on=fault != 'no_strong_on')
    assert result['state'] == 'UNCERTAIN'
    assert 'changed_area_relaxation' not in result


@pytest.mark.parametrize('fraction, expected', [(.69,'ON'),(.71,'ON'),(.85,'ON'),
                                               (.89,'ON'),(.91,'ON'),(.99,'ON'),(.995,'UNCERTAIN'),(1.,'UNCERTAIN')])
def test_outer_panel_fraction_boundary(fraction, expected):
    before, after = patch_image(), patch_image(2)
    after.gray[_RING & ~_PANEL_RING] += 50
    indices = np.flatnonzero(_CENTER)
    after.gray.flat[indices[:int(fraction*len(indices))]] += 80
    result = compare_patches(before, after, 'ON', stationary=True, strong_on=True,
                             allow_area_relaxation=True)
    assert result['state'] == expected
    if fraction > .70:
        assert result['panel_max_changed_fraction'] == .99


def test_pose_check_delay_cannot_relax_an_expired_frame(monkeypatch):
    before, after = patch_image(), patch_image(2)
    after.gray[_RING & ~_PANEL_RING] += 50
    ids = np.flatnonzero(_CENTER); after.gray.flat[ids[:int(.85*len(ids))]] += 80
    clock = [after.captured+.01]
    monkeypatch.setattr('piper_handeye.paired_button_light.time.time', lambda: clock[0])
    def delayed():
        clock[0] += .4
        return True
    r = compare_patches(before, after, 'ON', stationary=delayed, strong_on=True,
                         allow_area_relaxation=True)
    assert r['state'] == 'UNCERTAIN'
    assert 'changed_area_relaxation' not in r


@pytest.mark.parametrize('pair_record', GLARE)
@pytest.mark.parametrize('size_factor', [1,2])
def test_archived_unlit_glare_stays_unconfirmed_even_with_wrong_on_flags(pair_record, size_factor):
    before, after = (extract_patch(glare_snapshot(pair_record[k])) for k in ('before','after'))
    before.box *= size_factor; after.box *= size_factor
    after.captured = time.time()
    assert compare_patches(before, after, 'ON', stationary=True, strong_on=True,
                          allow_area_relaxation=True)['state'] == 'UNCERTAIN'


@pytest.mark.parametrize('fault', ['uniform_offset','uniform_gain','shadow','covered',
                                   'saturated','unchanged','view_shift','view_scale'])
def test_existing_adverse_image_controls_still_reject(fault):
    before, after = patch_image(increase=70), patch_image(2,increase=70)
    if fault == 'uniform_offset': after.gray += 35
    if fault == 'uniform_gain': after.gray *= 1.5
    if fault == 'shadow': after.gray[:,:32] += 50
    if fault == 'covered': after.gray[:] = 200
    if fault == 'saturated': after.gray[8:56,8:56] = 255
    if fault == 'view_shift': after.box = before.box+np.array([10,0,10,0])
    if fault == 'view_scale': after.box = before.box+np.array([0,0,20,20])
    assert compare_patches(before,after,'ON',stationary=True,strong_on=True,
                          allow_area_relaxation=True)['state'] == 'UNCERTAIN'


def test_off_comparison_unchanged_even_with_area_opt_in():
    before, after = patch_image(), patch_image(2,exposure=35)
    assert compare_patches(before,after,'OFF',stationary=True,strong_on=True,
        allow_area_relaxation=True) == compare_patches(before,after,'OFF',stationary=True,strong_on=True)


def test_runtime_report_describes_actual_qualified_numeric_area_limit():
    from run_button_light_closed_loop import LIGHT_EVIDENCE_POLICY
    from piper_handeye.paired_button_light import _NUMERIC_POST_PRESS_MAX_CHANGED_FRACTION
    policy = LIGHT_EVIDENCE_POLICY['numeric_post_press_changed_area_relaxation']
    assert policy['max_changed_fraction'] == _NUMERIC_POST_PRESS_MAX_CHANGED_FRACTION == .99
    assert policy['denominator'] == 'button_center_analysis_mask_pixels'
    assert policy['base_max_changed_fraction'] == .70
