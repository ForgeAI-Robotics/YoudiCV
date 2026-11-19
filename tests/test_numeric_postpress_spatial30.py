"""Only stationary numeric post-press strong-ON gets a 30-gray quadrant cap."""
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import piper_handeye.paired_button_light as light
from tests.test_paired_button_light import patch_image

FIXTURE = Path(__file__).parent / 'fixtures/numeric_postpress_spatial30'
FRAMES = json.loads((FIXTURE / 'manifest.json').read_text())['frames']


def snapshot(frame, label='2'):
    path = FIXTURE / frame['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == frame['sha256']
    return dict(path=str(path), fresh=True, frame_sequence=frame['sequence'],
                capture_time_unix_sec=frame['captured'], image_size=[640, 360],
                target_detection=dict(label=label, confidence=.95,
                    frame_sequence=frame['sequence'], bbox_xyxy=frame['box']))


def paired(monkeypatch, frame, *, pose=True, ready=True):
    pair = light.PairedButtonLight(require_stationary_reference=True,
                                   stationary_check=lambda: pose)
    pair.begin_attempt()
    pair.references.extend(light.extract_patch(snapshot(f)) for f in FRAMES[:2])
    pair.frozen = True
    pair.stationary_reference_ready = ready
    pair.press_started = FRAMES[1]['captured'] + .001
    pair.last_reference_sequence = FRAMES[1]['sequence']
    monkeypatch.setattr(light.time, 'time', lambda: frame['captured'] + .01)
    return pair


@pytest.mark.parametrize('frame', FRAMES[2:6], ids=lambda f: str(f['sequence']))
def test_recorded_lit_two_passes_only_new_stationary_strong_on_branch(monkeypatch, frame):
    p = paired(monkeypatch, frame)
    old_limit = light._NUMERIC_STRONG_ON_MAX_SPATIAL_ERROR
    monkeypatch.setattr(light, '_NUMERIC_STRONG_ON_MAX_SPATIAL_ERROR', 12.)
    old = p.classify(snapshot(frame), 'ON', stationary=True, strong_on=True)
    assert old['state'] == 'UNCERTAIN'
    monkeypatch.setattr(light, '_NUMERIC_STRONG_ON_MAX_SPATIAL_ERROR', old_limit)
    # Passing before registration avoids the extra ORB pass; no new inference.
    hook = Mock(side_effect=AssertionError('already qualified ON needs no registration'))
    monkeypatch.setattr(light, 'register_reference', hook)
    new = p.classify(snapshot(frame), 'ON', stationary=True, strong_on=True)
    assert new['state'] == 'ON'
    assert 12 < new['background_spatial_error'] <= 30
    assert new['background_max_spatial_error'] == 30
    assert new['numeric_strong_on_strict_attempt']['background_max_spatial_error'] == 12
    assert new['background_max_mean_error'] == 20
    assert new['background_max_p90_error'] == 50
    assert new['background_inlier_max_error'] == 30
    assert new['background_inlier_fraction'] >= .75
    assert new['changed_area_relaxation']['max_changed_fraction'] == .99
    hook.assert_not_called()


@pytest.mark.parametrize('fault', ['moving', 'pose', 'not_ready', 'no_strong_on', 'arrow'])
def test_relaxation_not_available_without_numeric_stationary_authorization(monkeypatch, fault):
    f = FRAMES[2]
    p = paired(monkeypatch, f, pose=fault != 'pose', ready=fault != 'not_ready')
    # The observer never supplies numeric strong_on for arrows.
    r = p.classify(snapshot(f, 'down' if fault == 'arrow' else '2'), 'ON',
                   stationary=fault != 'moving',
                   strong_on=fault not in {'no_strong_on', 'arrow'})
    assert r['state'] == 'UNCERTAIN'
    assert 'numeric_strong_on_background_attempt' not in r


@pytest.mark.parametrize('state', ['OFF', 'UNCERTAIN'])
def test_not_on_never_relaxes_spatial_background_limit(monkeypatch, state):
    f = FRAMES[2]
    r = paired(monkeypatch, f).classify(snapshot(f), state,
                                      stationary=True, strong_on=True)
    assert r['state'] == 'UNCERTAIN'
    assert r['background_max_spatial_error'] == 12
    assert 'numeric_strong_on_background_attempt' not in r


def test_later_off_frame_does_not_become_false_on(monkeypatch):
    f = FRAMES[-1]
    r = paired(monkeypatch, f).classify(snapshot(f), 'OFF', stationary=True)
    assert r['state'] != 'ON'
    assert r['background_max_spatial_error'] == 12


def test_normal_background_and_prepress_off_keep_twelve():
    a, b = patch_image(), patch_image(2)
    result = light.compare_patches(a, b, 'OFF', stationary=True, strong_on=True)
    assert result['state'] == 'OFF'
    assert result['background_max_spatial_error'] == 12
    assert result['background_max_mean_error'] == 10
    assert result['background_max_p90_error'] == 25


@pytest.mark.parametrize('cap,expected', [(12., False), (30., True), (31., True)])
def test_only_spatial_cap_changes_with_keyword(cap, expected):
    f = FRAMES[2]
    a, b = (light.extract_patch(snapshot(x)) for x in [FRAMES[0], f])
    residual, audit = light.normalize_background(a.gray, b.gray,
        residual_limits=(20., 50.), inlier_max_error=30., max_spatial_error=cap)
    assert (residual is not None) is expected
    assert audit['background_max_spatial_error'] == cap
    assert audit['background_max_mean_error'] == 20
    assert audit['background_max_p90_error'] == 50


def test_more_than_thirty_spatial_error_still_rejected():
    a = patch_image().gray
    b = a.copy()
    b[:32, :32] += 90
    residual, audit = light.normalize_background(a, b, residual_limits=(20.,50.),
                                                 inlier_max_error=30., max_spatial_error=30.)
    assert residual is None
    assert audit['background_spatial_error'] > 30
    assert audit['normalization_rejection'] == 'nonuniform_background_change'
