"""Numeral-only local gate; archived failure and whole/paired-path controls."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.button_light_process import OpenCvButtonLightProcess
from tests.test_button_light_hardware import FakeVision
from tools.opencv_light_worker import RelativeLuminance


FIXTURES = Path(__file__).parent / 'fixtures/numeric_local_on100'


def stroke(tmp_path, contrast):
    gray = np.full((100, 100), 35, np.uint8)
    gray[29:70, 48:52] = 35 + contrast
    path = tmp_path / f'stroke_{contrast}.png'
    assert cv2.imwrite(str(path), gray)
    return path


@pytest.mark.parametrize('target', [str(n) for n in range(1, 17)])
@pytest.mark.parametrize('contrast, expected', [(85, 'OFF'), (99, 'OFF'),
                                               (100, 'ON'), (101, 'ON')])
def test_every_numeral_local_boundary(tmp_path, target, contrast, expected):
    result = RelativeLuminance().predict(stroke(tmp_path, contrast), target, [0, 0, 100, 100])
    assert result['legacy_state'] == 'OFF'
    assert result['state'] == expected
    assert result['local_on_contrast_threshold'] == 100
    assert result['local_decision_policy'] == 'on_at_100_else_legacy'


@pytest.mark.parametrize('target', ['up', 'down', 'open', 'B1', '0', '17'])
def test_other_labels_keep_existing_gate(tmp_path, target):
    result = RelativeLuminance().predict(stroke(tmp_path, 90), target, [0, 0, 100, 100])
    assert result['local_on_contrast_threshold'] == 85
    assert result['local_decision_policy'] == 'on_at_85_else_legacy'


def test_existing_whole_on_is_preserved_below_new_local_gate(tmp_path):
    gray = np.full((100, 100), 35, np.uint8)
    gray[27:72, 27:72] = 95
    path = tmp_path / 'whole_on.png'
    assert cv2.imwrite(str(path), gray)
    result = RelativeLuminance().predict(path, '1', [0, 0, 100, 100])
    assert max(r['contrast'] for r in result['local_regions']) == 60
    assert result['legacy_state'] == result['state'] == 'ON'
    assert result['local_on_contrast_threshold'] == 100


def test_explicit_legacy_still_uses_original_whole_result(tmp_path):
    result = RelativeLuminance(algorithm='legacy').predict(
        stroke(tmp_path, 150), '1', [0, 0, 100, 100])
    assert result['legacy_state'] == result['state'] == 'OFF'


@pytest.mark.parametrize('contrast, strong', [(99, False), (100, True), (101, True)])
def test_numeric_paired_relaxation_requires_current_local_gate(contrast, strong):
    result = dict(state='ON', legacy_state='ON', local_on_contrast_threshold=100,
                  local_regions=[dict(contrast=contrast, fraction=.1, area_px=20)])
    paired = SimpleNamespace(classify=Mock(return_value={'state': 'ON'}))
    observer = FreshLightObserver(FakeVision(), SimpleNamespace(observe=Mock(return_value=result)),
                                  '15', paired_light=paired)
    assert observer.observe('near') == 'ON'
    assert bool(paired.classify.call_args.kwargs.get('strong_on')) is strong


def test_real_first_press_reference_false_on_no_longer_uses_85_override():
    frames = json.loads((FIXTURES / 'manifest.json').read_text())['frames']
    for frame in frames:
        path = FIXTURES / frame['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frame['sha256']
        result = RelativeLuminance().predict(path, '1', frame['bbox'])
        assert result['legacy_state'] == result['state'] == 'OFF'
        assert max(r['contrast'] for r in result['local_regions']) < 100
        if frame['previous_state'] == 'ON':
            assert max(r['contrast'] for r in result['local_regions']) >= 85


def test_resident_worker_applies_both_policies_without_restart(tmp_path):
    path = stroke(tmp_path, 90)
    with OpenCvButtonLightProcess() as worker:
        pid = worker.process.pid
        for target, gate in [('1', 100), ('up', 85), ('16', 100), ('down', 85)]:
            result = worker.observe(path, target, [0, 0, 100, 100])
            assert result['local_on_contrast_threshold'] == gate
            assert worker.process.pid == pid
