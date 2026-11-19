"""Numeric glare/reference regressions; all local ON thresholds are85."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest

from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.button_light_process import OpenCvButtonLightProcess
from piper_handeye.paired_button_light import PairedButtonLight
from tools.opencv_light_worker import RelativeLuminance


FIXTURES = Path(__file__).parent / 'fixtures/numeric_local_threshold65'
FRAMES = json.loads((FIXTURES / 'manifest.json').read_text())['fixtures']
OFF = [frame for frame in FRAMES if frame['expected'] == 'OFF']
ON = [frame for frame in FRAMES if frame['expected'] == 'ON']
ARROWS = Path(__file__).parent / 'fixtures/arrow_light_region'
ARROW_FRAMES = json.loads((ARROWS / 'manifest.json').read_text())['fixtures']


@pytest.mark.parametrize('frame', FRAMES, ids=lambda f: f['file'])
def test_real_unlit_highlight_is_off_and_actual_lit_two_remains_on(frame):
    result = RelativeLuminance().predict(
        FIXTURES / frame['file'], frame['target'], frame['bbox'])
    assert result['state'] == frame['expected']
    assert result['local_on_contrast_threshold'] == 85
    assert result['local_decision_policy'] == 'on_at_85_else_legacy'


@pytest.mark.parametrize('label', [*map(str, range(1, 17)), 'B1', 'B2'])
@pytest.mark.parametrize('contrast,expected', [(84, 'OFF'), (85, 'ON')])
def test_non_call_buttons_have_an_85_local_boundary(tmp_path, label, contrast, expected):
    gray = np.full((100, 100), 35, np.uint8)
    gray[29:70, 48:52] = 35 + contrast
    image = tmp_path / 'local_stroke.png'
    assert cv2.imwrite(str(image), gray)
    result = RelativeLuminance().predict(image, label, [0, 0, 100, 100])
    assert result['legacy_state'] == 'OFF'
    assert result['state'] == expected


@pytest.mark.parametrize('label', ['up', 'down', 'UP', 'DOWN'])
def test_call_arrows_have_the_85_threshold_and_keep_real_lights_on(label):
    frame = next(f for f in ARROW_FRAMES
                 if f['target'] == label.casefold() and f['expected'] == 'ON')
    result = RelativeLuminance().predict(ARROWS / frame['file'], label, frame['bbox'])
    assert result['state'] == 'ON'
    assert result['local_on_contrast_threshold'] == 85
    assert result['local_decision_policy'] == 'on_at_85_else_legacy'


def test_overall_on_is_retained_when_local_contrast_is_below_85(tmp_path):
    gray = np.full((100, 100), 35, np.uint8)
    gray[27:72, 27:72] = 80
    image = tmp_path / 'broad_light.png'
    assert cv2.imwrite(str(image), gray)
    result = RelativeLuminance().predict(image, '2', [0, 0, 100, 100])
    assert result['legacy_state'] == result['state'] == 'ON'
    assert max(r['contrast'] for r in result['local_regions']) < 85


def reference_observer(frames, tmp_path):
    class Vision:
        sequence = 0

        def request(self, *args, **kwargs):
            frame = frames[self.sequence % len(frames)]
            self.sequence += 1
            return dict(fresh=True, frame_sequence=self.sequence,
                        capture_time_unix_sec=time.time(), image_size=[640, 360],
                        path=str(FIXTURES / frame['file']),
                        target_detection=dict(frame_sequence=self.sequence,
                                              bbox_xyxy=frame['bbox'], confidence=.97))

    pair = PairedButtonLight(tmp_path, require_stationary_reference=True)
    pair.begin_attempt()
    observer = FreshLightObserver(
        Vision(), SimpleNamespace(observe=RelativeLuminance().predict), '2',
        paired_light=pair)
    return observer, pair


def test_failed_numeric_scene_can_build_two_comparable_off_references(tmp_path):
    observer, pair = reference_observer(OFF[:3], tmp_path)
    result = observer.capture_stationary_off_reference(timeout_sec=1)
    assert pair.stationary_reference_ready and result['reference_count'] >= 2
    assert all(x['state'] == 'OFF' for x in observer.audit)
    pair.freeze()


def test_actual_lit_two_cannot_authorize_press_as_an_off_reference(tmp_path):
    observer, pair = reference_observer(ON, tmp_path)
    with pytest.raises(RuntimeError, match='press withheld'):
        observer.capture_stationary_off_reference(timeout_sec=.15)
    assert not pair.references and not pair.stationary_reference_ready


def test_production_worker_switches_threshold_per_request_without_restarting():
    arrow = next(f for f in ARROW_FRAMES if f['target'] == 'up' and f['expected'] == 'ON')
    requests = [(FIXTURES, OFF[0], '2', 'OFF', 85),
                (ARROWS, arrow, 'up', 'ON', 85),
                (FIXTURES, OFF[0], '4', 'OFF', 85),
                (FIXTURES, ON[0], '2', 'ON', 85)]
    with OpenCvButtonLightProcess() as worker:
        pid = worker.process.pid
        for directory, frame, label, expected, threshold in requests:
            result = worker.observe(directory / frame['file'], label, frame['bbox'])
            assert result['state'] == expected
            assert result['local_on_contrast_threshold'] == threshold
            assert result['local_decision_policy'] == f'on_at_{threshold}_else_legacy'
            assert worker.process.pid == pid
