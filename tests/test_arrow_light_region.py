"""Distinguish a decorative reflection from the arrow without authorizing motion."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest

from piper_handeye.arrow_light_region import resolve_arrow_highlight
from piper_handeye.button_light_flow import ButtonLightFlow
from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.button_light_process import OpenCvButtonLightProcess
from piper_handeye.initial_light_guard import InitialLightGuard
from tools.opencv_light_worker import RelativeLuminance
from test_button_light_flow import FakeHardware


FIXTURES = Path(__file__).parent / 'fixtures/arrow_light_region'
FRAMES = json.loads((FIXTURES / 'manifest.json').read_text())['fixtures']


def frame_gray(frame):
    # Geometry-only tests deliberately include highlights down to60 to keep
    # testing stripe rejection independently of the production ON gate85.
    image = cv2.imread(str(FIXTURES / frame['file']))
    left, top, right, bottom = map(round, frame['bbox'])
    gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    x1, x2 = int(.275 * width), int(.725 * width)
    y1, y2 = int(.275 * height), int(.725 * height)
    ring = np.ones(gray.shape, dtype=bool);ring[y1:y2, x1:x2] = False
    median = float(np.median(gray[ring]))
    center = gray[y1:y2, x1:x2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (center > median + 30).astype(np.uint8), connectivity=8)
    mask = np.zeros(gray.shape, dtype=bool)
    for label in range(1, count):
        if (stats[label, cv2.CC_STAT_AREA] >= max(3, int(np.ceil(center.size * .03)))
                and float(center[labels == label].mean()) - median >= 60):
            mask[y1:y2, x1:x2] |= labels == label
    return gray, mask, median


@pytest.mark.parametrize('frame', FRAMES, ids=lambda f: f['file'])
def test_original_real_camera_frames(frame):
    result = RelativeLuminance().predict(FIXTURES / frame['file'], frame['target'], frame['bbox'])
    assert result['state'] == frame['expected']
    if frame['group'] == 'sun_stripe_off':
        assert result['legacy_state'] == 'OFF'
        assert max(r['contrast'] for r in result['local_regions']) < 85
        assert result['local_on_contrast_threshold'] == 85
        assert 'arrow_highlight_guard' not in result
        guarded = InitialLightGuard('down', FIXTURES).filter(result)
        assert guarded['state'] == 'OFF'


@pytest.mark.parametrize('frame', [f for f in FRAMES if f['expected'] == 'ON'], ids=lambda f: f['file'])
def test_real_lit_diagonals_cannot_become_off_when_overall_classifier_disagrees(frame):
    gray, mask, median = frame_gray(frame)
    state, audit = resolve_arrow_highlight(gray, mask, frame['target'], median, 'OFF')
    assert state == 'ON'


def test_production_worker_loads_geometry_and_reuses_process_from_another_directory(tmp_path, monkeypatch):
    # Exercise the actual script import path, not merely the in-process class.
    monkeypatch.chdir(tmp_path);monkeypatch.delenv('PYTHONPATH', raising=False)
    with OpenCvButtonLightProcess() as worker:
        pid = worker.process.pid
        for frame in FRAMES:
            result = worker.observe(FIXTURES / frame['file'], frame['target'], frame['bbox'])
            assert result['state'] == frame['expected']
            assert worker.process.pid == pid


def test_numeric_label_uses_100_threshold_without_arrow_geometry(tmp_path):
    # This is an unlit DOWN reflection relabelled as "1" to check routing,
    # not a true illuminated numeral. Its local contrast is about76.6.
    frame = FRAMES[0]
    result = RelativeLuminance().predict(FIXTURES / frame['file'], '1', frame['bbox'])
    assert result['state'] == result['legacy_state'] == 'OFF'
    assert result['local_on_contrast_threshold'] == 100
    assert 'arrow_highlight_guard' not in result


def test_explicit_legacy_algorithm_is_unchanged():
    frame = FRAMES[0]
    result = RelativeLuminance(algorithm='legacy').predict(FIXTURES / frame['file'], 'down', frame['bbox'])
    assert result['state'] == result['legacy_state'] == 'OFF'
    assert 'arrow_highlight_guard' not in result


@pytest.mark.parametrize('size', [14, 20, 27])
def test_resolution_insufficient_to_verify_arrow_remains_uncertain(size):
    frame = FRAMES[0];gray, mask, median = frame_gray(frame)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    state, audit = resolve_arrow_highlight(gray, mask, 'down', median, 'OFF')
    assert state == 'UNCERTAIN' and audit['reason'] == 'arrow_geometry_unavailable'


@pytest.mark.parametrize('target', ['up', 'down'])
def test_bright_bar_without_resolved_arrow_is_not_off(target):
    gray = np.full((64, 64), 30, np.uint8)
    gray[16:48, 38:41] = 255
    mask = gray > 200
    state, audit = resolve_arrow_highlight(gray, mask, target, 100., 'OFF')
    assert state == 'UNCERTAIN' and audit['reason'] == 'arrow_diagonals_not_resolved'


def test_gray_zone_overall_result_is_not_promoted_to_off():
    gray, mask, median = frame_gray(FRAMES[0])
    state, _ = resolve_arrow_highlight(gray, mask, 'down', median, 'UNCERTAIN')
    assert state == 'UNCERTAIN'


def test_flipped_up_arrow_can_exclude_the_same_decorative_stripe():
    gray, mask, median = frame_gray(FRAMES[0])
    state, audit = resolve_arrow_highlight(np.flipud(gray), np.flipud(mask), 'up', median, 'OFF')
    assert state == 'OFF' and audit['reason'] == 'isolated_stripe_outside_unlit_arrow'


def test_only_one_bright_diagonal_does_not_authorize_off():
    frame = next(f for f in FRAMES if f['target'] == 'down' and f['expected'] == 'ON')
    gray, mask, median = frame_gray(frame)
    gray[:, :gray.shape[1] // 2] = 20
    mask[:] = False
    height, width = gray.shape
    mask[int(.28 * height):int(.68 * height), int(.6 * width):int(.6 * width) + 2] = True
    state, _ = resolve_arrow_highlight(gray, mask, 'down', median, 'OFF')
    assert state == 'UNCERTAIN'


def test_uniform_overexposure_cannot_establish_an_unlit_arrow():
    gray, mask, median = frame_gray(FRAMES[0]);gray[:] = 255
    state, _ = resolve_arrow_highlight(gray, mask, 'down', median, 'OFF')
    assert state == 'UNCERTAIN'


def test_reflection_on_arrow_cannot_be_excluded_as_external_stripe():
    frame = FRAMES[0];gray, mask, median = frame_gray(frame)
    # Move the highlight to the symbol's own central stem.
    mask[:] = False;mask[8:20, 16:18] = True
    state, _ = resolve_arrow_highlight(gray, mask, 'down', median, 'OFF')
    assert state == 'UNCERTAIN'


@pytest.mark.parametrize('dx,dy', [(-1,0), (1,0), (0,-1), (0,1), (-2,1), (2,-1)])
def test_small_detector_jitter_never_reinstates_a_false_on(dx, dy):
    frame = FRAMES[0];box = deepcopy(frame['bbox'])
    box[0] += dx;box[2] += dx;box[1] += dy;box[3] += dy
    result = RelativeLuminance().predict(FIXTURES / frame['file'], 'down', box)
    assert result['state'] in {'OFF', 'UNCERTAIN'}


def test_fragmented_stripe_is_grouped_without_inventing_bright_pixels():
    frame = FRAMES[0];box = deepcopy(frame['bbox']);box[0] += 1;box[2] += 1
    result = RelativeLuminance().predict(FIXTURES / frame['file'], 'down', box)
    assert result['state'] == 'OFF'
    gray, mask, median = frame_gray({**frame, 'bbox': box})
    state, audit = resolve_arrow_highlight(gray, mask, 'down', median, 'OFF')
    assert state == 'OFF'
    highlights = audit['highlights']
    assert len(highlights) == 1
    assert highlights[0]['area_px'] == sum(r['area_px'] for r in result['local_regions'] if r['contrast'] >= 60.)


def test_true_initial_on_keeps_wait_without_pressing():
    frame = next(f for f in FRAMES if f['target'] == 'down' and f['expected'] == 'ON')
    on = RelativeLuminance().predict(FIXTURES / frame['file'], 'down', frame['bbox'])['state']
    scene = FakeHardware([on] * 3 + ['OFF'] * 3, raw_frames=True)
    result = ButtonLightFlow(scene).run()
    assert result.press_attempts == 0 and result.status == 'arrived_after_initial_on'


def test_real_reflection_off_can_supply_initial_streak_and_enter_original_press_flow(tmp_path):
    class Vision:
        sequence = 0
        def request(self, *_args, **_kwargs):
            frame = FRAMES[self.sequence % 6];self.sequence += 1
            return dict(fresh=True, frame_sequence=self.sequence, path=str(FIXTURES / frame['file']),
                image_size=[640, 360], capture_time_unix_sec=time.time(),
                target_detection=dict(frame_sequence=self.sequence, bbox_xyxy=frame['bbox'], confidence=.94))
    observer = FreshLightObserver(Vision(), SimpleNamespace(observe=RelativeLuminance().predict), 'down',
                                  initial_light_guard=InitialLightGuard('down', tmp_path / 'evidence'))
    states = [observer.observe('waiting') for _ in range(3)]
    assert states == ['OFF'] * 3
    # Real image observation feeds the existing state machine; FakeHardware
    # records requested actions only and cannot reach a physical device.
    scene = FakeHardware(states + ['ON'] * 3 + ['OFF'] * 3, ['ON', 'ON'], raw_frames=True)
    result = ButtonLightFlow(scene).run()
    assert result.status == 'arrived_after_verified_press' and result.press_attempts == 1
    assert len([action for action in scene.actions if action[0] == 'press']) == 1
