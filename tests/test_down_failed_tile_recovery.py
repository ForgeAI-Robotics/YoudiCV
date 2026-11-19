import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from piper_handeye.button_pose import ButtonDetection
from piper_handeye.call_button_detection import retry_call_button_tile
from piper_handeye.detection_enhancement import detect_with_enhancement_fallback

YY, XX = np.mgrid[:360, :640]
IMAGE = np.stack([XX, YY, XX*0], axis=2).astype(np.float32)
UP = (190, 90, 225, 125)
DOWN = (190, 135, 225, 170)


def det(label, score, box):
    return ButtonDetection(0, label, score, tuple(box))


class Detector:
    def __init__(self, *, recovered_score=None, competing=None, down_box=DOWN,
                 strong_open=False, up_score=.95):
        self.calls = []
        self.recovered_score = recovered_score
        self.competing = competing
        self.down_box = down_box
        self.raw = [det('up', up_score, UP)]
        if strong_open:
            self.raw.append(det('open', .95, down_box))

    def detect(self, image):
        if image.shape[:2] == (360, 640):
            self.calls.append('raw')
            return self.raw
        x, y = map(int, image[0, 0, :2])
        self.calls.append((x, y, image.shape[1], image.shape[0]))
        result = list(self.raw[:1])
        if (x, y) != (0, 0) and self.recovered_score is not None:
            result.append(det('down', self.recovered_score, self.down_box))
            if self.competing:
                result.append(det(self.competing[0], self.competing[1], self.down_box))
        return [det(d.label, d.confidence, (d.bbox_xyxy[0]-x, d.bbox_xyxy[1]-y,
                                           d.bbox_xyxy[2]-x, d.bbox_xyxy[3]-y)) for d in result]


def frame(detector, now, *, target='down', retry=retry_call_button_tile):
    with patch('piper_handeye.down_button_recovery.time', SimpleNamespace(monotonic=lambda: now)):
        return retry(detector, IMAGE, detector.raw, target, .70)


def test_only_up_does_not_cache_down_tile_or_trap_the_next_frame():
    detector = Detector()
    audits = [frame(detector, i*.1)[1] for i in range(5)]
    assert audits[0]['tile_xyxy'] == audits[1]['tile_xyxy'] == [0, 0, 320, 320]
    assert len({tuple(a['tile_xyxy']) for a in audits[2:]}) == 3
    assert all(not a['down_recovery']['target_accepted'] for a in audits)
    assert detector._call_tile_hints == {}
    assert all(a['accepted'] for a in audits)  # UP remains a search cue, not DOWN.


def test_two_misses_then_current_alternative_down_recovers_and_caches_good_tile():
    detector = Detector(recovered_score=.96)
    for i in range(2):
        output, audit = frame(detector, i*.1)
        assert not any(d.label == 'down' for d in output)
    output, audit = frame(detector, .2)
    assert audit['down_recovery']['target_accepted']
    assert audit['down_recovery']['used_alternative']
    assert next(d for d in output if d.label == 'down').bbox_xyxy == DOWN
    assert detector._call_tile_hints[(640, 360, 'down')] == tuple(audit['tile_xyxy'])
    # Re-infer this same region on the next new frame, never reuse its box.
    detector.recovered_score = None
    output, new_audit = frame(detector, .3)
    assert new_audit['tile_xyxy'] == audit['tile_xyxy']
    assert not any(d.label == 'down' for d in output)
    assert len(detector.calls) == 4


@pytest.mark.parametrize('score', [.15, .50, .699])
def test_recovery_does_not_lower_confidence_threshold(score):
    detector = Detector(recovered_score=score)
    for i in range(7):
        output, audit = frame(detector, i*.1)
        assert not any(d.label == 'down' for d in output)
        assert not audit['down_recovery']['target_accepted']


def test_strong_open_conflict_remains_rejected_not_relabelled():
    detector = Detector(recovered_score=.99, strong_open=True)
    for i in range(5):
        output, audit = frame(detector, i*.1)
        assert not any(d.label == 'down' for d in output)
        assert any(d.label == 'open' for d in output)
    assert audit['reason'] == 'conflicting_call_labels'


@pytest.mark.parametrize('label', ['open', 'M', 'up'])
def test_new_recovery_rejects_same_location_competing_class(label):
    detector = Detector(recovered_score=.96, competing=(label, .90))
    for i in range(5):
        output, audit = frame(detector, i*.1)
        assert not any(d.label == 'down' for d in output)


def test_alternative_arrow_far_from_real_up_cannot_become_panel_target():
    detector = Detector(recovered_score=.99, down_box=(70, 230, 105, 265))
    for i in range(5):
        output, _ = frame(detector, i*.1)
        assert not any(d.label == 'down' for d in output)


def test_clipped_alternative_down_is_not_a_complete_target():
    detector = Detector(recovered_score=.99, down_box=(190, 333, 225, 359))
    for i in range(5):
        output, audit = frame(detector, i*.1)
        assert not audit['down_recovery']['target_accepted']


def test_missing_strong_up_support_does_not_authorize_alternative_down():
    detector = Detector(recovered_score=.99, up_score=.30)
    for i in range(5):
        output, _ = frame(detector, i*.1)
        assert not any(d.label == 'down' for d in output)


def test_existing_good_raw_down_never_performs_local_retry():
    detector = Detector()
    detector.raw = [det('down', .90, DOWN)]
    output, audit = frame(detector, 0.)
    assert output == detector.raw and not audit['attempted'] and not detector.calls


def test_existing_good_original_tile_retains_same_region_without_recovery():
    detector = Detector()
    detector.detect = lambda image: [det('up', .95, UP), det('down', .91, DOWN)]
    for i in range(5):
        output, audit = frame(detector, i*.1)
        assert audit['tile_xyxy'] == [0, 0, 320, 320]
        assert not audit['down_recovery']['used_alternative']
        assert audit['down_recovery']['target_accepted']
        assert next(d for d in output if d.label == 'down').confidence == .91


def test_camera_move_or_idle_resets_old_recovery_cycle():
    for kind in ['move', 'idle']:
        detector = Detector()
        for i in range(3):
            frame(detector, i*.1)
        if kind == 'move':
            detector.raw = [det('up', .95, (190, 115, 225, 150))]
        _, audit = frame(detector, 2. if kind == 'idle' else .3)
        assert not audit['down_recovery']['used_alternative']
        assert audit['down_recovery']['consecutive_misses'] == 1


def test_switch_to_up_discards_down_schedule_without_affecting_up_output():
    detector = Detector()
    for i in range(3):
        frame(detector, i*.1)
    output, audit = frame(detector, .3, target='up')
    assert output == detector.raw and not audit['attempted']
    assert detector._down_retry_tiles is None


def test_no_extra_inference_or_clahe_added_to_each_frame():
    detector = Detector(recovered_score=.96)
    with patch('piper_handeye.detection_enhancement.enhance_detection_image') as enhance:
        for _ in range(4):
            count = len(detector.calls)
            detect_with_enhancement_fallback(detector, IMAGE, 'down', .70)
            assert len(detector.calls)-count == 2
    enhance.assert_not_called()
    assert np.array_equal(IMAGE[:, :, 0], XX)


def test_up_retry_outputs_match_frozen_baseline_across_cached_and_uncached_tiles():
    path = Path(__file__).parent/'fixtures/down_failure_fallback/legacy_call_button_detection.py'
    spec = importlib.util.spec_from_file_location('piper_handeye._frozen_up_retry', path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    class Scripted:
        def __init__(self):self.n = 0;self.regions = []
        def detect(self, image):
            self.regions.append(image.copy());self.n += 1
            return [] if self.n % 2 else [det('up', .95, (20, 40, 55, 75))]
    left, right = Scripted(), Scripted()
    for _ in range(8):
        old, a = legacy.retry_call_button_tile(left, IMAGE, [], 'up', .70)
        new, b = retry_call_button_tile(right, IMAGE, [], 'up', .70)
        assert old == new
        assert {k:v for k,v in a.items() if k != 'retry_inference_sec'} == {
            k:v for k,v in b.items() if k != 'retry_inference_sec'}
    assert all(np.array_equal(a, b) for a, b in zip(left.regions, right.regions))
