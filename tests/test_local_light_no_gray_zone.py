"""Regression frames from the real down-button reference failure and lit buttons."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.button_light_process import OpenCvButtonLightProcess
from piper_handeye.paired_button_light import PairedButtonLight
from tools.opencv_light_worker import RelativeLuminance


FIXTURES = Path(__file__).parent / 'fixtures/local_light_no_gray_zone'
FRAMES = json.loads((FIXTURES / 'manifest.json').read_text())['fixtures']


@pytest.mark.parametrize('frame', FRAMES, ids=lambda f: f['file'])
def test_real_unlit_arrow_can_remain_off_and_real_lights_remain_on(frame):
    result = RelativeLuminance().predict(
        FIXTURES / frame['file'], frame['label'], frame['bbox'])
    assert result['state'] == frame['expected']
    assert result['local_uncertainty_enabled'] is False
    if frame['label'] == 'down':
        assert result['legacy_state'] == 'OFF'


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

    paired = PairedButtonLight(tmp_path, require_stationary_reference=True)
    paired.begin_attempt()
    classifier = RelativeLuminance()
    observer = FreshLightObserver(
        Vision(), SimpleNamespace(observe=classifier.predict), frames[0]['label'],
        paired_light=paired)
    return observer, paired


def test_failed_down_scene_now_supplies_two_stationary_off_references(tmp_path):
    observer, paired = reference_observer(FRAMES[:2], tmp_path)
    result = observer.capture_stationary_off_reference(timeout_sec=1)
    assert result['reference_count'] == 2
    assert result['sequences'][1] > result['sequences'][0] >= 1
    assert paired.stationary_reference_ready
    assert all(e['state'] == 'OFF' for e in observer.audit)
    paired.freeze()


def test_true_lit_button_still_cannot_supply_an_off_reference(tmp_path):
    observer, paired = reference_observer([FRAMES[2]], tmp_path)
    with pytest.raises(RuntimeError, match='press withheld'):
        observer.capture_stationary_off_reference(timeout_sec=.1)
    assert not paired.stationary_reference_ready and not paired.references


def test_missing_button_box_still_means_uncertain():
    result = RelativeLuminance().predict(FIXTURES / FRAMES[0]['file'], 'down')
    assert result['state'] == 'UNCERTAIN'
    assert result['reason'] == 'target_bbox_required'


def test_production_subprocess_loads_new_policy_and_reuses_worker():
    with OpenCvButtonLightProcess() as worker:
        pid = worker.process.pid
        for frame in FRAMES:
            result = worker.observe(FIXTURES / frame['file'], frame['label'], frame['bbox'])
            assert result['state'] == frame['expected']
            threshold = (100 if str(frame['label']).isdigit()
                         and 1 <= int(frame['label']) <= 16 else 85)
            assert result['local_on_contrast_threshold'] == threshold
            assert result['local_decision_policy'] == f'on_at_{threshold}_else_legacy'
            assert worker.process.pid == pid
