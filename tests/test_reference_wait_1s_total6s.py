"""Exercise delayed RGB delivery through the actual collector without hardware."""
from types import SimpleNamespace
import time

import pytest

import piper_handeye.button_light_hardware as hardware
from piper_handeye.paired_button_light import PairedButtonLight
from tests.test_paired_button_light import patch_image, snapshot


def scheduled_observer(tmp_path, monkeypatch, arrivals, *, image_age=0., state='OFF'):
    class Clock:
        elapsed = 0.
        epoch = time.time()

        def monotonic(self):
            return self.elapsed

        def time(self):
            return self.epoch + self.elapsed

        def sleep(self, seconds):
            self.elapsed += seconds

    clock = Clock()
    monkeypatch.setattr(hardware, 'time', clock)

    class Vision:
        sequence = 0

        def request(self, *args, **kwargs):
            clock.sleep(.001)
            if self.sequence >= len(arrivals) or clock.elapsed < arrivals[self.sequence]:
                return {'fresh': False}
            self.sequence += 1
            frame = patch_image(self.sequence)
            frame.captured = clock.time() - image_age
            result = snapshot(tmp_path, frame)
            result['target_detection']['label'] = '2'
            return result

    paired = PairedButtonLight(require_stationary_reference=True)
    paired.begin_attempt()
    observer = hardware.FreshLightObserver(
        Vision(), SimpleNamespace(observe=lambda *a, **kw: {'state': state}),
        '2', paired_light=paired)
    return observer, paired, clock


def test_delayed_off_frames_400ms_apart_now_form_a_reference(tmp_path, monkeypatch):
    observer, paired, clock = scheduled_observer(tmp_path, monkeypatch, [.4, .8])
    result = observer.capture_stationary_off_reference()
    assert result['sequences'] == [1, 2]
    assert paired.stationary_reference_ready
    assert .8 <= clock.elapsed < 1.
    assert all(o['state'] == 'OFF' for o in observer.audit)


def test_first_good_pair_after_old_three_second_limit_is_accepted(tmp_path, monkeypatch):
    observer, paired, clock = scheduled_observer(tmp_path, monkeypatch, [3.4, 3.8])
    result = observer.capture_stationary_off_reference()
    assert result['sequences'] == [1, 2]
    assert paired.stationary_reference_ready
    assert 3.8 <= clock.elapsed < 4.


def test_missing_images_still_stop_at_total_ten_second_budget(tmp_path, monkeypatch):
    observer, paired, clock = scheduled_observer(tmp_path, monkeypatch, [])
    with pytest.raises(RuntimeError, match='press withheld'):
        observer.capture_stationary_off_reference()
    assert not paired.stationary_reference_ready and not paired.references
    assert 10. <= clock.elapsed < 10.03


@pytest.mark.parametrize('image_age, state', [(.36, 'OFF'), (0., 'ON'), (0., 'UNCERTAIN')])
def test_longer_wait_does_not_accept_old_or_non_off_images(tmp_path, monkeypatch, image_age, state):
    observer, paired, _ = scheduled_observer(
        tmp_path, monkeypatch, [.4, .8], image_age=image_age, state=state)
    with pytest.raises(RuntimeError, match='press withheld'):
        observer.capture_stationary_off_reference()
    assert not paired.stationary_reference_ready and not paired.references
    if image_age > .35:
        assert observer.audit
        assert all(o['state'] == 'UNCERTAIN' for o in observer.audit)


def test_explicit_short_collection_budget_still_caps_each_wait(tmp_path, monkeypatch):
    observer, paired, clock = scheduled_observer(tmp_path, monkeypatch, [.4, .8])
    with pytest.raises(RuntimeError, match='press withheld'):
        observer.capture_stationary_off_reference(timeout_sec=.1)
    assert not paired.references
    assert .1 <= clock.elapsed < .13
