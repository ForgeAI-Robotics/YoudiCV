"""Numeric tasks must acquire OFF at the exact saved post-press camera pose."""
import ast
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

import run_button_light_closed_loop as runner
from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.paired_button_light import PairedButtonLight
from tests.test_paired_button_light import patch_image, snapshot
from tests.test_up_stationary_light_reference import executor_fixture


@pytest.mark.parametrize('label', [*map(str, range(1, 17)), 'up', 'down'])
def test_supported_buttons_require_stationary_reference(label):
    assert runner.requires_stationary_press_reference(label)
    assert runner.is_numeric_button(label) == (label not in ('up', 'down'))


@pytest.mark.parametrize('label', ['0', '17', '2.0', 'open', '', None, True])
def test_invalid_targets_do_not_enter_numeric_reference_policy(label):
    assert not runner.is_numeric_button(label)
    assert not runner.requires_stationary_press_reference(label)


@pytest.mark.parametrize('label', ['1', '2', '4', '16'])
def test_numeric_reference_requires_two_new_stationary_off_frames(tmp_path, label):
    class Vision:
        n = 0
        def request(self, *a, **kw):
            self.n += 1
            patch = patch_image(self.n)
            if self.n == 1:
                patch.captured -= 10
            return snapshot(tmp_path, patch)
    paired = PairedButtonLight(tmp_path/'evidence', require_stationary_reference=True)
    paired.begin_attempt()
    obs = FreshLightObserver(Vision(), NS(observe=lambda *a, **kw: {'state': 'OFF'}), label, paired_light=paired)
    result = obs.capture_stationary_off_reference(timeout_sec=1)
    assert result['sequences'] == [2, 3]
    assert paired.stationary_reference_ready
    assert all(p.captured >= result['capture_barrier_unix'] for p in paired.references)
    paired.freeze()
    current = patch_image(4, increase=80)
    current.captured = max(time.time(), paired.press_started + .001)
    assert paired.classify(snapshot(tmp_path, current), 'ON')['state'] == 'ON'


@pytest.mark.parametrize('when', ['before_capture', 'during_capture'])
def test_nonfinite_encoder_reference_never_starts_press(when):
    executor, trace, joints = executor_fixture()
    if when == 'before_capture':
        joints[2] = np.nan
    def capture():
        if when == 'during_capture':
            joints[2] = np.nan
        return {}
    with pytest.raises(RuntimeError, match='reference|observation pose'):
        executor.press_with_stationary_reference(capture, already_at_observation=True)
    assert 'press' not in trace and 'rearm' not in trace


@pytest.mark.parametrize('label', ['2', '4', '16'])
def test_numeric_post_press_stationary_check_uses_saved_observation_endpoint(label):
    tree = ast.parse(Path(runner.__file__).read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'hold_at_prepress_if_visible')
    vision = Mock()
    namespace = dict(vars(runner), args=NS(target_label=label), vision=vision)
    exec(compile(ast.Module(body=[function], type_ignores=[]), runner.__file__, 'exec'), namespace)
    assert namespace['hold_at_prepress_if_visible']() is False
    vision.request.assert_not_called()
