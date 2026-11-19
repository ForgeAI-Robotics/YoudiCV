import numpy as np
import pytest
from piper_handeye.paired_button_light import compare_patches
from piper_handeye.assistant_gui_events import safety_pause_message, AssistantEventPresenter
from tests.test_paired_button_light import patch_image


@pytest.mark.parametrize('offset', [17, 19, 24, 40, -15])
@pytest.mark.parametrize('gain', [.75, 1., 1.4])
def test_lamp_increase_survives_compensated_brightness_and_contrast(offset, gain):
    before = patch_image()
    after = patch_image(2)
    after.gray = np.clip(gain * after.gray + offset, 0, 255)
    after.gray[25:39, 25:39] += 70
    result = compare_patches(before, after, 'ON')
    assert result['state'] == 'ON'
    assert result['background_gain'] == pytest.approx(gain, abs=.02)


@pytest.mark.parametrize('offset,gain', [(24, 1), (10, 1.5), (0, 1.8), (-10, .7)])
def test_unchanged_white_symbol_under_global_change_is_not_a_new_light(offset, gain):
    before = patch_image(increase=70)
    after = patch_image(2, increase=70)
    after.gray = np.clip(gain * after.gray + offset, 0, 255)
    assert compare_patches(before, after, 'ON')['state'] == 'UNCERTAIN'
    assert compare_patches(before, after, 'OFF')['state'] == 'OFF'


def test_flat_background_cannot_hide_gain_as_a_small_additive_offset():
    before = patch_image(); before.gray[:] = 20; before.gray[25:39,25:39] = 100
    after = patch_image(2); after.gray = before.gray * 1.4
    result = compare_patches(before, after, 'ON')
    assert result['exposure_offset'] == pytest.approx(8)
    assert result['state'] == 'UNCERTAIN'
    assert result['normalization_rejection'] == 'insufficient_background_texture'


@pytest.mark.parametrize('fault', ['clipped', 'shadow', 'textureless', 'gain', 'occluded'])
def test_unexplainable_background_cannot_authorize_press_success(fault):
    before = patch_image(); after = patch_image(2, increase=80)
    if fault == 'clipped': after.gray[:32] = 255
    if fault == 'shadow': after.gray[:, :32] += 50
    if fault == 'textureless':
        before.gray[:] = 40; after.gray[:] = 70; after.gray[25:39,25:39] = 160
    if fault == 'gain': after.gray = np.clip(after.gray * 3, 0, 255)
    if fault == 'occluded': after.gray[:] = 200
    result = compare_patches(before, after, 'ON')
    assert result['state'] == 'UNCERTAIN'
    assert result.get('normalization_rejection')


@pytest.mark.parametrize('evidence,text', [
    ({'reason':'matrix_monitor_failure_latched','underlying_reason':'matrix_camera_pose_stale'}, '相机位姿数据过期'),
    ({'reason':'matrix_camera_pose_stale'}, '相机位姿数据过期'),
    ({'reason':'matrix_monitor_failure_latched'}, '安全感知数据异常'),
    ({'reason':'matrix_prediction_coverage_insufficient'}, '安全感知数据异常'),
    ({'reason':'unexpected_depth_matrix_residual','underlying_reason':'matrix_camera_pose_stale'}, '检测到异物'),
    ({'reason':'rgb_scene_change_unverified_near_panel'}, '尚未确认异物'),
    ({}, '原因待核实'),
])
def test_safety_messages_distinguish_evidence_in_both_event_paths(evidence, text):
    message = safety_pause_message(evidence)
    assert text in message
    presenter = AssistantEventPresenter()
    result = presenter.consume({'phase':'visual_safety_paused','value':evidence})
    assert result[0].text == message
