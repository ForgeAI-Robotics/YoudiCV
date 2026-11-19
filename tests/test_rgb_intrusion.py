import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from piper_handeye.depth_safety import evaluate_rgb_frame_intrusion


def scene_pair():
    rng = np.random.default_rng(4)
    reference = rng.integers(30, 220, (360, 640, 3), dtype=np.uint8)
    reference = cv2.GaussianBlur(reference, (0, 0), 3)
    transform = np.float32([[1.002, 0.0, 2.0], [0.0, 1.002, 1.0]])
    moved = cv2.warpAffine(
        reference, transform, (640, 360), borderMode=cv2.BORDER_REPLICATE
    )
    return reference, moved


def test_camera_motion_is_registered_without_false_intrusion():
    reference, moved = scene_pair()
    result = evaluate_rgb_frame_intrusion(reference, moved)
    assert result["rgb_registration_score"] > 0.95
    assert not result["rgb_intrusion_raw"]


def test_new_hand_sized_foreground_is_detected():
    reference, moved = scene_pair()
    moved[90:260, 250:390] = (20, 100, 190)
    result = evaluate_rgb_frame_intrusion(reference, moved)
    assert result["rgb_intrusion_raw"]
    assert result["rgb_largest_component_pixels"] > 1000


def test_global_exposure_change_is_not_an_intrusion():
    reference, moved = scene_pair()
    brighter = np.clip(moved.astype(np.int16) + 35, 0, 255).astype(np.uint8)
    result = evaluate_rgb_frame_intrusion(reference, brighter)
    assert not result["rgb_intrusion_raw"]
