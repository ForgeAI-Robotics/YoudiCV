import numpy as np
import pytest
import time
from types import SimpleNamespace

import piper_handeye.depth_safety as depth_safety_module
from piper_handeye.depth_matrix_safety import DepthMatrixReference

from piper_handeye.depth_safety import (
    DepthSafetyConfig,
    DepthSafetyReference,
    DepthPanelSafetyMonitor,
    evaluate_panel_foreground,
    compensate_reference_for_camera_motion,
)


def reference():
    return DepthSafetyReference(
        sequence=10,
        received_monotonic=1.0,
        bbox_xyxy=(80.0, 40.0, 120.0, 80.0),
        panel_depth_m=0.50,
    )


CAMERA_MATRIX = np.array(
    [[100.0, 0.0, 100.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]
)


def tracked_monitor(reference_age_sec=1.0):
    monitor = DepthPanelSafetyMonitor(
        None, lambda: None, DepthSafetyConfig(), camera_matrix=CAMERA_MATRIX
    )
    monitor._target_label = "4"
    monitor._reference = DepthSafetyReference(
        sequence=10,
        received_monotonic=time.monotonic() - reference_age_sec,
        bbox_xyxy=(80.0, 40.0, 120.0, 80.0),
        panel_depth_m=0.50,
        base_to_panel_m=(0.0, 0.0, 0.50),
    )
    monitor.update_camera_pose(time.monotonic(), np.eye(4))
    return monitor


def test_camera_forward_motion_offsets_depth_and_reprojects_bbox():
    ref = DepthSafetyReference(
        sequence=10,
        received_monotonic=1.0,
        bbox_xyxy=(80.0, 40.0, 120.0, 80.0),
        panel_depth_m=0.50,
        base_to_panel_m=(0.0, 0.0, 0.50),
    )
    current = np.eye(4)
    current[2, 3] = 0.10
    adjusted, audit = compensate_reference_for_camera_motion(
        ref, current, CAMERA_MATRIX
    )
    assert np.isclose(adjusted.panel_depth_m, 0.40)
    assert np.isclose(audit["motion_depth_offset_m"], -0.10)
    assert np.allclose(adjusted.bbox_xyxy, (75.0, 35.0, 125.0, 85.0))


def test_clear_panel_is_not_blocked():
    depth = np.full((120, 200), 0.50, dtype=np.float32)
    result = evaluate_panel_foreground(depth, reference(), DepthSafetyConfig())
    assert not result["blocked_raw"]
    assert result["foreground_pixels"] == 0


def test_probe_wider_roi_does_not_mutate_active_config():
    monitor = DepthPanelSafetyMonitor(None, lambda: None, DepthSafetyConfig())
    monitor._reference = reference()
    frame = SimpleNamespace(depth_m=np.full((120, 200), 0.50, dtype=np.float32))
    probes = monitor.probe_roi_scales(frame, [3.2, 7.0])
    assert [item["roi_scale"] for item in probes] == [3.2, 7.0]
    assert probes[1]["valid_pixels"] > probes[0]["valid_pixels"]
    assert monitor.config.roi_scale == 3.2


def test_foreground_hand_sized_patch_is_blocked():
    depth = np.full((120, 200), 0.50, dtype=np.float32)
    depth[20:100, 45:155] = 0.30
    result = evaluate_panel_foreground(depth, reference(), DepthSafetyConfig())
    assert result["blocked_raw"]
    assert result["foreground_fraction"] > 0.08


def test_pressing_tip_inside_center_exclusion_is_ignored():
    depth = np.full((120, 200), 0.50, dtype=np.float32)
    depth[42:78, 82:118] = 0.20
    result = evaluate_panel_foreground(depth, reference(), DepthSafetyConfig())
    assert not result["blocked_raw"]


def test_partial_hand_entering_outside_tool_corridor_is_detected():
    depth = np.full((120, 200), 0.50, dtype=np.float32)
    depth[20:100, 45:75] = 0.30
    result = evaluate_panel_foreground(depth, reference(), DepthSafetyConfig())
    assert result["foreground_pixels"] >= 120
    assert result["blocked_raw"]
    assert result["foreground_bbox_xyxy"] == [45, 20, 75, 100]
    assert result["foreground_region_pixels"]["left_of_tool"] > 0


def test_small_depth_speckle_does_not_trip_guard():
    depth = np.full((120, 200), 0.50, dtype=np.float32)
    depth[15:20, 45:50] = 0.20
    result = evaluate_panel_foreground(depth, reference(), DepthSafetyConfig())
    assert not result["blocked_raw"]


def test_sparse_full_frame_noise_does_not_form_blocking_component():
    depth = np.full((120, 200), 0.50, dtype=np.float32)
    depth[::4, ::4] = 0.20
    config = DepthSafetyConfig(roi_scale=20.0)
    result = evaluate_panel_foreground(depth, reference(), config)
    assert result["foreground_pixels"] >= config.min_foreground_pixels
    assert result["largest_foreground_component_pixels"] == 1
    assert not result["blocked_raw"]


def test_small_coherent_obstacle_trips_in_full_frame_roi():
    depth = np.full((120, 200), 0.50, dtype=np.float32)
    depth[90:102, 155:170] = 0.20
    config = DepthSafetyConfig(roi_scale=20.0)
    result = evaluate_panel_foreground(depth, reference(), config)
    assert result["foreground_fraction"] < 0.04
    assert result["largest_foreground_component_pixels"] == 180
    assert result["blocked_raw"]


def test_partial_hand_above_four_percent_trips_guard():
    depth = np.full((120, 200), 0.50, dtype=np.float32)
    # Enter from above/left of the central tool corridor so more than four
    # percent of the still-monitored ring is occupied.
    depth[15:35, 45:105] = 0.30
    result = evaluate_panel_foreground(depth, reference(), DepthSafetyConfig())
    assert result["foreground_pixels"] >= 120
    assert result["foreground_fraction"] >= 0.04
    assert result["blocked_raw"]


def test_stale_target_fails_safe_instead_of_allowing_blind_motion():
    monitor = DepthPanelSafetyMonitor(None, lambda: None, DepthSafetyConfig())
    monitor._target_label = "4"
    monitor._reference = DepthSafetyReference(
        sequence=10,
        received_monotonic=time.monotonic() - 10.0,
        bbox_xyxy=(80.0, 40.0, 120.0, 80.0),
        panel_depth_m=0.50,
    )
    monitor._latched_blocked = False
    for sequence in range(11, 14):
        frame = SimpleNamespace(
            sequence=sequence,
            received_monotonic=time.monotonic(),
            depth_m=np.full((120, 200), 0.50, dtype=np.float32),
        )
        monitor._process(frame)
    status = monitor.status()
    assert not status["ready"]
    assert status["blocked"]
    assert not status["monitoring_available"]
    assert status["reason"] == "target_reference_stale"


def test_paused_monitor_cannot_clear_latch_from_stale_camera_roi():
    monitor = DepthPanelSafetyMonitor(None, lambda: None, DepthSafetyConfig())
    monitor._target_label = "4"
    monitor._reference = DepthSafetyReference(
        sequence=10,
        received_monotonic=time.monotonic() - 1.0,
        bbox_xyxy=(80.0, 40.0, 120.0, 80.0),
        panel_depth_m=0.50,
    )
    monitor._latched_blocked = True
    monitor.set_motion_paused(True)
    clear_depth = np.full((120, 200), 0.50, dtype=np.float32)
    for sequence in range(11, 31):
        monitor._process(
            SimpleNamespace(
                sequence=sequence,
                received_monotonic=time.monotonic(),
                depth_m=clear_depth,
            )
        )
    status = monitor.status()
    assert not status["ready"]
    assert status["blocked"]
    assert status["clear_streak"] == 0
    assert status["reason"] == "target_reference_stale"


def test_bounded_cached_ring_allows_short_clear_tool_occlusion():
    monitor = tracked_monitor()
    monitor._latched_blocked = False
    monitor._process(
        SimpleNamespace(
            sequence=11,
            received_monotonic=time.monotonic(),
            depth_m=np.full((120, 200), 0.50, dtype=np.float32),
        )
    )
    status = monitor.status()
    assert status["ready"]
    assert not status["blocked"]
    assert status["coverage_mode"] == "kinematic_reference_tracking"
    assert status["clear_streak"] == 0


def test_hand_in_cached_ring_still_latches_during_blind_grace():
    monitor = tracked_monitor()
    monitor._latched_blocked = False
    blocked_depth = np.full((120, 200), 0.50, dtype=np.float32)
    blocked_depth[20:100, 45:155] = 0.30
    for sequence in range(11, 14):
        monitor._process(
            SimpleNamespace(
                sequence=sequence,
                received_monotonic=time.monotonic(),
                depth_m=blocked_depth,
            )
        )
    status = monitor.status()
    assert status["blocked"]
    assert status["reason"] == "unexpected_foreground"


def test_guard_requires_full_clear_streak_before_releasing_obstacle():
    config = DepthSafetyConfig(blocked_frames=3, clear_frames=15)
    monitor = DepthPanelSafetyMonitor(None, lambda: None, config)
    monitor._target_label = "4"
    monitor._reference = DepthSafetyReference(
        sequence=10,
        received_monotonic=time.monotonic(),
        bbox_xyxy=(80.0, 40.0, 120.0, 80.0),
        panel_depth_m=0.50,
    )
    blocked_depth = np.full((120, 200), 0.50, dtype=np.float32)
    blocked_depth[20:100, 45:155] = 0.30
    clear_depth = np.full((120, 200), 0.50, dtype=np.float32)

    for sequence in range(11, 14):
        monitor._process(
            SimpleNamespace(
                sequence=sequence,
                received_monotonic=time.monotonic(),
                depth_m=blocked_depth,
            )
        )
    assert monitor.status()["blocked"]

    for sequence in range(14, 28):
        monitor._process(
            SimpleNamespace(
                sequence=sequence,
                received_monotonic=time.monotonic(),
                depth_m=clear_depth,
            )
        )
    assert monitor.status()["blocked"]
    monitor._process(
        SimpleNamespace(
            sequence=28,
            received_monotonic=time.monotonic(),
            depth_m=clear_depth,
        )
    )
    assert not monitor.status()["blocked"]
    assert monitor.status()["last_obstruction"]["reason"] == "unexpected_foreground"


def matrix_monitor(monkeypatch, metrics):
    config = DepthSafetyConfig(
        matrix_downsample_factor=1,
        matrix_blocked_frames=2,
        matrix_unavailable_frames=2,
    )
    monitor = DepthPanelSafetyMonitor(
        None, lambda: None, config, camera_matrix=np.eye(3)
    )
    monitor._target_label = "15"
    monitor._latched_blocked = False
    monitor._matrix_reference = DepthMatrixReference(
        depth_m=np.full((4, 4), 0.20),
        camera_matrix=np.eye(3),
        base_points_m=np.zeros((1, 3)),
        tool_mask=np.zeros((4, 4), dtype=bool),
        tool_depth_m=np.full((4, 4), np.nan),
        reference_base_to_camera=np.eye(4),
    )
    monitor.update_camera_pose(time.monotonic(), np.eye(4))
    monkeypatch.setattr(
        depth_safety_module,
        "predict_depth_matrix",
        lambda *args, **kwargs: np.full((4, 4), 0.20),
    )
    monkeypatch.setattr(
        depth_safety_module,
        "predict_panel_plane_depth",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        depth_safety_module,
        "evaluate_depth_matrix_residual",
        lambda *args, **kwargs: {
            **metrics,
            "_debug": {
                "actual_depth_m": np.full((4, 4), np.nan),
                "predicted_depth_m": np.full((4, 4), 0.20),
                "expected_depth_m": np.full((4, 4), 0.20),
                "unexpected_mask": np.zeros((4, 4), dtype=bool),
            },
        },
    )
    monkeypatch.setattr(
        depth_safety_module,
        "evaluate_rgb_frame_intrusion",
        lambda *args, **kwargs: {
            "rgb_intrusion_available": True,
            "rgb_registration_score": 1.0,
            "rgb_difference_threshold": 1.0,
            "rgb_changed_pixels": 0,
            "rgb_largest_component_pixels": 0,
            "rgb_intrusion_raw": False,
            "_rgb_debug_mask": np.zeros((4, 4), dtype=bool),
        },
    )
    monitor._previous_color = np.zeros((4, 4, 3), dtype=np.uint8)
    return monitor


def matrix_frame(sequence):
    return SimpleNamespace(
        sequence=sequence,
        received_monotonic=time.monotonic(),
        received_unix=time.time(),
        depth_m=np.full((4, 4), np.nan),
        color=np.zeros((4, 4, 3), dtype=np.uint8),
    )


def test_recovered_monitor_failure_is_not_mislabeled_as_depth_residual(monkeypatch):
    monitor = matrix_monitor(
        monkeypatch,
        {
            "matrix_monitoring_available": True,
            "blocked_raw": False,
            "coverage_fraction": 0.90,
        },
    )
    for sequence in (8, 9, 10):
        monitor._set_failure(sequence, "matrix_camera_pose_stale")
    monitor._process(matrix_frame(11))
    status = monitor.status()
    assert status["blocked"]
    assert status["reason"] == "matrix_monitor_failure_latched"
    assert status["last_obstruction"]["reason"] == "matrix_camera_pose_stale"


@pytest.mark.parametrize('difference,stale', [(.34, False), (.36, True), (-.36, True)])
def test_pose_timing_keeps_signed_difference_and_original_350ms_protection(monkeypatch, difference, stale):
    monitor = matrix_monitor(monkeypatch, dict(matrix_monitoring_available=True,
                                               blocked_raw=False, coverage_fraction=.9))
    monkeypatch.setattr(depth_safety_module, 'time', SimpleNamespace(monotonic=lambda: 100.))
    monitor._latest_camera_pose = (100.-difference, np.eye(4))
    for seq in (21, 22, 23):
        frame = SimpleNamespace(sequence=seq, received_monotonic=100., received_unix=200.,
                                depth_m=np.full((4, 4), .2), color=np.zeros((4, 4, 3), dtype=np.uint8))
        monitor._process(frame)
    status = monitor.status()
    assert status['blocked'] is stale
    last = monitor.pose_timing_evidence()['frames']['recent'][-1]
    assert last['frame_minus_pose_sec'] == pytest.approx(difference)
    assert last['pose_frame_max_difference_sec'] == .35
    if stale:
        assert status['last_obstruction']['reason'] == 'matrix_camera_pose_stale'
        assert status['last_obstruction']['pose_timing']['frame_minus_pose_sec'] == pytest.approx(difference)
    else:
        assert status['reason'] == 'clear'


def test_optional_snapshot_arrays_remain_none_in_async_copy(monkeypatch, tmp_path):
    monitor = DepthPanelSafetyMonitor(
        None,
        lambda: None,
        DepthSafetyConfig(hazard_snapshot_directory=str(tmp_path)),
    )
    received = {}

    def fake_save(*, frame, debug, status):
        received.update(debug)

    monkeypatch.setattr(monitor, "_save_hazard_snapshot", fake_save)
    monitor._schedule_hazard_snapshot(
        matrix_frame(12),
        {
            "actual_depth_m": np.full((4, 4), 0.2),
            "predicted_depth_m": None,
            "expected_depth_m": None,
            "unexpected_mask": np.zeros((4, 4), dtype=bool),
        },
        {"reason": "test"},
    )
    deadline = time.monotonic() + 0.5
    while monitor.status()["hazard_snapshot_pending"] and time.monotonic() < deadline:
        time.sleep(0.001)
    assert received["predicted_depth_m"] is None
    assert received["expected_depth_m"] is None
    assert monitor.status()["hazard_snapshot_error"] is None


def test_near_mode_disables_depth_and_rgb_change_detection_by_default(monkeypatch):
    monitor = matrix_monitor(
        monkeypatch,
        {
            "matrix_monitoring_available": False,
            "blocked_raw": False,
            "coverage_fraction": 0.03,
        },
    )
    monitor.set_depth_degraded_commit(True)
    monitor._process(matrix_frame(11))
    assert monitor.status()["depth_degraded_commit_armed"]
    assert monitor.status()["depth_degraded_commit"]
    status = monitor.status()
    assert status["ready"]
    assert not status["blocked"]
    assert status["monitoring_available"]
    assert status["depth_degraded_commit"]
    assert status["reason"] == "near_panel_depth_and_rgb_change_suppressed"
    assert status["coverage_mode"] == "near_rgb_change_suppressed"
    assert status["depth_monitoring_suppressed"]
    assert status["rgb_change_detection_suppressed"]


def test_near_mode_ignores_depth_residual_with_rgb_change_suppressed(monkeypatch):
    monitor = matrix_monitor(
        monkeypatch,
        {
            "matrix_monitoring_available": False,
            "blocked_raw": False,
            "matrix_residual_detected_raw": True,
            "matrix_largest_component_pixels": 500,
            "coverage_fraction": 0.03,
        },
    )
    monitor.set_depth_degraded_commit(True)
    monitor._process(matrix_frame(11))
    monitor._process(matrix_frame(12))
    status = monitor.status()
    assert not status["blocked"]
    assert status["reason"] == "near_panel_depth_and_rgb_change_suppressed"
    assert status["depth_degraded_commit"]


def test_near_mode_remains_active_until_explicitly_disabled(monkeypatch):
    metrics = {
        "matrix_monitoring_available": False,
        "blocked_raw": False,
        "matrix_residual_detected_raw": False,
        "coverage_fraction": 0.03,
    }
    monitor = matrix_monitor(monkeypatch, metrics)
    monitor.set_depth_degraded_commit(True)
    monitor._process(matrix_frame(11))
    monitor._process(matrix_frame(12))
    assert monitor.status()["depth_degraded_commit"]
    metrics["matrix_monitoring_available"] = True
    for sequence in (13, 14, 15):
        monitor._process(matrix_frame(sequence))
        assert monitor.status()["depth_degraded_commit"]
    monitor.set_depth_degraded_commit(False)
    assert not monitor.status()["depth_degraded_commit"]
    assert not monitor.status()["depth_degraded_commit_armed"]


def test_rgb_registration_is_not_run_while_depth_monitoring_is_available(monkeypatch):
    monitor = matrix_monitor(
        monkeypatch,
        {
            "matrix_monitoring_available": True,
            "blocked_raw": False,
            "coverage_fraction": 0.90,
        },
    )
    monitor._previous_color = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(
        depth_safety_module,
        "evaluate_rgb_frame_intrusion",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("RGB fallback ran while depth was healthy")
        ),
    )
    monitor._process(matrix_frame(11))
    assert monitor.status()["monitoring_available"]


def test_near_mode_ignores_rgb_frame_differences_by_default(monkeypatch):
    monitor = matrix_monitor(
        monkeypatch,
        {
            "matrix_monitoring_available": True,
            "blocked_raw": True,
            "coverage_fraction": 0.80,
        },
    )
    monkeypatch.setattr(
        depth_safety_module,
        "evaluate_rgb_frame_intrusion",
        lambda *args, **kwargs: {
            "rgb_intrusion_available": True,
            "rgb_registration_score": 0.9,
            "rgb_difference_threshold": 1.0,
            "rgb_changed_pixels": 600,
            "rgb_largest_component_pixels": 600,
            "rgb_intrusion_raw": True,
            "_rgb_debug_mask": np.ones((4, 4), dtype=bool),
        },
    )
    monitor.set_depth_degraded_commit(True)
    monitor._process(matrix_frame(11))
    monitor._process(matrix_frame(12))
    status = monitor.status()
    assert not status["blocked"]
    assert status["reason"] == "near_panel_depth_and_rgb_change_suppressed"
    assert status["rgb_change_detection_suppressed"] is True
    assert status["rgb_object_confirmed"] is False


def test_rgb_pause_snapshot_keeps_both_frames_and_overlay(tmp_path):
    pytest.importorskip("cv2")
    monitor = DepthPanelSafetyMonitor(
        None,
        lambda: None,
        DepthSafetyConfig(hazard_snapshot_directory=str(tmp_path)),
    )
    frame = matrix_frame(99)
    frame.color = np.zeros((24, 32, 3), dtype=np.uint8)
    previous = frame.color.copy()
    previous[5:12, 8:16] = (20, 30, 40)
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:6, 4:8] = True
    monitor._save_hazard_snapshot(
        frame=frame,
        debug={
            "actual_depth_m": np.full((4, 4), np.nan),
            "predicted_depth_m": None,
            "expected_depth_m": None,
            "unexpected_mask": np.zeros((4, 4), dtype=bool),
            "rgb_previous_bgr": previous,
            "rgb_current_bgr": frame.color,
            "rgb_intrusion_mask": mask,
        },
        status={"reason": "rgb_scene_change_unverified_near_panel"},
    )
    saved = next(tmp_path.iterdir())
    assert (saved / "rgb_current.png").is_file()
    assert (saved / "rgb_previous.png").is_file()
    assert (saved / "rgb_change_mask.png").is_file()
    assert (saved / "rgb_change_overlay.png").is_file()
