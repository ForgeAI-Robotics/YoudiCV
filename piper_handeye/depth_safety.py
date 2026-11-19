from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np
import yaml

from .retention import prune_old_entries
from .pose_timing import PoseTimingTrace
from scipy import ndimage

from .depth_matrix_safety import (
    DepthMatrixReference,
    build_depth_matrix_reference,
    evaluate_absolute_near_field,
    evaluate_depth_matrix_residual,
    predict_panel_plane_depth,
    predict_depth_matrix,
    downsample_depth_matrix,
)


@dataclass(frozen=True)
class DepthSafetyConfig:
    roi_scale: float = 3.2
    center_exclusion_scale: float = 1.15
    foreground_margin_m: float = 0.06
    min_foreground_pixels: int = 120
    min_foreground_fraction: float = 0.04
    blocked_frames: int = 3
    clear_frames: int = 15
    reference_max_age_sec: float = 0.40
    reference_blind_grace_sec: float = 8.00
    camera_pose_max_age_sec: float = 0.35
    frame_max_age_sec: float = 0.25
    matrix_min_coverage_fraction: float = 0.35
    matrix_splat_radius: int = 2
    matrix_border_ignore_px: int = 20
    matrix_alignment_tolerance_px: int = 3
    matrix_tool_ignore_dilation_px: int = 3
    matrix_predicted_edge_ignore_dilation_px: int = 0
    matrix_fixed_tool_roi_normalized: tuple[float, float, float, float] | None = None
    matrix_panel_plane_tolerance_m: float = 0.03
    matrix_reaction_max_depth_m: float = 0.30
    matrix_downsample_factor: int = 2
    matrix_min_component_pixels: int = 30
    matrix_degraded_min_component_pixels: int = 300
    matrix_blocked_frames: int = 2
    matrix_unavailable_frames: int = 3
    matrix_recovered_frames: int = 3
    rgb_intrusion_min_component_pixels: int = 450
    rgb_intrusion_blocked_frames: int = 2
    rgb_intrusion_min_diff: int = 22
    # The close-range depth pause must not turn panel/button appearance changes
    # into obstacle stops. Enable only for setups with a separately validated
    # RGB obstacle classifier; raw frame differencing is disabled by default.
    near_rgb_change_detection_enabled: bool = False
    hazard_snapshot_directory: str | None = None


def camera_fixed_tool_ignore_mask(
    shape: tuple[int, int],
    learned_mask: np.ndarray,
    normalized_roi: tuple[float, float, float, float] | None,
    dilation_px: int,
) -> np.ndarray:
    """Combine the learned tool silhouette with a tight camera-rigid tip ROI."""
    learned = np.asarray(learned_mask, dtype=bool)
    if learned.shape != tuple(shape):
        raise ValueError("learned tool mask shape mismatch")
    result = learned.copy()
    if normalized_roi is not None:
        values = tuple(float(value) for value in normalized_roi)
        if len(values) != 4 or not (
            0.0 <= values[0] < values[2] <= 1.0
            and 0.0 <= values[1] < values[3] <= 1.0
        ):
            raise ValueError("fixed tool ROI must be normalized xyxy in 0..1")
        height, width = shape
        x1 = max(0, min(width, int(np.floor(values[0] * width))))
        y1 = max(0, min(height, int(np.floor(values[1] * height))))
        x2 = max(0, min(width, int(np.ceil(values[2] * width))))
        y2 = max(0, min(height, int(np.ceil(values[3] * height))))
        result[y1:y2, x1:x2] = True
    iterations = int(dilation_px)
    return (
        ndimage.binary_dilation(result, iterations=iterations)
        if iterations > 0 else result
    )


def evaluate_rgb_frame_intrusion(
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
    *,
    min_component_pixels: int = 450,
    min_diff: int = 22,
    border_ignore_px: int = 20,
) -> dict:
    """Detect a new coherent RGB foreground after compensating camera motion."""
    import cv2
    previous = cv2.resize(previous_bgr, (320, 180), interpolation=cv2.INTER_AREA)
    current = cv2.resize(current_bgr, (320, 180), interpolation=cv2.INTER_AREA)
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY).astype(np.float32)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Remove global exposure changes before registration and differencing.
    for image in (previous_gray, current_gray):
        mean, std = cv2.meanStdDev(image)
        image -= float(mean[0, 0])
        image /= max(float(std[0, 0]), 8.0)
    warp = np.eye(2, 3, dtype=np.float32)
    score, warp = cv2.findTransformECC(
        current_gray,
        previous_gray,
        warp,
        cv2.MOTION_AFFINE,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4),
        None,
        3,
    )
    aligned = cv2.warpAffine(
        previous_gray,
        warp,
        (current_gray.shape[1], current_gray.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    difference = np.abs(current_gray - aligned)
    median = float(np.median(difference))
    mad = float(np.median(np.abs(difference - median)))
    # Normalized grayscale makes this threshold insensitive to auto exposure.
    normalized_floor = float(min_diff) / 32.0
    threshold = max(normalized_floor, median + 6.0 * max(mad, 0.02))
    changed = difference >= threshold
    border = max(0, int(border_ignore_px) // 2)
    if border:
        changed[:border, :] = False
        changed[-border:, :] = False
        changed[:, :border] = False
        changed[:, -border:] = False
    changed = cv2.morphologyEx(
        changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    ).astype(bool)
    labels, count = ndimage.label(changed, structure=np.ones((3, 3), dtype=int))
    areas = np.bincount(labels.ravel())[1:] if count else np.empty(0, dtype=int)
    largest = int(np.max(areas)) if areas.size else 0
    return {
        "rgb_intrusion_available": True,
        "rgb_registration_score": float(score),
        "rgb_difference_threshold": float(threshold),
        "rgb_changed_pixels": int(np.count_nonzero(changed)),
        "rgb_largest_component_pixels": largest,
        "rgb_intrusion_raw": largest >= int(min_component_pixels),
        "_rgb_debug_mask": changed,
    }


@dataclass(frozen=True)
class DepthSafetyReference:
    sequence: int
    received_monotonic: float
    bbox_xyxy: tuple[float, float, float, float]
    panel_depth_m: float
    base_to_panel_m: tuple[float, float, float] | None = None
    base_panel_normal: tuple[float, float, float] | None = None


def compensate_reference_for_camera_motion(
    reference: DepthSafetyReference,
    base_to_camera: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[DepthSafetyReference, dict]:
    """Project a fixed panel reference into the current eye-in-hand camera."""
    if reference.base_to_panel_m is None:
        raise RuntimeError("reference has no base-frame panel point")
    pose = np.asarray(base_to_camera, dtype=float)
    intrinsics = np.asarray(camera_matrix, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("base_to_camera must be a finite 4x4 transform")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    panel_base = np.r_[np.asarray(reference.base_to_panel_m, dtype=float), 1.0]
    panel_camera = np.linalg.inv(pose) @ panel_base
    expected_depth = float(panel_camera[2])
    if not np.isfinite(expected_depth) or expected_depth <= 0.08:
        raise RuntimeError("compensated panel point is outside the camera")
    u = float(intrinsics[0, 0] * panel_camera[0] / expected_depth + intrinsics[0, 2])
    v = float(intrinsics[1, 1] * panel_camera[1] / expected_depth + intrinsics[1, 2])
    x1, y1, x2, y2 = reference.bbox_xyxy
    scale = float(np.clip(reference.panel_depth_m / expected_depth, 0.5, 4.0))
    half_w = max(1.0, (x2 - x1) * scale * 0.5)
    half_h = max(1.0, (y2 - y1) * scale * 0.5)
    adjusted = replace(
        reference,
        bbox_xyxy=(u - half_w, v - half_h, u + half_w, v + half_h),
        panel_depth_m=expected_depth,
    )
    return adjusted, {
        "motion_depth_offset_m": expected_depth - float(reference.panel_depth_m),
        "expected_panel_depth_m": expected_depth,
        "projected_center_xy": [u, v],
        "projected_bbox_scale": scale,
    }


def _scaled_box(box, scale: float, width: int, height: int):
    x1, y1, x2, y2 = (float(value) for value in box)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_w = max(1.0, (x2 - x1) * float(scale) * 0.5)
    half_h = max(1.0, (y2 - y1) * float(scale) * 0.5)
    return (
        max(0, int(np.floor(cx - half_w))),
        max(0, int(np.floor(cy - half_h))),
        min(width, int(np.ceil(cx + half_w))),
        min(height, int(np.ceil(cy + half_h))),
    )


def evaluate_panel_foreground(
    depth_m: np.ndarray,
    reference: DepthSafetyReference,
    config: DepthSafetyConfig,
) -> dict:
    """Detect unexpected depth in a ring around the selected button.

    The small center exclusion is where the robot's own pressing tip is
    expected to appear. A hand or body normally occupies the larger ring as
    well; losing the YOLO reference is handled independently by the monitor.
    """
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    height, width = depth.shape
    outer = _scaled_box(reference.bbox_xyxy, config.roi_scale, width, height)
    inner = _scaled_box(
        reference.bbox_xyxy, config.center_exclusion_scale, width, height
    )
    mask = np.zeros((height, width), dtype=bool)
    mask[outer[1] : outer[3], outer[0] : outer[2]] = True
    mask[inner[1] : inner[3], inner[0] : inner[2]] = False
    valid = mask & np.isfinite(depth) & (depth > 0.08) & (depth < 5.0)
    threshold = float(reference.panel_depth_m) - float(config.foreground_margin_m)
    foreground = valid & (depth < threshold)
    valid_pixels = int(np.count_nonzero(valid))
    foreground_pixels = int(np.count_nonzero(foreground))
    fraction = foreground_pixels / valid_pixels if valid_pixels else 0.0
    foreground_bbox = None
    if foreground_pixels:
        ys, xs = np.nonzero(foreground)
        foreground_bbox = [
            int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        ]
    foreground_regions = {
        "left_of_tool": int(np.count_nonzero(foreground[:, : inner[0]])),
        "right_of_tool": int(np.count_nonzero(foreground[:, inner[2] :])),
        "above_tool": int(np.count_nonzero(foreground[: inner[1], :])),
        "below_tool": int(np.count_nonzero(foreground[inner[3] :, :])),
    }
    largest_component_pixels = 0
    largest_component_bbox = None
    if foreground_pixels:
        labels, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=int))
        if count:
            areas = np.bincount(labels.ravel())[1:]
            index = 1 + int(np.argmax(areas))
            largest_component_pixels = int(areas[index - 1])
            ys, xs = np.nonzero(labels == index)
            largest_component_bbox = [
                int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            ]
    blocked = (
        foreground_pixels >= int(config.min_foreground_pixels)
        and largest_component_pixels >= int(config.min_foreground_pixels)
    )
    return {
        "blocked_raw": bool(blocked),
        "reason_raw": "unexpected_foreground" if blocked else "clear",
        "foreground_pixels": foreground_pixels,
        "valid_pixels": valid_pixels,
        "foreground_fraction": float(fraction),
        "foreground_bbox_xyxy": foreground_bbox,
        "foreground_region_pixels": foreground_regions,
        "largest_foreground_component_pixels": largest_component_pixels,
        "largest_foreground_component_bbox_xyxy": largest_component_bbox,
        "foreground_threshold_m": threshold,
        "outer_roi_xyxy": list(outer),
        "center_exclusion_xyxy": list(inner),
    }


class DepthPanelSafetyMonitor:
    """Fail-safe, debounced RGB-D guard running independently of YOLO speed."""

    def __init__(
        self,
        camera,
        sample_provider: Callable[[], object | None],
        config: DepthSafetyConfig,
        camera_matrix: np.ndarray | None = None,
        numeric_layout_task_settings: dict | None = None,
    ) -> None:
        self.camera = camera
        self.sample_provider = sample_provider
        self.config = config
        self.numeric_layout_task_settings = dict(numeric_layout_task_settings or {})
        self.camera_matrix = (
            None if camera_matrix is None else np.asarray(camera_matrix, dtype=float)
        )
        self._lock = threading.Lock()
        self._pose_timing = PoseTimingTrace()
        self._frame_timing = PoseTimingTrace()
        self._last_pose_update_timing = None
        self._last_frame_timing = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._reference: DepthSafetyReference | None = None
        self._target_label: str | None = None
        self._blocked_streak = 0
        self._clear_streak = 0
        self._latched_blocked = True
        self._motion_paused = False
        self._last_obstruction: dict | None = None
        self._latest_camera_pose: tuple[float, np.ndarray] | None = None
        self._matrix_reference: DepthMatrixReference | None = None
        self._mode = "predicted_depth_matrix"
        self._raw_block_started_unix: float | None = None
        self._raw_block_started_sequence: int | None = None
        self._matrix_latched_frame_unix: float | None = None
        self._matrix_unavailable_streak = 0
        self._matrix_recovered_streak = 0
        self._previous_color: np.ndarray | None = None
        self._rgb_intrusion_streak = 0
        self._matrix_latch_reason: str | None = None
        self._depth_degraded_commit_armed = False
        self._depth_degraded_commit = False
        self._depth_degraded_snapshot_saved = False
        self._hazard_snapshot_path: str | None = None
        self._hazard_snapshot_pending = False
        self._hazard_snapshot_error: str | None = None
        self._status = {
            "ready": False,
            "blocked": True,
            "reason": "target_reference_not_established",
            "frame_sequence": 0,
        }

    def reset(self, target_label: str) -> None:
        with self._lock:
            self._pose_timing = PoseTimingTrace()
            self._frame_timing = PoseTimingTrace()
            self._last_pose_update_timing = None
            self._last_frame_timing = None
            self._target_label = str(target_label)
            self._reference = None
            self._reference_position_source = None
            self._blocked_streak = 0
            self._clear_streak = 0
            self._latched_blocked = True
            self._motion_paused = False
            self._last_obstruction = None
            self._matrix_reference = None
            self._raw_block_started_unix = None
            self._raw_block_started_sequence = None
            self._matrix_latched_frame_unix = None
            self._matrix_unavailable_streak = 0
            self._matrix_recovered_streak = 0
            self._previous_color = None
            self._rgb_intrusion_streak = 0
            self._matrix_recovered_streak = 0
            self._matrix_latch_reason = None
            self._matrix_latch_reason = None
            self._depth_degraded_commit_armed = False
            self._depth_degraded_commit = False
            self._depth_degraded_snapshot_saved = False
            self._hazard_snapshot_path = None
            self._hazard_snapshot_pending = False
            self._hazard_snapshot_error = None
            self._mode = "predicted_depth_matrix"
            self._status = {
                "ready": False,
                "blocked": True,
                "reason": "target_reference_not_established",
                "frame_sequence": 0,
                "target_label": self._target_label,
            }

    def set_mode(self, mode: str) -> dict:
        value = str(mode)
        if value not in {"predicted_depth_matrix", "absolute_near_field"}:
            raise ValueError("unsupported visual safety mode")
        with self._lock:
            self._mode = value
            self._blocked_streak = 0
            self._clear_streak = 0
            self._latched_blocked = False
            self._matrix_unavailable_streak = 0
            self._matrix_recovered_streak = 0
            self._matrix_latch_reason = None
        return {"mode": value}

    def start(self, target_label: str) -> None:
        if self._running:
            return
        self.reset(target_label)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _update_reference(self, sample) -> None:
        if sample is None:
            return
        pose=sample.pose
        reference_source='yolo'
        if pose is None and getattr(self,'numeric_layout_task_settings',{}).get('task_fallback_enabled') is True:
            from .numeric_layout_handoff import LayoutTaskPolicy
            from types import SimpleNamespace
            record=getattr(sample,'numeric_layout_localization',None)
            policy=LayoutTaskPolicy(self.numeric_layout_task_settings,sample.target_label)
            evidence=policy.evidence(record,sequence=sample.sequence,
                                     capture=sample.frame.received_unix,depth=True)
            if evidence is not None:
                geometry=evidence['pose']
                # Local safety-reference geometry only: never publish this as
                # sample.pose, a detector result, a lamp state or a motion goal.
                pose=SimpleNamespace(p_camera=np.asarray(geometry['p_camera_m']),
                    normal_camera=np.asarray(geometry['normal_camera']),
                    detection=SimpleNamespace(bbox_xyxy=evidence['bbox_xyxy']))
                reference_source='panel_layout_inferred'
        if pose is None:return
        with self._lock:
            if sample.target_label != self._target_label:
                return
            if self._reference is not None and sample.sequence <= self._reference.sequence:
                return
            base_to_panel = None
            base_panel_normal = None
            camera_pose = self._latest_camera_pose
            if camera_pose is not None:
                pose_stamp, base_to_camera = camera_pose
                if abs(pose_stamp - sample.frame.received_monotonic) <= float(
                    self.config.camera_pose_max_age_sec
                ):
                    panel_camera = np.r_[np.asarray(pose.p_camera, dtype=float), 1.0]
                    base_to_panel = tuple((base_to_camera @ panel_camera)[:3])
                    normal = base_to_camera[:3, :3] @ np.asarray(
                        pose.normal_camera, dtype=float
                    )
                    normal /= np.linalg.norm(normal)
                    base_panel_normal = tuple(normal)
            self._reference = DepthSafetyReference(
                sequence=int(sample.sequence),
                received_monotonic=float(sample.frame.received_monotonic),
                bbox_xyxy=tuple(pose.detection.bbox_xyxy),
                panel_depth_m=float(pose.p_camera[2]),
                base_to_panel_m=base_to_panel,
                base_panel_normal=base_panel_normal,
            )
            self._reference_position_source=reference_source

    def update_camera_pose(
        self, timestamp_monotonic: float, base_to_camera: np.ndarray
    ) -> dict:
        pose = np.asarray(base_to_camera, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("base_to_camera must be a finite 4x4 transform")
        requested = time.monotonic()
        with self._lock:
            acquired = time.monotonic()
            previous = self._last_pose_update_timing
            stamp = float(timestamp_monotonic)
            accepted = self._latest_camera_pose is None or stamp > self._latest_camera_pose[0]
            if accepted:
                self._latest_camera_pose = (stamp, pose.copy())
            evidence = dict(request_monotonic=requested, acquired_monotonic=acquired,
                            pose_sample_monotonic=stamp, lock_wait_sec=acquired-requested,
                            pose_age_at_apply_sec=acquired-stamp, accepted=accepted,
                            request_interval_sec=None if previous is None else requested-previous['request_monotonic'])
            self._last_pose_update_timing = evidence
        self._pose_timing.record(evidence, slow=evidence['lock_wait_sec'] > .05
                                or evidence['pose_age_at_apply_sec'] > .25)
        return evidence

    def pose_timing_evidence(self) -> dict:
        """Diagnostic read only; fetch full bounded buffers once after a task."""
        return dict(updates=self._pose_timing.snapshot(), frames=self._frame_timing.snapshot(),
                    pose_frame_max_difference_sec=float(self.config.camera_pose_max_age_sec))

    def capture_matrix_reference(self, depth_frames_m: np.ndarray) -> dict:
        started = time.perf_counter()
        with self._lock:
            camera_pose = self._latest_camera_pose
            reference = self._reference
            if camera_pose is None:
                raise RuntimeError("camera pose is not available")
            if reference is None:
                raise RuntimeError("button/panel reference is not established")
            pose_stamp, pose = camera_pose
            if time.monotonic() - pose_stamp > float(self.config.camera_pose_max_age_sec):
                raise RuntimeError("camera pose is stale")
            panel_depth = float(reference.panel_depth_m)
        checked = time.perf_counter()
        factor = int(self.config.matrix_downsample_factor)
        reduced_frames = downsample_depth_matrix(
            depth_frames_m, factor, reducer="median"
        )
        downsampled = time.perf_counter()
        reduced_intrinsics = np.asarray(self.camera_matrix, dtype=float).copy()
        reduced_intrinsics[0, :] /= factor
        reduced_intrinsics[1, :] /= factor
        reduced_intrinsics[2, :] = [0.0, 0.0, 1.0]
        model = build_depth_matrix_reference(
            reduced_frames,
            reduced_intrinsics,
            pose,
            panel_depth_m=panel_depth,
            foreground_margin_m=self.config.foreground_margin_m,
            panel_point_base_m=reference.base_to_panel_m,
            panel_normal_base=reference.base_panel_normal,
        )
        built = time.perf_counter()
        with self._lock:
            self._matrix_reference = model
            self._blocked_streak = 0
            self._clear_streak = 0
            # Capture is allowed only after the clear ring preflight. Start
            # the new matrix model unlatched; it must still see the configured
            # consecutive blocked frames before requesting a stop.
            self._latched_blocked = False
            self._last_obstruction = None
            self._raw_block_started_unix = None
            self._raw_block_started_sequence = None
            self._matrix_latched_frame_unix = None
            self._matrix_unavailable_streak = 0
        committed = time.perf_counter()
        return {
            "reference_timing_sec": {
                "validate_and_lock": checked - started,
                "downsample": downsampled - checked,
                "build_model": built - downsampled,
                "commit": committed - built,
                "total": committed - started,
            },
            "reference_depth_shape": list(model.depth_m.shape),
            "reference_world_points": int(len(model.base_points_m)),
            "reference_tool_pixels": int(np.count_nonzero(model.tool_mask)),
            "matrix_downsample_factor": factor,
        }

    def disable_matrix_reference(self) -> None:
        with self._lock:
            self._matrix_reference = None
            self._matrix_latch_reason = None
            self._matrix_unavailable_streak = 0
            self._matrix_recovered_streak = 0
            self._depth_degraded_commit_armed = False
            self._depth_degraded_commit = False
            self._depth_degraded_snapshot_saved = False

    def set_depth_degraded_commit(self, enabled: bool) -> dict:
        """Arm RGB-only safety for a bounded final approach and retreat.

        The next processed frame suppresses moving depth residuals and starts
        affine-registered RGB intrusion monitoring. Explicit disable restores
        normal depth-matrix monitoring at the observation pose.
        """
        value = bool(enabled)
        with self._lock:
            self._depth_degraded_commit_armed = value
            if not value:
                self._depth_degraded_commit = False
            self._depth_degraded_snapshot_saved = False
            self._status["depth_degraded_commit_armed"] = value
            self._status["depth_degraded_commit"] = self._depth_degraded_commit
            return {
                "enabled": self._depth_degraded_commit,
                "armed": value,
                "visual_safety": dict(self._status),
            }

    def _set_failure(self, sequence: int, reason: str, **extra) -> None:
        self._blocked_streak += 1
        self._clear_streak = 0
        if self._blocked_streak >= int(self.config.blocked_frames):
            self._latched_blocked = True
            self._matrix_latch_reason = "monitor_failure"
        self._status = {
            "ready": False,
            "blocked": bool(self._latched_blocked),
            "reason": reason,
            "frame_sequence": int(sequence),
            "blocked_streak": self._blocked_streak,
            "clear_streak": self._clear_streak,
            **extra,
        }
        if self._latched_blocked:
            self._last_obstruction = dict(self._status)

    @staticmethod
    def _colorize_depth(depth_m: np.ndarray, maximum_m: float = 0.50) -> np.ndarray:
        import cv2

        depth = np.asarray(depth_m, dtype=float)
        valid = np.isfinite(depth) & (depth > 0.08) & (depth < 5.0)
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        normalized[valid] = np.clip(
            (depth[valid] - 0.08) / max(1e-6, maximum_m - 0.08) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        return colored

    def _save_hazard_snapshot(
        self,
        *,
        frame,
        debug: dict,
        status: dict,
    ) -> None:
        import cv2

        root_value = self.config.hazard_snapshot_directory
        if not root_value:
            return
        sequence = int(frame.sequence)
        stamp = datetime.fromtimestamp(
            float(getattr(frame, "received_unix", time.time()))
        ).strftime("%Y%m%d_%H%M%S_%f")
        output = Path(root_value).expanduser() / f"{stamp}_seq{sequence}"
        output.mkdir(parents=True, exist_ok=False)
        actual = np.asarray(debug["actual_depth_m"], dtype=float)
        predicted = debug.get("predicted_depth_m")
        expected = debug.get("expected_depth_m")
        unexpected = np.asarray(debug["unexpected_mask"], dtype=bool)

        actual_mm = np.zeros(actual.shape, dtype=np.uint16)
        actual_valid = np.isfinite(actual) & (actual > 0.0)
        actual_mm[actual_valid] = np.clip(
            np.rint(actual[actual_valid] * 1000.0), 0, 65535
        ).astype(np.uint16)
        cv2.imwrite(str(output / "actual_depth_mm.png"), actual_mm)

        # Preserve metric depth, NaNs and individual masks so a future stop can
        # be replayed exactly against revised safety logic. The diagnostic PNG
        # is useful for inspection but is intentionally lossy.
        np.savez_compressed(
            output / "depth_debug_arrays.npz",
            **{
                key: np.asarray(value)
                for key, value in debug.items()
                if value is not None
            },
        )

        # Preserve the actual evidence behind an RGB fallback pause. A depth
        # montage alone hides whether the RGB trigger came from an obstacle,
        # camera motion, or a changing indicator lamp.
        rgb_current = debug.get("rgb_current_bgr")
        rgb_previous = debug.get("rgb_previous_bgr")
        rgb_mask = debug.get("rgb_intrusion_mask")
        if rgb_current is not None:
            current = np.asarray(rgb_current, dtype=np.uint8)
            cv2.imwrite(str(output / "rgb_current.png"), current)
            if rgb_previous is not None:
                previous = np.asarray(rgb_previous, dtype=np.uint8)
                cv2.imwrite(str(output / "rgb_previous.png"), previous)
            if rgb_mask is not None:
                mask = np.asarray(rgb_mask, dtype=bool)
                mask_u8 = (mask.astype(np.uint8) * 255)
                cv2.imwrite(str(output / "rgb_change_mask.png"), mask_u8)
                if mask.shape[:2] == current.shape[:2]:
                    overlay = current.copy()
                    overlay[mask] = (
                        0.45 * overlay[mask] + 0.55 * np.array([0, 0, 255])
                    ).astype(np.uint8)
                    cv2.imwrite(str(output / "rgb_change_overlay.png"), overlay)
                else:
                    # The detector works at 320x180; scale its mask back to
                    # source resolution for a convenient human-readable overlay.
                    full_mask = cv2.resize(
                        mask_u8,
                        (current.shape[1], current.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0
                    overlay = current.copy()
                    overlay[full_mask] = (
                        0.45 * overlay[full_mask]
                        + 0.55 * np.array([0, 0, 255])
                    ).astype(np.uint8)
                    cv2.imwrite(str(output / "rgb_change_overlay.png"), overlay)

        panels = [self._colorize_depth(actual)]
        labels = ["ACTUAL DEPTH"]
        if predicted is not None:
            panels.append(self._colorize_depth(np.asarray(predicted, dtype=float)))
            labels.append("PREDICTED DEPTH")
        residual = np.zeros(actual.shape, dtype=float)
        if expected is not None:
            expected_array = np.asarray(expected, dtype=float)
            comparable = actual_valid & np.isfinite(expected_array)
            residual[comparable] = np.maximum(
                0.0, expected_array[comparable] - actual[comparable]
            )
        residual_u8 = np.clip(residual / 0.15 * 255.0, 0, 255).astype(np.uint8)
        residual_color = cv2.applyColorMap(residual_u8, cv2.COLORMAP_INFERNO)
        residual_color[~actual_valid] = 0
        residual_color[unexpected] = (0, 0, 255)
        panels.append(residual_color)
        labels.append("RESIDUAL (RED = HAZARD)")

        color = np.asarray(frame.color).copy()
        tile_h, tile_w = actual.shape
        color = cv2.resize(color, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        panels.insert(0, color)
        labels.insert(0, "RGB")
        while len(panels) < 4:
            panels.append(np.zeros_like(panels[0]))
            labels.append("")
        for panel, label in zip(panels, labels):
            cv2.putText(
                panel, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
        composite = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:4])))
        cv2.imwrite(str(output / "hazard_depth_diagnostic.png"), composite)
        metadata = {
            key: value for key, value in status.items()
            if not key.startswith("_") and key != "last_obstruction"
        }
        (output / "metadata.yaml").write_text(
            yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        with self._lock:
            self._hazard_snapshot_path = str(output)
        prune_old_entries(output.parent)

    def _schedule_hazard_snapshot(self, frame, debug: dict, status: dict) -> None:
        if not self.config.hazard_snapshot_directory:
            return
        frame_copy = type("HazardFrame", (), {})()
        frame_copy.sequence = int(frame.sequence)
        frame_copy.received_unix = float(getattr(frame, "received_unix", time.time()))
        frame_copy.color = np.asarray(frame.color).copy()
        # Preserve optional arrays as None. Converting None with np.asarray
        # creates an object scalar which later looks present and makes the
        # asynchronous image writer fail before publishing its path.
        debug_copy = {
            key: (None if value is None else np.asarray(value).copy())
            for key, value in debug.items()
        }
        self._hazard_snapshot_pending = True
        self._hazard_snapshot_error = None
        threading.Thread(
            target=self._save_hazard_snapshot_worker,
            kwargs={"frame": frame_copy, "debug": debug_copy, "status": dict(status)},
            daemon=True,
        ).start()

    def _save_hazard_snapshot_worker(self, *, frame, debug: dict, status: dict) -> None:
        try:
            self._save_hazard_snapshot(frame=frame, debug=debug, status=status)
        except Exception as exc:
            with self._lock:
                self._hazard_snapshot_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._hazard_snapshot_pending = False

    def _process(self, frame) -> None:
        began = time.monotonic()
        timing = dict(frame_sequence=int(frame.sequence),
                      frame_received_monotonic=float(frame.received_monotonic),
                      processing_started_monotonic=began)
        try:
            self._process_with_timing(frame, timing)
        finally:
            # Evidence collection never replaces the original safety outcome.
            try:
                timing['processing_sec'] = time.monotonic()-began
                if 'lock_acquired_monotonic' in timing:
                    timing['lock_scope_sec'] = time.monotonic()-timing['lock_acquired_monotonic']
                with self._lock:
                    timing['reason'] = self._status.get('reason')
                    timing['blocked'] = self._status.get('blocked')
                    self._last_frame_timing = dict(timing)
                self._frame_timing.record(timing, slow=(
                    timing.get('reason') == 'matrix_camera_pose_stale'
                    or timing['processing_sec'] > .1))
            except Exception:
                pass

    def _process_with_timing(self, frame, timing) -> None:
        reference_started = time.monotonic()
        self._update_reference(self.sample_provider())
        now = time.monotonic()
        timing['reference_update_sec'] = now-reference_started
        lock_requested = time.monotonic()
        with self._lock:
            acquired = time.monotonic()
            timing['lock_wait_sec'] = acquired-lock_requested
            timing['lock_acquired_monotonic'] = acquired
            reference = self._reference
            target_label = self._target_label
            if now - float(frame.received_monotonic) > self.config.frame_max_age_sec:
                self._set_failure(frame.sequence, "depth_frame_stale")
                return
            matrix_reference = self._matrix_reference
            mode = self._mode
            if mode == "absolute_near_field":
                actual = downsample_depth_matrix(
                    frame.depth_m,
                    self.config.matrix_downsample_factor,
                    reducer="min",
                )
                ignored = camera_fixed_tool_ignore_mask(
                    actual.shape,
                    np.zeros_like(actual, dtype=bool),
                    self.config.matrix_fixed_tool_roi_normalized,
                    int(self.config.matrix_tool_ignore_dilation_px),
                )
                metrics = evaluate_absolute_near_field(
                    actual,
                    reaction_max_depth_m=self.config.matrix_reaction_max_depth_m,
                    min_component_pixels=self.config.matrix_min_component_pixels,
                    border_ignore_px=self.config.matrix_border_ignore_px,
                    ignored_mask=ignored,
                    include_debug_arrays=True,
                )
                debug = metrics.pop("_debug")
                was_latched = self._latched_blocked
                if metrics["blocked_raw"]:
                    self._blocked_streak += 1
                    self._clear_streak = 0
                    if self._blocked_streak >= int(
                        self.config.matrix_blocked_frames
                    ):
                        self._latched_blocked = True
                else:
                    self._clear_streak += 1
                    self._blocked_streak = 0
                    if self._clear_streak >= int(self.config.clear_frames):
                        self._latched_blocked = False
                self._status = {
                    "ready": True,
                    "monitoring_available": True,
                    "coverage_mode": "absolute_near_field",
                    "blocked": bool(self._latched_blocked),
                    "reason": (
                        "absolute_near_field_object"
                        if self._latched_blocked else "clear"
                    ),
                    "target_label": target_label,
                    "motion_paused": bool(self._motion_paused),
                    "frame_sequence": int(frame.sequence),
                    "blocked_streak": self._blocked_streak,
                    "clear_streak": self._clear_streak,
                    **metrics,
                }
                if self._latched_blocked and metrics["blocked_raw"]:
                    self._last_obstruction = dict(self._status)
                    if not was_latched:
                        self._schedule_hazard_snapshot(frame, debug, self._status)
                return
            if matrix_reference is not None:
                camera_pose = self._latest_camera_pose
                checked = time.monotonic()
                timing.update(pose_checked_monotonic=checked,
                              frame_age_at_check_sec=checked-float(frame.received_monotonic),
                              pose_frame_max_difference_sec=float(self.config.camera_pose_max_age_sec),
                              motion_paused=bool(self._motion_paused))
                if camera_pose is not None:
                    timing.update(pose_sample_monotonic=float(camera_pose[0]),
                                  pose_age_at_check_sec=checked-float(camera_pose[0]),
                                  frame_minus_pose_sec=float(frame.received_monotonic)-float(camera_pose[0]))
                if (
                    camera_pose is None
                    or abs(float(frame.received_monotonic) - camera_pose[0])
                    > float(self.config.camera_pose_max_age_sec)
                ):
                    self._set_failure(
                        frame.sequence,
                        "matrix_camera_pose_stale",
                        monitoring_available=False,
                        target_label=target_label,
                        pose_timing=dict(timing),
                    )
                    return
                prediction_started = time.monotonic()
                predicted = predict_depth_matrix(
                    matrix_reference,
                    camera_pose[1],
                    splat_radius=self.config.matrix_splat_radius,
                )
                panel_plane = predict_panel_plane_depth(
                    matrix_reference, camera_pose[1]
                )
                residual_started = time.monotonic()
                timing['depth_prediction_sec'] = residual_started-prediction_started
                metrics = evaluate_depth_matrix_residual(
                    downsample_depth_matrix(
                        frame.depth_m,
                        self.config.matrix_downsample_factor,
                        reducer="min",
                    ),
                    predicted,
                    foreground_margin_m=self.config.foreground_margin_m,
                    min_component_pixels=self.config.matrix_min_component_pixels,
                    min_coverage_fraction=self.config.matrix_min_coverage_fraction,
                    border_ignore_px=self.config.matrix_border_ignore_px,
                    alignment_tolerance_px=(
                        self.config.matrix_alignment_tolerance_px
                    ),
                    reaction_max_depth_m=self.config.matrix_reaction_max_depth_m,
                    known_safe_plane_depth_m=panel_plane,
                    known_safe_plane_tolerance_m=(
                        self.config.matrix_panel_plane_tolerance_m
                    ),
                    ignored_mask=camera_fixed_tool_ignore_mask(
                        matrix_reference.depth_m.shape,
                        matrix_reference.tool_mask,
                        self.config.matrix_fixed_tool_roi_normalized,
                        int(self.config.matrix_tool_ignore_dilation_px),
                    ),
                    predicted_edge_ignore_dilation_px=(
                        self.config.matrix_predicted_edge_ignore_dilation_px
                    ),
                    include_debug_arrays=True,
                )
                timing['depth_residual_sec'] = time.monotonic()-residual_started
                debug = metrics.pop("_debug")
                rgb_metrics = {
                    "rgb_intrusion_available": False,
                    "rgb_intrusion_raw": False,
                    "rgb_intrusion_suppressed": False,
                }
                if (
                    self._depth_degraded_commit_armed
                    and self.config.near_rgb_change_detection_enabled
                    and self._previous_color is not None
                ):
                    try:
                        previous_color = np.asarray(self._previous_color).copy()
                        current_color = np.asarray(frame.color).copy()
                        rgb_metrics = evaluate_rgb_frame_intrusion(
                            previous_color,
                            current_color,
                            min_component_pixels=(
                                self.config.rgb_intrusion_min_component_pixels
                            ),
                            min_diff=self.config.rgb_intrusion_min_diff,
                            border_ignore_px=self.config.matrix_border_ignore_px,
                        )
                        rgb_debug_mask = rgb_metrics.pop("_rgb_debug_mask", None)
                        if rgb_debug_mask is not None:
                            debug["rgb_intrusion_mask"] = rgb_debug_mask
                        # RGB differencing only reports scene change; it cannot
                        # establish that the changed pixels are an obstacle.
                        # Keep both source frames with the safety snapshot so
                        # every fallback stop can be reviewed and replayed.
                        debug["rgb_previous_bgr"] = previous_color
                        debug["rgb_current_bgr"] = current_color
                    except Exception:
                        rgb_metrics = {
                            "rgb_intrusion_available": False,
                            "rgb_intrusion_raw": False,
                            "rgb_intrusion_suppressed": False,
                        }
                elif self._depth_degraded_commit_armed:
                    # Do not interpret button illumination, reflections, or
                    # other appearance-only changes as an obstacle in the
                    # near-panel zone. The image stream remains alive, while
                    # near-field depth intrusion monitoring is also suppressed.
                    rgb_metrics = {
                        "rgb_intrusion_available": True,
                        "rgb_intrusion_raw": False,
                        "rgb_intrusion_suppressed": True,
                    }
                self._previous_color = np.asarray(frame.color).copy()
                if self._depth_degraded_commit_armed:
                    # Keep the near-panel sensor policy active through retreat:
                    # unreliable close-range depth residuals and, by default,
                    # raw RGB differencing are both excluded from obstacle
                    # decisions. Collision validation and the independent stop
                    # bridge remain active. A separately validated RGB object
                    # classifier can opt back into RGB intrusion monitoring.
                    self._depth_degraded_commit = True
                    self._matrix_unavailable_streak = (
                        self._matrix_unavailable_streak + 1
                        if not metrics["matrix_monitoring_available"]
                        else 0
                    )
                    rgb_detection_enabled = bool(
                        self.config.near_rgb_change_detection_enabled
                    )
                    if rgb_detection_enabled and rgb_metrics["rgb_intrusion_raw"]:
                        self._rgb_intrusion_streak += 1
                    else:
                        self._rgb_intrusion_streak = 0
                    rgb_latched = bool(
                        rgb_detection_enabled
                        and self._rgb_intrusion_streak
                        >= int(self.config.rgb_intrusion_blocked_frames)
                    )
                    was_latched = self._latched_blocked
                    if rgb_latched:
                        self._latched_blocked = True
                        self._matrix_latch_reason = "rgb_scene_change"
                    preexisting_latch = bool(
                        self._latched_blocked
                        and self._matrix_latch_reason != "rgb_scene_change"
                    )
                    if (
                        not rgb_detection_enabled
                        and self._matrix_latch_reason == "rgb_scene_change"
                    ):
                        # A scene-change latch is exactly the class disabled by
                        # this mode; retain any independent depth/health latch.
                        self._latched_blocked = False
                        self._matrix_latch_reason = None
                        self._blocked_streak = 0
                        preexisting_latch = False
                    rgb_available = bool(rgb_metrics["rgb_intrusion_available"])
                    blocked = bool(
                        self._latched_blocked or not rgb_available
                    )
                    reason = (
                        "rgb_scene_change_unverified_near_panel"
                        if self._matrix_latch_reason == "rgb_scene_change"
                        else "preexisting_safety_latch"
                        if preexisting_latch
                        else "near_rgb_monitor_unavailable"
                        if not rgb_available
                        else "near_panel_depth_and_rgb_change_suppressed"
                    )
                    self._status = {
                        "ready": bool(rgb_available and not preexisting_latch),
                        "blocked": blocked,
                        "reason": reason,
                        "monitoring_available": rgb_available,
                        "coverage_mode": (
                            "near_rgb_affine"
                            if rgb_detection_enabled
                            else "near_rgb_change_suppressed"
                        ),
                        "depth_monitoring_suppressed": True,
                        "rgb_change_detection_suppressed": not rgb_detection_enabled,
                        "depth_degraded_commit_armed": True,
                        "depth_degraded_commit": True,
                        "target_label": target_label,
                        "motion_paused": bool(self._motion_paused),
                        "frame_sequence": int(frame.sequence),
                        "rgb_intrusion_streak": self._rgb_intrusion_streak,
                        "rgb_object_confirmed": False,
                        **rgb_metrics,
                        **metrics,
                    }
                    if blocked:
                        self._last_obstruction = dict(self._status)
                        if not was_latched:
                            self._schedule_hazard_snapshot(
                                frame, debug, self._status
                            )
                    return
                if not metrics["matrix_monitoring_available"]:
                    self._matrix_recovered_streak = 0
                    self._matrix_unavailable_streak += 1
                    residual_now = bool(
                        metrics.get("matrix_largest_component_pixels", 0)
                        >= int(self.config.matrix_degraded_min_component_pixels)
                    )
                    if rgb_metrics["rgb_intrusion_raw"]:
                        self._rgb_intrusion_streak += 1
                    else:
                        self._rgb_intrusion_streak = 0
                    rgb_latched = self._rgb_intrusion_streak >= int(
                        self.config.rgb_intrusion_blocked_frames
                    )
                    if residual_now:
                        self._blocked_streak += 1
                        self._clear_streak = 0
                        if self._blocked_streak >= int(
                            self.config.matrix_blocked_frames
                        ):
                            self._latched_blocked = True
                            self._matrix_latch_reason = "residual"
                    if self._depth_degraded_commit_armed:
                        # Activate one frame before the ordinary coverage latch.
                        # This is driven by actual camera degradation, not by a
                        # large fixed distance from the panel.
                        activation_frames = max(
                            2, int(self.config.matrix_unavailable_frames) - 1
                        )
                        if (
                            not residual_now
                            and self._matrix_unavailable_streak >= activation_frames
                        ):
                            self._depth_degraded_commit = True
                        residual_latched = residual_now or rgb_latched or (
                            self._latched_blocked
                            and self._matrix_latch_reason == "residual"
                        )
                        if not residual_latched:
                            self._latched_blocked = False
                            self._blocked_streak = 0
                            self._clear_streak = 0
                        self._status = {
                            "ready": not residual_latched,
                            "blocked": bool(residual_latched),
                            "reason": (
                                    (
                                        "rgb_scene_change_unverified_during_matrix_coverage_loss"
                                        if rgb_latched
                                        else "residual_latched_during_matrix_coverage_loss"
                                    )
                                if residual_latched
                                else (
                                    "matrix_coverage_insufficient_degraded_commit"
                                    if self._depth_degraded_commit
                                    else "matrix_prediction_coverage_transient"
                                )
                            ),
                            "monitoring_available": False,
                            "depth_degraded_commit_armed": True,
                            "depth_degraded_commit": self._depth_degraded_commit,
                            "target_label": target_label,
                            "frame_sequence": int(frame.sequence),
                            "matrix_unavailable_streak": self._matrix_unavailable_streak,
                            "rgb_intrusion_streak": self._rgb_intrusion_streak,
                            "rgb_object_confirmed": False,
                            **rgb_metrics,
                            **metrics,
                        }
                        if self._depth_degraded_commit and not self._depth_degraded_snapshot_saved:
                            self._depth_degraded_snapshot_saved = True
                            self._schedule_hazard_snapshot(
                                frame, debug, self._status
                            )
                        return
                    was_latched = self._latched_blocked
                    unavailable_latched = self._matrix_unavailable_streak >= int(
                        self.config.matrix_unavailable_frames
                    )
                    if unavailable_latched:
                        self._latched_blocked = True
                        self._matrix_latch_reason = "coverage"
                    self._status = {
                        "ready": not unavailable_latched,
                        "blocked": bool(self._latched_blocked),
                        "reason": (
                            "matrix_prediction_coverage_insufficient"
                            if unavailable_latched
                            else "matrix_prediction_coverage_transient"
                        ),
                        "monitoring_available": False,
                        "target_label": target_label,
                        "frame_sequence": int(frame.sequence),
                        "matrix_unavailable_streak": self._matrix_unavailable_streak,
                        **metrics,
                    }
                    if unavailable_latched:
                        self._last_obstruction = dict(self._status)
                        if not was_latched:
                            self._schedule_hazard_snapshot(
                                frame, debug, self._status
                            )
                    return
                self._matrix_unavailable_streak = 0
                self._matrix_recovered_streak += 1
                self._rgb_intrusion_streak = 0
                if (
                    self._depth_degraded_commit
                    and self._matrix_recovered_streak
                    >= int(self.config.matrix_recovered_frames)
                ):
                    self._depth_degraded_commit = False
                was_latched = self._latched_blocked
                first_raw_residual = False
                if metrics["blocked_raw"]:
                    if self._blocked_streak == 0:
                        first_raw_residual = True
                        self._raw_block_started_unix = float(
                            getattr(frame, "received_unix", time.time())
                        )
                        self._raw_block_started_sequence = int(frame.sequence)
                    self._blocked_streak += 1
                    self._clear_streak = 0
                    if self._blocked_streak >= int(self.config.matrix_blocked_frames):
                        if not self._latched_blocked:
                            self._matrix_latched_frame_unix = float(
                                getattr(frame, "received_unix", time.time())
                            )
                        self._latched_blocked = True
                        self._matrix_latch_reason = "residual"
                else:
                    self._clear_streak += 1
                    self._blocked_streak = 0
                    self._raw_block_started_unix = None
                    self._raw_block_started_sequence = None
                    if self._clear_streak >= int(self.config.clear_frames):
                        self._latched_blocked = False
                        self._matrix_latched_frame_unix = None
                        self._matrix_latch_reason = None
                if self._latched_blocked:
                    matrix_reason = (
                        "unexpected_depth_matrix_residual"
                        if self._matrix_latch_reason == "residual"
                        else "matrix_monitor_failure_latched"
                    )
                else:
                    matrix_reason = "clear"
                self._status = {
                    "ready": True,
                    "monitoring_available": True,
                    "coverage_mode": "predicted_depth_matrix",
                    "blocked": bool(self._latched_blocked),
                    "reason": matrix_reason,
                    "target_label": target_label,
                    "motion_paused": bool(self._motion_paused),
                    "frame_sequence": int(frame.sequence),
                    "blocked_streak": self._blocked_streak,
                    "clear_streak": self._clear_streak,
                    "matrix_first_raw_frame_sequence": self._raw_block_started_sequence,
                    "matrix_first_raw_frame_unix_sec": self._raw_block_started_unix,
                    "matrix_latched_frame_unix_sec": self._matrix_latched_frame_unix,
                    "matrix_debounce_sec": (
                        self._matrix_latched_frame_unix
                        - self._raw_block_started_unix
                        if self._matrix_latched_frame_unix is not None
                        and self._raw_block_started_unix is not None
                        else None
                    ),
                    **metrics,
                }
                # Preserve the first comparable residual frame as well as the
                # eventual latch frame.  The latter can be a sparse/degraded
                # frame and is insufficient to distinguish a fixed panel edge
                # from a newly introduced object.
                if first_raw_residual:
                    transient = {**self._status, "reason": "matrix_residual_first_frame"}
                    self._schedule_hazard_snapshot(frame, debug, transient)
                if self._latched_blocked and metrics["blocked_raw"]:
                    self._last_obstruction = dict(self._status)
                    if not was_latched:
                        self._schedule_hazard_snapshot(frame, debug, self._status)
                return
            if reference is None:
                self._set_failure(frame.sequence, "target_reference_not_established")
                return
            reference_age = now - reference.received_monotonic
            evaluation_reference = reference
            compensation = {}
            if reference_age > self.config.reference_max_age_sec:
                grace_deadline = (
                    float(self.config.reference_max_age_sec)
                    + float(self.config.reference_blind_grace_sec)
                )
                tracking_allowed = (
                    not self._motion_paused
                    and not self._latched_blocked
                    and reference_age <= grace_deadline
                )
                if not tracking_allowed:
                    self._set_failure(
                        frame.sequence,
                        "target_reference_stale",
                        monitoring_available=False,
                        motion_paused=bool(self._motion_paused),
                        target_label=target_label,
                        reference_sequence=int(reference.sequence),
                        reference_age_sec=float(reference_age),
                    )
                    return
                camera_pose = self._latest_camera_pose
                if (
                    self.camera_matrix is None
                    or camera_pose is None
                    or abs(float(frame.received_monotonic) - camera_pose[0])
                    > float(self.config.camera_pose_max_age_sec)
                ):
                    self._set_failure(
                        frame.sequence,
                        "motion_compensation_unavailable",
                        monitoring_available=False,
                        motion_paused=bool(self._motion_paused),
                        target_label=target_label,
                        reference_sequence=int(reference.sequence),
                        reference_age_sec=float(reference_age),
                    )
                    return
                try:
                    evaluation_reference, compensation = (
                        compensate_reference_for_camera_motion(
                            reference, camera_pose[1], self.camera_matrix
                        )
                    )
                except (RuntimeError, ValueError) as exc:
                    self._set_failure(
                        frame.sequence,
                        "motion_compensation_unavailable",
                        monitoring_available=False,
                        compensation_error=str(exc),
                        motion_paused=bool(self._motion_paused),
                        target_label=target_label,
                        reference_sequence=int(reference.sequence),
                        reference_age_sec=float(reference_age),
                    )
                    return
            metrics = evaluate_panel_foreground(
                frame.depth_m, evaluation_reference, self.config
            )
            if reference_age > self.config.reference_max_age_sec:
                if not metrics["blocked_raw"]:
                    # A short, bounded tool-occlusion interval may continue
                    # only while the cached ring itself remains clear. It does
                    # not contribute clear frames and can never release a
                    # previously latched obstacle.
                    self._blocked_streak = 0
                    self._clear_streak = 0
                    self._status = {
                        "ready": True,
                        "monitoring_available": True,
                        "coverage_mode": "kinematic_reference_tracking",
                        "blocked": False,
                        "reason": "clear_cached_ring",
                        "motion_paused": False,
                        "target_label": target_label,
                        "frame_sequence": int(frame.sequence),
                        "reference_sequence": int(reference.sequence),
                        "reference_age_sec": float(reference_age),
                        "blind_grace_remaining_sec": float(
                            grace_deadline - reference_age
                        ),
                        "blocked_streak": 0,
                        "clear_streak": 0,
                        **compensation,
                        **metrics,
                    }
                    return
            if metrics["blocked_raw"]:
                self._blocked_streak += 1
                self._clear_streak = 0
                if self._blocked_streak >= int(self.config.blocked_frames):
                    self._latched_blocked = True
            else:
                self._clear_streak += 1
                self._blocked_streak = 0
                if self._clear_streak >= int(self.config.clear_frames):
                    self._latched_blocked = False
            self._status = {
                "ready": True,
                "monitoring_available": True,
                "coverage_mode": (
                    "fresh_target"
                    if reference_age <= self.config.reference_max_age_sec
                    else "kinematic_reference_tracking"
                ),
                "blocked": bool(self._latched_blocked),
                "reason": (
                    "unexpected_foreground"
                    if self._latched_blocked
                    else "clear"
                ),
                "target_label": target_label,
                "motion_paused": bool(self._motion_paused),
                "frame_sequence": int(frame.sequence),
                "reference_sequence": int(reference.sequence),
                "reference_age_sec": float(reference_age),
                "blocked_streak": self._blocked_streak,
                "clear_streak": self._clear_streak,
                **compensation,
                **metrics,
            }
            if self._latched_blocked and metrics["blocked_raw"]:
                self._last_obstruction = dict(self._status)

    def _run(self) -> None:
        sequence = 0
        while self._running:
            try:
                frame = self.camera.wait_for_sequence(sequence, timeout=1.0)
                sequence = int(frame.sequence)
                self._process(frame)
            except TimeoutError:
                with self._lock:
                    self._set_failure(sequence, "depth_stream_timeout")
            except Exception as exc:
                with self._lock:
                    self._set_failure(
                        sequence, f"depth_guard_error:{type(exc).__name__}:{exc}"
                    )

    def status(self) -> dict:
        with self._lock:
            result = dict(self._status)
            if getattr(self,'_reference_position_source',None)=='panel_layout_inferred':
                result['reference_position_source']='panel_layout_inferred'
            result['pose_timing'] = dict(last_update=self._last_pose_update_timing,
                                        last_frame=self._last_frame_timing)
            result["last_obstruction"] = (
                dict(self._last_obstruction)
                if self._last_obstruction is not None
                else None
            )
            result["hazard_snapshot_path"] = self._hazard_snapshot_path
            result["hazard_snapshot_pending"] = self._hazard_snapshot_pending
            result["hazard_snapshot_error"] = self._hazard_snapshot_error
            result["depth_degraded_commit"] = self._depth_degraded_commit
            result["depth_degraded_commit_armed"] = self._depth_degraded_commit_armed
            return result

    def probe_roi_scales(self, frame, roi_scales) -> list[dict]:
        """Evaluate wider ROIs without changing the active safety state."""
        with self._lock:
            reference = self._reference
            if reference is None:
                raise RuntimeError("target reference is not established")
            results = []
            for value in roi_scales:
                scale = float(value)
                if not 1.0 < scale <= 20.0:
                    raise ValueError("probe ROI scale must be in (1, 20]")
                metrics = evaluate_panel_foreground(
                    frame.depth_m,
                    reference,
                    replace(self.config, roi_scale=scale),
                )
                results.append({"roi_scale": scale, **metrics})
            return results

    def set_motion_paused(self, paused: bool) -> None:
        """Allow a cached camera-frame reference only while the arm is held.

        A reference does not geometrically age while the eye-in-hand camera is
        stationary. This lets an obstacle clear even if the pressing tool now
        hides the target from YOLO.
        """
        with self._lock:
            self._motion_paused = bool(paused)

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
