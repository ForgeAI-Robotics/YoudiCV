from __future__ import annotations

from dataclasses import dataclass
import ctypes
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from .geometry import transform


@dataclass(frozen=True)
class ButtonDetection:
    """A transport-independent YOLO button detection in color-image pixels."""

    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def center_pixel(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5


@dataclass(frozen=True)
class ButtonCameraPose:
    """Button surface point and outward panel normal in the RGB camera frame."""

    detection: ButtonDetection
    p_camera: np.ndarray
    normal_camera: np.ndarray
    camera_to_button: np.ndarray
    plane_rms_m: float
    plane_inliers: int
    panel_support_detections: int
    local_global_depth_delta_m: float
    depth_median_m: float
    center_depth_m: float | None
    panel_plane_fit: dict[str, Any] | None = None

    @property
    def transform(self) -> np.ndarray:
        """Compatibility alias used by the existing target transform chain."""
        return self.camera_to_button

    @property
    def camera_to_marker(self) -> np.ndarray:
        """Legacy name retained while the downstream marker code is migrated."""
        return self.camera_to_button


class YoloButtonDetector:
    """Lazy Ultralytics adapter so geometry tests do not require PyTorch."""

    def __init__(self, model_path: str | Path, **predict_options: Any) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {path}")
        header = path.read_bytes()[:64]
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"YOLO model is only a Git LFS pointer, not downloaded weights: {path}"
            )
        # NVIDIA's Jetson PyTorch wheel may install cuSPARSELt inside the
        # active Conda environment without adding that directory to the
        # process loader path. Preloading the exact library is local to this
        # process and avoids requiring a global LD_LIBRARY_PATH modification.
        cusparselt = Path(sys.prefix) / "lib" / "libcusparseLt.so.0"
        if cusparselt.is_file():
            try:
                ctypes.CDLL(str(cusparselt), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        try:
            from ultralytics import YOLO
            import torch
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Ultralytics/PyTorch is unavailable; activate a Jetson-compatible YOLO environment"
            ) from exc
        self.predict_options = dict(predict_options)
        if str(self.predict_options.get("device", "")).casefold() == "auto":
            self.predict_options["device"] = "0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(path)

    def detect(self, image: Any) -> list[ButtonDetection]:
        return self.detect_many([image], batch_size=1)[0]

    def detect_many(
        self, images: list[Any], *, batch_size: int = 16
    ) -> list[list[ButtonDetection]]:
        """Run the production detector in batches while preserving input order."""
        if not images:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        results = self.model.predict(
            source=images, batch=int(batch_size), verbose=False,
            **self.predict_options,
        )
        return [self._convert_result(result) for result in results]

    def _convert_result(self, result: Any) -> list[ButtonDetection]:
        if result.boxes is None:
            return []
        output: list[ButtonDetection] = []
        for xyxy, confidence, class_id in zip(
            result.boxes.xyxy.cpu(),
            result.boxes.conf.cpu(),
            result.boxes.cls.cpu(),
        ):
            index = int(class_id.item())
            output.append(
                ButtonDetection(
                    class_id=index,
                    label=str(self.model.names[index]),
                    confidence=float(confidence.item()),
                    bbox_xyxy=tuple(map(float, xyxy.tolist())),
                )
            )
        return output


def select_button(
    detections: list[ButtonDetection],
    target_label: str,
    min_confidence: float,
) -> ButtonDetection:
    matches = [
        item
        for item in detections
        if item.label.casefold() == target_label.casefold()
        and item.confidence >= min_confidence
    ]
    if not matches:
        raise RuntimeError(
            f"button {target_label!r} not detected above confidence {min_confidence:.2f}"
        )
    matches.sort(key=lambda item: item.confidence, reverse=True)
    if len(matches) > 1 and matches[1].confidence >= matches[0].confidence - 0.05:
        raise RuntimeError(
            f"button {target_label!r} is ambiguous: "
            f"confidences {matches[0].confidence:.3f} and {matches[1].confidence:.3f}"
        )
    return matches[0]


def _scaled_box(
    bbox_xyxy: tuple[float, float, float, float], scale: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_w, half_h = (x2 - x1) * 0.5 * scale, (y2 - y1) * 0.5 * scale
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _box_mask(shape: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = box
    left = int(np.clip(np.floor(x1), 0, width))
    right = int(np.clip(np.ceil(x2), 0, width))
    top = int(np.clip(np.floor(y1), 0, height))
    bottom = int(np.clip(np.ceil(y2), 0, height))
    mask = np.zeros((height, width), dtype=bool)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = True
    return mask


def _backproject(
    pixels_uv: np.ndarray, depth_z: np.ndarray, camera_matrix: np.ndarray
) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=float).reshape(-1, 2)
    depth = np.asarray(depth_z, dtype=float).reshape(-1)
    homogeneous = np.c_[pixels, np.ones(len(pixels))]
    rays = (np.linalg.inv(camera_matrix) @ homogeneous.T).T
    rays *= (depth / rays[:, 2])[:, None]
    return rays


def _fit_plane(
    points: np.ndarray,
    min_inliers: int,
    distance_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    current = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(current) < min_inliers:
        raise RuntimeError(f"only {len(current)} panel depth points; require {min_inliers}")
    for _ in range(5):
        center = np.mean(current, axis=0)
        _, _, vh = np.linalg.svd(current - center, full_matrices=False)
        normal = vh[-1]
        distances = np.abs((current - center) @ normal)
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        threshold = max(float(distance_threshold_m), median + 3.0 * 1.4826 * mad)
        kept = current[distances <= threshold]
        if len(kept) < min_inliers:
            raise RuntimeError(
                f"only {len(kept)} planar depth inliers; require {min_inliers}"
            )
        if len(kept) == len(current):
            break
        current = kept
    center = np.mean(current, axis=0)
    _, _, vh = np.linalg.svd(current - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if float(normal @ (-center)) < 0.0:
        normal = -normal
    residuals = (current - center) @ normal
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    return center, normal, rms, len(current)


# Fallback limits are deliberately independent of the accepted original fit.
# No model, extra image, motion, or absolute foreground-depth cutoff is used.
_ROBUST_PANEL_MAX_PROBES = 2048
_ROBUST_PANEL_MAX_TRIALS = 32
_ROBUST_PANEL_BUDGET_SEC = 0.20


def _fit_panel_plane_robust(
    points: np.ndarray, pixels_uv: np.ndarray, supports: list[ButtonDetection],
    detection: ButtonDetection, local_fit: tuple, ray: np.ndarray, *,
    min_inliers: int, distance_threshold_m: float, max_rms_m: float,
    outer_scale: float,
) -> tuple[tuple[np.ndarray, np.ndarray, float, int], dict[str, Any]]:
    """Bounded fallback for a failed pooled fit, never a depth/safety waiver.

    A candidate needs >=70% of all pooled points within the original distance
    threshold, >=3 spatially spread independent rings (>=60% inliers each),
    and compatibility with a valid target-local plane. These extra checks
    apply only to rescued fits, leaving original successful estimates intact.
    """
    started = time.monotonic()
    def check_budget():
        if time.monotonic() - started > _ROBUST_PANEL_BUDGET_SEC:
            raise RuntimeError("robust panel fallback exceeded 0.20 s budget")

    if len(supports) < 3:
        raise RuntimeError("robust panel fallback requires 3 independent support regions")
    local_center, local_normal, local_rms, _ = local_fit
    if not np.isfinite(local_rms) or local_rms > max_rms_m:
        raise RuntimeError("robust panel fallback target-local plane is not reliable")
    rng = np.random.default_rng(0)  # Repeatable same-frame geometry, no global RNG.
    probe = points if len(points) <= _ROBUST_PANEL_MAX_PROBES else points[
        rng.choice(len(points), _ROBUST_PANEL_MAX_PROBES, replace=False)]
    best = None
    best_count = -1
    for _ in range(_ROBUST_PANEL_MAX_TRIALS):
        check_budget()
        a, b, c = probe[rng.choice(len(probe), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length < 1e-10:
            continue
        normal /= length
        offset = float(normal @ a)
        count = int(np.count_nonzero(np.abs(probe @ normal - offset) <= distance_threshold_m))
        if count > best_count:
            best, best_count = (normal, offset), count
    if best is None:
        raise RuntimeError("robust panel fallback has no nondegenerate plane")
    normal, offset = best
    initial_mask = np.abs(points @ normal - offset) <= distance_threshold_m
    if np.count_nonzero(initial_mask) < max(min_inliers, 0.70 * len(points)):
        raise RuntimeError("robust panel fallback lacks 70% pooled depth consensus")
    check_budget()
    fit = _fit_plane(points[initial_mask], min_inliers, distance_threshold_m)
    center, normal, rms, count = fit
    if not np.isfinite(rms) or rms > max_rms_m:
        raise RuntimeError("robust panel fallback exceeds original plane RMS limit")
    kept = initial_mask & (np.abs((points - center) @ normal) <= distance_threshold_m)
    fraction = float(np.mean(kept))
    if np.count_nonzero(kept) < min_inliers or fraction < 0.70:
        raise RuntimeError("robust panel fallback refined plane lacks 70% depth consensus")

    # Check actual pooled pixels; duplicate/overlapping detections cannot create
    # artificial spatial coverage. Coordinates match _box_mask's clipped integer
    # boxes, but no HxW temporary mask is needed per reference region here.
    qualified = []
    for item in supports:
        x1, y1, x2, y2 = _scaled_box(item.bbox_xyxy, outer_scale)
        region = ((pixels_uv[:, 0] >= np.floor(x1)) & (pixels_uv[:, 0] < np.ceil(x2))
                  & (pixels_uv[:, 1] >= np.floor(y1)) & (pixels_uv[:, 1] < np.ceil(y2)))
        total = int(np.count_nonzero(region))
        if total >= min_inliers and np.count_nonzero(region & kept) >= max(min_inliers, 0.60 * total):
            qualified.append(item.center_pixel)
    if len(qualified) < 3:
        raise RuntimeError("robust panel fallback lacks 3 supported reference regions")
    centers = np.asarray(qualified, dtype=float)
    x1, y1, x2, y2 = detection.bbox_xyxy
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    normalized = centers / np.array([width, height])
    spread = np.ptp(normalized, axis=0)
    covariance = np.cov(normalized, rowvar=False, bias=True)
    if np.any(spread < 1.0) or np.linalg.eigvalsh(covariance)[0] < 0.0625:
        raise RuntimeError("robust panel fallback reference regions are not spatially spread")
    local_denominator, denominator = float(local_normal @ ray), float(normal @ ray)
    if abs(local_denominator) < 1e-8 or abs(denominator) < 1e-8:
        raise RuntimeError("robust panel fallback target ray is parallel to plane")
    local_scale = float(local_normal @ local_center) / local_denominator
    scale = float(normal @ center) / denominator
    depth_delta = abs((local_scale - scale) * float(ray[2]))
    angle = float(np.arccos(np.clip(local_normal @ normal, -1.0, 1.0)))
    if scale <= 0 or local_scale <= 0 or depth_delta > 2.0 * max_rms_m or angle > np.deg2rad(15.0):
        raise RuntimeError("robust panel fallback disagrees with target-local plane")
    check_budget()
    return fit, dict(
        method="bounded_ransac_fallback_v1", pooled_points=len(points),
        candidate_inliers=int(np.count_nonzero(initial_mask)),
        consensus_inliers=int(np.count_nonzero(kept)), consensus_fraction=fraction,
        qualified_support_regions=len(qualified), support_spread_button_units=spread.tolist(),
        local_depth_delta_m=float(depth_delta), local_normal_angle_rad=angle,
        probe_points=len(probe), trials=_ROBUST_PANEL_MAX_TRIALS,
        distance_threshold_m=float(distance_threshold_m), max_plane_rms_m=float(max_rms_m),
        elapsed_sec=time.monotonic() - started, budget_sec=_ROBUST_PANEL_BUDGET_SEC,
    )


def estimate_button_camera_pose(
    detection: ButtonDetection,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    outer_scale: float = 1.8,
    inner_scale: float = 1.05,
    min_depth_m: float = 0.08,
    max_depth_m: float = 2.0,
    min_plane_pixels: int = 120,
    plane_distance_threshold_m: float = 0.004,
    max_plane_rms_m: float = 0.006,
    button_surface_offset_m: float = 0.0,
    panel_support_detections: list[ButtonDetection] | tuple[ButtonDetection, ...] | None = None,
    min_panel_support_detections: int = 0,
) -> ButtonCameraPose:
    """Estimate a 6D button frame from a YOLO box and D2C-aligned depth.

    The panel is fitted from a ring around the button. ``p_camera`` is the
    center-pixel ray intersected with that plane. Button +Z points out of the
    panel toward the observing camera, matching the previous marker convention.
    """

    depth = np.asarray(depth_m, dtype=float)
    matrix = np.asarray(camera_matrix, dtype=float)
    if depth.ndim != 2 or matrix.shape != (3, 3):
        raise ValueError("depth must be HxW and camera_matrix must be 3x3")
    if not 1.0 < outer_scale or not 0.0 < inner_scale < outer_scale:
        raise ValueError("require outer_scale > 1 and 0 < inner_scale < outer_scale")
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    target_outer = _box_mask(
        depth.shape, _scaled_box(detection.bbox_xyxy, outer_scale)
    )
    target_inner = _box_mask(
        depth.shape, _scaled_box(detection.bbox_xyxy, inner_scale)
    )
    target_ring = target_outer & ~target_inner & valid
    local_ys, local_xs = np.nonzero(target_ring)
    if len(local_xs) < min_plane_pixels:
        raise RuntimeError(
            f"only {len(local_xs)} valid local panel pixels around button; "
            f"require {min_plane_pixels}"
        )
    local_points = _backproject(
        np.c_[local_xs, local_ys], depth[local_ys, local_xs], matrix
    )
    local_fit = _fit_plane(
        local_points, min_plane_pixels, plane_distance_threshold_m
    )

    local_center, local_normal, _, _ = local_fit

    # The target ring is exactly where the tool, button rim, glare and RGB-D
    # registration errors are most likely.  When other high-confidence panel
    # buttons are visible, fit their rings instead and intersect the target's
    # RGB center ray with that broad, independent panel plane.
    supports = []
    for item in panel_support_detections or ():
        if item is detection or (
            item.label == detection.label
            and np.allclose(item.bbox_xyxy, detection.bbox_xyxy, atol=1.0)
        ):
            continue
        supports.append(item)
    if len(supports) < int(min_panel_support_detections):
        raise RuntimeError(
            f"only {len(supports)} independent panel button supports; "
            f"require {int(min_panel_support_detections)}"
        )

    if supports:
        support_outer = np.zeros(depth.shape, dtype=bool)
        support_inner = np.zeros(depth.shape, dtype=bool)
        for item in supports:
            support_outer |= _box_mask(
                depth.shape, _scaled_box(item.bbox_xyxy, outer_scale)
            )
            support_inner |= _box_mask(
                depth.shape, _scaled_box(item.bbox_xyxy, inner_scale)
            )
        ring = support_outer & ~support_inner & valid
    else:
        ring = target_ring
    ys, xs = np.nonzero(ring)
    if len(xs) < min_plane_pixels:
        raise RuntimeError(
            f"only {len(xs)} valid panel pixels around button; require {min_plane_pixels}"
        )
    points = _backproject(np.c_[xs, ys], depth[ys, xs], matrix)
    panel_plane_fit = None
    try:
        plane_center, normal_camera, plane_rms, plane_inliers = _fit_plane(
            points, min_plane_pixels, plane_distance_threshold_m
        )
        if plane_rms > max_plane_rms_m:
            raise RuntimeError(
                f"panel plane RMS {plane_rms * 1000:.1f} mm exceeds {max_plane_rms_m * 1000:.1f} mm"
            )
    except RuntimeError as original_error:
        ray = np.linalg.inv(matrix) @ np.array([*detection.center_pixel, 1.0])
        try:
            fit, panel_plane_fit = _fit_panel_plane_robust(
                points, np.c_[xs, ys], supports, detection, local_fit, ray,
                min_inliers=min_plane_pixels, distance_threshold_m=plane_distance_threshold_m,
                max_rms_m=max_plane_rms_m, outer_scale=outer_scale,
            )
        except RuntimeError as fallback_error:
            raise RuntimeError(f"{original_error}; {fallback_error}") from fallback_error
        plane_center, normal_camera, plane_rms, plane_inliers = fit
        panel_plane_fit["original_failure"] = str(original_error)

    u, v = detection.center_pixel
    ray = np.linalg.inv(matrix) @ np.array([u, v, 1.0])
    denominator = float(normal_camera @ ray)
    if abs(denominator) < 1e-8:
        raise RuntimeError("button center ray is parallel to the fitted panel plane")
    scale = float(normal_camera @ plane_center) / denominator
    if scale <= 0.0:
        raise RuntimeError("fitted button point is behind the camera")
    p_camera = scale * ray + float(button_surface_offset_m) * normal_camera

    local_denominator = float(local_normal @ ray)
    if abs(local_denominator) < 1e-8:
        raise RuntimeError("button center ray is parallel to the local panel plane")
    local_scale = float(local_normal @ local_center) / local_denominator
    local_global_depth_delta_m = abs(float((local_scale - scale) * ray[2]))

    center_box = _box_mask(depth.shape, _scaled_box(detection.bbox_xyxy, 0.35))
    center_values = depth[center_box & valid]
    center_depth = float(np.median(center_values)) if len(center_values) else None

    camera_x = np.array([1.0, 0.0, 0.0])
    button_x = camera_x - normal_camera * float(camera_x @ normal_camera)
    if np.linalg.norm(button_x) < 1e-6:
        camera_y = np.array([0.0, 1.0, 0.0])
        button_x = camera_y - normal_camera * float(camera_y @ normal_camera)
    button_x /= np.linalg.norm(button_x)
    button_y = np.cross(normal_camera, button_x)
    button_y /= np.linalg.norm(button_y)
    rotation = np.column_stack((button_x, button_y, normal_camera))
    camera_to_button = transform(rotation, p_camera)
    return ButtonCameraPose(
        detection=detection,
        p_camera=p_camera,
        normal_camera=normal_camera,
        camera_to_button=camera_to_button,
        plane_rms_m=plane_rms,
        plane_inliers=plane_inliers,
        panel_support_detections=len(supports),
        local_global_depth_delta_m=local_global_depth_delta_m,
        depth_median_m=float(np.median(depth[ring])),
        center_depth_m=center_depth,
        panel_plane_fit=panel_plane_fit,
    )
