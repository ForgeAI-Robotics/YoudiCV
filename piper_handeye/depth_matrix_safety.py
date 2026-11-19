from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class DepthMatrixReference:
    depth_m: np.ndarray
    camera_matrix: np.ndarray
    base_points_m: np.ndarray
    tool_mask: np.ndarray
    tool_depth_m: np.ndarray
    reference_base_to_camera: np.ndarray
    panel_point_base_m: np.ndarray | None = None
    panel_normal_base: np.ndarray | None = None


def downsample_depth_matrix(
    depth_m: np.ndarray, factor: int, *, reducer: str
) -> np.ndarray:
    """Downsample depth while preserving invalid pixels as NaN."""
    depth = np.asarray(depth_m, dtype=float)
    factor = int(factor)
    if depth.ndim not in (2, 3) or factor < 1:
        raise ValueError("depth must be HxW or NxHxW and factor must be positive")
    squeeze = depth.ndim == 2
    if squeeze:
        depth = depth[None, ...]
    count, height, width = depth.shape
    out_h, out_w = height // factor, width // factor
    cropped = depth[:, : out_h * factor, : out_w * factor]
    blocks = cropped.reshape(count, out_h, factor, out_w, factor)
    valid = np.where((blocks > 0.08) & (blocks < 5.0), blocks, np.nan)
    with np.errstate(all="ignore"):
        if reducer == "min":
            result = np.nanmin(valid, axis=(2, 4))
        elif reducer == "median":
            if factor == 2:
                # A four-element median avoids nanmedian's general masked-array
                # machinery for millions of tiny blocks. NaNs sort last; keep
                # the exact odd/even valid-count semantics, including all-NaN.
                values = valid.transpose(0, 1, 3, 2, 4).reshape(
                    count, out_h, out_w, 4
                ).copy()
                values.sort(axis=-1)
                counts = np.sum(~np.isnan(values), axis=-1)
                lower = np.maximum((counts - 1) // 2, 0)
                upper = counts // 2
                left = np.take_along_axis(values, lower[..., None], axis=-1)[..., 0]
                right = np.take_along_axis(values, upper[..., None], axis=-1)[..., 0]
                result = (left + right) * 0.5
            else:
                result = np.nanmedian(valid, axis=(2, 4))
        else:
            raise ValueError("reducer must be 'min' or 'median'")
    return result[0] if squeeze else result


def _largest_component(mask: np.ndarray) -> tuple[int, list[int] | None]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if not count:
        return 0, None
    areas = np.bincount(labels.ravel())[1:]
    index = 1 + int(np.argmax(areas))
    ys, xs = np.nonzero(labels == index)
    return int(areas[index - 1]), [
        int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    ]


def evaluate_absolute_near_field(
    depth_m: np.ndarray,
    *,
    reaction_max_depth_m: float,
    min_component_pixels: int,
    border_ignore_px: int = 0,
    ignored_mask: np.ndarray | None = None,
    include_debug_arrays: bool = False,
) -> dict:
    """Detect any coherent geometry inside an absolute camera-depth gate."""
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError("absolute near-field depth must be HxW")
    limit = float(reaction_max_depth_m)
    if not 0.08 < limit < 5.0:
        raise ValueError("reaction_max_depth_m must be in (0.08, 5.0)")
    near = np.isfinite(depth) & (depth > 0.08) & (depth < limit)
    ignored = np.zeros_like(near)
    if ignored_mask is not None:
        ignored = np.asarray(ignored_mask, dtype=bool)
        if ignored.shape != depth.shape:
            raise ValueError("ignored mask must match depth shape")
        near &= ~ignored
    border = int(border_ignore_px)
    if border > 0:
        near[:border, :] = False
        near[-border:, :] = False
        near[:, :border] = False
        near[:, -border:] = False
    pixels = int(np.count_nonzero(near))
    largest, bbox = _largest_component(near)
    blocked = largest >= int(min_component_pixels)
    result = {
        "near_field_pixels": pixels,
        "near_field_largest_component_pixels": largest,
        "near_field_largest_component_bbox_xyxy": bbox,
        "near_field_min_depth_m": (
            float(np.min(depth[near])) if pixels else None
        ),
        "near_field_reaction_max_depth_m": limit,
        "near_field_ignored_tool_pixels": int(np.count_nonzero(ignored)),
        "blocked_raw": bool(blocked),
        "reason_raw": "absolute_near_field_object" if blocked else "clear",
    }
    if include_debug_arrays:
        result["_debug"] = {
            "actual_depth_m": depth.copy(),
            "unexpected_mask": near.copy(),
            "ignored_mask": ignored.copy(),
        }
    return result


def build_depth_matrix_reference(
    depth_frames_m: np.ndarray,
    camera_matrix: np.ndarray,
    base_to_camera: np.ndarray,
    *,
    panel_depth_m: float,
    foreground_margin_m: float = 0.06,
    panel_point_base_m: np.ndarray | None = None,
    panel_normal_base: np.ndarray | None = None,
) -> DepthMatrixReference:
    """Build a clear-scene world model plus a camera-rigid gripper layer."""
    stack = np.asarray(depth_frames_m, dtype=float)
    if stack.ndim != 3 or stack.shape[0] < 3:
        raise ValueError("at least three HxW depth frames are required")
    intrinsics = np.asarray(camera_matrix, dtype=float)
    pose = np.asarray(base_to_camera, dtype=float)
    if intrinsics.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("invalid camera matrix or camera pose")
    valid_stack = np.where((stack > 0.08) & (stack < 5.0), stack, np.nan)
    with np.errstate(all="ignore"):
        depth = np.nanmedian(valid_stack, axis=0)
        nearest = np.nanmin(valid_stack, axis=0)
    height, width = depth.shape
    yy, xx = np.indices((height, width))
    # The eye-in-hand gripper is camera-rigid and appears in the lower part of
    # the clear reference image. Preserve it at fixed pixels instead of moving
    # it with the world point cloud.
    tool_candidate = (
        np.isfinite(nearest)
        & (yy >= int(0.55 * height))
        & (nearest < float(panel_depth_m) - float(foreground_margin_m))
    )
    tool_labels, tool_count = ndimage.label(
        tool_candidate, structure=np.ones((3, 3), dtype=int)
    )
    tool_seed = np.zeros_like(tool_candidate)
    if tool_count:
        areas = np.bincount(tool_labels.ravel())
        keep = np.flatnonzero(areas >= 20)
        keep = keep[keep != 0]
        if keep.size:
            tool_seed = np.isin(tool_labels, keep)
    tool_mask = ndimage.binary_dilation(tool_seed, iterations=4)
    valid = np.isfinite(depth) & ~tool_mask
    z = depth[valid]
    x = (xx[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (yy[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1]
    camera_points = np.column_stack((x, y, z, np.ones_like(z)))
    base_points = (pose @ camera_points.T).T[:, :3]
    return DepthMatrixReference(
        depth_m=depth,
        camera_matrix=intrinsics.copy(),
        base_points_m=base_points,
        tool_mask=tool_mask,
        tool_depth_m=np.where(tool_mask, nearest, np.nan),
        reference_base_to_camera=pose.copy(),
        panel_point_base_m=(
            None
            if panel_point_base_m is None
            else np.asarray(panel_point_base_m, dtype=float).copy()
        ),
        panel_normal_base=(
            None
            if panel_normal_base is None
            else np.asarray(panel_normal_base, dtype=float).copy()
        ),
    )


def predict_panel_plane_depth(
    reference: DepthMatrixReference,
    current_base_to_camera: np.ndarray,
) -> np.ndarray | None:
    """Render the detected target panel plane at the current camera pose."""
    if reference.panel_point_base_m is None or reference.panel_normal_base is None:
        return None
    pose = np.asarray(current_base_to_camera, dtype=float)
    camera_to_base_rotation = pose[:3, :3]
    base_to_current_camera = np.linalg.inv(pose)
    point_camera = (
        base_to_current_camera
        @ np.r_[np.asarray(reference.panel_point_base_m, dtype=float), 1.0]
    )[:3]
    normal_camera = camera_to_base_rotation.T @ np.asarray(
        reference.panel_normal_base, dtype=float
    )
    normal_norm = float(np.linalg.norm(normal_camera))
    if normal_norm < 1e-9:
        raise ValueError("panel normal must be nonzero")
    normal_camera /= normal_norm
    height, width = reference.depth_m.shape
    yy, xx = np.indices((height, width))
    k = reference.camera_matrix
    ray_x = (xx - k[0, 2]) / k[0, 0]
    ray_y = (yy - k[1, 2]) / k[1, 1]
    denominator = (
        normal_camera[0] * ray_x
        + normal_camera[1] * ray_y
        + normal_camera[2]
    )
    numerator = float(normal_camera @ point_camera)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = numerator / denominator
    depth[(np.abs(denominator) < 1e-6) | (depth <= 0.08) | (depth >= 5.0)] = np.nan
    return depth


def predict_depth_matrix(
    reference: DepthMatrixReference,
    current_base_to_camera: np.ndarray,
    *,
    splat_radius: int = 2,
) -> np.ndarray:
    """Render the clear reference scene from the current camera pose."""
    pose = np.asarray(current_base_to_camera, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("current_base_to_camera must be a finite 4x4 transform")
    height, width = reference.depth_m.shape
    points = np.column_stack(
        (reference.base_points_m, np.ones(len(reference.base_points_m)))
    )
    camera_points = (np.linalg.inv(pose) @ points.T).T[:, :3]
    z = camera_points[:, 2]
    valid = np.isfinite(z) & (z > 0.08) & (z < 5.0)
    camera_points = camera_points[valid]
    z = z[valid]
    k = reference.camera_matrix
    u = np.rint(k[0, 0] * camera_points[:, 0] / z + k[0, 2]).astype(int)
    v = np.rint(k[1, 1] * camera_points[:, 1] / z + k[1, 2]).astype(int)
    predicted = np.full(height * width, np.inf, dtype=float)
    radius = int(splat_radius)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px, py = u + dx, v + dy
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            np.minimum.at(predicted, py[inside] * width + px[inside], z[inside])
    predicted = predicted.reshape(height, width)
    predicted[~np.isfinite(predicted)] = np.nan
    # Camera-rigid tool pixels remain fixed relative to the RGB-D sensor.
    tool_valid = reference.tool_mask & np.isfinite(reference.tool_depth_m)
    existing = predicted[tool_valid]
    tool = reference.tool_depth_m[tool_valid]
    predicted[tool_valid] = np.where(np.isfinite(existing), np.minimum(existing, tool), tool)
    return predicted


def evaluate_depth_matrix_residual(
    actual_depth_m: np.ndarray,
    predicted_depth_m: np.ndarray,
    *,
    foreground_margin_m: float = 0.06,
    min_component_pixels: int = 120,
    min_coverage_fraction: float = 0.35,
    border_ignore_px: int = 0,
    alignment_tolerance_px: int = 0,
    reaction_max_depth_m: float | None = None,
    known_safe_plane_depth_m: np.ndarray | None = None,
    known_safe_plane_tolerance_m: float = 0.03,
    ignored_mask: np.ndarray | None = None,
    predicted_edge_ignore_dilation_px: int = 0,
    include_debug_arrays: bool = False,
) -> dict:
    """Detect coherent new geometry closer than the predicted clear scene."""
    actual = np.asarray(actual_depth_m, dtype=float)
    predicted = np.asarray(predicted_depth_m, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("actual and predicted depth matrices must share HxW shape")
    tolerance = int(alignment_tolerance_px)
    if tolerance < 0 or tolerance > 20:
        raise ValueError("alignment_tolerance_px must be in 0..20")
    predicted_support = np.isfinite(predicted)
    expected = predicted
    if tolerance:
        # Reprojection is continuous but the depth image is discrete.  When
        # the eye-in-hand camera approaches the panel, a small calibration or
        # pose error shifts panel edges by several pixels.  Compare against
        # the nearest expected safe surface in a bounded neighbourhood so the
        # known panel does not become a false foreground strip.  A genuinely
        # closer object still remains closer than every nearby safe surface.
        expected = ndimage.minimum_filter(
            np.where(np.isfinite(predicted), predicted, np.inf),
            size=2 * tolerance + 1,
            mode="constant",
            cval=np.inf,
        )
        # The neighbourhood is only an alignment correction for pixels that
        # the reference renderer actually predicts.  A minimum filter also
        # dilates finite values into unsupported/disoccluded image regions;
        # treating those invented values as a baseline turns newly revealed
        # fixed scene boundaries into false foreground obstacles.
        expected[~predicted_support] = np.nan
        expected[~np.isfinite(expected)] = np.nan
    actual_valid = np.isfinite(actual) & (actual > 0.08) & (actual < 5.0)
    ignored = np.zeros_like(actual_valid)
    if ignored_mask is not None:
        ignored = np.asarray(ignored_mask, dtype=bool)
        if ignored.shape != actual.shape:
            raise ValueError("ignored mask must match actual depth shape")
    edge_dilation = int(predicted_edge_ignore_dilation_px)
    if edge_dilation < 0 or edge_dilation > 20:
        raise ValueError("predicted_edge_ignore_dilation_px must be in 0..20")
    predicted_edge = np.zeros_like(actual_valid)
    if edge_dilation:
        # A fixed panel boundary is a predicted depth discontinuity.  Small
        # hand-eye/reprojection errors otherwise turn it into a false nearby
        # object.  Mask only a narrow dilated edge band, never an arbitrary
        # image border or interior panel region.
        edge_threshold = float(foreground_margin_m) * 0.5
        horizontal = (
            np.isfinite(predicted[:, 1:])
            & np.isfinite(predicted[:, :-1])
            & (np.abs(predicted[:, 1:] - predicted[:, :-1]) >= edge_threshold)
        )
        vertical = (
            np.isfinite(predicted[1:, :])
            & np.isfinite(predicted[:-1, :])
            & (np.abs(predicted[1:, :] - predicted[:-1, :]) >= edge_threshold)
        )
        predicted_edge[:, 1:] |= horizontal
        predicted_edge[:, :-1] |= horizontal
        predicted_edge[1:, :] |= vertical
        predicted_edge[:-1, :] |= vertical
        predicted_edge = ndimage.binary_dilation(
            predicted_edge, iterations=edge_dilation
        )
        ignored |= predicted_edge
    plane_match = np.zeros_like(actual_valid)
    if known_safe_plane_depth_m is not None:
        plane = np.asarray(known_safe_plane_depth_m, dtype=float)
        if plane.shape != actual.shape:
            raise ValueError("known safe plane depth must match actual depth shape")
        plane_tolerance = float(known_safe_plane_tolerance_m)
        if not 0.0 < plane_tolerance < float(foreground_margin_m):
            raise ValueError(
                "known safe plane tolerance must be positive and below foreground margin"
            )
        plane_match = (
            actual_valid
            & np.isfinite(plane)
            & (np.abs(actual - plane) <= plane_tolerance)
        )
    expected_valid = np.isfinite(expected)
    comparable = actual_valid & (expected_valid | plane_match)
    valid_count = int(np.count_nonzero(actual_valid))
    comparable_count = int(np.count_nonzero(comparable))
    expected_count = int(np.count_nonzero(expected_valid))
    frame_pixel_count = int(actual.size)
    # Report prediction support separately, but define monitoring coverage as
    # the fraction of predicted surfaces the camera actually observed.
    # Dividing by only the actual valid pixels made a nearly blind depth frame
    # report 100%.  Requiring the model itself to fill a fixed fraction of the
    # whole image would instead reject valid close/cropped panel views.
    prediction_support = expected_count / frame_pixel_count if frame_pixel_count else 0.0
    observed_expected = (
        int(np.count_nonzero(actual_valid & expected_valid)) / expected_count
        if expected_count
        else 0.0
    )
    actual_comparable = comparable_count / valid_count if valid_count else 0.0
    residual_foreground = (
        actual_valid
        & ~ignored
        & np.isfinite(expected)
        & ~plane_match
        & (actual < expected - float(foreground_margin_m))
    )
    if reaction_max_depth_m is None:
        unexpected = residual_foreground
    else:
        max_depth = float(reaction_max_depth_m)
        if not 0.08 < max_depth < 5.0:
            raise ValueError("reaction_max_depth_m must be in (0.08, 5.0)")
        unexpected = residual_foreground & (actual < max_depth)
    border = int(border_ignore_px)
    if border > 0:
        unexpected[:border, :] = False
        unexpected[-border:, :] = False
        unexpected[:, :border] = False
        unexpected[:, -border:] = False
    pixels = int(np.count_nonzero(unexpected))
    largest, bbox = _largest_component(unexpected)
    coverage = observed_expected
    available = coverage >= float(min_coverage_fraction)
    # A coherent foreground residual remains meaningful even when much of the
    # expected panel has fallen outside the sensor's usable close range.  The
    # coverage gate controls whether absence of depth is trustworthy; it must
    # not erase positive evidence from the valid pixels that remain.
    residual_detected = largest >= int(min_component_pixels)
    blocked = available and residual_detected
    residuals = expected[unexpected] - actual[unexpected]
    result = {
        "matrix_monitoring_available": bool(available),
        "matrix_coverage_fraction": float(coverage),
        "matrix_prediction_support_fraction": float(prediction_support),
        "matrix_observed_expected_fraction": float(observed_expected),
        "matrix_actual_comparable_fraction": float(actual_comparable),
        "matrix_expected_valid_pixels": expected_count,
        "matrix_frame_pixels": frame_pixel_count,
        "matrix_comparable_pixels": comparable_count,
        "matrix_actual_valid_pixels": valid_count,
        "matrix_unexpected_pixels": pixels,
        "matrix_largest_component_pixels": largest,
        "matrix_largest_component_bbox_xyxy": bbox,
        "matrix_max_foreground_residual_m": (
            float(np.max(residuals)) if residuals.size else 0.0
        ),
        "matrix_border_ignore_px": border,
        "matrix_alignment_tolerance_px": tolerance,
        "matrix_known_safe_plane_pixels": int(np.count_nonzero(plane_match)),
        "matrix_ignored_tool_pixels": int(np.count_nonzero(ignored)),
        "matrix_predicted_edge_ignored_pixels": int(np.count_nonzero(predicted_edge)),
        "matrix_reaction_max_depth_m": (
            None if reaction_max_depth_m is None else float(reaction_max_depth_m)
        ),
        "matrix_residual_pixels_before_depth_gate": int(
            np.count_nonzero(residual_foreground)
        ),
        "matrix_residual_detected_raw": bool(residual_detected),
        "blocked_raw": bool(blocked),
        "reason_raw": (
            "matrix_prediction_coverage_insufficient"
            if not available
            else "unexpected_depth_matrix_residual" if blocked else "clear"
        ),
    }
    if include_debug_arrays:
        result["_debug"] = {
            "actual_depth_m": actual.copy(),
            "predicted_depth_m": predicted.copy(),
            "expected_depth_m": expected.copy(),
            "unexpected_mask": unexpected.copy(),
            "ignored_mask": ignored.copy(),
            "predicted_edge_mask": predicted_edge.copy(),
            "plane_match_mask": plane_match.copy(),
        }
    return result
