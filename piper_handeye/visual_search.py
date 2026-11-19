from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation



@dataclass(frozen=True)
class ButtonGridPosition:
    row: int
    column: int


@dataclass(frozen=True)
class UpwardSearchPlan:
    target_label: str
    target: ButtonGridPosition
    visible_labels: tuple[str, ...]
    highest_visible_row: int
    predicted_lift_m: float
    lift_offsets_m: tuple[float, ...]
    camera_tilt_offsets_deg: tuple[float, ...]
    direction: str


@dataclass(frozen=True)
class ButtonPixelPrediction:
    center_xy: tuple[float, float]
    row_gap_px: float
    column_gap_px: float
    fit_rms_px: float
    labels: tuple[str, ...]
    layout: str
    target_column: int
    rows_above_visible: float
    anchor_label: str
    column_vertical_offset_px: float


def button_grid_position(
    label: str, layout: str = "column_major"
) -> ButtonGridPosition:
    """Map a two-column numbering pattern to a common row.

    This legacy helper describes column-local order only. Runtime target
    inference uses the live YOLO geometry and does not use this table.
    """
    value = int(str(label))
    if layout == "row_major":
        if 1 <= value <= 16:
            return ButtonGridPosition(row=(value - 1) // 2, column=(value - 1) % 2)
        raise ValueError(f"button {label!r} is outside the two-column 1..16 layout")
    if layout != "column_major":
        raise ValueError(f"unsupported button layout {layout!r}")
    if 1 <= value <= 8:
        return ButtonGridPosition(row=value - 1, column=0)
    if 9 <= value <= 16:
        return ButtonGridPosition(row=value - 9, column=1)
    raise ValueError(f"button {label!r} is outside the learned two-column 1..16 layout")


def plan_upward_search(
    target_label: str,
    visible_labels: Iterable[str],
    *,
    origin_highest_visible_row: int = 2,
    row_pitch_m: float = 0.049,
    step_m: float = 0.04,
    margin_m: float = 0.06,
    max_lift_m: float = 0.24,
    camera_tilt_step_deg: float = 6.0,
    camera_tilt_max_deg: float = 30.0,
    layout: str = "column_major",
    camera_rows_above_visible: float | None = None,
    camera_target_column: int | None = None,
) -> UpwardSearchPlan:
    visible = tuple(dict.fromkeys(str(value) for value in visible_labels))
    if camera_rows_above_visible is not None:
        rows_above = float(camera_rows_above_visible)
        if not np.isfinite(rows_above) or rows_above < -0.25:
            raise ValueError("camera-inferred target would require a downward search")
        target = ButtonGridPosition(
            row=max(0, int(round(max(0.0, rows_above)))),
            column=int(camera_target_column or 0),
        )
        if str(target_label) in visible:
            return UpwardSearchPlan(
                str(target_label), target, visible, 0, 0.0, (), (),
                "already_visible",
            )
        predicted = min(
            float(max_lift_m), max(float(step_m), rows_above * float(row_pitch_m))
        )
        limit = min(float(max_lift_m), predicted + float(margin_m))
        count = max(1, int(math.ceil(limit / float(step_m))))
        offsets = tuple(dict.fromkeys(
            min(float(max_lift_m), index * float(step_m))
            for index in range(1, count + 1)
        ))
        tilt_offsets = tuple(
            min(float(camera_tilt_max_deg), index * float(camera_tilt_step_deg))
            for index in range(1, len(offsets) + 1)
        )
        return UpwardSearchPlan(
            str(target_label), target, visible, 0, predicted, offsets,
            tilt_offsets, "up_with_wrist",
        )

    target = button_grid_position(target_label, layout)
    visible_positions = []
    for label in visible:
        try:
            visible_positions.append(button_grid_position(label, layout))
        except (TypeError, ValueError):
            continue
    highest = max(
        [item.row for item in visible_positions] or [int(origin_highest_visible_row)]
    )
    if str(target_label) in visible:
        return UpwardSearchPlan(
            str(target_label), target, visible, highest, 0.0, (), (),
            "already_visible"
        )
    if target.row <= highest:
        # The origin is the lowest permitted camera pose. A missing slot inside
        # the observed span is therefore an occlusion/poor-view case (button 8
        # beside button 1 is the onsite example), never evidence that we should
        # search below origin. Reuse the normal grid-predicted upward/wrist
        # search; do not invent a target-specific motion branch.
        small_lift = min(float(step_m), float(max_lift_m))
        return UpwardSearchPlan(
            str(target_label), target, visible, highest, small_lift,
            (small_lift,), (min(float(camera_tilt_step_deg), float(camera_tilt_max_deg)),),
            "up_with_wrist"
        )
    predicted = min(
        float(max_lift_m), (target.row - highest) * float(row_pitch_m)
    )
    limit = min(float(max_lift_m), predicted + float(margin_m))
    count = max(1, int(math.ceil(limit / float(step_m))))
    offsets = tuple(
        min(float(max_lift_m), index * float(step_m))
        for index in range(1, count + 1)
    )
    offsets = tuple(dict.fromkeys(offsets))
    tilt_offsets = tuple(
        min(float(camera_tilt_max_deg), index * float(camera_tilt_step_deg))
        for index in range(1, len(offsets) + 1)
    )
    return UpwardSearchPlan(
        str(target_label), target, visible, highest, predicted, offsets,
        tilt_offsets, "up_with_wrist"
    )


def camera_center_error_px(
    bbox_xyxy: Iterable[float], image_size: Iterable[int]
) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    width, height = (float(value) for value in image_size)
    return np.asarray([(x1 + x2 - width) * 0.5, (y1 + y2 - height) * 0.5])


def predict_button_center_from_grid(
    target_label: str, detections: Iterable[dict]
) -> ButtonPixelPrediction:
    """Infer numeric columns and target location from the live YOLO boxes.

    No remembered button-to-row table is used. Numeric detections are split by
    their observed X positions, then each visible column learns its own
    label-to-Y progression. A missing target may be extrapolated within or
    above a column, but never below the lowest visible numeric button because
    the recorded origin is the lowest permitted observation pose.
    """
    if not str(target_label).strip().isdigit():
        raise RuntimeError(
            f"target {target_label!r} is not a numeric floor; it must be visible directly"
        )
    numeric = []
    boundary_labels = []
    for item in detections:
        try:
            label = str(item["label"])
            bbox = item["bbox_xyxy"]
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite([x1, y1, x2, y2])):
            continue
        try:
            value = int(label)
        except ValueError:
            boundary_labels.append(label)
            continue
        numeric.append((value, label, (x1 + x2) * 0.5, (y1 + y2) * 0.5))
    if len(numeric) < 4:
        raise RuntimeError("not enough live numeric YOLO boxes to infer columns")

    ordered = sorted(numeric, key=lambda item: item[2])
    x_values = np.asarray([item[2] for item in ordered], dtype=float)
    gaps = np.diff(x_values)
    if len(gaps) == 0:
        raise RuntimeError("live YOLO boxes do not show two numeric columns")
    split = int(np.argmax(gaps)) + 1
    if float(gaps[split - 1]) < 20.0 or split < 2 or len(ordered) - split < 2:
        raise RuntimeError("live YOLO column split is not reliable")
    groups = (ordered[:split], ordered[split:])

    target_value = int(str(target_label))
    column_models = []
    for column, group in enumerate(groups):
        values = np.asarray([item[0] for item in group], dtype=float)
        ys = np.asarray([item[3] for item in group], dtype=float)
        if len(np.unique(values)) < 2:
            continue
        design = np.column_stack((np.ones(len(values)), values))
        coefficients, _, rank, _ = np.linalg.lstsq(design, ys, rcond=None)
        if rank < 2:
            continue
        fitted = design @ coefficients
        rms = float(np.sqrt(np.mean((fitted - ys) ** 2)))
        slope = float(coefficients[1])
        if not np.isfinite(rms) or rms > 6.0 or abs(slope) < 5.0:
            continue
        distinct = sorted(set(int(value) for value in values))
        increments = [
            abs(second - first)
            for first, second in zip(distinct, distinct[1:])
            if second != first
        ]
        sequence_step = 0
        for increment in increments:
            sequence_step = math.gcd(sequence_step, int(increment))
        sequence_step = max(1, sequence_step)
        if (target_value - distinct[0]) % sequence_step:
            continue
        predicted_y = float(coefficients[0] + slope * target_value)
        row_pitch_px = abs(slope) * sequence_step
        lowest_visible_y = float(np.max(ys))
        highest_visible_y = float(np.min(ys))
        # A predicted center more than half a learned row below the lowest
        # visible numeric box contradicts the origin-is-lowest search policy.
        if predicted_y > lowest_visible_y + 0.5 * row_pitch_px:
            continue
        rows_above = max(0.0, (highest_visible_y - predicted_y) / row_pitch_px)
        label_distance_rows = min(
            abs(target_value - value) / sequence_step for value in distinct
        )
        anchor_value = min(
            distinct, key=lambda value: abs(target_value - value)
        )
        anchor_label = next(
            item[1] for item in group if item[0] == anchor_value
        )
        column_models.append({
            "column": column,
            "group": group,
            "rms": rms,
            "slope": slope,
            "row_pitch_px": row_pitch_px,
            "sequence_step": sequence_step,
            "predicted_y": predicted_y,
            "rows_above": rows_above,
            "label_distance_rows": label_distance_rows,
            "anchor_label": anchor_label,
        })
    if not column_models:
        raise RuntimeError(
            "camera evidence gives no upward-only column for the requested button"
        )
    column_models.sort(
        key=lambda model: (
            model["label_distance_rows"], model["rms"], model["column"]
        )
    )
    selected = column_models[0]
    if (
        len(column_models) > 1
        and abs(
            column_models[1]["label_distance_rows"]
            - selected["label_distance_rows"]
        ) < 0.25
        and abs(column_models[1]["rms"] - selected["rms"]) < 1.0
    ):
        raise RuntimeError("live YOLO column inference is ambiguous")
    group = selected["group"]
    center_x = float(np.mean([item[2] for item in group]))
    all_labels = tuple(item[1] for item in numeric)
    column_centers_x = [
        float(np.mean([item[2] for item in candidate])) for candidate in groups
    ]
    # Report the observed vertical mismatch of nearest cross-column pairs
    # (for example 2/9 and 3/10), rather than extrapolating unlike absolute
    # labels to a fictitious common label.
    row_pitch_px = float(selected["row_pitch_px"])
    paired_vertical_offsets = []
    for left in groups[0]:
        right = min(groups[1], key=lambda item: abs(item[3] - left[3]))
        difference = float(right[3] - left[3])
        if abs(difference) <= 0.6 * row_pitch_px:
            paired_vertical_offsets.append(difference)
    column_vertical_offset_px = (
        float(np.median(paired_vertical_offsets))
        if paired_vertical_offsets else float("nan")
    )
    return ButtonPixelPrediction(
        center_xy=(center_x, float(selected["predicted_y"])),
        row_gap_px=float(selected["slope"] * selected["sequence_step"]),
        column_gap_px=float(column_centers_x[1] - column_centers_x[0]),
        fit_rms_px=float(selected["rms"]),
        labels=all_labels,
        layout="camera_inferred_columns",
        target_column=int(selected["column"]),
        rows_above_visible=float(selected["rows_above"]),
        anchor_label=str(selected["anchor_label"]),
        column_vertical_offset_px=column_vertical_offset_px,
    )


def is_button_centered(
    bbox_xyxy: Iterable[float],
    image_size: Iterable[int],
    *,
    tolerance_px: tuple[float, float] = (55.0, 45.0),
) -> bool:
    error = np.abs(camera_center_error_px(bbox_xyxy, image_size))
    return bool(
        error[0] <= float(tolerance_px[0])
        and error[1] <= float(tolerance_px[1])
    )


def centered_camera_translation_base(
    base_to_camera: np.ndarray,
    button_point_base_m: Iterable[float],
    *,
    max_translation_m: float = 0.10,
) -> np.ndarray:
    """Translate the camera parallel to its image plane to center a button.

    Orientation and optical depth stay unchanged. Moving the camera by the
    target's camera-frame X/Y offset places the target on the optical axis.
    """
    pose = np.asarray(base_to_camera, dtype=float)
    point = np.asarray(button_point_base_m, dtype=float)
    if pose.shape != (4, 4) or point.shape != (3,):
        raise ValueError("expected a 4x4 camera pose and a three-vector point")
    point_camera = (np.linalg.inv(pose) @ np.r_[point, 1.0])[:3]
    if not np.isfinite(point_camera).all() or point_camera[2] <= 0.08:
        raise ValueError("button is not in front of the camera")
    translation_camera = np.asarray([point_camera[0], point_camera[1], 0.0])
    magnitude = float(np.linalg.norm(translation_camera))
    if magnitude > float(max_translation_m):
        raise RuntimeError(
            f"camera centering correction {magnitude:.3f}m exceeds "
            f"{float(max_translation_m):.3f}m"
        )
    return pose[:3, :3] @ translation_camera


def raised_tcp_target(
    origin_base_to_tcp: np.ndarray, lift_m: float
) -> np.ndarray:
    pose = np.asarray(origin_base_to_tcp, dtype=float).copy()
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("origin TCP pose must be a finite 4x4 matrix")
    if not 0.0 <= float(lift_m) <= 0.24:
        raise ValueError("search lift must be in 0..0.24 m")
    pose[2, 3] += float(lift_m)
    return pose


def raised_and_tilted_camera_tcp_target(
    origin_base_to_tcp: np.ndarray,
    tcp_to_camera: np.ndarray,
    lift_m: float,
    camera_tilt_deg: float,
) -> np.ndarray:
    """Raise the eye-in-hand camera and tilt its optical axis upward."""
    tcp = np.asarray(origin_base_to_tcp, dtype=float)
    handeye = np.asarray(tcp_to_camera, dtype=float)
    if tcp.shape != (4, 4) or handeye.shape != (4, 4):
        raise ValueError("TCP and hand-eye poses must both be 4x4")
    if not 0.0 <= float(lift_m) <= 0.24:
        raise ValueError("search lift must be in 0..0.24 m")
    if not 0.0 <= float(camera_tilt_deg) <= 30.0:
        raise ValueError("camera search tilt must be in 0..30 degrees")
    desired_camera = (tcp @ handeye).copy()
    desired_camera[2, 3] += float(lift_m)
    desired_camera[:3, :3] = desired_camera[:3, :3] @ Rotation.from_euler(
        "x", float(camera_tilt_deg), degrees=True
    ).as_matrix()
    return desired_camera @ np.linalg.inv(handeye)


def pixel_centered_camera_tcp_target(
    current_base_to_tcp: np.ndarray,
    tcp_to_camera: np.ndarray,
    bbox_xyxy: Iterable[float],
    image_size: Iterable[int],
    camera_matrix: np.ndarray,
    calibration_size: Iterable[int],
    *,
    max_correction_deg: float = 12.0,
) -> np.ndarray:
    """Use wrist orientation to bring a transient detection toward center."""
    tcp = np.asarray(current_base_to_tcp, dtype=float)
    handeye = np.asarray(tcp_to_camera, dtype=float)
    matrix = np.asarray(camera_matrix, dtype=float)
    width, height = (float(value) for value in image_size)
    calib_width, calib_height = (float(value) for value in calibration_size)
    if tcp.shape != (4, 4) or handeye.shape != (4, 4) or matrix.shape != (3, 3):
        raise ValueError("invalid TCP, hand-eye, or camera matrix shape")
    dx, dy = camera_center_error_px(bbox_xyxy, image_size)
    fx = float(matrix[0, 0]) * width / calib_width
    fy = float(matrix[1, 1]) * height / calib_height
    pitch_x = -math.atan2(float(dy), fy)
    yaw_y = math.atan2(float(dx), fx)
    limit = math.radians(float(max_correction_deg))
    pitch_x = float(np.clip(pitch_x, -limit, limit))
    yaw_y = float(np.clip(yaw_y, -limit, limit))
    desired_camera = (tcp @ handeye).copy()
    desired_camera[:3, :3] = desired_camera[:3, :3] @ Rotation.from_rotvec(
        [pitch_x, yaw_y, 0.0]
    ).as_matrix()
    return desired_camera @ np.linalg.inv(handeye)


def predicted_pixel_aim_tcp_target(
    origin_base_to_tcp: np.ndarray,
    tcp_to_camera: np.ndarray,
    predicted_center_xy: Iterable[float],
    camera_matrix: np.ndarray,
    calibration_size: Iterable[int],
    image_size: Iterable[int],
    panel_depth_m: float,
    lift_m: float,
    max_aim_angle_deg: float = 30.0,
) -> np.ndarray:
    """Lift once and pitch the camera toward an extrapolated button row."""
    tcp = np.asarray(origin_base_to_tcp, dtype=float)
    handeye = np.asarray(tcp_to_camera, dtype=float)
    matrix = np.asarray(camera_matrix, dtype=float).copy()
    width, height = (float(value) for value in image_size)
    calib_width, calib_height = (float(value) for value in calibration_size)
    u, v = (float(value) for value in predicted_center_xy)
    depth = float(panel_depth_m)
    if not 0.15 <= depth <= 1.20:
        raise ValueError("panel depth is outside the aiming range")
    if not 0.0 <= float(lift_m) <= 0.24:
        raise ValueError("search lift must be in 0..0.24 m")
    matrix[0, :] *= width / calib_width
    matrix[1, :] *= height / calib_height
    matrix[2, :] = [0.0, 0.0, 1.0]
    ray = np.asarray([
        (u - matrix[0, 2]) / matrix[0, 0],
        (v - matrix[1, 2]) / matrix[1, 1],
        1.0,
    ])
    origin_camera = tcp @ handeye
    target_base = (origin_camera @ np.r_[ray * depth, 1.0])[:3]
    desired_camera = origin_camera.copy()
    desired_camera[2, 3] += float(lift_m)
    direction_camera = origin_camera[:3, :3].T @ (
        target_base - desired_camera[:3, 3]
    )
    # Match the proven 12/15/16 search: elbow lift plus wrist pitch toward the
    # predicted row.  Horizontal pixel displacement is deliberately ignored,
    # so a missing target in another column cannot introduce base/yaw motion.
    pitch = math.atan2(-float(direction_camera[1]), float(direction_camera[2]))
    limit = math.radians(float(max_aim_angle_deg))
    pitch = float(np.clip(pitch, -limit, limit))
    desired_camera[:3, :3] = (
        origin_camera[:3, :3]
        @ Rotation.from_euler("x", pitch).as_matrix()
    )
    return desired_camera @ np.linalg.inv(handeye)


def closer_camera_tcp_target(
    current_base_to_tcp: np.ndarray,
    tcp_to_camera: np.ndarray,
    advance_m: float,
) -> np.ndarray:
    """Move the camera forward along its optical axis without changing view."""
    tcp = np.asarray(current_base_to_tcp, dtype=float)
    handeye = np.asarray(tcp_to_camera, dtype=float)
    if tcp.shape != (4, 4) or handeye.shape != (4, 4):
        raise ValueError("TCP and hand-eye poses must both be 4x4")
    if not 0.0 <= float(advance_m) <= 0.10:
        raise ValueError("close-up advance must be in 0..0.10 m")
    camera = (tcp @ handeye).copy()
    camera[:3, 3] += camera[:3, 2] * float(advance_m)
    return camera @ np.linalg.inv(handeye)
