#!/usr/bin/env python3
"""Replan from six live RGB-D frames while moving, then retreat and return."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from itertools import product
from pathlib import Path
import threading
import time
from piper_handeye.pipeline_timing import traced

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from piper_handeye.realtime_planner import (
    FastIkResult,
    FixedRollFastIkPlanner,
    PRESS_SEGMENT_MAX_JOINT_DELTA_RAD,
    RobotTransformHistory,
    SlidingButtonWindow,
    TimedButtonTransform,
)
from piper_handeye.retention import prune_old_entries
from piper_handeye.robot import PiperUdpRobot
from piper_handeye.robot import PIPER_JOINT_LIMITS
from piper_handeye.near_button_depth_pause import NearButtonDepthPause, NEAR_BUTTON_DEPTH_LEASE_SEC
from piper_handeye.kinematics import (
    elbow_position,
    elbow_positions_batch,
    forward_kinematics,
    forward_kinematics_batch,
    inverse_kinematics,
)
from piper_handeye.rolling_speed import (
    RollingSpeedConfig,
    plan_receding_approach,
)
from piper_handeye.timed_trajectory import (
    PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
    PIPER_MAX_VELOCITIES,
    TrajectoryAudit,
    TimedTrajectoryUdpClient,
    generate_continuous_waypoint_trajectory,
    generate_blended_press_inverse_return,
    generate_press_cycle_trajectory,
    generate_timed_joint_trajectory,
    generate_two_zone_joint_trajectory,
    joint_target_at_panel_distance,
    reverse_timed_trajectory,
)
from piper_handeye.trajectory_splice import (
    PreparedSpliceUdpClient,
    generate_splice_trajectory,
    select_future_handoff,
)
from piper_handeye.vision_rpc import VisionRpcClient
from piper_handeye.call_button_search import (
    down_search_anchor, up_search_anchor, choose_down_search_j5_step,
)
from piper_handeye.observation_visibility import complete_button_box
from piper_handeye.missing_button_capture import save_missing_button_view
from piper_handeye.numeric_visibility import (
    wait_numeric_visibility, recover_numeric_target,
)
from piper_handeye.numeric_search_evidence import numeric_search_evidence
from piper_handeye.safety_stop import SafetyStopUdpClient, VisualSafetySupervisor
from piper_handeye.visual_search import (
    button_grid_position,
    centered_camera_translation_base,
    closer_camera_tcp_target,
    is_button_centered,
    pixel_centered_camera_tcp_target,
    plan_upward_search,
    predict_button_center_from_grid,
    predicted_pixel_aim_tcp_target,
)


ROOT = Path(__file__).resolve().parent
TERMINAL = {"replacement_complete", "fallback_complete", "failed", "aborted"}
ONLINE_IK_MAX_JOINT_DELTA_RAD = 1.00
MAX_PERPENDICULAR_DEVIATION_DEG = 35.0
ELBOW_HEIGHT_NUMERICAL_TOLERANCE_M = 0.0001
LATE_SPLICE_SCHEDULING_LEAD_SEC = 1.0
DEFAULT_PRESS_TRAVEL_M = 0.007
DEFAULT_FALLBACK_HOLD_SEC = 10.0
SAFETY_POSE_PUBLISH_PERIOD_SEC = 0.02


def tcp_panel_distances(
    target_surface_distance_m: float,
    press_travel_m: float,
    material_tip_offset_m: float,
) -> tuple[float, float]:
    """Convert material-surface gaps to distances for the unchanged TCP.

    A positive material offset means the replaceable tip protrudes from the
    current TCP toward the panel.  The TCP therefore stays farther from the
    panel by the same amount.  No robot, URDF, hand-eye, or FK TCP definition
    is changed by this conversion.
    """
    prepress_tcp_distance = float(
        target_surface_distance_m + material_tip_offset_m
    )
    pressed_tcp_distance = float(
        target_surface_distance_m - press_travel_m + material_tip_offset_m
    )
    return prepress_tcp_distance, pressed_tcp_distance


def motion_stage_sequence(press_travel_m: float) -> list[str]:
    """Describe the executed stages without folding press travel into approach."""
    stages = [
        "approach_to_second_capture_prepress",
        "encoder_verify_prepress",
    ]
    if float(press_travel_m) > 0.0:
        stages.extend(
            [
                "independent_press_forward",
                "independent_press_reverse_to_prepress",
            ]
        )
    stages.append("continuous_return_to_origin")
    return stages


def should_center_visible_target(
    *,
    auto_pointing_enabled: bool,
    legacy_center_requested: bool,
    target_visible_at_origin: bool,
    target_was_pointed: bool,
    incomplete_view: bool = False,
) -> bool:
    """Keep legacy centering opt-in; direct visible targets retain their box."""
    return bool(
        auto_pointing_enabled
        and (incomplete_view or (not target_was_pointed and legacy_center_requested))
    )


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def refresh_handoff_after_compute(
    active,
    handoff,
    *,
    active_elapsed_sec: float,
    scheduling_lead_sec: float = LATE_SPLICE_SCHEDULING_LEAD_SEC,
):
    """Move an expired or near-expired splice point to a future active point."""
    elapsed = max(0.0, float(active_elapsed_sec))
    lead = float(scheduling_lead_sec)
    remaining = float(handoff.active_time_sec) - elapsed
    if remaining >= lead:
        return handoff, False, remaining
    refreshed = select_future_handoff(
        active,
        active_elapsed_sec=elapsed,
        lead_sec=lead,
    )
    return refreshed, True, float(refreshed.active_time_sec) - elapsed


def reuse_task_preview(guard_ik, target_position_m, *, max_joint_delta_rad=1.0):
    """Reuse only an identical endpoint from this live task's checked preview."""
    started = time.perf_counter()
    desired = guard_ik.desired_base_to_tcp.copy()
    desired[:3, 3] = target_position_m
    for entry in guard_ik.downstream_preview_solutions:
        if not np.allclose(entry.desired_base_to_tcp, desired, rtol=0., atol=1e-10):
            continue
        joints = entry.joints.copy()
        if not np.isfinite(joints).all() or np.any(joints < PIPER_JOINT_LIMITS[:, 0]) or np.any(joints > PIPER_JOINT_LIMITS[:, 1]):
            continue
        actual = forward_kinematics(joints)
        position_error = float(np.linalg.norm(actual[:3, 3]-desired[:3, 3]))
        orientation_error = float(np.rad2deg(Rotation.from_matrix(
            desired[:3, :3].T @ actual[:3, :3]).magnitude()))
        delta = float(np.max(np.abs(joints-guard_ik.joints)))
        if position_error > .001 or orientation_error > .5 or delta > max_joint_delta_rad or entry.max_step_rad > .25:
            continue
        return FastIkResult(
            joints=joints, desired_base_to_tcp=desired,
            solve_sec=time.perf_counter()-started, position_error_m=position_error,
            orientation_error_deg=orientation_error, max_joint_delta_rad=delta,
            continuation_steps=entry.steps, max_continuation_joint_step_rad=entry.max_step_rad,
            reused_solution=True,
        )
    return None


@traced('ik.refine_precomputed', 'ik')
def refine_precomputed_ik(
    desired_template,
    target_position_m,
    seed_joints,
    *,
    max_joint_delta_rad: float = 0.25,
) -> FastIkResult:
    """Apply a translation locally, using fixed-orientation continuation if needed."""
    desired = np.asarray(desired_template, dtype=float).copy()
    desired[:3, 3] = np.asarray(target_position_m, dtype=float)
    seed = np.asarray(seed_joints, dtype=float)
    started = time.perf_counter()
    joints, position_error, angle_error = inverse_kinematics(
        desired,
        seed,
        regularization=0.001,
        max_nfev=100,
        solver_tolerance=1e-8,
    )
    elapsed = time.perf_counter() - started
    orientation_error_deg = float(np.rad2deg(angle_error))
    joint_delta = float(np.max(np.abs(joints - seed)))
    direct_accepted = not (
        position_error > 0.001
        or orientation_error_deg > 0.5
        or joint_delta > float(max_joint_delta_rad)
    )
    continuation_steps = 1
    maximum_step = joint_delta
    if not direct_accepted:
        start_pose = forward_kinematics(seed)
        start_position = start_pose[:3, 3]
        translation_m = float(np.linalg.norm(desired[:3, 3] - start_position))
        rotation_deg = float(np.rad2deg(
            Rotation.from_matrix(
                start_pose[:3, :3].T @ desired[:3, :3]
            ).magnitude()
        ))
        continuation_steps = max(
            2,
            int(np.ceil(translation_m / 0.01)),
            int(np.ceil(rotation_deg / 2.5)),
        )
        rotations = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(
                np.stack((start_pose[:3, :3], desired[:3, :3]))
            ),
        )
        joints = seed.copy()
        maximum_step = 0.0
        for index in range(1, continuation_steps + 1):
            waypoint = desired.copy()
            fraction = index / continuation_steps
            waypoint[:3, 3] = (
                start_position * (1.0 - fraction)
                + desired[:3, 3] * fraction
            )
            waypoint[:3, :3] = rotations([fraction]).as_matrix()[0]
            next_joints, position_error, angle_error = inverse_kinematics(
                waypoint,
                joints,
                regularization=0.001,
                max_nfev=100,
                solver_tolerance=1e-8,
            )
            orientation_error_deg = float(np.rad2deg(angle_error))
            step_delta = float(np.max(np.abs(next_joints - joints)))
            if (
                position_error > 0.001
                or orientation_error_deg > 0.5
                or step_delta > 0.25
            ):
                raise RuntimeError(
                    "fixed-orientation continuation IK failed: "
                    f"step={index}/{continuation_steps} "
                    f"position={position_error:.6f}m "
                    f"orientation={orientation_error_deg:.3f}deg "
                    f"joint_step={step_delta:.3f}rad"
                )
            maximum_step = max(maximum_step, step_delta)
            joints = next_joints
        joint_delta = float(np.max(np.abs(joints - seed)))
        if joint_delta > float(max_joint_delta_rad):
            raise RuntimeError(
                "fixed-orientation continuation exceeds total joint delta: "
                f"{joint_delta:.3f}rad"
            )
        elapsed = time.perf_counter() - started
    return FastIkResult(
        joints=joints,
        desired_base_to_tcp=desired,
        solve_sec=float(elapsed),
        position_error_m=float(position_error),
        orientation_error_deg=orientation_error_deg,
        max_joint_delta_rad=joint_delta,
        continuation_steps=continuation_steps,
        max_continuation_joint_step_rad=maximum_step,
    )


def solve_independent_press_ik(
    prepress_joints,
    pressed_position_m,
) -> FastIkResult:
    """Solve the short press locally from the verified prepress configuration."""
    seed = np.asarray(prepress_joints, dtype=float)
    if seed.shape != (6,) or not np.isfinite(seed).all():
        raise ValueError("prepress_joints must contain six finite radians")
    return refine_precomputed_ik(
        forward_kinematics(seed),
        np.asarray(pressed_position_m, dtype=float),
        seed,
        max_joint_delta_rad=PRESS_SEGMENT_MAX_JOINT_DELTA_RAD,
    )


def constrain_button_ray_to_reference_plane(
    base_to_camera,
    measured_base_to_button,
    reference_base_to_button,
) -> tuple[np.ndarray, float]:
    """Keep live RGB direction but freeze depth to the initialized panel plane."""
    camera = np.asarray(base_to_camera, dtype=float)
    measured = np.asarray(measured_base_to_button, dtype=float)
    reference = np.asarray(reference_base_to_button, dtype=float)
    if any(value.shape != (4, 4) for value in (camera, measured, reference)):
        raise ValueError("camera, measured button and reference plane must be 4x4")
    origin = camera[:3, 3]
    ray = measured[:3, 3] - origin
    normal = reference[:3, 2].copy()
    normal /= np.linalg.norm(normal)
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-8:
        raise RuntimeError("live button RGB ray is parallel to reference panel plane")
    scale = float(normal @ (reference[:3, 3] - origin)) / denominator
    if scale <= 0.0:
        raise RuntimeError("reference panel intersection is behind the camera")
    constrained = measured.copy()
    raw_position = measured[:3, 3].copy()
    constrained[:3, 3] = origin + scale * ray

    # Retain the live image's in-plane direction while using the one frozen
    # panel normal for every capture in this task.
    axis_x = measured[:3, 0] - normal * float(normal @ measured[:3, 0])
    if np.linalg.norm(axis_x) < 1e-8:
        axis_x = reference[:3, 0].copy()
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    constrained[:3, :3] = np.column_stack((axis_x, axis_y, normal))
    correction = float(np.linalg.norm(constrained[:3, 3] - raw_position))
    return constrained, correction


def load_handeye(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return (
        np.asarray(data["gripper_to_camera"], dtype=float),
        np.asarray(data["camera_matrix"], dtype=float),
        np.asarray(data["distortion"], dtype=float),
    )


def make_button_window(
    config,
    *,
    size=6,
    max_position_rms_m=0.0015,
    max_position_jump_m=0.003,
    max_sample_age_sec=0.18,
    max_span_sec=0.70,
):
    window = SlidingButtonWindow(
        size=int(size),
        min_confidence=float(config["detector"].get("target_confidence", 0.70)),
        max_input_age_sec=float(max_sample_age_sec),
        max_latest_age_sec=float(max_sample_age_sec),
        max_span_sec=float(max_span_sec),
        max_plane_rms_m=float(
            config["button_geometry"].get("max_plane_rms_m", 0.006)
        ),
        max_position_rms_m=float(max_position_rms_m),
        max_normal_rms_deg=1.5,
        max_position_jump_m=float(max_position_jump_m),
    )
    settings=config.get('numeric_layout_localization') or {}
    if settings.get('task_fallback_enabled') is True and settings.get('mode')=='shadow':
        from piper_handeye.numeric_layout_handoff import SourceAwareButtonWindow
        return SourceAwareButtonWindow.from_window(window)
    return window


@traced('localization.stationary_collection', 'localization')
def collect_stationary_target(
    vision,
    robot,
    handeye,
    target_label,
    config,
    *,
    min_capture_unix_sec=None,
    frame_count=4,
):
    """Collect a fresh target before motion; never reuse a saved button pose."""
    from piper_handeye.numeric_layout_handoff import LayoutTaskVision, collect_stationary_task_target
    if isinstance(vision,LayoutTaskVision) and str(target_label)==vision.layout_task_policy.target:
        return collect_stationary_task_target(vision,robot,handeye,target_label,config,
            frame_count=frame_count,min_capture_unix_sec=min_capture_unix_sec)
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count not in (2, 4):
        raise ValueError("stationary frame_count must be 2 or 4")
    # Stationary acquisition includes RGB-D inference latency. Applying the
    # moving-arm 180 ms gate here drops valid ~230 ms observations and starves
    # the stationary window. Keep this budget local to encoder-guarded holds.
    # Recorded stationary samples spanned 1.677 s despite stable geometry and
    # individually fresh inputs. Allow 2 s only in this encoder-guarded hold;
    # Initial localization and search safety references still use four frames.
    # The closed-loop caller may explicitly request two for a later stationary
    # pre-press refresh. All age/geometry/encoder checks remain identical.
    # Return as soon as stable, without imposing a fixed two-second delay.
    def stationary_window():
        return make_button_window(
            config, size=frame_count, max_sample_age_sec=0.35, max_span_sec=2.0
        )

    window = stationary_window()
    acquisition_started = time.perf_counter()
    state_read_started = time.perf_counter()
    stationary_joints = robot.state(max_age=0.35).joints.copy()
    initial_state_read_sec = time.perf_counter() - state_read_started
    min_capture_unix_sec = max(
        time.time(),
        float(min_capture_unix_sec) if min_capture_unix_sec is not None else 0.0,
    )
    last_sequence = 0
    deadline = time.monotonic() + 3.0
    audit = []
    estimate = None
    counters = dict(requests=0, unchanged_reads=0, empty_reads=0, encoder_resets=0)
    request_total_sec = 0.
    request_max_sec = 0.
    robot_reads_total_sec = initial_state_read_sec
    previous_capture_time = None
    previous_received_time = None
    latest_source_rate_hz = None

    def optional_number(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def finish_diagnostics(status, final_estimate):
        accepted_rows = [row for row in audit if row.get('accepted')]
        processing = [row['timing']['source_processing_sec'] for row in audit
                      if row.get('timing', {}).get('source_processing_sec') is not None]
        span = final_estimate.span_sec
        summary = dict(status=status, target_label=str(target_label),
            policy=dict(frame_count=frame_count, input_max_age_sec=.35, latest_max_age_sec=.35,
                        max_span_sec=2., timeout_sec=3., max_encoder_drift_rad=.003),
            elapsed_sec=time.perf_counter() - acquisition_started, **counters,
            accepted_frames_total=len(accepted_rows), rejected_frames_total=len(audit)-len(accepted_rows),
            request_total_sec=request_total_sec, request_max_sec=request_max_sec,
            robot_state_reads_total_sec=robot_reads_total_sec,
            source_processing_mean_sec=float(np.mean(processing)) if processing else None,
            source_processing_max_sec=max(processing) if processing else None,
            observed_accepted_rate_hz=(final_estimate.sample_count-1)/span if span and span > 0 else None,
            source_valid_rate_hz=latest_source_rate_hz,
            window_sample_count=final_estimate.sample_count,
            window_span_sec=final_estimate.span_sec, window_reasons=list(final_estimate.reasons),
            timing_scope='source_processing combines detector and depth; age outside processing includes waiting and scheduling')
        # Preserve the row shape: callers check bbox_xyxy in the final two
        # rows. A separate summary row would invalidate those visibility checks.
        if audit:
            audit[-1]['acquisition_summary'] = summary
        try:
            print(json.dumps({'stationary_target_acquisition': summary}, ensure_ascii=False), flush=True)
        except Exception as exc:
            summary['diagnostic_emit_error'] = f'{type(exc).__name__}: {exc}'
        return summary

    while time.monotonic() < deadline:
        request_started = time.perf_counter()
        response = vision.request(
            "latest", target_label=str(target_label), window_size=6
        )
        request_sec = time.perf_counter() - request_started
        received_time = time.time()
        counters['requests'] += 1
        request_total_sec += request_sec
        request_max_sec = max(request_max_sec, request_sec)
        realtime = response.get('realtime') or {}
        latest_source_rate_hz = optional_number(realtime.get('sample_rate_hz'))
        sample = response.get("sample")
        if not sample or int(sample["sequence"]) <= last_sequence:
            counters['unchanged_reads' if sample else 'empty_reads'] += 1
            time.sleep(0.01)
            continue
        last_sequence = int(sample["sequence"])
        captured_time = float(sample['capture_time_unix_sec'])
        processing_sec = optional_number(sample.get('processing_sec'))
        timing = dict(request_roundtrip_sec=request_sec,
            response_received_unix_sec=received_time,
            server_time_unix_sec=optional_number(response.get('server_time_unix_sec')),
            source_processing_sec=processing_sec,
            age_outside_processing_sec=received_time-captured_time-processing_sec if processing_sec is not None else None,
            capture_interval_sec=captured_time-previous_capture_time if previous_capture_time is not None else None,
            consumed_interval_sec=received_time-previous_received_time if previous_received_time is not None else None,
            source_valid_rate_hz=latest_source_rate_hz)
        previous_capture_time = captured_time
        previous_received_time = received_time
        row = dict(sequence=last_sequence, capture_time_unix_sec=captured_time,
                   sample_age_sec=received_time-captured_time, timing=timing)
        if (
            min_capture_unix_sec is not None
            and float(sample["capture_time_unix_sec"])
            < float(min_capture_unix_sec)
        ):
            audit.append({**row, 'accepted': False, 'reason': 'captured_before_stationary_pose'})
            continue
        state_read_started = time.perf_counter()
        state = robot.state(max_age=0.35)
        state_read_sec = time.perf_counter() - state_read_started
        robot_reads_total_sec += state_read_sec
        timing['robot_state_read_sec'] = state_read_sec
        if float(np.max(np.abs(state.joints - stationary_joints))) > 0.003:
            # Discard the entire acquisition epoch if the arm is still settling.
            stationary_joints = state.joints.copy()
            min_capture_unix_sec = time.time()
            window = stationary_window()
            counters['encoder_resets'] += 1
            audit.append({**row, "accepted": False,
                          "reason": "encoder_motion_during_stationary_capture"})
            continue
        base_to_button = (
            state.base_to_gripper
            @ handeye
            @ np.asarray(sample["camera_to_button"], dtype=float)
        )
        checked_at = time.time()
        accepted = window.add(
            TimedButtonTransform(
                sequence=last_sequence,
                capture_time_unix_sec=float(sample["capture_time_unix_sec"]),
                base_to_button=base_to_button,
                confidence=float(sample["confidence"]),
                plane_rms_m=float(sample["plane_rms_m"]),
            ), now=checked_at,
        )
        rejection_reasons = []
        age = checked_at-captured_time
        if age < -.05 or age > window.max_input_age_sec:
            rejection_reasons.append('sample_age_out_of_range')
        if float(sample['confidence']) < window.min_confidence:
            rejection_reasons.append('confidence_below_threshold')
        if float(sample['plane_rms_m']) > window.max_plane_rms_m:
            rejection_reasons.append('plane_rms_above_threshold')
        audit.append(
            {
                **row,
                "accepted": bool(accepted),
                "sample_age_sec": age,
                "rejection_reasons": rejection_reasons,
                "confidence": float(sample["confidence"]),
                "plane_rms_m": float(sample["plane_rms_m"]),
                "base_to_button_position_m": base_to_button[:3, 3].tolist(),
                "bbox_xyxy": list(sample["bbox_xyxy"]),
            }
        )
        if accepted:
            estimate = window.estimate()
            if estimate.stable:
                finish_diagnostics('ready', estimate)
                return estimate, audit
    estimate = window.estimate()
    diagnostics = finish_diagnostics('timeout', estimate)
    failure_details = {
        "target_label": str(target_label),
        "samples": audit,
        "window_reasons": list(estimate.reasons),
        "acquisition_summary": diagnostics,
    }
    try:
        print(yaml.safe_dump({'stationary_target_failure': failure_details}, sort_keys=False), flush=True)
    except Exception as exc:
        diagnostics['failure_log_error'] = f'{type(exc).__name__}: {exc}'
    save_missing_button_view(vision, target_label, "stationary_target_acquisition_failed")
    failure = RuntimeError("no stable fresh button target before motion")
    failure.stationary_target_failure = failure_details
    raise failure


def concatenate_stopped_trajectories(first, second, *, hold_sec=0.0):
    """Join two zero-velocity-boundary trajectories with an optional hold."""
    output = [dict(point) for point in first]
    offset = float(output[-1]["time_from_start_sec"])
    if hold_sec > 0.0:
        count = max(1, int(np.ceil(float(hold_sec) / 0.10)))
        for stamp in np.linspace(0.10, float(hold_sec), count):
            output.append(
                {
                    "time_from_start_sec": offset + float(stamp),
                    "positions": list(first[-1]["positions"]),
                    "velocities": [0.0] * 6,
                    "accelerations": [0.0] * 6,
                }
            )
        offset += float(hold_sec)
    for point in second[1:]:
        value = dict(point)
        value["time_from_start_sec"] = offset + float(point["time_from_start_sec"])
        output.append(value)
    return output, offset


def visible_target(response, target_label, min_confidence):
    direct = next(
        (
            item
            for item in response.get("detections", [])
            if str(item.get("label")) == str(target_label)
            and float(item.get("confidence", 0.0)) >= float(min_confidence)
        ),
        None,
    )
    if direct is not None:return direct
    from piper_handeye.numeric_layout_handoff import selected_button_region,region_is_inferred
    region=selected_button_region(response)
    return region if region_is_inferred(region) and region['label']==str(target_label) else None


def complete_visible_target(response, target_label, min_confidence):
    target=visible_target(response,target_label,min_confidence)
    return target if target is not None and complete_button_box(
        target.get('bbox_xyxy'),response.get('image_size')) else None


def wait_for_down_visibility(vision, *, min_confidence, sequence_barrier,
                             timeout_sec, required_consecutive):
    """Stable DOWN authorizes localization; stable UP only authorizes search."""
    deadline=time.monotonic()+float(timeout_sec)
    last_sequence=int(sequence_barrier)
    previous={"target": None, "anchor": None}
    streak={"target": 0, "anchor": 0}
    while time.monotonic()<deadline:
        response=vision.request("visible_buttons",min_confidence=.50)
        sequence=response.get('frame_sequence')
        if sequence is None or int(sequence)<=last_sequence:
            time.sleep(.01);continue
        last_sequence=int(sequence)
        try:age=time.time()-float(response.get('capture_time_unix_sec',0))
        except (TypeError,ValueError):age=float('inf')
        if not -.05<=age<=.35:
            streak={"target":0,"anchor":0};previous={"target":None,"anchor":None}
            continue
        detections={"target":complete_visible_target(response,'down',min_confidence),
                    "anchor":down_search_anchor(response,min_confidence)}
        for key,item in detections.items():
            if item is None:
                streak[key]=0;previous[key]=None;continue
            box=np.asarray(item['bbox_xyxy'],dtype=float)
            center=(box[:2]+box[2:])*.5
            streak[key]=(streak[key]+1 if previous[key] is not None and
                         np.linalg.norm(center-previous[key])<=8 else 1)
            previous[key]=center
        if streak['target']>=required_consecutive:
            return {**response,'visibility_resolution':'stable_down_target'},True
        if streak['anchor']>=required_consecutive:
            return {**response,'visibility_resolution':'up_anchor_search_only'},False
    save_missing_button_view(vision,'down','down_target_and_search_anchor_unavailable')
    raise RuntimeError('no stable DOWN target or UP safety anchor; guarded search unavailable')


def wait_for_stable_visibility_after_target_switch(
    vision,
    *,
    target_label,
    min_confidence,
    sequence_barrier,
    timeout_sec=None,
    required_consecutive=2,
    check_hold=None,
):
    """Wait for new-label inference frames instead of racing the YOLO worker."""
    if timeout_sec is None:
        timeout_sec = 4.0 if str(target_label).casefold() in {"up", "down"} or str(target_label).isdigit() else 1.5
    if str(target_label).isdigit():
        try:
            return wait_numeric_visibility(
                vision, target_label=target_label, min_confidence=min_confidence,
                sequence_barrier=sequence_barrier, timeout_sec=timeout_sec,
                required_consecutive=required_consecutive, check_hold=check_hold)
        except RuntimeError as exc:
            if str(exc).startswith('no stable fresh visibility after target switch'):
                save_missing_button_view(vision,target_label,'target_switch_visibility_timeout')
            raise
    if str(target_label).casefold() == 'down':
        return wait_for_down_visibility(vision,min_confidence=min_confidence,
            sequence_barrier=sequence_barrier,timeout_sec=timeout_sec,
            required_consecutive=required_consecutive)
    deadline = time.monotonic() + float(timeout_sec)
    last_sequence = int(sequence_barrier)
    previous_mode = None
    previous_center = None
    consecutive = 0
    while time.monotonic() < deadline:
        response = vision.request("visible_buttons", min_confidence=0.50)
        sequence = response.get("frame_sequence")
        if sequence is None or int(sequence) <= last_sequence:
            time.sleep(0.01)
            continue
        last_sequence = int(sequence)
        if str(target_label).casefold() == "up":
            try:
                age = time.time()-float(response.get("capture_time_unix_sec", 0))
            except (TypeError, ValueError):
                age = float("inf")
            if not -.05 <= age <= .35:
                consecutive=0; previous_mode=None; previous_center=None
                continue
            # An edge UP is a search cue only. It must never be returned as a
            # qualified observation, even when the DOWN reference is missing.
            target = complete_visible_target(response, target_label, min_confidence)
        else:
            target = visible_target(response, target_label, min_confidence)
        if target is not None:
            box = np.asarray(target["bbox_xyxy"], dtype=float)
            center = np.asarray(
                [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]
            )
            mode = "target_visible"
        else:
            if not str(target_label).strip().isdigit():
                if str(target_label).casefold() != "up":
                    previous_mode = None
                    previous_center = None
                    consecutive = 0
                    continue
                anchor = up_search_anchor(response, min_confidence)
                if anchor is None:
                    previous_mode = None
                    previous_center = None
                    consecutive = 0
                    continue
                box = np.asarray(anchor["bbox_xyxy"], dtype=float)
                center = np.asarray(
                    [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]
                )
                mode = ("down_anchor_visible" if anchor['label'] == 'down'
                        else "up_edge_anchor_visible")
            else:
                try:
                    prediction = predict_button_center_from_grid(
                        target_label, response.get("detections", [])
                    )
                except RuntimeError:
                    previous_mode = None
                    previous_center = None
                    consecutive = 0
                    continue
                center = np.asarray(prediction.center_xy, dtype=float)
                mode = "grid_prediction"
        if (
            mode == previous_mode
            and previous_center is not None
            and float(np.linalg.norm(center - previous_center)) <= 8.0
        ):
            consecutive += 1
        else:
            consecutive = 1
        previous_mode = mode
        previous_center = center
        if consecutive >= int(required_consecutive):
            if mode == "up_edge_anchor_visible":
                response = {**response, "visibility_resolution": "up_edge_search_only"}
            elif mode == "down_anchor_visible":
                response = {**response, "visibility_resolution": "down_anchor_search_only"}
            return response, target is not None
    save_missing_button_view(vision, target_label, "target_switch_visibility_timeout")
    raise RuntimeError(
        "no stable fresh visibility after target switch within "
        f"{float(timeout_sec):.1f}s"
    )


def wait_for_target_after_motion(
    vision, *, target_label, min_confidence, timeout_sec=1.5, require_complete=False
):
    """Allow several fresh YOLO frames to acquire a target after a move."""
    deadline = time.monotonic() + float(timeout_sec)
    last_sequence = -1
    last_response = None
    captured_after=time.time();consecutive=0
    while time.monotonic() < deadline:
        response = vision.request("visible_buttons", min_confidence=0.50)
        last_response = response
        sequence = int(response.get("frame_sequence", -1))
        if sequence <= last_sequence:
            time.sleep(0.01)
            continue
        last_sequence = sequence
        target = visible_target(response, target_label, min_confidence)
        if require_complete:
            target=complete_visible_target(response,target_label,min_confidence)
            stamp=float(response.get('capture_time_unix_sec',0))
            if stamp<captured_after or not -.05<=time.time()-stamp<=.35:target=None
            consecutive=consecutive+1 if target is not None else 0
            if consecutive<2:continue
        if target is not None:
            return response, target
    save_missing_button_view(vision, target_label, "target_missing_after_motion")
    return last_response or {"detections": []}, None


def perpendicular_orientation_candidates(button_rotation, roll_deg):
    """Yield the perpendicular tool pose first, then <=35-degree offsets.

    Offsets are expressed around the TCP's three local axes.  Every individual
    axis is bounded by 35 degrees; the candidates are ordered by the smallest
    combined deviation so the perpendicular solution is always preferred.
    """
    base = np.asarray(button_rotation, dtype=float) @ (
        Rotation.from_euler("z", np.deg2rad(float(roll_deg))).as_matrix()
        @ Rotation.from_euler("x", np.pi).as_matrix()
    )
    offsets = sorted(
        product((-35.0, -20.0, -10.0, 0.0, 10.0, 20.0, 35.0), repeat=3),
        key=lambda values: (sum(value * value for value in values), values),
    )
    for values in offsets:
        yield values, base @ Rotation.from_euler("xyz", values, degrees=True).as_matrix()


def alignment_anchor_label(target_label, visible_labels, layout="column_major"):
    """Choose the closest known grid button, preferring the target column."""
    target = button_grid_position(str(target_label), layout)
    candidates = []
    for label in visible_labels:
        try:
            position = button_grid_position(str(label), layout)
        except (TypeError, ValueError):
            continue
        same_column = position.column == target.column
        # A button below the target is the useful upward-search reference.
        upward_distance = max(0, target.row - position.row)
        candidates.append((not same_column, upward_distance, abs(target.row - position.row), str(label)))
    if not candidates:
        raise RuntimeError("no numeric button is available for camera alignment")
    return min(candidates)[3]


def interpolate_trajectory_joints(points, elapsed_sec):
    """Interpolate the commanded joint state for safety-pose publication."""
    times = np.asarray([point["time_from_start_sec"] for point in points], dtype=float)
    joints = np.asarray([point["positions"] for point in points], dtype=float)
    elapsed = float(np.clip(float(elapsed_sec), times[0], times[-1]))
    return np.asarray([
        np.interp(elapsed, times, joints[:, index]) for index in range(6)
    ])


def trajectory_audit_payload(audit):
    """Convert trajectory audit dataclasses to YAML-safe primitives."""
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in audit.__dict__.items()
    }


def first_time_within_panel_distance(points, base_to_button, threshold_m):
    normal = np.asarray(base_to_button[:3, 2], dtype=float)
    origin = np.asarray(base_to_button[:3, 3], dtype=float)
    for point in points:
        tcp = forward_kinematics(np.asarray(point["positions"], dtype=float))[:3, 3]
        distance = float((tcp - origin) @ normal)
        if distance <= float(threshold_m):
            return float(point["time_from_start_sec"]), distance
    raise RuntimeError("active trajectory never enters near-vision distance")


def base_x_distance_frame(base_to_button):
    """Return a distance-only frame whose positive distance is toward -base-X."""
    frame = np.asarray(base_to_button, dtype=float).copy()
    if frame.shape != (4, 4) or not np.isfinite(frame).all():
        raise ValueError("base_to_button must be a finite 4x4 transform")
    frame[:3, 2] = [-1.0, 0.0, 0.0]
    return frame


def compose_executed_approach(active, handoff, replacement):
    """Join the executed active prefix and its replacement into one timeline."""
    points = [dict(point) for point in active[: handoff.active_index + 1]]
    for point in replacement[1:]:
        value = dict(point)
        value["time_from_start_sec"] = (
            handoff.active_time_sec + float(point["time_from_start_sec"])
        )
        points.append(value)
    return points


def audit_sampled_trajectory(points):
    times = np.asarray([p["time_from_start_sec"] for p in points], dtype=float)
    joints = np.asarray([p["positions"] for p in points], dtype=float)
    velocities = np.asarray([p["velocities"] for p in points], dtype=float)
    xyz = forward_kinematics_batch(joints)[:, :3, 3]
    tcp_speeds = np.linalg.norm(np.gradient(xyz, times, axis=0), axis=1)
    joint_peak = np.max(np.abs(velocities), axis=0)
    return TrajectoryAudit(
        duration_sec=float(times[-1]),
        point_count=len(points),
        cartesian_path_m=float(
            np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
        ),
        cartesian_peak_speed_m_s=float(tcp_speeds.max()),
        joint_peak_velocity_rad_s=joint_peak,
        joint_velocity_fraction=joint_peak / PIPER_MAX_VELOCITIES,
    )


def execute_standard(client, points, token, label):
    validation = client.execute(
        points, token=token, task_id=f"{label}-validate-{time.time_ns()}", dry_run=True
    )
    result = client.execute(
        points, token=token, task_id=f"{label}-execute-{time.time_ns()}"
    )
    return validation, result


def wait_terminal(client, token, timeout_sec):
    deadline = time.monotonic() + timeout_sec
    checked_at = time.monotonic()
    last_state = None
    transitions = []
    while time.monotonic() < deadline:
        status = client.execution_status(token=token)
        state = status.get("execution_status")
        now = time.monotonic()
        if state == "safety_paused":
            # A human may need an arbitrary amount of time to clear the work
            # area. Holding safely must not consume the execution timeout.
            deadline += now - checked_at
        checked_at = now
        if state != last_state:
            transitions.append({"local_monotonic": time.monotonic(), **status})
            last_state = state
        if state in TERMINAL:
            return status, transitions
        time.sleep(0.05)
    raise TimeoutError("online splice did not reach a terminal state")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--target-label", default="4")
    parser.add_argument("--vision-start-active-sec", type=float, default=2.2)
    parser.add_argument("--target-distance-m", type=float, default=0.01)
    parser.add_argument(
        "--material-tip-offset-m",
        type=float,
        default=0.0,
        help=(
            "signed replaceable-material protrusion from the unchanged TCP "
            "toward the panel; positive keeps the TCP farther away"
        ),
    )
    parser.add_argument(
        "--press-travel-m",
        type=float,
        default=DEFAULT_PRESS_TRAVEL_M,
        help=(
            "independent post-approach press displacement; the approach still "
            "ends at --target-distance-m"
        ),
    )
    parser.add_argument("--press-speed-m-s", type=float, default=0.05)
    parser.add_argument("--press-joint-velocity-scale", type=float, default=0.75)
    parser.add_argument(
        "--snapshot-at-prepress",
        action="store_true",
        help="save RGB and 16-bit depth after encoder-verified prepress",
    )
    parser.add_argument(
        "--snapshot-at-pointing",
        action="store_true",
        help="save RGB and depth immediately after stopping at the observer pose",
    )
    parser.add_argument("--handoff-remaining-sec", type=float, default=0.8)
    parser.add_argument("--approach-speed-m-s", type=float, default=0.15)
    parser.add_argument("--near-speed-m-s", type=float, default=0.10)
    parser.add_argument(
        "--final-depth-degraded-commit",
        action="store_true",
        help=(
            "experimental: inside the final bounded distance, permit only "
            "depth-matrix coverage loss while retaining residual/collision stops"
        ),
    )
    parser.add_argument(
        "--final-depth-degraded-distance-m", type=float, default=0.05
    )
    parser.add_argument(
        "--final-depth-degraded-speed-scale", type=float, default=0.50
    )
    parser.add_argument("--retreat-speed-m-s", type=float, default=0.15)
    parser.add_argument("--return-speed-m-s", type=float, default=0.25)
    parser.add_argument("--joint-velocity-scale", type=float, default=0.65)
    parser.add_argument("--return-joint-velocity-scale", type=float, default=0.75)
    parser.add_argument("--duration-margin", type=float, default=1.02)
    parser.add_argument(
        "--fallback-hold-sec", type=float, default=DEFAULT_FALLBACK_HOLD_SEC
    )
    parser.add_argument("--near-zone-start-m", type=float, default=0.20)
    parser.add_argument("--near-approach-speed-m-s", type=float, default=0.10)
    parser.add_argument("--receding-schedule", action="store_true")
    parser.add_argument("--dynamic-fallback-only", action="store_true")
    parser.add_argument(
        "--hold-at-fallback",
        action="store_true",
        help=(
            "diagnostic mode: find the closest reachable guarded endpoint, "
            "execute only that fallback, and hold there without returning"
        ),
    )
    parser.add_argument("--pass-through-approach", action="store_true")
    parser.add_argument("--inverse-return", action="store_true")
    parser.add_argument("--predicted-ready-sec", type=float, default=1.0)
    parser.add_argument("--coarse-uncertainty-m", type=float, default=0.02)
    parser.add_argument("--control-latency-sec", type=float, default=0.12)
    parser.add_argument("--near-vision-start-distance-m", type=float, default=0.20)
    parser.add_argument("--minimum-handoff-lead-sec", type=float, default=0.45)
    parser.add_argument(
        "--vision-ready-browse",
        action="store_true",
        help=(
            "if the target is not visible at origin, follow the recorded "
            "approach path to a distant viewing pose, capture it, then return "
            "to origin before the normal approach"
        ),
    )
    parser.add_argument("--browse-panel-distance-m", type=float, default=0.30)
    parser.add_argument("--browse-speed-m-s", type=float, default=0.12)
    parser.add_argument(
        "--auto-pointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "if the target is not visible, make one layout-predicted rise to "
            "locate it; after acquisition use the same direct motion-time "
            "capture and destination replacement as an initially visible target"
        ),
    )
    parser.add_argument(
        "--center-before-approach",
        action="store_true",
        help=(
            "opt in to the legacy extra wrist-centering and close-up stages; "
            "disabled by default"
        ),
    )
    parser.add_argument("--pointing-hold-sec", type=float, default=1.0)
    parser.add_argument("--search-step-m", type=float, default=0.04)
    parser.add_argument("--search-max-lift-m", type=float, default=0.24)
    parser.add_argument("--search-close-step-m", type=float, default=0.08)
    parser.add_argument("--search-min-panel-distance-m", type=float, default=0.35)
    parser.add_argument("--pointing-only", action="store_true")
    parser.add_argument(
        "--hold-at-pointing",
        action="store_true",
        help="with --pointing-only, remain at the validated target-view pose",
    )
    parser.add_argument(
        "--start-from-current-pointing",
        action="store_true",
        help=(
            "start from an already held target-view pose and include that "
            "pose in the continuous return to the recorded origin"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pause-depth-near-button", action="store_true",
                        help="explicit test: permit only depth-matrix coverage loss inside the configured final distance, max 15 s")
    parser.add_argument("--acknowledge-clear-workspace", action="store_true")
    parser.add_argument(
        "--visual-safety", action="store_true",
        help="stop and latch if the RGB-D panel guard detects an obstruction",
    )
    parser.add_argument(
        "--return-without-visual-safety",
        action="store_true",
        help=(
            "after a separate press cycle, disable RGB-D guarding and execute "
            "the exact inverse of the selected approach"
        ),
    )
    args = parser.parse_args()
    if not (args.execute and args.acknowledge_clear_workspace):
        raise RuntimeError("online test requires both physical acknowledgements")
    if not 0.0 <= args.target_distance_m <= 0.08:
        raise ValueError("--target-distance-m must be in 0..0.08 m")
    if not -0.03 <= args.material_tip_offset_m <= 0.03:
        raise ValueError("--material-tip-offset-m must be in -0.03..0.03 m")
    if not 0.0 <= args.press_travel_m <= 0.04:
        raise ValueError("--press-travel-m must be in 0..0.04 m")
    if not 0.01 <= args.press_speed_m_s <= 0.15:
        raise ValueError("--press-speed-m-s must be in 0.01..0.15 m/s")
    if not 0.05 <= args.press_joint_velocity_scale <= 0.75:
        raise ValueError("--press-joint-velocity-scale must be in 0.05..0.75")
    if args.press_travel_m > 0.0 and args.target_distance_m < 0.01:
        raise ValueError("independent press requires at least 0.01 m prepress distance")
    if args.target_distance_m - args.press_travel_m < -0.015 - 1e-9:
        raise ValueError("press endpoint must not exceed 0.015 m beyond the panel")
    prepress_tcp_distance_m, press_endpoint_tcp_distance_m = tcp_panel_distances(
        args.target_distance_m,
        args.press_travel_m,
        args.material_tip_offset_m,
    )
    downstream_preview_distances_m = (
        (prepress_tcp_distance_m, press_endpoint_tcp_distance_m)
        if args.press_travel_m > 0.0
        else (prepress_tcp_distance_m,)
    )
    if not -0.015 <= press_endpoint_tcp_distance_m <= 0.11:
        raise ValueError("material offset puts the TCP outside the permitted panel range")
    if not 0.20 <= args.handoff_remaining_sec <= 2.0:
        raise ValueError("--handoff-remaining-sec must be in 0.20..2.0 s")
    if not -1.0 <= args.vision_start_active_sec <= 10.0:
        raise ValueError("--vision-start-active-sec must be -1 or in 0..10 s")
    if not 0.01 <= args.approach_speed_m_s <= 0.15:
        raise ValueError("--approach-speed-m-s must be in 0.01..0.15 m/s")
    near_speed_limit = 0.15 if args.pass_through_approach else 0.10
    if not 0.005 <= args.near_speed_m_s <= near_speed_limit:
        raise ValueError(
            f"--near-speed-m-s must be in 0.005..{near_speed_limit:.2f} m/s"
        )
    if not 0.02 <= args.final_depth_degraded_distance_m <= 0.10:
        raise ValueError(
            "--final-depth-degraded-distance-m must be in 0.02..0.10 m"
        )
    if not 0.10 <= args.final_depth_degraded_speed_scale <= 0.50:
        raise ValueError(
            "--final-depth-degraded-speed-scale must be in 0.10..0.50"
        )
    if args.pause_depth_near_button and not (args.visual_safety and args.final_depth_degraded_commit):
        raise ValueError("--pause-depth-near-button requires --visual-safety and --final-depth-degraded-commit")
    if args.final_depth_degraded_commit and not args.visual_safety:
        raise ValueError("--final-depth-degraded-commit requires --visual-safety")
    if args.final_depth_degraded_commit and not args.receding_schedule:
        raise ValueError(
            "--final-depth-degraded-commit requires --receding-schedule"
        )
    if (
        args.visual_safety
        and not args.pointing_only
        and not args.dynamic_fallback_only
        and args.target_distance_m < args.final_depth_degraded_distance_m
        and not args.final_depth_degraded_commit
    ):
        raise ValueError(
            "near-panel motion requires --final-depth-degraded-commit; "
            "otherwise expected depth-matrix coverage loss would stop and cache "
            "the active trajectory"
        )
    if not 0.01 <= args.retreat_speed_m_s <= 0.15:
        raise ValueError("--retreat-speed-m-s must be in 0.01..0.15 m/s")
    if not 0.01 <= args.return_speed_m_s <= PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S:
        raise ValueError("--return-speed-m-s must be in 0.01..0.25 m/s")
    if not 0.05 <= args.joint_velocity_scale <= 0.75:
        raise ValueError("--joint-velocity-scale must be in 0.05..0.75")
    if not 0.05 <= args.return_joint_velocity_scale <= 0.75:
        raise ValueError("--return-joint-velocity-scale must be in 0.05..0.75")
    if not 1.02 <= args.duration_margin <= 1.20:
        raise ValueError("--duration-margin must be in 1.02..1.20")
    if not 0.0 <= args.fallback_hold_sec <= 15.0:
        raise ValueError("--fallback-hold-sec must be in 0..15.0 s")
    if args.near_zone_start_m != 0.0 and not 0.11 <= args.near_zone_start_m <= 0.30:
        raise ValueError("--near-zone-start-m must be 0 or in 0.11..0.30 m")
    if not 0.01 <= args.near_approach_speed_m_s <= args.approach_speed_m_s:
        raise ValueError("near approach speed must be in 0.01..approach speed")
    if not 0.0 <= args.predicted_ready_sec <= 5.0:
        raise ValueError("--predicted-ready-sec must be in 0..5.0 s")
    if not 0.005 <= args.coarse_uncertainty_m <= 0.05:
        raise ValueError("--coarse-uncertainty-m must be in 0.005..0.05 m")
    if not 0.02 <= args.control_latency_sec <= 0.50:
        raise ValueError("--control-latency-sec must be in 0.02..0.50 s")
    if args.dynamic_fallback_only and not args.receding_schedule:
        raise ValueError("--dynamic-fallback-only requires --receding-schedule")
    if args.hold_at_fallback:
        if not args.dynamic_fallback_only:
            raise ValueError("--hold-at-fallback requires --dynamic-fallback-only")
        if not args.visual_safety:
            raise ValueError("--hold-at-fallback requires --visual-safety")
        if args.press_travel_m != 0.0:
            raise ValueError("--hold-at-fallback requires zero press travel")
    if args.return_without_visual_safety and not args.visual_safety:
        raise ValueError("--return-without-visual-safety requires --visual-safety")
    if args.return_without_visual_safety and not args.inverse_return:
        raise ValueError(
            "--return-without-visual-safety requires --inverse-return so the "
            "10 cm waypoint return cannot be selected"
        )
    if args.pass_through_approach and not args.receding_schedule:
        raise ValueError("--pass-through-approach requires --receding-schedule")
    if not 0.11 <= args.near_vision_start_distance_m <= 0.34:
        raise ValueError("--near-vision-start-distance-m must be in 0.11..0.34 m")
    if not 0.30 <= args.minimum_handoff_lead_sec <= 0.80:
        raise ValueError("--minimum-handoff-lead-sec must be in 0.30..0.80 s")
    if not 0.25 <= args.browse_panel_distance_m <= 0.40:
        raise ValueError("--browse-panel-distance-m must be in 0.25..0.40 m")
    if not 0.02 <= args.browse_speed_m_s <= 0.15:
        raise ValueError("--browse-speed-m-s must be in 0.02..0.15 m/s")
    if not 0.6 <= args.pointing_hold_sec <= 2.0:
        raise ValueError("--pointing-hold-sec must be in 0.6..2.0 s")
    if not 0.02 <= args.search_step_m <= 0.06:
        raise ValueError("--search-step-m must be in 0.02..0.06 m")
    if not 0.08 <= args.search_max_lift_m <= 0.24:
        raise ValueError("--search-max-lift-m must be in 0.08..0.24 m")
    if not 0.02 <= args.search_close_step_m <= 0.10:
        raise ValueError("--search-close-step-m must be in 0.02..0.10 m")
    if not 0.32 <= args.search_min_panel_distance_m <= 0.45:
        raise ValueError("--search-min-panel-distance-m must be in 0.32..0.45 m")
    auto_pointing_enabled = bool(args.auto_pointing and args.receding_schedule)
    if args.pointing_only and not auto_pointing_enabled:
        raise ValueError("--pointing-only requires --receding-schedule and auto pointing")
    if args.hold_at_pointing and not args.pointing_only:
        raise ValueError("--hold-at-pointing requires --pointing-only")
    if args.start_from_current_pointing and not auto_pointing_enabled:
        raise ValueError(
            "--start-from-current-pointing requires receding auto pointing"
        )
    if args.start_from_current_pointing and args.pointing_only:
        raise ValueError(
            "--start-from-current-pointing is an approach mode, not pointing-only"
        )
    if args.center_before_approach and not auto_pointing_enabled:
        raise ValueError("--center-before-approach requires auto pointing")

    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    safety_stop_cfg = config["safety_stop"]
    hold = yaml.safe_load(args.hold_report.expanduser().resolve().read_text())
    origin = np.asarray(hold["stage_1_states"][0]["joints_rad"], dtype=float)
    preapproach = np.asarray(hold["stage_1"]["planned_joints_rad"][-1], dtype=float)
    roll_deg = float(hold["stage_1"]["selected_approach_roll_deg"])
    handeye, full_camera_matrix, _ = load_handeye(
        resolve_path(config["calibration"]["handeye_yaml"], config_path)
    )
    saved_coarse_button = np.asarray(
        hold["stage_1"]["base_to_button"], dtype=float
    )
    coarse_button = saved_coarse_button.copy()
    initial_live_estimate = None
    initial_live_samples = []
    vision_ready_browse = None
    pointing_joints = origin.copy()
    pointing_audit = None
    carry_pointing_pose = False
    target_was_pointed = False
    numeric_target_recovered = False
    probe_segments = []
    initial_coarse_guard_ik = None
    initial_coarse_guard_ik_mode = None
    initial_receding_plan = None
    processing_latency = {}
    origin_processing_started_perf = None
    origin_phase_started_perf = None

    def record_origin_phase(name):
        nonlocal origin_phase_started_perf
        now = time.perf_counter()
        processing_latency[name] = float(now - origin_phase_started_perf)
        origin_phase_started_perf = now

    rise_motion_completed_perf = None
    guard_ik_ready_perf = None
    safety_anchor_label = str(args.target_label)
    approach_safety_target_label = str(args.target_label)
    pointing_safety_preparation = None
    closest_reachable_search = None
    initial_prepress_seed_joints = None
    initial_prepress_desired_tcp = None
    initial_prepress_seed_audit = None
    initialized_panel_plane = None
    initialized_panel_plane_source = None
    if args.receding_schedule:
        service_cfg = config["vision_service"]
        initial_vision = VisionRpcClient(
            service_cfg["host"], int(service_cfg["port"]),
            service_cfg["token"], timeout=2.0,
        )
        from piper_handeye.numeric_layout_handoff import task_vision,require_layout_server_support,LayoutTaskVision
        initial_vision=task_vision(initial_vision,config,args.target_label)
        if isinstance(initial_vision,LayoutTaskVision):
            require_layout_server_support(initial_vision.request('health'),config,args.target_label)
        target_switch = initial_vision.request(
            "set_target", target_label=str(args.target_label)
        )
        target_switch_barrier = int(target_switch.get("barrier_sequence", 0))
        initial_robot = PiperUdpRobot(config["robot"])
        probe_guard = None
        probe_pose_stop = threading.Event()
        probe_pose_thread = None
        try:
            initial_robot.open()
            initial_state = initial_robot.state(max_age=2.0)
            numeric_hold_joints = initial_state.joints.copy()
            numeric_stop_checked_at = -float('inf')

            def check_numeric_hold():
                nonlocal numeric_stop_checked_at
                state = initial_robot.state(max_age=.35)
                if (state.raw.get('hardware_ok') != 1
                        or not np.isfinite(state.joints).all()
                        or np.max(np.abs(state.joints-numeric_hold_joints)) > .003):
                    raise RuntimeError('arm moved or hardware failed during stationary numeric observation')
                if time.monotonic()-numeric_stop_checked_at >= 1.:
                    stop_cfg = config['safety_stop']
                    health = SafetyStopUdpClient(stop_cfg['host'],int(stop_cfg['port']),
                                                 timeout_sec=2.).health(token=str(stop_cfg['token']))
                    numeric_stop_checked_at = time.monotonic()
                    if health.get('latched'):
                        raise RuntimeError('independent safety stop latched during numeric observation')
            origin_processing_started_perf = time.perf_counter()
            origin_phase_started_perf = origin_processing_started_perf
            initial_error = float(np.max(np.abs(initial_state.joints - origin)))
            if initial_state.raw.get("hardware_ok") != 1:
                raise RuntimeError("robot hardware is not OK before live aim")
            if not args.start_from_current_pointing and initial_error > 0.03:
                raise RuntimeError(
                    f"arm is not at recorded origin before live aim: {initial_error:.4f} rad"
                )
            if args.start_from_current_pointing:
                pointing_joints = initial_state.joints.copy()
            initial_vision.request(
                "safety_camera_pose",
                timestamp_monotonic=time.monotonic(),
                base_to_camera=(initial_state.base_to_gripper @ handeye).tolist(),
            )
            trajectory_cfg = config["timed_trajectory"]
            browse_token = str(trajectory_cfg["token"])
            browse_client = TimedTrajectoryUdpClient(
                trajectory_cfg["host"], int(trajectory_cfg["port"]), timeout_sec=30.0
            )
            if args.visual_safety:
                probe_pose_vision = VisionRpcClient(
                    service_cfg["host"], int(service_cfg["port"]),
                    service_cfg["token"], timeout=0.20,
                )

                def publish_probe_pose():
                    latest = -1.0
                    while not probe_pose_stop.is_set():
                        try:
                            sample = initial_robot.state(max_age=1.0)
                            if sample.timestamp > latest:
                                probe_pose_vision.request(
                                    "safety_camera_pose",
                                    timestamp_monotonic=float(sample.timestamp),
                                    base_to_camera=(
                                        sample.base_to_gripper @ handeye
                                    ).tolist(),
                                )
                                latest = sample.timestamp
                        except (OSError, RuntimeError, TimeoutError):
                            pass
                        time.sleep(0.005)

                probe_pose_thread = threading.Thread(
                    target=publish_probe_pose, daemon=True
                )
                probe_pose_thread.start()

                def arm_probe_guard():
                    nonlocal probe_guard
                    if probe_guard is not None:
                        return
                    reference_started = time.perf_counter()
                    initial_vision.request(
                        "safety_capture_matrix_reference", frame_count=25
                    )
                    processing_latency["origin_guard_matrix_reference_sec"] = time.perf_counter() - reference_started
                    preflight_started = time.perf_counter()
                    initial_vision.request(
                        "safety_set_mode", mode="absolute_near_field"
                    )
                    stop_cfg = config["safety_stop"]
                    probe_guard = VisualSafetySupervisor(
                        initial_vision,
                        SafetyStopUdpClient(
                            stop_cfg["host"], int(stop_cfg["port"]),
                            timeout_sec=2.0,
                        ),
                        str(stop_cfg["token"]),
                    )
                    probe_guard.start()
                    probe_guard.preflight(3.0)
                    probe_guard.arm()
                    processing_latency["origin_guard_mode_preflight_sec"] = time.perf_counter() - preflight_started
            probe_segments = []
            probe_events = []
            search_plan = None
            pixel_prediction = None
            probe_error = None

            def execute_probe_segment(points, label):
                validation = browse_client.execute(
                    points,
                    token=browse_token,
                    task_id=f"{label}-validate-{time.time_ns()}",
                    dry_run=True,
                )
                # Register the inverse before sending the physical request.
                # A UDP response timeout does not prove the arm stayed still.
                probe_segments.append(points)
                target_joints = np.asarray(points[-1]["positions"], dtype=float)

                def execute_once(active_points, active_label):
                    predicted_pose_stop = threading.Event()
                    predicted_pose_thread = None
                    if probe_guard is not None:
                        trajectory_started = time.monotonic() + 0.15

                    def publish_commanded_probe_pose():
                        while not predicted_pose_stop.is_set():
                            now = time.monotonic()
                            try:
                                if probe_guard.stop_active.is_set():
                                    # Once cancelled, the commanded timeline is
                                    # no longer authoritative. The arm is held,
                                    # so repeatedly timestamp the latest actual
                                    # encoder pose to let the guard evaluate the
                                    # stopped scene and release only when clear.
                                    state = initial_robot.state(max_age=1.0)
                                    base_to_camera = (
                                        state.base_to_gripper @ handeye
                                    )
                                else:
                                    commanded = interpolate_trajectory_joints(
                                        active_points, now - trajectory_started
                                    )
                                    base_to_camera = (
                                        forward_kinematics(commanded) @ handeye
                                    )
                                probe_pose_vision.request(
                                    "safety_camera_pose",
                                    timestamp_monotonic=now,
                                    base_to_camera=base_to_camera.tolist(),
                                )
                            except (OSError, RuntimeError, TimeoutError):
                                pass
                            time.sleep(0.02)

                        predicted_pose_thread = threading.Thread(
                            target=publish_commanded_probe_pose, daemon=True
                        )
                        predicted_pose_thread.start()
                    try:
                        return browse_client.execute(
                            active_points,
                            token=browse_token,
                            task_id=f"{active_label}-execute-{time.time_ns()}",
                        )
                    finally:
                        predicted_pose_stop.set()
                        if predicted_pose_thread is not None:
                            predicted_pose_thread.join(timeout=2.0)

                active_points = points
                for interruption in range(4):
                    try:
                        return validation, execute_once(active_points, label)
                    except RuntimeError:
                        if (
                            probe_guard is None
                            or not (
                                probe_guard.triggered.is_set()
                                or probe_guard.stop_active.is_set()
                                or probe_guard.resumed.is_set()
                            )
                            or interruption >= 3
                        ):
                            raise
                        # Prevent the supervisor from rearming against the old
                        # commanded timeline while a replacement is generated.
                        probe_guard.disarm()
                        # A deliberate camera cover may last arbitrarily long.
                        # Keep the robot stopped and retain the live task until
                        # the supervisor observes a stable clear view and
                        # explicitly releases its latch.
                        while not probe_guard.resumed.wait(timeout=1.0):
                            if probe_guard.error:
                                raise RuntimeError(
                                    "visual safety recovery failed: "
                                    f"{probe_guard.error}"
                                )
                        actual = initial_robot.state(max_age=2.0)
                        active_points, resume_audit = generate_timed_joint_trajectory(
                            actual.joints,
                            target_joints,
                            cartesian_speed_m_s=min(float(args.browse_speed_m_s), 0.10),
                            joint_velocity_scale=float(args.joint_velocity_scale),
                            point_period_sec=0.10,
                            duration_margin=float(args.duration_margin),
                        )
                        resume_validation = browse_client.execute(
                            active_points,
                            token=browse_token,
                            task_id=(
                                f"{label}-resume-{interruption + 1}-validate-"
                                f"{time.time_ns()}"
                            ),
                            dry_run=True,
                        )
                        probe_events.append({
                            "kind": "visual_obstruction_resume",
                            "interruption": interruption + 1,
                            "start_joints_rad": actual.joints.tolist(),
                            "audit": trajectory_audit_payload(resume_audit),
                            "validation": resume_validation,
                        })
                        probe_guard.rearm_after_recovery()
                raise AssertionError("unreachable")
            record_origin_phase("origin_setup_sec")
            try:
                origin_elbow_z = float(elbow_position(origin)[2])
                try:
                    # A full six-frame target window cannot succeed when the
                    # detector already confirms that the requested label is
                    # outside the origin view. Skip the old three-second
                    # timeout and begin the proven upward search immediately.
                    visible, target_visible_at_origin = (
                        wait_for_stable_visibility_after_target_switch(
                            initial_vision,
                            target_label=args.target_label,
                            min_confidence=config["detector"].get(
                                "target_confidence", 0.70
                            ),
                            sequence_barrier=target_switch_barrier,
                            check_hold=check_numeric_hold,
                        )
                    )
                    record_origin_phase("origin_visibility_sec")
                    if not target_visible_at_origin:
                        raise RuntimeError(
                            "no stable fresh button target before motion"
                        )
                    initial_live_estimate, initial_live_samples = (
                        collect_stationary_target(
                            initial_vision, initial_robot, handeye,
                            args.target_label, config,
                        )
                    )
                    initialized_panel_plane = (
                        initial_live_estimate.base_to_button.copy()
                    )
                    initialized_panel_plane_source = "requested_target_at_origin"
                except RuntimeError as exc:
                    if (
                        not auto_pointing_enabled
                        or str(exc) != "no stable fresh button target before motion"
                    ):
                        raise
                    detections = visible.get("detections", [])
                    visible_labels = [item["label"] for item in detections]
                    min_confidence = float(
                        config["detector"].get("target_confidence", 0.70)
                    )
                    search_label = str(args.target_label).casefold()
                    up_anchor = (up_search_anchor(visible, min_confidence)
                                 if search_label == "up" else None)
                    up_down_search = (
                        search_label == "up"
                        and up_anchor is not None
                    ) or (search_label == "down" and
                          down_search_anchor(visible, min_confidence) is not None)
                    if up_down_search:
                        pixel_prediction = None
                        safety_anchor = (str(up_anchor['label'])
                                         if search_label == "up" else "up")
                    else:
                        if str(args.target_label).isdigit():
                            detections, pixel_prediction, search_evidence = numeric_search_evidence(
                                visible, args.target_label, min_confidence)
                            visible_labels = [item['label'] for item in detections]
                            if search_evidence['source'] == 'numeric_tile_context':
                                probe_events.append(dict(kind='numeric_search_neighbour_context',
                                                         **search_evidence))
                        else:
                            pixel_prediction = predict_button_center_from_grid(
                                args.target_label, detections
                            )
                        safety_anchor = pixel_prediction.anchor_label
                    if not up_down_search:
                        safety_anchor_label = str(safety_anchor)
                        initial_vision.request(
                            "set_target", target_label=safety_anchor
                        )
                        anchor_capture_joints = initial_robot.state(max_age=.35).joints.copy()
                        anchor_estimate, anchor_samples = collect_stationary_target(
                            initial_vision, initial_robot, handeye,
                            safety_anchor, config,
                        )
                        initialized_panel_plane = (
                            anchor_estimate.base_to_button.copy()
                        )
                        initialized_panel_plane_source = (
                            f"origin_alignment_anchor_{safety_anchor}"
                        )
                        record_origin_phase("origin_anchor_sampling_sec")
                        anchor_state = initial_robot.state(max_age=2.0)
                        anchor_camera = anchor_state.base_to_gripper @ handeye
                        panel_depth_m = float(
                            (np.linalg.inv(anchor_camera) @ np.r_[
                                anchor_estimate.base_to_button[:3, 3], 1.0
                            ])[2]
                        )
                    else:
                        # The depth-panel guard needs a detected button pose
                        # before it can build its matrix reference. Use the
                        # selected high-confidence call button only as a panel
                        # safety reference. An edge UP cue cannot authorize a
                        # target observation or press before J5 reacquisition.
                        safety_anchor_label = str(safety_anchor)
                        initial_vision.request(
                            "set_target", target_label=safety_anchor_label
                        )
                        anchor_estimate, anchor_samples = collect_stationary_target(
                            initial_vision,
                            initial_robot,
                            handeye,
                            safety_anchor_label,
                            config,
                        )
                        initialized_panel_plane = (
                            anchor_estimate.base_to_button.copy()
                        )
                        initialized_panel_plane_source = f"{safety_anchor_label}_safety_reference"
                        record_origin_phase("origin_anchor_sampling_sec")
                        # Sampling and the independent safety worker run
                        # concurrently. Confirm the reference is installed
                        # before arming the guard that consumes it.
                        reference_deadline = time.monotonic() + 1.0
                        while True:
                            try:
                                initial_vision.request(
                                    "safety_probe", roi_scales=[3.2]
                                )
                                break
                            except RuntimeError as reference_exc:
                                if (
                                    "reference is not established"
                                    not in str(reference_exc)
                                    or time.monotonic() >= reference_deadline
                                ):
                                    raise
                                time.sleep(0.02)
                    if str(args.target_label).isdigit() and not up_down_search:
                        recovered_plan = plan_upward_search(
                            args.target_label,visible_labels,
                            step_m=float(args.search_step_m),max_lift_m=float(args.search_max_lift_m),
                            camera_rows_above_visible=pixel_prediction.rows_above_visible,
                            camera_target_column=pixel_prediction.target_column)
                        if recovered_plan.direction == 'already_visible':
                            # A weak candidate may have recovered while the
                            # reference was sampled. Rejoin the normal path;
                            # never classify "already visible" as search failure.
                            initial_vision.request('set_perception_target',target_label=str(args.target_label))
                            initial_live_estimate, initial_live_samples, pointing_joints, reused = recover_numeric_target(
                                args.target_label,safety_anchor,anchor_estimate,anchor_samples,
                                anchor_capture_joints,initial_robot,
                                lambda: collect_stationary_target(initial_vision,initial_robot,handeye,
                                                                  args.target_label,config),
                                image_size=config['depth'].get('capture_size',[640,360]),
                                min_confidence=min_confidence)
                            initialized_panel_plane = initial_live_estimate.base_to_button.copy()
                            initialized_panel_plane_source = 'requested_target_recovered_at_origin'
                            target_visible_at_origin = True
                            numeric_target_recovered = True
                            probe_events.append(dict(kind='numeric_target_recovered_without_search',
                                                     target_label=str(args.target_label),reused_reference=reused))
                    if args.visual_safety and not numeric_target_recovered:
                        arm_probe_guard()
                    record_origin_phase("origin_guard_setup_sec")
                    if up_down_search:
                        # UP keeps the validated negative-J5 search. DOWN
                        # chooses its wrist direction from calibrated FK and
                        # a high-confidence UP plane, never a weak DOWN guess.
                        current_joints = initial_robot.state(max_age=2.0).joints.copy()
                        probe_events.append({
                            "kind": f"{search_label}_button_j5_search_started",
                            "trigger_label": (
                                "up_edge_without_complete_down" if search_label == "up" and safety_anchor == "up"
                                else "down_complete_up_missing_or_clipped" if search_label == "up"
                                else "up_anchor_down_unstable"
                            ),
                            "safety_anchor_label": safety_anchor_label,
                            "start_joints_rad": current_joints.tolist(),
                            "direction": "negative_j5" if search_label == "up" else "calibrated_view_improvement",
                            "step_rad": 0.05,
                            "step_count_limit": 4,
                            "total_j5_limit_rad": 0.20,
                        })
                        initial_vision.request(
                            "set_perception_target", target_label=search_label
                        )
                        found_call = False
                        cumulative_j5_travel = 0.0
                        for search_step in range(1, 5):
                            next_joints = current_joints.copy()
                            direction_audit = {}
                            if search_label == "down":
                                next_joints, direction_audit = choose_down_search_j5_step(
                                    current_joints, handeye, anchor_estimate.base_to_button[:3, 3],
                                    full_camera_matrix,
                                    config["calibration"].get("rgb_calibration_size", [1920, 1080]),
                                    visible["image_size"])
                            else:
                                next_joints[4] -= 0.05
                            if (
                                next_joints[4] < PIPER_JOINT_LIMITS[4, 0]
                                or next_joints[4] > PIPER_JOINT_LIMITS[4, 1]
                            ):
                                raise RuntimeError(
                                    "call-button J5 search reached its joint limit"
                                )
                            segment, segment_audit = generate_timed_joint_trajectory(
                                current_joints,
                                next_joints,
                                cartesian_speed_m_s=float(args.browse_speed_m_s),
                                joint_velocity_scale=0.30,
                                point_period_sec=0.10,
                                duration_margin=float(args.duration_margin),
                            )
                            segment_joints = np.asarray(
                                [point["positions"] for point in segment]
                            )
                            if float(np.max(np.abs(
                                segment_joints[:, 4] - current_joints[4]
                            ))) > 0.050001:
                                raise RuntimeError("J5 search step exceeded 0.05 rad")
                            validation, result = execute_probe_segment(
                                segment, f"{search_label}-button-j5-search-{search_step}"
                            )
                            actual = initial_robot.state(max_age=2.0)
                            if search_label == "down" or safety_anchor == "up":
                                actual_step = abs(float(actual.joints[4] - current_joints[4]))
                                cumulative_j5_travel += actual_step
                                if actual_step > .055 or cumulative_j5_travel > .205:
                                    raise RuntimeError(f"{search_label.upper()} J5 search exceeded encoder travel budget")
                            current_joints = actual.joints.copy()
                            initial_vision.request(
                                "safety_camera_pose",
                                timestamp_monotonic=time.monotonic(),
                                base_to_camera=(
                                    actual.base_to_gripper @ handeye
                                ).tolist(),
                            )
                            fresh, call_detection = wait_for_target_after_motion(
                                initial_vision,
                                target_label=search_label,
                                min_confidence=min_confidence,
                                timeout_sec=1.5 if search_label == "down" else 1.0,
                                require_complete=True,
                            )
                            step_event = {
                                "kind": f"{search_label}_button_j5_search_step",
                                "step": search_step,
                                "direction_prediction": direction_audit,
                                "j5_rad": float(current_joints[4]),
                                "audit": trajectory_audit_payload(segment_audit),
                                "validation": validation,
                                "result": result,
                                "visible_labels": [
                                    item["label"]
                                    for item in fresh.get("detections", [])
                                ],
                                "found": call_detection is not None,
                            }
                            probe_events.append(step_event)
                            if call_detection is None:
                                continue
                            pointing_joints = current_joints.copy()
                            target_was_pointed = True
                            stationary_after_unix_sec = time.time()
                            try:
                                initial_live_estimate, initial_live_samples = (
                                    collect_stationary_target(
                                        initial_vision, initial_robot, handeye, search_label,
                                        config, min_capture_unix_sec=stationary_after_unix_sec))
                            except RuntimeError as acquire_error:
                                if (search_label != "down" or
                                        str(acquire_error) != "no stable fresh button target before motion"):
                                    raise
                                initial_live_estimate = None
                                initial_live_samples = []
                                probe_events.append({"kind": "down_search_depth_not_stable",
                                                     "step": search_step, "error": str(acquire_error)})
                                continue
                            initialized_panel_plane = (
                                initial_live_estimate.base_to_button.copy()
                            )
                            initialized_panel_plane_source = (
                                f"{search_label}_button_depth_after_j5_search"
                            )
                            probe_events.append({
                                "kind": f"{search_label}_button_acquired_after_j5_search",
                                "source_bbox_xyxy": list(
                                    call_detection["bbox_xyxy"]
                                ),
                                "stable_samples": len(initial_live_samples),
                            })
                            found_call = True
                            break
                        if not found_call:
                            raise RuntimeError(
                                f"{search_label.upper()} button not fully visible within four J5 steps (total travel <=0.20 rad)"
                            )
                    elif not numeric_target_recovered:
                        # Direct grid-prediction search: retain the existing
                        # numeric-floor behavior unchanged.
                        current_joints = origin.copy()
                        origin_tcp = forward_kinematics(origin)
                        initial_vision.request(
                            "set_perception_target", target_label=str(args.target_label)
                        )
                        search_plan = plan_upward_search(
                            args.target_label,
                            visible_labels,
                            step_m=float(args.search_step_m),
                            max_lift_m=float(args.search_max_lift_m),
                            camera_rows_above_visible=(
                                pixel_prediction.rows_above_visible
                            ),
                            camera_target_column=pixel_prediction.target_column,
                        )
                        if search_plan.direction != "up_with_wrist":
                            raise RuntimeError(
                                "selected button is not visible and the learned layout "
                                "does not permit an upward-only search"
                            )
                    if not up_down_search and not numeric_target_recovered:
                        stop_cfg = config["safety_stop"]
                        reset = SafetyStopUdpClient(
                            stop_cfg["host"], int(stop_cfg["port"]), timeout_sec=2.0
                        ).reset_latch(token=str(stop_cfg["token"]))
                        # Use the layout-predicted height first.  Starting from the
                        # highest candidate works for 12/15/16 but overshoots a
                        # near top-row target such as 8, forcing the wrist to look
                        # back down after a needless 20 cm rise.
                        predicted_search_lift_m = round(
                            float(search_plan.predicted_lift_m), 3
                        )
                        search_candidates = sorted(
                            {
                                predicted_search_lift_m,
                                0.20,
                                0.16,
                                0.12,
                                0.08,
                            },
                            key=lambda value: (
                                abs(value - predicted_search_lift_m),
                                -value,
                            ),
                        )
                        selected_search = None
                        rejected_search_candidates = []
                        for candidate_lift_m in search_candidates:
                            if candidate_lift_m > float(args.search_max_lift_m):
                                continue
                            candidate_tcp = predicted_pixel_aim_tcp_target(
                                origin_tcp,
                                handeye,
                                pixel_prediction.center_xy,
                                full_camera_matrix,
                                config["calibration"].get(
                                    "rgb_calibration_size", [1920, 1080]
                                ),
                                visible["image_size"],
                                panel_depth_m,
                                candidate_lift_m,
                            )
                            candidate_joints, position_error, angle_error = inverse_kinematics(
                                candidate_tcp,
                                current_joints,
                                regularization=0.01,
                                minimum_elbow_z_m=origin_elbow_z,
                                max_nfev=120,
                                solver_tolerance=1e-8,
                                fixed_joint_indices=(0,),
                            )
                            joint_step = float(
                                np.max(np.abs(candidate_joints - current_joints))
                            )
                            if (
                                position_error <= 0.001
                                and np.rad2deg(angle_error) <= 0.5
                                and joint_step <= ONLINE_IK_MAX_JOINT_DELTA_RAD
                            ):
                                selected_search = (
                                    candidate_lift_m, candidate_tcp, candidate_joints
                                )
                                break
                            rejected_search_candidates.append(
                                {
                                    "lift_m": candidate_lift_m,
                                    "position_error_m": float(position_error),
                                    "angle_error_deg": float(np.rad2deg(angle_error)),
                                    "max_joint_delta_rad": joint_step,
                                }
                            )
                        if selected_search is None:
                            raise RuntimeError(
                                "no reachable upward visual-search pose within safety limits"
                            )
                        for index, lift_m in enumerate(
                            (selected_search[0],), 1
                        ):
                            desired_tcp = selected_search[1]
                            next_joints = selected_search[2]
                            segment, segment_audit = generate_timed_joint_trajectory(
                                current_joints,
                                next_joints,
                                cartesian_speed_m_s=float(args.browse_speed_m_s),
                                joint_velocity_scale=0.40,
                                point_period_sec=0.20,
                                duration_margin=float(args.duration_margin),
                            )
                            segment_joints = np.asarray(
                                [point["positions"] for point in segment]
                            )
                            rise_base_yaw_error = float(np.max(np.abs(
                                segment_joints[:, 0] - origin[0]
                            )))
                            if rise_base_yaw_error > 1e-10:
                                raise RuntimeError(
                                    "visual search violated the hard base-Z yaw lock: "
                                    f"{rise_base_yaw_error:.12f} rad"
                                )
                            path_z = forward_kinematics_batch(
                                segment_joints
                            )[:, 2, 3]
                            if float(np.min(path_z)) < float(origin_tcp[2, 3]) - 0.002:
                                raise RuntimeError(
                                    "visual search would move below the recorded origin"
                                )
                            elbow_z = elbow_positions_batch(
                                segment_joints
                            )[:, 2]
                            if (
                                float(np.min(elbow_z))
                                < origin_elbow_z - ELBOW_HEIGHT_NUMERICAL_TOLERANCE_M
                            ):
                                raise RuntimeError(
                                    "visual search would lower the elbow below its "
                                    "recorded-origin height"
                                )
                            record_origin_phase("origin_rise_planning_sec")
                            processing_latency[
                                "origin_to_rise_command_sec"
                            ] = float(
                                time.perf_counter() - origin_processing_started_perf
                            )
                            validation, result = execute_probe_segment(
                                segment,
                                f"visual-search-up-{index}",
                            )
                            rise_motion_completed_perf = time.perf_counter()
                            probe_events.append(
                                {
                                    "kind": "gap_predicted_one_shot_aim",
                                    "lift_m": float(lift_m),
                                    "rejected_ik_candidates": rejected_search_candidates,
                                    "predicted_center_xy": list(
                                        pixel_prediction.center_xy
                                    ),
                                    "panel_depth_m": panel_depth_m,
                                    "audit": trajectory_audit_payload(segment_audit),
                                    "validation": validation,
                                    "result": result,
                                }
                            )
                            current_joints = next_joints
                            stationary_after_unix_sec = time.time()
                            moved_state = initial_robot.state(max_age=2.0)
                            initial_vision.request(
                                "safety_camera_pose",
                                timestamp_monotonic=time.monotonic(),
                                base_to_camera=(
                                    moved_state.base_to_gripper @ handeye
                                ).tolist(),
                            )
                            if args.snapshot_at_pointing:
                                observer_snapshot = initial_vision.request("snapshot")
                                probe_events.append({
                                    "kind": "observer_snapshot",
                                    **observer_snapshot,
                                })
                                print(
                                    yaml.safe_dump(
                                        {"observer_snapshot": observer_snapshot},
                                        sort_keys=False,
                                    ),
                                    end="",
                                    flush=True,
                                )
                            visible, transient_target = wait_for_target_after_motion(
                                initial_vision,
                                target_label=args.target_label,
                                min_confidence=config["detector"].get(
                                    "target_confidence", 0.70
                                ),
                            )
                            if (
                                transient_target is not None
                                and not args.center_before_approach
                            ):
                                # The rise is only an acquisition move. Once the
                                # target is visible, hold this pose and collect the
                                # initial stable window; do not add wrist centering,
                                # close-up, or another vertical stage. The normal
                                # motion-time capture will refine the destination.
                                pointing_joints = moved_state.joints.copy()
                                target_was_pointed = True
                                probe_events.append({
                                    "kind": "rise_located_target_direct_handoff",
                                    "source_bbox_xyxy": list(
                                        transient_target["bbox_xyxy"]
                                    ),
                                })
                                initial_live_estimate, initial_live_samples = (
                                    collect_stationary_target(
                                        initial_vision,
                                        initial_robot,
                                        handeye,
                                        args.target_label,
                                        config,
                                        min_capture_unix_sec=(
                                            stationary_after_unix_sec
                                        ),
                                    )
                                )
                                # The high target may have too few neighbouring
                                # buttons for an absolute depth fit. Keep its RGB
                                # direction, but intersect it with the broad panel
                                # plane initialized from the origin anchor.
                                if initialized_panel_plane is None:
                                    raise RuntimeError(
                                        "top-button acquisition has no initialized panel plane"
                                    )
                                corrected_target, initial_plane_correction_m = (
                                    constrain_button_ray_to_reference_plane(
                                        moved_state.base_to_gripper @ handeye,
                                        initial_live_estimate.base_to_button,
                                        initialized_panel_plane,
                                    )
                                )
                                initial_live_estimate = replace(
                                    initial_live_estimate,
                                    base_to_button=corrected_target,
                                )
                                probe_events.append({
                                    "kind": "target_rgb_ray_on_origin_anchor_plane",
                                    "panel_plane_source": initialized_panel_plane_source,
                                    "raw_depth_correction_m": initial_plane_correction_m,
                                })
                                break
                            if transient_target is not None:
                                desired_center_tcp = pixel_centered_camera_tcp_target(
                                    moved_state.base_to_gripper,
                                    handeye,
                                    transient_target["bbox_xyxy"],
                                    visible["image_size"],
                                    full_camera_matrix,
                                    config["calibration"].get(
                                        "rgb_calibration_size", [1920, 1080]
                                    ),
                                )
                                centered_search_joints, position_error, angle_error = (
                                    inverse_kinematics(
                                        desired_center_tcp,
                                        moved_state.joints,
                                        regularization=0.01,
                                        minimum_elbow_z_m=origin_elbow_z,
                                    )
                                )
                                if (
                                    position_error > 0.001
                                    or np.rad2deg(angle_error) > 0.5
                                    or float(np.max(np.abs(
                                        centered_search_joints - moved_state.joints
                                    ))) > 0.50
                                ):
                                    continue
                                center_segment, center_audit = (
                                    generate_timed_joint_trajectory(
                                        moved_state.joints,
                                        centered_search_joints,
                                        cartesian_speed_m_s=float(args.browse_speed_m_s),
                                        joint_velocity_scale=0.40,
                                        point_period_sec=0.20,
                                        duration_margin=float(args.duration_margin),
                                    )
                                )
                                center_elbow_z = elbow_positions_batch(
                                    np.asarray([
                                        point["positions"] for point in center_segment
                                    ])
                                )[:, 2]
                                if (
                                    float(np.min(center_elbow_z))
                                    < origin_elbow_z
                                    - ELBOW_HEIGHT_NUMERICAL_TOLERANCE_M
                                ):
                                    continue
                                validation, result = execute_probe_segment(
                                    center_segment,
                                    f"visual-search-wrist-center-{index}",
                                )
                                probe_events.append({
                                    "kind": "wrist_pixel_center",
                                    "source_bbox_xyxy": list(
                                        transient_target["bbox_xyxy"]
                                    ),
                                    "audit": trajectory_audit_payload(center_audit),
                                    "validation": validation,
                                    "result": result,
                                })
                                current_joints = centered_search_joints
                                target_was_pointed = True
                                pointing_joints = centered_search_joints.copy()
                                time.sleep(0.40)
                                try:
                                    initial_live_estimate, initial_live_samples = (
                                        collect_stationary_target(
                                            initial_vision,
                                            initial_robot,
                                            handeye,
                                            args.target_label,
                                            config,
                                        )
                                    )
                                except RuntimeError as exc:
                                    if str(exc) != "no stable fresh button target before motion":
                                        raise
                                    break
                                else:
                                    break
                        if (
                            initial_live_estimate is None
                            and not target_was_pointed
                            and not args.center_before_approach
                        ):
                            save_missing_button_view(
                                initial_vision, args.target_label, "target_missing_after_single_rise"
                            )
                            raise RuntimeError(
                                "selected button was not visible after the single "
                                "rise; extra close-up and centering are disabled"
                            )
                        if initial_live_estimate is None and not target_was_pointed:
                            close_state = initial_robot.state(max_age=2.0)
                            close_camera = close_state.base_to_gripper @ handeye
                            expected_panel_depth = float(
                                (np.linalg.inv(close_camera) @ np.r_[
                                    anchor_estimate.base_to_button[:3, 3], 1.0
                                ])[2]
                            )
                            close_advance = min(
                                float(args.search_close_step_m),
                                expected_panel_depth
                                - float(args.search_min_panel_distance_m),
                            )
                            if close_advance >= 0.01:
                                close_tcp = closer_camera_tcp_target(
                                    close_state.base_to_gripper,
                                    handeye,
                                    close_advance,
                                )
                                close_joints, position_error, angle_error = (
                                    inverse_kinematics(
                                        close_tcp,
                                        close_state.joints,
                                        regularization=0.01,
                                        minimum_elbow_z_m=origin_elbow_z,
                                    )
                                )
                                if (
                                    position_error > 0.001
                                    or np.rad2deg(angle_error) > 0.5
                                    or float(np.max(np.abs(
                                        close_joints - close_state.joints
                                    ))) > 0.50
                                ):
                                    raise RuntimeError("close-up search IK is unsafe")
                                close_segment, close_audit = (
                                    generate_timed_joint_trajectory(
                                        close_state.joints,
                                        close_joints,
                                        cartesian_speed_m_s=float(args.browse_speed_m_s),
                                        joint_velocity_scale=0.35,
                                        point_period_sec=0.20,
                                        duration_margin=float(args.duration_margin),
                                    )
                                )
                                close_elbow_z = elbow_positions_batch(
                                    np.asarray([
                                        point["positions"] for point in close_segment
                                    ])
                                )[:, 2]
                                close_tcp_z = forward_kinematics_batch(
                                    np.asarray([
                                        point["positions"] for point in close_segment
                                    ])
                                )[:, 2, 3]
                                if (
                                    float(np.min(close_elbow_z))
                                    < origin_elbow_z
                                    - ELBOW_HEIGHT_NUMERICAL_TOLERANCE_M
                                    or float(np.min(close_tcp_z))
                                    < float(origin_tcp[2, 3]) - 0.002
                                ):
                                    raise RuntimeError("close-up search path is unsafe")
                                validation, result = execute_probe_segment(
                                    close_segment, "visual-search-close-up"
                                )
                                probe_events.append({
                                    "kind": "close_up_recapture",
                                    "advance_m": close_advance,
                                    "expected_panel_depth_before_m": expected_panel_depth,
                                    "expected_panel_distance_after_m": (
                                        expected_panel_depth - close_advance
                                    ),
                                    "audit": trajectory_audit_payload(close_audit),
                                    "validation": validation,
                                    "result": result,
                                })
                                time.sleep(0.50)
                                initial_live_estimate, initial_live_samples = (
                                    collect_stationary_target(
                                        initial_vision,
                                        initial_robot,
                                        handeye,
                                        args.target_label,
                                        config,
                                    )
                                )
                        if initial_live_estimate is None and not target_was_pointed:
                            raise RuntimeError(
                                "selected button was not found within the upward search limit"
                            )

                view_requires_adjustment=bool(initial_live_samples and not complete_button_box(
                    initial_live_samples[-1].get('bbox_xyxy'),config['depth'].get('capture_size',[640,360])))
                if should_center_visible_target(
                    auto_pointing_enabled=auto_pointing_enabled,
                    legacy_center_requested=args.center_before_approach,
                    target_visible_at_origin=target_visible_at_origin,
                    target_was_pointed=target_was_pointed,
                    incomplete_view=view_requires_adjustment,
                ):
                    if args.visual_safety:
                        arm_probe_guard()
                    origin_tcp_z = float(forward_kinematics(origin)[2, 3])
                    image_size = config["depth"].get("capture_size", [640, 360])
                    # A close-up reacquisition can legitimately find the button
                    # far from centre.  Do not reject that measurement solely
                    # because it needs more than one safe camera translation:
                    # recapture after every <=75 mm correction instead.
                    for center_step in range(1, 4):
                        state_at_target = initial_robot.state(max_age=2.0)
                        base_to_camera = state_at_target.base_to_gripper @ handeye
                        full_correction = centered_camera_translation_base(
                            base_to_camera,
                            initial_live_estimate.base_to_button[:3, 3],
                            max_translation_m=0.30,
                        )
                        full_distance = float(np.linalg.norm(full_correction))
                        selected_center = None
                        rejected_center_steps = []
                        for candidate_limit_m in (0.075, 0.050, 0.035, 0.020):
                            candidate_distance = min(
                                full_distance, candidate_limit_m
                            )
                            if candidate_distance <= 0.002:
                                selected_center = (np.zeros(3), None, None, None)
                                break
                            correction = full_correction * (
                                candidate_distance / full_distance
                            )
                            desired_tcp = state_at_target.base_to_gripper.copy()
                            desired_tcp[:3, 3] += correction
                            if float(desired_tcp[2, 3]) < origin_tcp_z - 0.002:
                                rejected_center_steps.append({
                                    "step_m": candidate_distance,
                                    "reason": "below_origin_tcp",
                                })
                                continue
                            centered_joints, position_error, angle_error = inverse_kinematics(
                                desired_tcp, state_at_target.joints, regularization=0.01,
                                minimum_elbow_z_m=origin_elbow_z,
                            )
                            joint_delta = float(np.max(np.abs(
                                centered_joints - state_at_target.joints
                            )))
                            if (
                                position_error > 0.001
                                or np.rad2deg(angle_error) > 0.5
                                or joint_delta > 0.50
                            ):
                                rejected_center_steps.append({
                                    "step_m": candidate_distance,
                                    "reason": "ik_limits",
                                    "position_error_m": float(position_error),
                                    "angle_error_deg": float(np.rad2deg(angle_error)),
                                    "max_joint_delta_rad": joint_delta,
                                })
                                continue
                            segment, segment_audit = generate_timed_joint_trajectory(
                                state_at_target.joints,
                                centered_joints,
                                cartesian_speed_m_s=float(args.browse_speed_m_s),
                                joint_velocity_scale=0.40,
                                point_period_sec=0.20,
                                duration_margin=float(args.duration_margin),
                            )
                            segment_joints = np.asarray(
                                [point["positions"] for point in segment]
                            )
                            elbow_z = elbow_positions_batch(segment_joints)[:, 2]
                            tcp_z = forward_kinematics_batch(segment_joints)[:, 2, 3]
                            if (
                                float(np.min(elbow_z)) < origin_elbow_z
                                - ELBOW_HEIGHT_NUMERICAL_TOLERANCE_M
                                or float(np.min(tcp_z)) < origin_tcp_z - 0.002
                            ):
                                rejected_center_steps.append({
                                    "step_m": candidate_distance,
                                    "reason": "path_limits",
                                })
                                continue
                            selected_center = (
                                correction, centered_joints, segment, segment_audit
                            )
                            break
                        if selected_center is None:
                            # Translation with the search-view orientation was
                            # not reachable.  Prefer a panel-perpendicular TCP
                            # pose, then allow the smallest safe offset with
                            # every local axis individually bounded to +/-20°.
                            for offsets_deg, orientation in perpendicular_orientation_candidates(
                                initial_live_estimate.base_to_button[:3, :3],
                                roll_deg,
                            ):
                                for candidate_limit_m in (0.075, 0.050, 0.035, 0.020):
                                    candidate_distance = min(
                                        full_distance, candidate_limit_m
                                    )
                                    if candidate_distance <= 0.002:
                                        continue
                                    correction = full_correction * (
                                        candidate_distance / full_distance
                                    )
                                    desired_tcp = state_at_target.base_to_gripper.copy()
                                    desired_tcp[:3, 3] += correction
                                    desired_tcp[:3, :3] = orientation
                                    if float(desired_tcp[2, 3]) < origin_tcp_z - 0.002:
                                        continue
                                    centered_joints, position_error, angle_error = inverse_kinematics(
                                        desired_tcp, state_at_target.joints,
                                        regularization=0.01,
                                        minimum_elbow_z_m=origin_elbow_z,
                                    )
                                    joint_delta = float(np.max(np.abs(
                                        centered_joints - state_at_target.joints
                                    )))
                                    if (
                                        position_error > 0.001
                                        or np.rad2deg(angle_error) > 0.5
                                        or joint_delta > 0.90
                                    ):
                                        continue
                                    segment, segment_audit = generate_timed_joint_trajectory(
                                        state_at_target.joints, centered_joints,
                                        cartesian_speed_m_s=float(args.browse_speed_m_s),
                                        joint_velocity_scale=0.40,
                                        point_period_sec=0.20,
                                        duration_margin=float(args.duration_margin),
                                    )
                                    segment_joints = np.asarray([
                                        point["positions"] for point in segment
                                    ])
                                    if (
                                        float(np.min(elbow_positions_batch(segment_joints)[:, 2]))
                                        < origin_elbow_z - ELBOW_HEIGHT_NUMERICAL_TOLERANCE_M
                                        or float(np.min(forward_kinematics_batch(segment_joints)[:, 2, 3]))
                                        < origin_tcp_z - 0.002
                                    ):
                                        continue
                                    selected_center = (
                                        correction, centered_joints, segment,
                                        segment_audit, offsets_deg,
                                    )
                                    break
                                if selected_center is not None:
                                    break
                        if selected_center is None:
                            raise RuntimeError("no safe camera-centering pose within 35 degrees")
                        if len(selected_center) == 4:
                            correction, centered_joints, segment, segment_audit = selected_center
                            orientation_offsets_deg = (0.0, 0.0, 0.0)
                        else:
                            (
                                correction, centered_joints, segment,
                                segment_audit, orientation_offsets_deg,
                            ) = selected_center
                        if segment is not None:
                            validation, result = execute_probe_segment(
                                segment, "camera-pointing"
                            )
                            probe_events.append(
                                {
                                    "kind": "center",
                                    "center_step": center_step,
                                    "translation_base_m": correction.tolist(),
                                    "remaining_translation_m": (
                                        full_correction - correction
                                    ).tolist(),
                                    "rejected_step_candidates": rejected_center_steps,
                                    "perpendicular_orientation_offset_deg": list(
                                        orientation_offsets_deg
                                    ),
                                    "audit": trajectory_audit_payload(segment_audit),
                                    "validation": validation,
                                    "result": result,
                                }
                            )
                            pointing_joints = centered_joints.copy()
                        centered_state = initial_robot.state(max_age=2.0)
                        initial_vision.request(
                            "safety_camera_pose",
                            timestamp_monotonic=time.monotonic(),
                            base_to_camera=(
                                centered_state.base_to_gripper @ handeye
                            ).tolist(),
                        )
                        initial_live_estimate, initial_live_samples = (
                            collect_stationary_target(
                                initial_vision, initial_robot, handeye,
                                args.target_label, config,
                            )
                        )
                        if ((view_requires_adjustment and all(complete_button_box(
                                s.get('bbox_xyxy'),image_size) for s in initial_live_samples[-2:]))
                            or (not view_requires_adjustment and is_button_centered(
                                initial_live_samples[-1]["bbox_xyxy"], image_size))):
                            break
                    else:
                        raise RuntimeError(
                            "button remains outside the camera-center tolerance"
                        )
                if not initial_live_samples or not all(complete_button_box(
                        s.get('bbox_xyxy'),config['depth'].get('capture_size',[640,360]))
                        for s in initial_live_samples[-2:]):
                    raise RuntimeError('button observation view incomplete after bounded adjustment')
                if initial_live_estimate is not None and not args.pointing_only:
                    # The first physical leg ends at the dynamic guard, not at
                    # contact. Solving that actual endpoint avoids a costly
                    # nine-step virtual contact solve while retaining the same
                    # 1 mm / 0.5 degree acceptance checks. The motion-time
                    # second capture still computes the final target normally.
                    start_panel_distance = float(
                        (
                            forward_kinematics(pointing_joints)[:3, 3]
                            - initial_live_estimate.base_to_button[:3, 3]
                        ) @ initial_live_estimate.base_to_button[:3, 2]
                    )
                    initial_receding_plan = plan_receding_approach(
                        start_panel_distance,
                        float(args.predicted_ready_sec),
                        RollingSpeedConfig(
                            max_speed_m_s=float(args.approach_speed_m_s),
                            coarse_target_uncertainty_m=float(
                                args.coarse_uncertainty_m
                            ),
                            control_latency_sec=float(
                                args.control_latency_sec
                            ),
                        ),
                    )
                    safety_prepare_thread = None
                    safety_prepare_result = {}
                    if args.visual_safety and probe_guard is not None:
                        # The target is now visible and stable at the raised
                        # pose. Build its approach safety reference while the
                        # independent guard-endpoint IK is running. No motion
                        # is permitted during this preparation.
                        probe_guard.disarm()

                        def prepare_pointing_safety():
                            started_prepare = time.perf_counter()
                            try:
                                initial_vision.request(
                                    "set_target",
                                    target_label=str(args.target_label),
                                )
                                initial_vision.request(
                                    "safety_set_mode",
                                    mode="predicted_depth_matrix",
                                )
                                ring_started = time.perf_counter()
                                ring = probe_guard.preflight(3.0)
                                ring_sec = time.perf_counter() - ring_started
                                matrix_started = time.perf_counter()
                                matrix = initial_vision.request(
                                    "safety_capture_matrix_reference",
                                    frame_count=25,
                                )
                                matrix_sec = time.perf_counter() - matrix_started
                                final_started = time.perf_counter()
                                final = probe_guard.preflight(3.0)
                                final_sec = time.perf_counter() - final_started
                                safety_prepare_result.update(
                                    ring_preflight=ring,
                                    matrix_reference=matrix,
                                    matrix_preflight=final,
                                    ring_preflight_sec=float(ring_sec),
                                    matrix_capture_sec=float(matrix_sec),
                                    matrix_preflight_sec=float(final_sec),
                                    total_sec=float(
                                        time.perf_counter() - started_prepare
                                    ),
                                )
                            except BaseException as exc:
                                safety_prepare_result["error"] = exc

                        safety_prepare_thread = threading.Thread(
                            target=prepare_pointing_safety,
                            daemon=True,
                        )
                        safety_prepare_thread.start()
                    try:
                        requested_guard_m = float(
                            initial_receding_plan.dynamic_guard_distance_m
                        )
                        try:
                            initial_coarse_guard_ik = FixedRollFastIkPlanner(
                                roll_deg=roll_deg,
                                stand_off_m=requested_guard_m,
                                max_joint_delta_rad=(
                                    ONLINE_IK_MAX_JOINT_DELTA_RAD
                                ),
                                max_perpendicular_deviation_deg=(
                                    MAX_PERPENDICULAR_DEVIATION_DEG
                                ),
                                downstream_base_x_distances_m=(
                                    downstream_preview_distances_m
                                ),
                            ).solve(
                                initial_live_estimate.base_to_button,
                                pointing_joints,
                            )
                            initial_coarse_guard_ik_mode = (
                                "live_geometry_current_joint_branch"
                            )
                        except RuntimeError as primary_ik_error:
                            if not args.hold_at_fallback:
                                raise
                            attempts = [{
                                "stand_off_m": requested_guard_m,
                                "reachable": False,
                                "error": str(primary_ik_error),
                            }]
                            maximum_guard_m = min(
                                0.30, start_panel_distance - 0.03
                            )
                            candidates = sorted({
                                value for value in (
                                    0.10, 0.12, 0.15, 0.18,
                                    0.22, 0.26, 0.30,
                                )
                                if requested_guard_m + 0.005 < value
                                <= maximum_guard_m + 1e-9
                            })
                            selected_guard_m = None
                            selected_guard_ik = None
                            last_unreachable_m = requested_guard_m
                            for candidate_guard_m in candidates:
                                try:
                                    candidate_ik = FixedRollFastIkPlanner(
                                        roll_deg=roll_deg,
                                        stand_off_m=float(candidate_guard_m),
                                        max_joint_delta_rad=(
                                            ONLINE_IK_MAX_JOINT_DELTA_RAD
                                        ),
                                        max_perpendicular_deviation_deg=(
                                            MAX_PERPENDICULAR_DEVIATION_DEG
                                        ),
                                        downstream_base_x_distances_m=(
                                            downstream_preview_distances_m
                                        ),
                                    ).solve(
                                        initial_live_estimate.base_to_button,
                                        pointing_joints,
                                    )
                                except RuntimeError as candidate_error:
                                    attempts.append({
                                        "stand_off_m": float(candidate_guard_m),
                                        "reachable": False,
                                        "error": str(candidate_error),
                                    })
                                    last_unreachable_m = float(candidate_guard_m)
                                    continue
                                attempts.append({
                                    "stand_off_m": float(candidate_guard_m),
                                    "reachable": True,
                                })
                                selected_guard_m = float(candidate_guard_m)
                                selected_guard_ik = candidate_ik
                                break
                            if selected_guard_ik is None:
                                closest_reachable_search = {
                                    "requested_guard_m": requested_guard_m,
                                    "maximum_guard_m": maximum_guard_m,
                                    "attempts": attempts,
                                    "result": "no_reachable_guard",
                                }
                                raise RuntimeError(
                                    "no reachable diagnostic hold endpoint up to "
                                    f"{maximum_guard_m:.3f} m"
                                ) from primary_ik_error
                            # Refine the first reachable interval twice without
                            # ever selecting an unvalidated closer endpoint.
                            for _ in range(2):
                                midpoint_m = round(
                                    (last_unreachable_m + selected_guard_m) / 2.0,
                                    4,
                                )
                                if selected_guard_m - midpoint_m < 0.002:
                                    break
                                try:
                                    midpoint_ik = FixedRollFastIkPlanner(
                                        roll_deg=roll_deg,
                                        stand_off_m=midpoint_m,
                                        max_joint_delta_rad=(
                                            ONLINE_IK_MAX_JOINT_DELTA_RAD
                                        ),
                                        max_perpendicular_deviation_deg=(
                                            MAX_PERPENDICULAR_DEVIATION_DEG
                                        ),
                                        downstream_base_x_distances_m=(
                                            downstream_preview_distances_m
                                        ),
                                    ).solve(
                                        initial_live_estimate.base_to_button,
                                        pointing_joints,
                                    )
                                except RuntimeError as midpoint_error:
                                    attempts.append({
                                        "stand_off_m": midpoint_m,
                                        "reachable": False,
                                        "error": str(midpoint_error),
                                    })
                                    last_unreachable_m = midpoint_m
                                else:
                                    attempts.append({
                                        "stand_off_m": midpoint_m,
                                        "reachable": True,
                                    })
                                    selected_guard_m = midpoint_m
                                    selected_guard_ik = midpoint_ik
                            initial_coarse_guard_ik = selected_guard_ik
                            initial_coarse_guard_ik_mode = (
                                "global_closest_reachable_search"
                            )
                            usable_m = max(
                                0.0, start_panel_distance - selected_guard_m
                            )
                            initial_receding_plan = replace(
                                initial_receding_plan,
                                dynamic_guard_distance_m=selected_guard_m,
                                brake_entry_distance_m=max(
                                    selected_guard_m,
                                    float(
                                        initial_receding_plan.brake_entry_distance_m
                                    ),
                                ),
                                usable_approach_distance_m=usable_m,
                                expected_guard_arrival_sec=(
                                    usable_m
                                    / max(
                                        float(initial_receding_plan.speed_cap_m_s),
                                        1e-9,
                                    )
                                ),
                            )
                            closest_reachable_search = {
                                "requested_guard_m": requested_guard_m,
                                "maximum_guard_m": maximum_guard_m,
                                "attempts": attempts,
                                "selected_guard_m": selected_guard_m,
                                "result": "reachable_guard_selected",
                            }
                        guard_ik_ready_perf = time.perf_counter()
                    finally:
                        if safety_prepare_thread is not None:
                            safety_prepare_thread.join(timeout=8.0)
                    if (
                        safety_prepare_thread is not None
                        and safety_prepare_thread.is_alive()
                    ):
                        raise RuntimeError(
                            "pointing safety preparation did not finish"
                        )
                    if safety_prepare_result.get("error") is not None:
                        raise RuntimeError(
                            "pointing safety preparation failed: "
                            f"{safety_prepare_result['error']}"
                        ) from safety_prepare_result["error"]
                    if safety_prepare_thread is not None:
                        pointing_safety_preparation = safety_prepare_result
                        approach_safety_target_label = str(args.target_label)
                    if rise_motion_completed_perf is not None:
                        processing_latency[
                            "rise_motion_to_guard_ik_ready_sec"
                        ] = float(
                            guard_ik_ready_perf - rise_motion_completed_perf
                        )
                    if pointing_safety_preparation is not None:
                        processing_latency[
                            "pointing_safety_parallel_total_sec"
                        ] = float(pointing_safety_preparation["total_sec"])
                        processing_latency[
                            "pointing_safety_ring_preflight_sec"
                        ] = float(
                            pointing_safety_preparation[
                                "ring_preflight_sec"
                            ]
                        )
                        processing_latency[
                            "pointing_safety_matrix_capture_sec"
                        ] = float(
                            pointing_safety_preparation[
                                "matrix_capture_sec"
                            ]
                        )
                        processing_latency[
                            "pointing_safety_matrix_preflight_sec"
                        ] = float(
                            pointing_safety_preparation[
                                "matrix_preflight_sec"
                            ]
                        )
            except BaseException as exc:
                probe_error = exc
            finally:
                # Exact probe return follows the already executed path and is
                # deliberately not allowed to be blocked by a changing view.
                if probe_guard is not None:
                    probe_guard.disarm()
                if args.return_without_visual_safety:
                    # Search uses the conservative absolute near-field mode.
                    # Restore normal sensing and clear its already-disarmed
                    # stop latch before the explicitly authorised exact
                    # reverse return, otherwise a stale search observation can
                    # strand the arm after it has stopped safely.
                    initial_vision.request(
                        "safety_set_mode", mode="predicted_depth_matrix"
                    )
                    initial_vision.request("safety_motion_state", paused=False)
                    SafetyStopUdpClient(
                        safety_stop_cfg["host"],
                        int(safety_stop_cfg["port"]),
                        timeout_sec=2.0,
                    ).reset_latch(token=str(safety_stop_cfg["token"]))
                return_events = []
                carry_pointing_pose = bool(
                    probe_error is None
                    and (not args.pointing_only or args.hold_at_pointing)
                    and (initial_live_estimate is not None or target_was_pointed)
                )
                for index, segment in enumerate(reversed(probe_segments), 1):
                    if carry_pointing_pose:
                        break
                    reverse_segment = reverse_timed_trajectory(segment)
                    actual_return_state = initial_robot.state(max_age=2.0)
                    expected_start = np.asarray(
                        reverse_segment[0]["positions"], dtype=float
                    )
                    start_mismatch = float(np.max(np.abs(
                        actual_return_state.joints - expected_start
                    )))
                    if start_mismatch > 0.03:
                        # A stopped/paused bridge can leave the encoder pose
                        # between cached points.  Never submit an inverse path
                        # whose first point no longer matches the real arm;
                        # regenerate a bounded trajectory from the encoder pose
                        # directly to the recorded origin instead.
                        fallback_points, fallback_audit = generate_timed_joint_trajectory(
                            actual_return_state.joints,
                            origin,
                            cartesian_speed_m_s=min(
                                float(args.return_speed_m_s),
                                # The generic encoder-replanned trajectory
                                # generator is certified to 0.15 m/s, while
                                # the dedicated direct-return utility permits
                                # a higher cap.  Use the stricter bound here.
                                0.15,
                            ),
                            joint_velocity_scale=float(
                                args.return_joint_velocity_scale
                            ),
                            point_period_sec=0.10,
                            duration_margin=float(args.duration_margin),
                        )
                        validation, result = execute_standard(
                            browse_client, fallback_points, browse_token,
                            f"pointing-probe-recovered-return-{index}",
                        )
                        return_events.append({
                            "kind": "encoder_replanned_return",
                            "start_mismatch_rad": start_mismatch,
                            "audit": trajectory_audit_payload(fallback_audit),
                            "validation": validation, "result": result,
                        })
                        break
                    validation, result = execute_standard(
                        browse_client, reverse_segment, browse_token,
                        f"pointing-probe-return-{index}",
                    )
                    return_events.append(
                        {
                            "kind": "exact_reverse",
                            "start_mismatch_rad": start_mismatch,
                            "validation": validation, "result": result,
                        }
                    )
                if (
                    args.start_from_current_pointing
                    and not carry_pointing_pose
                    and not probe_segments
                ):
                    # The held pointing path predates this process, so there
                    # is no cached inverse. Regenerate from encoders and rely
                    # on the central bridge's fail-closed MoveIt validation.
                    actual_return_state = initial_robot.state(max_age=2.0)
                    fallback_points, fallback_audit = generate_timed_joint_trajectory(
                        actual_return_state.joints,
                        origin,
                        cartesian_speed_m_s=min(
                            float(args.return_speed_m_s), 0.15
                        ),
                        joint_velocity_scale=float(
                            args.return_joint_velocity_scale
                        ),
                        point_period_sec=0.10,
                        duration_margin=float(args.duration_margin),
                    )
                    validation, result = execute_standard(
                        browse_client,
                        fallback_points,
                        browse_token,
                        "current-pointing-failure-return",
                    )
                    return_events.append({
                        "kind": "current_pointing_encoder_return",
                        "audit": trajectory_audit_payload(fallback_audit),
                        "validation": validation,
                        "result": result,
                    })
                returned_state = initial_robot.state(max_age=2.0)
                returned_error = float(
                    np.max(np.abs(returned_state.joints - origin))
                )
                pointing_audit = {
                    "enabled": auto_pointing_enabled,
                    "search_plan": (
                        None
                        if search_plan is None
                        else {
                            "target_label": search_plan.target_label,
                            "target_row": search_plan.target.row,
                            "target_column": search_plan.target.column,
                            "visible_labels": list(search_plan.visible_labels),
                            "highest_visible_row": search_plan.highest_visible_row,
                            "predicted_lift_m": search_plan.predicted_lift_m,
                            "lift_offsets_m": list(search_plan.lift_offsets_m),
                            "camera_tilt_offsets_deg": list(
                                search_plan.camera_tilt_offsets_deg
                            ),
                            "direction": search_plan.direction,
                        }
                    ),
                    "pixel_grid_prediction": (
                        None
                        if pixel_prediction is None
                        else {
                            "center_xy": list(pixel_prediction.center_xy),
                            "row_gap_px": pixel_prediction.row_gap_px,
                            "column_gap_px": pixel_prediction.column_gap_px,
                            "fit_rms_px": pixel_prediction.fit_rms_px,
                            "labels": list(pixel_prediction.labels),
                            "layout": pixel_prediction.layout,
                            "target_column": pixel_prediction.target_column,
                            "rows_above_visible": (
                                pixel_prediction.rows_above_visible
                            ),
                            "anchor_label": pixel_prediction.anchor_label,
                            "column_vertical_offset_px": (
                                pixel_prediction.column_vertical_offset_px
                            ),
                        }
                    ),
                    "outbound_events": probe_events,
                    "return_events": return_events,
                    "pointing_joints_rad": pointing_joints.tolist(),
                    "returned_origin_error_rad": returned_error,
                    "visual_safety_events": (
                        [] if probe_guard is None else list(probe_guard.events)
                    ),
                }
                if (
                    returned_error > 0.03
                    and probe_error is None
                    and not carry_pointing_pose
                ):
                    probe_error = RuntimeError(
                        "pointing probe failed to return to origin: "
                        f"{returned_error:.4f} rad"
                    )
            if probe_error is not None:
                raise probe_error
            if initial_live_estimate is not None:
                coarse_button = initial_live_estimate.base_to_button.copy()
        finally:
            probe_pose_stop.set()
            if probe_pose_thread is not None:
                probe_pose_thread.join(timeout=2.0)
            if probe_guard is not None:
                probe_guard.close()
            initial_robot.close()
        if guard_ik_ready_perf is not None:
            processing_latency[
                "guard_ik_ready_to_probe_cleanup_sec"
            ] = float(time.perf_counter() - guard_ik_ready_perf)
        if args.pointing_only:
            output = (
                ROOT / "pointing_only_runs" /
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            output.mkdir(parents=True, exist_ok=False)
            prune_old_entries(output.parent)
            report = {
                "status": (
                    "pointing_complete_held_at_target_view"
                    if args.hold_at_pointing
                    else "pointing_complete_returned_origin"
                ),
                "target_label": str(args.target_label),
                "contact_attempted": False,
                "held_position": bool(args.hold_at_pointing),
                "returned_to_origin": not args.hold_at_pointing,
                "held_joints_rad": (
                    pointing_joints.tolist()
                    if args.hold_at_pointing else None
                ),
                "auto_pointing": pointing_audit,
                "live_target_base_position_m": coarse_button[:3, 3].tolist(),
            }
            path = output / "pointing_report.yaml"
            path.write_text(
                yaml.safe_dump(report, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), end="")
            return 0
    if args.receding_schedule and initial_live_estimate is not None:
        warm_start_started = time.perf_counter()
        warm_start_target = initial_live_estimate.base_to_button.copy()
        warm_start_target[0, 3] -= prepress_tcp_distance_m
        warm_start_seed = (
            initial_coarse_guard_ik.joints
            if initial_coarse_guard_ik is not None
            else pointing_joints
        )
        try:
            if initial_coarse_guard_ik is not None:
                try:
                    warm_start_ik = reuse_task_preview(
                        initial_coarse_guard_ik, warm_start_target[:3, 3],
                        max_joint_delta_rad=ONLINE_IK_MAX_JOINT_DELTA_RAD,
                    )
                    warm_start_mode = "validated_preview_reused"
                    if warm_start_ik is None:
                        warm_start_ik = refine_precomputed_ik(
                            initial_coarse_guard_ik.desired_base_to_tcp,
                            warm_start_target[:3, 3], warm_start_seed,
                            max_joint_delta_rad=ONLINE_IK_MAX_JOINT_DELTA_RAD,
                        )
                        warm_start_mode = "local_from_guard_ik"
                except RuntimeError:
                    warm_start_ik = FixedRollFastIkPlanner(
                        roll_deg=roll_deg,
                        stand_off_m=0.0,
                        max_joint_delta_rad=ONLINE_IK_MAX_JOINT_DELTA_RAD,
                        max_perpendicular_deviation_deg=(
                            MAX_PERPENDICULAR_DEVIATION_DEG
                        ),
                    ).solve(warm_start_target, warm_start_seed)
                    warm_start_mode = "global_after_local_guard_ik_failure"
            else:
                warm_start_ik = FixedRollFastIkPlanner(
                    roll_deg=roll_deg,
                    stand_off_m=0.0,
                    max_joint_delta_rad=ONLINE_IK_MAX_JOINT_DELTA_RAD,
                    max_perpendicular_deviation_deg=(
                        MAX_PERPENDICULAR_DEVIATION_DEG
                    ),
                ).solve(warm_start_target, warm_start_seed)
                warm_start_mode = "global_orientation_search"
            initial_prepress_seed_joints = warm_start_ik.joints.copy()
            initial_prepress_desired_tcp = (
                warm_start_ik.desired_base_to_tcp.copy()
            )
            initial_prepress_seed_audit = {
                "status": "ready",
                "mode": warm_start_mode,
                "solve_ms": float(warm_start_ik.solve_sec * 1000.0),
                "wall_ms": float(
                    (time.perf_counter() - warm_start_started) * 1000.0
                ),
                "position_error_m": float(warm_start_ik.position_error_m),
                "continuation_steps": int(warm_start_ik.continuation_steps),
            }
        except RuntimeError as exc:
            initial_prepress_seed_audit = {
                "status": "unavailable",
                "wall_ms": float(
                    (time.perf_counter() - warm_start_started) * 1000.0
                ),
                "error": f"{type(exc).__name__}: {exc}",
            }
        if initial_prepress_seed_joints is None:
            # Do not start the timed approach and attempt an expensive global
            # IK search while the active trajectory is already running.
            recovery_robot = PiperUdpRobot(config["robot"])
            try:
                recovery_robot.open()
                recovery_state = recovery_robot.state(max_age=2.0)
                recovery_error = float(
                    np.max(np.abs(recovery_state.joints - origin))
                )
                if recovery_error > 0.03:
                    recovery_points, _ = generate_timed_joint_trajectory(
                        recovery_state.joints,
                        origin,
                        cartesian_speed_m_s=min(
                            float(args.return_speed_m_s), 0.15
                        ),
                        joint_velocity_scale=float(
                            args.return_joint_velocity_scale
                        ),
                        point_period_sec=0.10,
                        duration_margin=float(args.duration_margin),
                    )
                    execute_standard(
                        browse_client,
                        recovery_points,
                        browse_token,
                        "ik-profile-preflight-return",
                    )
            finally:
                recovery_robot.close()
            raise RuntimeError(
                "near-target IK profile could not be validated before motion"
            )

    active_endpoint = preapproach
    receding_plan = initial_receding_plan
    coarse_guard_ik = initial_coarse_guard_ik
    active_speed_m_s = float(args.approach_speed_m_s)
    pass_through_joints = None
    pass_through_time_sec = None
    pointing_arrival_sec = None
    pointing_min_elbow_z_m = None
    if args.receding_schedule:
        start_panel_distance = float(
            (
                forward_kinematics(pointing_joints)[:3, 3]
                - coarse_button[:3, 3]
            )
            @ coarse_button[:3, 2]
        )
        if not 0.15 <= start_panel_distance <= 0.80:
            raise RuntimeError(
                f"coarse start panel distance is invalid: {start_panel_distance:.4f} m"
            )
        if receding_plan is None:
            receding_plan = plan_receding_approach(
                start_panel_distance,
                float(args.predicted_ready_sec),
                RollingSpeedConfig(
                    max_speed_m_s=float(args.approach_speed_m_s),
                    coarse_target_uncertainty_m=float(
                        args.coarse_uncertainty_m
                    ),
                    control_latency_sec=float(args.control_latency_sec),
                ),
            )
        if coarse_guard_ik is None:
            coarse_guard_ik = FixedRollFastIkPlanner(
                roll_deg=roll_deg,
                stand_off_m=float(receding_plan.dynamic_guard_distance_m),
                max_joint_delta_rad=ONLINE_IK_MAX_JOINT_DELTA_RAD,
                max_perpendicular_deviation_deg=(
                    MAX_PERPENDICULAR_DEVIATION_DEG
                ),
                downstream_base_x_distances_m=(
                    downstream_preview_distances_m
                ),
            ).solve(coarse_button, pointing_joints)
            initial_coarse_guard_ik_mode = "global_orientation_search"
        active_endpoint = coarse_guard_ik.joints.copy()
        actual_guard = float(
            (
                forward_kinematics(active_endpoint)[:3, 3]
                - coarse_button[:3, 3]
            ) @ coarse_button[:3, 2]
        )
        if abs(actual_guard - receding_plan.dynamic_guard_distance_m) > 0.002:
            raise RuntimeError("dynamic guard joint target does not match panel distance")
        active_speed_m_s = float(receding_plan.speed_cap_m_s)
        if args.pass_through_approach:
            pass_through_joints, _, actual_brake_entry = (
                joint_target_at_panel_distance(
                    pointing_joints,
                    active_endpoint,
                    coarse_button,
                    receding_plan.brake_entry_distance_m,
                )
            )
            if abs(
                actual_brake_entry - receding_plan.brake_entry_distance_m
            ) > 0.002:
                raise RuntimeError(
                    "pass-through target does not match brake-entry distance"
                )
            approach_leg, approach_audit = generate_continuous_waypoint_trajectory(
                pointing_joints,
                pass_through_joints,
                active_endpoint,
                first_cartesian_speed_m_s=active_speed_m_s,
                second_cartesian_speed_m_s=active_speed_m_s,
                first_joint_velocity_scale=float(args.joint_velocity_scale),
                second_joint_velocity_scale=float(args.joint_velocity_scale),
                waypoint_velocity_fraction=1.20,
                point_period_sec=0.10,
                duration_margin=float(args.duration_margin),
            )
            approach_waypoint_sec = float(approach_audit.waypoint_time_sec)
        else:
            approach_leg, approach_audit = generate_timed_joint_trajectory(
                pointing_joints, active_endpoint,
                cartesian_speed_m_s=active_speed_m_s,
                joint_velocity_scale=float(args.joint_velocity_scale),
                point_period_sec=0.10,
                duration_margin=float(args.duration_margin),
            )
            approach_waypoint_sec = None
        if carry_pointing_pose or np.allclose(origin, pointing_joints, atol=1e-6):
            pointing_leg = [
                {
                    "time_from_start_sec": 0.0,
                    "positions": pointing_joints.tolist(),
                    "velocities": [0.0] * 6,
                    "accelerations": [0.0] * 6,
                },
                {
                    "time_from_start_sec": 0.10,
                    "positions": pointing_joints.tolist(),
                    "velocities": [0.0] * 6,
                    "accelerations": [0.0] * 6,
                },
            ]
        else:
            pointing_leg, _ = generate_timed_joint_trajectory(
                origin,
                pointing_joints,
                cartesian_speed_m_s=float(args.browse_speed_m_s),
                joint_velocity_scale=0.40,
                point_period_sec=0.10,
                duration_margin=float(args.duration_margin),
            )
        pointing_elbow_z = elbow_positions_batch(
            np.asarray([point["positions"] for point in pointing_leg])
        )[:, 2]
        pointing_min_elbow_z_m = float(np.min(pointing_elbow_z))
        origin_elbow_z = float(elbow_position(origin)[2])
        if (
            pointing_min_elbow_z_m
            < origin_elbow_z - ELBOW_HEIGHT_NUMERICAL_TOLERANCE_M
        ):
            raise RuntimeError(
                "formal camera-pointing path would lower the elbow below its "
                "recorded-origin height"
            )
        pointing_arrival_sec = float(pointing_leg[-1]["time_from_start_sec"])
        active, approach_offset_sec = concatenate_stopped_trajectories(
            pointing_leg,
            approach_leg,
            hold_sec=(0.0 if carry_pointing_pose else float(args.pointing_hold_sec)),
        )
        active_audit = audit_sampled_trajectory(active)
        if approach_waypoint_sec is not None:
            pass_through_time_sec = approach_offset_sec + approach_waypoint_sec
        transition_fraction = None
        transition_distance = None
        automatic_vision_start_sec = max(0.0, pointing_arrival_sec - 0.10)
    elif args.near_zone_start_m > 0.0:
        _, transition_fraction, transition_distance = joint_target_at_panel_distance(
            origin, preapproach, coarse_button, float(args.near_zone_start_m)
        )
        active, active_audit = generate_two_zone_joint_trajectory(
            origin, preapproach,
            transition_fraction=transition_fraction,
            fast_cartesian_speed_m_s=float(args.approach_speed_m_s),
            near_cartesian_speed_m_s=float(args.near_approach_speed_m_s),
            joint_velocity_scale=float(args.joint_velocity_scale),
            point_period_sec=0.10,
            duration_margin=float(args.duration_margin),
        )
        automatic_vision_start_sec = float(active_audit.transition_time_sec)
    else:
        transition_fraction = None
        transition_distance = None
        active, active_audit = generate_timed_joint_trajectory(
            origin, preapproach, cartesian_speed_m_s=float(args.approach_speed_m_s),
            joint_velocity_scale=float(args.joint_velocity_scale), point_period_sec=0.10,
            duration_margin=float(args.duration_margin),
        )
        automatic_vision_start_sec = 0.0
    if guard_ik_ready_perf is not None:
        processing_latency[
            "guard_ik_ready_to_active_geometry_sec"
        ] = float(time.perf_counter() - guard_ik_ready_perf)
    # Preserve the motion-only approach before appending the endpoint hold.
    # Its reversal is the only fallback return path, so an online-target
    # rejection cannot silently select the old direct return geometry.
    active_motion = [dict(point) for point in active]
    if args.fallback_hold_sec > 0.0:
        hold_count = int(np.ceil(float(args.fallback_hold_sec) / 0.10))
        hold_times = np.linspace(
            float(active[-1]["time_from_start_sec"]) + 0.10,
            float(active[-1]["time_from_start_sec"]) + float(args.fallback_hold_sec),
            hold_count,
        )
        for stamp in hold_times:
            active.append(
                {
                    "time_from_start_sec": float(stamp),
                    "positions": active_endpoint.tolist(),
                    "velocities": [0.0] * 6,
                    "accelerations": [0.0] * 6,
                }
            )
    active_duration_sec = float(active[-1]["time_from_start_sec"])
    near_vision_entry_distance_m = None
    if (
        args.receding_schedule
        and not auto_pointing_enabled
        and args.vision_start_active_sec < 0.0
    ):
        automatic_vision_start_sec, near_vision_entry_distance_m = (
            first_time_within_panel_distance(
                active, coarse_button,
                float(args.near_vision_start_distance_m),
            )
        )
    vision_start_active_sec = (
        automatic_vision_start_sec
        if auto_pointing_enabled or args.vision_start_active_sec < 0.0
        else float(args.vision_start_active_sec)
    )
    active_motion_duration_sec = float(active_audit.duration_sec)
    if pass_through_time_sec is None:
        desired_handoff = select_future_handoff(
            active,
            active_elapsed_sec=max(
                0.0,
                active_motion_duration_sec
                - args.handoff_remaining_sec
                - 0.20,
            ),
            lead_sec=0.20,
        )
    else:
        desired_handoff = select_future_handoff(
            active,
            active_elapsed_sec=max(0.0, pass_through_time_sec - 0.20),
            lead_sec=0.20,
        )
    trajectory_cfg = config["timed_trajectory"]
    token = str(trajectory_cfg["token"])
    splice = PreparedSpliceUdpClient(trajectory_cfg["host"], 5011, timeout_sec=40.0)
    standard = TimedTrajectoryUdpClient(
        trajectory_cfg["host"], int(trajectory_cfg["port"]), timeout_sec=20.0
    )
    smooth_carried_return = bool(
        carry_pointing_pose
        and (
            len(probe_segments) == 1
            or args.start_from_current_pointing
        )
    )
    carried_pointing_returns = []
    if carry_pointing_pose and not smooth_carried_return:
        for index, segment in enumerate(reversed(probe_segments), 1):
            reverse_segment = reverse_timed_trajectory(segment)
            validation = standard.execute(
                reverse_segment,
                token=token,
                task_id=(
                    f"carried-pointing-return-{index}-validate-{time.time_ns()}"
                ),
                dry_run=True,
            )
            carried_pointing_returns.append((reverse_segment, validation))
    if smooth_carried_return:
        # The browse/rise move has already been executed before ``active`` is
        # started.  Return through that same pointing pose in one C2 command,
        # rather than stopping there and submitting a second trajectory.
        precomputed_return, precomputed_return_audit = (
            generate_continuous_waypoint_trajectory(
                active_endpoint,
                pointing_joints,
                origin,
                first_cartesian_speed_m_s=float(args.retreat_speed_m_s),
                second_cartesian_speed_m_s=float(args.return_speed_m_s),
                first_joint_velocity_scale=float(args.joint_velocity_scale),
                second_joint_velocity_scale=float(
                    args.return_joint_velocity_scale
                ),
                point_period_sec=0.10,
                duration_margin=float(args.duration_margin),
            )
        )
    else:
        precomputed_return = reverse_timed_trajectory(active_motion)
        precomputed_return_audit = audit_sampled_trajectory(precomputed_return)
    return_prevalidation = standard.execute(
        precomputed_return, token=token,
        task_id=f"online-return-prevalidate-{time.time_ns()}", dry_run=True,
    )
    if guard_ik_ready_perf is not None:
        processing_latency[
            "guard_ik_ready_to_return_prevalidated_sec"
        ] = float(time.perf_counter() - guard_ik_ready_perf)
    service_cfg = config["vision_service"]
    vision = VisionRpcClient(
        service_cfg["host"], int(service_cfg["port"]),
        service_cfg["token"], timeout=2.0,
    )
    from piper_handeye.numeric_layout_handoff import (
        task_vision,require_layout_server_support,LayoutTaskVision,target_batch,timed_target,
    )
    vision=task_vision(vision,config,args.target_label)
    if isinstance(vision,LayoutTaskVision):
        require_layout_server_support(vision.request('health'),config,args.target_label)
    pose_vision = VisionRpcClient(
        service_cfg["host"], int(service_cfg["port"]),
        service_cfg["token"], timeout=0.20,
    )
    commit_vision = VisionRpcClient(
        service_cfg["host"], int(service_cfg["port"]),
        service_cfg["token"], timeout=0.50,
    )
    safety_supervisor = None
    if args.visual_safety:
        safety_cfg = config["safety_stop"]
        safety_supervisor = VisualSafetySupervisor(
            vision,
            SafetyStopUdpClient(
                safety_cfg["host"], int(safety_cfg["port"]), timeout_sec=2.0
            ),
            str(safety_cfg["token"]),
        )
        safety_supervisor.start()
    suffix = str(time.time_ns())
    active_id = f"online-active-{suffix}"
    prepared_active = splice.prepare(active, trajectory_id=active_id, token=token)
    if guard_ik_ready_perf is not None:
        processing_latency[
            "guard_ik_ready_to_active_prepared_sec"
        ] = float(time.perf_counter() - guard_ik_ready_perf)
    robot = PiperUdpRobot(config["robot"])
    history = RobotTransformHistory(max_gap_sec=0.12)
    stop_monitor = threading.Event()
    output = ROOT / "online_replan_no_contact_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    prune_old_entries(output.parent)
    report = {
        "status": "prepared",
        "executed_motion": False,
        "contact_attempted": False,
        "returned_to_origin": False,
        "target_label": str(args.target_label),
        "runtime_arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "target_roles": {
            "requested_button": str(args.target_label),
            "perception_target": str(args.target_label),
            "search_safety_anchor": str(safety_anchor_label),
            "approach_safety_target": str(
                approach_safety_target_label
            ),
            "approach_reference_prepared_at_pointing": bool(
                pointing_safety_preparation is not None
            ),
        },
        "motion_strategy": (
            "legacy_rise_search_center_then_replan"
            if auto_pointing_enabled and args.center_before_approach
            else "single_rise_locate_direct_approach_motion_capture_replan"
            if auto_pointing_enabled
            else "visible_direct_approach_motion_capture_replan"
        ),
        "motion_stage_sequence": motion_stage_sequence(args.press_travel_m),
        "processing_latency": processing_latency,
        "prepared_active": prepared_active,
        "coarse_guard_ik": (
            None
            if coarse_guard_ik is None
            else {
                "target_panel_distance_m": float(
                    receding_plan.dynamic_guard_distance_m
                ),
                "mode": initial_coarse_guard_ik_mode,
                "solve_ms": coarse_guard_ik.solve_sec * 1000.0,
                "position_error_m": coarse_guard_ik.position_error_m,
                "orientation_error_deg": coarse_guard_ik.orientation_error_deg,
                "total_joint_delta_rad": coarse_guard_ik.max_joint_delta_rad,
                "virtual_continuation_steps": (
                    coarse_guard_ik.continuation_steps
                ),
                "maximum_continuation_joint_step_rad": (
                    coarse_guard_ik.max_continuation_joint_step_rad
                ),
                "live_orientation_offset_deg": (
                    None
                    if coarse_guard_ik.orientation_offset_deg is None
                    else list(coarse_guard_ik.orientation_offset_deg)
                ),
                "projected_orientation_offset_deg": (
                    None
                    if coarse_guard_ik.projected_orientation_offset_deg is None
                    else list(
                        coarse_guard_ik.projected_orientation_offset_deg
                    )
                ),
                "direct_candidate_count": (
                    coarse_guard_ik.direct_candidate_count
                ),
                "continuation_attempt_count": (
                    coarse_guard_ik.continuation_attempt_count
                ),
                "downstream_preview_distances_m": list(
                    coarse_guard_ik.downstream_preview_distances_m
                ),
                "downstream_preview_max_joint_step_rad": (
                    coarse_guard_ik.downstream_preview_max_joint_step_rad
                ),
                "downstream_preview_min_joint_limit_margin_rad": (
                    coarse_guard_ik.downstream_preview_min_joint_limit_margin_rad
                ),
            }
        ),
        "closest_reachable_search": closest_reachable_search,
        "initial_prepress_ik_warm_start": initial_prepress_seed_audit,
        "active_duration_sec": active_duration_sec,
        "active_motion_duration_sec": active_motion_duration_sec,
        "fallback_hold_sec": float(args.fallback_hold_sec),
        "near_zone": {
            "start_distance_m": (
                None if transition_distance is None else float(transition_distance)
            ),
            "transition_fraction": (
                None if transition_fraction is None else float(transition_fraction)
            ),
            "transition_time_sec": (
                None
                if transition_fraction is None
                else float(active_audit.transition_time_sec)
            ),
            "requested_speed_m_s": float(args.near_approach_speed_m_s),
            "audited_peak_speed_m_s": (
                None
                if transition_fraction is None
                else float(active_audit.near_zone_peak_speed_m_s)
            ),
        },
        "initial_target_source": (
            "fresh_live_six_frame_window"
            if initial_live_estimate is not None
            else "saved_hold_report"
        ),
        "initial_live_samples": initial_live_samples,
        "second_capture": {
            "enabled": True,
            "window_size": 6,
            "status": "pending",
            "trajectory_replacement_scheduled": False,
        },
        "vision_ready_browse": vision_ready_browse,
        "auto_pointing": pointing_audit,
        "pointing_arrival_active_sec": pointing_arrival_sec,
        "pointing_elbow_height_guard": {
            "origin_z_m": float(elbow_position(origin)[2]),
            "planned_min_z_m": pointing_min_elbow_z_m,
            "allowed_numerical_tolerance_m": ELBOW_HEIGHT_NUMERICAL_TOLERANCE_M,
        },
        "saved_to_live_target_delta_m": (
            None
            if initial_live_estimate is None
            else (
                coarse_button[:3, 3] - saved_coarse_button[:3, 3]
            ).tolist()
        ),
        "vision_start_active_sec": vision_start_active_sec,
        "near_vision_start_distance_m": float(
            args.near_vision_start_distance_m
        ),
        "near_vision_entry_distance_m": near_vision_entry_distance_m,
        "desired_handoff_active_time_sec": float(desired_handoff.active_time_sec),
        "handoff_active_time_sec": None,
        "fallback_distance_m": (
            0.10
            if receding_plan is None
            else float(receding_plan.dynamic_guard_distance_m)
        ),
        "pass_through_approach": {
            "enabled": bool(args.pass_through_approach),
            "brake_entry_distance_m": (
                None
                if receding_plan is None
                else float(receding_plan.brake_entry_distance_m)
            ),
            "pass_through_time_sec": pass_through_time_sec,
            "pass_through_joint_speed_rad_s": (
                None
                if pass_through_time_sec is None
                else float(approach_audit.waypoint_joint_speed_rad_s)
            ),
        },
        "receding_schedule": (
            None
            if receding_plan is None
            else {
                **receding_plan.__dict__,
                "coarse_guard_ik_error_m": float(
                    coarse_guard_ik.position_error_m
                ),
                "active_speed_m_s": active_speed_m_s,
                "streaming_goal_replacement": False,
                "maximum_visual_splices": 1,
            }
        ),
        "online_target_distance_m": float(args.target_distance_m),
        "material_tip_offset_m": float(args.material_tip_offset_m),
        "tcp_prepress_distance_m": prepress_tcp_distance_m,
        "press": {
            "stage_type": "independent_after_encoder_verified_prepress",
            "merged_into_approach_target": False,
            "translation_axis": "base_x",
            "prepress_xyz_constraint": (
                f"x=button.x-{prepress_tcp_distance_m:.3f}m; "
                "y=button.y; z=button.z"
            ),
            "requested_material_surface_distance_m": float(args.target_distance_m),
            "material_tip_offset_m": float(args.material_tip_offset_m),
            "prepress_tcp_distance_m": prepress_tcp_distance_m,
            "travel_m": float(args.press_travel_m),
            "endpoint_material_surface_distance_m": float(
                args.target_distance_m - args.press_travel_m
            ),
            "endpoint_tcp_distance_m": press_endpoint_tcp_distance_m,
            "requested_speed_m_s": float(args.press_speed_m_s),
            "joint_velocity_scale": float(args.press_joint_velocity_scale),
        },
        "requested_speeds_m_s": {
            "approach": float(args.approach_speed_m_s),
            "near_contact": float(args.near_speed_m_s),
            "retreat_10cm": float(args.retreat_speed_m_s),
            "return": float(args.return_speed_m_s),
            "joint_velocity_scale": float(args.joint_velocity_scale),
            "return_joint_velocity_scale": float(args.return_joint_velocity_scale),
            "duration_margin": float(args.duration_margin),
        },
        "audited_approach_peak_speed_m_s": float(
            active_audit.cartesian_peak_speed_m_s
        ),
        "precomputed_return": {
            "duration_sec": float(precomputed_return_audit.duration_sec),
            "peak_speed_m_s": float(
                precomputed_return_audit.cartesian_peak_speed_m_s
            ),
            "validation": return_prevalidation,
        },
        "rejections": {},
        "return_visual_safety_policy": (
            "disabled_before_prevalidated_continuous_return"
            if args.return_without_visual_safety
            else "armed_through_return"
        ),
    }
    live_audit_path = ROOT / "latest_online_replan_audit.yaml"

    def persist_live_audit(stage: str) -> None:
        snapshot = dict(report)
        snapshot["audit_stage"] = stage
        snapshot["audit_written_unix_sec"] = time.time()
        live_audit_path.write_text(
            yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    persist_live_audit("prepared_before_motion")
    robot_open = False
    motion_started = False
    continuous_return = None
    continuous_return_audit = None
    continuous_return_validation = None
    inverse_return = None
    inverse_return_audit = None
    inverse_return_validation = None
    inverse_return_includes_carried_pointing = False
    target_joints = None
    pressed_joints = None
    press_cycle = None
    press_cycle_audit = None
    press_cycle_validation = None
    press_return = None
    press_return_audit = None
    press_return_validation = None
    press_lateral_error_m = None
    pose_sample_lock = threading.Lock()
    latest_pose_sample = None
    return_visual_safety_disabled = False
    degraded_commit_lock = threading.Lock()
    degraded_commit_target = None
    degraded_commit_active = threading.Event()
    degraded_commit_allowed = threading.Event()
    near_depth_pause = NearButtonDepthPause(
        entry_distance_m=float(args.final_depth_degraded_distance_m)
    )
    report["near_button_depth_pause"] = {
        "requested": bool(args.pause_depth_near_button),
        "distance_m": float(args.final_depth_degraded_distance_m),
        "max_duration_sec": NEAR_BUTTON_DEPTH_LEASE_SEC, "encoder_lease_sec": 0.25,
        "events": near_depth_pause.events,
    }
    return_safety_transition_lock = threading.Lock()
    effective_near_speed_m_s = float(args.near_speed_m_s)
    if args.final_depth_degraded_commit:
        effective_near_speed_m_s *= float(
            args.final_depth_degraded_speed_scale
        )
    report["final_depth_degraded_commit"] = {
        "requested": bool(args.final_depth_degraded_commit),
        "armed": False,
        "active": False,
        "distance_gate_m": float(args.final_depth_degraded_distance_m),
        "near_speed_scale": float(args.final_depth_degraded_speed_scale),
        "effective_near_speed_m_s": effective_near_speed_m_s,
        "policy": (
            "armed_by_encoder_gate; activated_only_by_consecutive_coverage_loss; "
            "valid-pixel residual and collision stops retained"
        ),
    }

    def transition_to_return_safety_policy():
        nonlocal return_visual_safety_disabled
        with return_safety_transition_lock:
            degraded_commit_allowed.clear()
            if (
                args.return_without_visual_safety
                and safety_supervisor is not None
                and not return_visual_safety_disabled
            ):
                safety_supervisor.disarm()
                vision.request("safety_disable_matrix_reference")
                return_visual_safety_disabled = True
                report["return_visual_safety_disabled"] = True
            if degraded_commit_active.is_set():
                try:
                    vision.request(
                        "safety_set_depth_degraded_commit", enabled=False
                    )
                finally:
                    degraded_commit_active.clear()
                    report["final_depth_degraded_commit"].update(
                        armed=False,
                        active=False,
                        deactivated_before_return_unix_sec=time.time(),
                    )

    def wait_for_depth_recovery(command_finished, *, required_frames=3):
        """Drop the coverage exception only after live depth really recovers."""
        stable = 0
        while not command_finished.is_set():
            response = commit_vision.request("safety")["visual_safety"]
            if (
                response.get("monitoring_available", False)
                and response.get("ready", False)
                and not response.get("blocked", True)
            ):
                stable += 1
                if stable >= int(required_frames):
                    transition_to_return_safety_policy()
                    report["final_depth_degraded_commit"].update(
                        recovery_confirmed=True,
                        recovery_frame_sequence=response.get("frame_sequence"),
                        recovery_coverage_fraction=response.get(
                            "matrix_coverage_fraction"
                        ),
                    )
                    return
            else:
                stable = 0
            command_finished.wait(timeout=0.03)

    def execute_return(points, task_id):
        if args.return_without_visual_safety or not degraded_commit_active.is_set():
            transition_to_return_safety_policy()
            return standard.execute(points, token=token, task_id=task_id)
        command_finished = threading.Event()
        recovery_errors = []

        def recover_policy_when_observable():
            try:
                wait_for_depth_recovery(command_finished)
            except Exception as exc:
                recovery_errors.append(exc)

        recovery_thread = threading.Thread(
            target=recover_policy_when_observable, daemon=True
        )
        recovery_thread.start()
        try:
            result = standard.execute(points, token=token, task_id=task_id)
        finally:
            command_finished.set()
            recovery_thread.join(timeout=2.0)
        if recovery_errors:
            raise RuntimeError(
                "return depth-recovery handoff failed: "
                f"{recovery_errors[0]}"
            )
        # A short return may finish before three new camera frames arrive.
        # Motion has stopped, so closing the bounded exception is now safe.
        if degraded_commit_active.is_set():
            transition_to_return_safety_policy()
        return result

    def execute_blended_press_return(points, audit, task_id):
        """Run press/release/return as one command and switch guard in flight."""
        # This command starts with a press, not a return. Even when return
        # monitoring stays enabled, its policy clears depth-degraded commit.
        # Applying it before submission cancels a valid near-panel press.
        command_finished = threading.Event()
        transition_errors = []
        # Keep visual monitoring through the complete release back across the
        # encoder-verified prepress point. The 20 ms margin prevents socket
        # submission latency from disabling the guard a little too early.
        transition_delay_sec = float(audit.prepress_pass_time_sec) + 0.02

        def delayed_transition():
            if command_finished.wait(timeout=transition_delay_sec):
                return
            try:
                # Passing prepress can still leave the camera in its blind
                # region. Keep the exception until live coverage itself has
                # recovered for consecutive frames.
                if args.final_depth_degraded_commit and degraded_commit_active.is_set():
                    wait_for_depth_recovery(command_finished)
                elif not command_finished.is_set():
                    transition_to_return_safety_policy()
                report["continuous_press_return_safety_handoff"] = {
                    "scheduled_after_start_sec": transition_delay_sec,
                    "prepress_pass_time_sec": float(audit.prepress_pass_time_sec),
                    "visual_safety_kept_through_prepress": not args.pause_depth_near_button,
                }
            except Exception as exc:
                transition_errors.append(exc)

        transition_thread = threading.Thread(
            target=delayed_transition, daemon=True
        )
        transition_thread.start()
        try:
            result = standard.execute(points, token=token, task_id=task_id)
        finally:
            command_finished.set()
            transition_thread.join(timeout=2.0)
        if transition_errors:
            raise RuntimeError(
                "continuous press-return safety handoff failed: "
                f"{transition_errors[0]}"
            )
        return result

    def finish_carried_pointing_return(label):
        """Finish the exact inverse browse path after approach returns to pointing."""
        if not carry_pointing_pose:
            return []
        current = robot.state(max_age=2.0)
        if float(np.max(np.abs(current.joints - origin))) <= 0.03:
            return []
        pointing_error = float(
            np.max(np.abs(current.joints - pointing_joints))
        )
        if pointing_error > 0.03:
            raise RuntimeError(
                "return did not reach carried pointing pose before exact browse "
                f"inverse: {pointing_error:.4f} rad"
            )
        events = []
        for index, (points, validation) in enumerate(
            carried_pointing_returns, 1
        ):
            before = robot.state(max_age=2.0)
            expected = np.asarray(points[0]["positions"], dtype=float)
            start_error = float(np.max(np.abs(before.joints - expected)))
            if start_error > 0.03:
                raise RuntimeError(
                    "exact browse return start mismatch: "
                    f"segment={index} error={start_error:.4f} rad"
                )
            result = execute_return(
                points,
                task_id=(
                    f"{label}-carried-pointing-return-{index}-{time.time_ns()}"
                ),
            )
            events.append(
                {
                    "segment": index,
                    "start_error_rad": start_error,
                    "validation": validation,
                    "result": result,
                }
            )
        return events

    def monitor():
        nonlocal latest_pose_sample
        latest = -1.0
        while not stop_monitor.is_set():
            try:
                sample = robot.state(max_age=1.0)
                if sample.timestamp > latest:
                    history.add(sample.timestamp, sample.base_to_gripper)
                    latest = sample.timestamp
                    with pose_sample_lock:
                        latest_pose_sample = sample
            except (TimeoutError, RuntimeError):
                pass
            time.sleep(0.002)

    def publish_safety_pose():
        latest = -1.0
        while not stop_monitor.is_set():
            with pose_sample_lock:
                sample = latest_pose_sample
            if args.pause_depth_near_button and safety_supervisor is not None:
                with degraded_commit_lock:
                    pause_target = (None if degraded_commit_target is None
                                    else degraded_commit_target.copy())
                if sample is not None and pause_target is not None:
                    safety_supervisor.depth_pause_until = near_depth_pause.update(
                        sample.base_to_gripper, pause_target, sample.timestamp,
                        time.monotonic(), enabled=degraded_commit_allowed.is_set(),
                        hardware_ok=sample.raw.get("hardware_ok") == 1,
                    )
                else:
                    safety_supervisor.depth_pause_until = 0.0
            if (
                safety_supervisor is not None
                and sample is not None
                and sample.timestamp > latest
            ):
                try:
                    pose_vision.request(
                        "safety_camera_pose",
                        timestamp_monotonic=float(sample.timestamp),
                        base_to_camera=(sample.base_to_gripper @ handeye).tolist(),
                    )
                    latest = sample.timestamp
                    if (
                        args.final_depth_degraded_commit
                        and degraded_commit_allowed.is_set()
                        and not degraded_commit_active.is_set()
                    ):
                        with degraded_commit_lock:
                            target = (
                                None
                                if degraded_commit_target is None
                                else degraded_commit_target.copy()
                            )
                        if target is not None:
                            target_frame = base_x_distance_frame(target)
                            distance_m = float(
                                (
                                    sample.base_to_gripper[:3, 3]
                                    - target[:3, 3]
                                ) @ target_frame[:3, 2]
                            )
                            if (
                                -0.005 <= distance_m
                                <= float(args.final_depth_degraded_distance_m)
                            ):
                                activation = commit_vision.request(
                                    "safety_set_depth_degraded_commit",
                                    enabled=True,
                                )
                                degraded_commit_active.set()
                                report["final_depth_degraded_commit"].update(
                                    armed=True,
                                    active=bool(activation.get("enabled", False)),
                                    armed_ever=True,
                                    armed_distance_m=distance_m,
                                    armed_robot_timestamp=float(
                                        sample.timestamp
                                    ),
                                    arm_response=activation,
                                )
                except (OSError, RuntimeError, TimeoutError):
                    pass
            # The RGB-D stream cannot usefully consume hundreds of robot-pose
            # RPCs per second.  A bounded 50 Hz publisher keeps the latest pose
            # comfortably inside the freshness gate without starving safety
            # polling and image processing on the vision service.
            time.sleep(SAFETY_POSE_PUBLISH_PERIOD_SEC)

    try:
        robot.open()
        robot_open = True
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        pose_thread = threading.Thread(target=publish_safety_pose, daemon=True)
        pose_thread.start()
        initial = robot.state(max_age=2.0)
        expected_start_joints = pointing_joints if carry_pointing_pose else origin
        start_error = float(np.max(np.abs(initial.joints - expected_start_joints)))
        if initial.raw.get("hardware_ok") != 1 or start_error > 0.03:
            raise RuntimeError(f"arm is not at planned approach start: {start_error:.4f} rad")
        history.add(initial.timestamp, initial.base_to_gripper)
        if safety_supervisor is not None:
            safety_stage_started = time.perf_counter()
            pose_vision.request(
                "safety_camera_pose",
                timestamp_monotonic=float(initial.timestamp),
                base_to_camera=(initial.base_to_gripper @ handeye).tolist(),
            )
            processing_latency["approach_safety_pose_publish_sec"] = float(
                time.perf_counter() - safety_stage_started
            )
            stage_started = time.perf_counter()
            vision.request(
                "safety_set_mode", mode="predicted_depth_matrix"
            )
            processing_latency["approach_safety_mode_switch_sec"] = float(
                time.perf_counter() - stage_started
            )
            if pointing_safety_preparation is not None:
                # The complete target-12 matrix was already built and cleared
                # at this same stationary pointing pose in parallel with IK.
                # Verify it once more immediately before motion; do not reset
                # it or repeat the two full establishment preflights.
                report["visual_safety_ring_preflight"] = (
                    pointing_safety_preparation["ring_preflight"]
                )
                report["visual_safety_matrix_reference"] = (
                    pointing_safety_preparation["matrix_reference"]
                )
                stage_started = time.perf_counter()
                report["visual_safety_preflight"] = (
                    safety_supervisor.preflight(3.0)
                )
                processing_latency[
                    "approach_reused_matrix_final_preflight_sec"
                ] = float(time.perf_counter() - stage_started)
                processing_latency[
                    "approach_matrix_reference_reused"
                ] = True
            else:
                # Without a raised, target-specific preparation, establish the
                # reference here using the original fail-safe sequence.
                if initial_live_estimate is None:
                    stage_started = time.perf_counter()
                    vision.request(
                        "set_target", target_label=str(args.target_label)
                    )
                    processing_latency["approach_target_reset_sec"] = float(
                        time.perf_counter() - stage_started
                    )
                stage_started = time.perf_counter()
                report["visual_safety_ring_preflight"] = (
                    safety_supervisor.preflight(3.0)
                )
                processing_latency["approach_ring_preflight_sec"] = float(
                    time.perf_counter() - stage_started
                )
                stage_started = time.perf_counter()
                report["visual_safety_matrix_reference"] = vision.request(
                    "safety_capture_matrix_reference", frame_count=25
                )
                processing_latency["approach_matrix_capture_sec"] = float(
                    time.perf_counter() - stage_started
                )
                stage_started = time.perf_counter()
                report["visual_safety_preflight"] = (
                    safety_supervisor.preflight(3.0)
                )
                processing_latency["approach_matrix_preflight_sec"] = float(
                    time.perf_counter() - stage_started
                )
            processing_latency["approach_safety_total_sec"] = float(
                time.perf_counter() - safety_stage_started
            )
        wall_minus_monotonic = time.time() - time.monotonic()
        started_local = time.monotonic()
        if safety_supervisor is not None:
            safety_supervisor.arm()
        if rise_motion_completed_perf is not None:
            processing_latency[
                "rise_motion_to_active_start_sec"
            ] = float(time.perf_counter() - rise_motion_completed_perf)
        stage_started = time.perf_counter()
        started = splice.start_active(active_id=active_id, token=token)
        processing_latency["active_start_request_sec"] = float(
            time.perf_counter() - stage_started
        )
        motion_started = True
        report["executed_motion"] = True
        collect_motion_elapsed = 0.30 + vision_start_active_sec
        while (
            time.monotonic() - started_local
            - (
                safety_supervisor.total_pause_sec()
                if safety_supervisor is not None else 0.0
            )
            < collect_motion_elapsed
        ):
            time.sleep(0.01)

        window = make_button_window(
            config,
            max_position_rms_m=0.0035,
            max_position_jump_m=0.0070,
            max_sample_age_sec=0.35,
            max_span_sec=1.0,
        )
        last_sequence = 0
        online_started = time.monotonic()
        online_start_pause_sec = (
            safety_supervisor.total_pause_sec()
            if safety_supervisor is not None else 0.0
        )
        estimate = None
        samples_audit = []
        completed_safety_events = (
            len(safety_supervisor.events) if safety_supervisor is not None else 0
        )
        report["online_window_resets_after_safety"] = []
        report["online_capture_timing"] = {"rpc_sec": 0.0, "rpc_calls": 0, "largest_batch": 0}
        while (
            time.monotonic() - online_started
            - (
                safety_supervisor.total_pause_sec() - online_start_pause_sec
                if safety_supervisor is not None else 0.0
            )
            < 3.0
        ):
            if (
                safety_supervisor is not None
                and len(safety_supervisor.events) != completed_safety_events
            ):
                completed_safety_events = len(safety_supervisor.events)
                window = make_button_window(
                    config,
                    max_position_rms_m=0.0035,
                    max_position_jump_m=0.0070,
                    max_sample_age_sec=0.35,
                    max_span_sec=1.0,
                )
                estimate = None
                online_started = time.monotonic()
                online_start_pause_sec = safety_supervisor.total_pause_sec()
                report["online_window_resets_after_safety"].append(
                    {
                        "completed_event_count": completed_safety_events,
                        "last_sequence_before_reset": last_sequence,
                    }
                )
            fetch_started = time.perf_counter()
            response = vision.request(
                "latest", target_label=str(args.target_label), window_size=6,
                after_sequence=last_sequence, include_realtime=False,
                max_batch_age_sec=0.35,
            )
            report["online_capture_timing"]["rpc_sec"] += time.perf_counter() - fetch_started
            report["online_capture_timing"]["rpc_calls"] += 1
            batch = target_batch(response,after_sequence=last_sequence)
            report["online_capture_timing"]["largest_batch"] = max(
                report["online_capture_timing"]["largest_batch"], len(batch)
            )
            if not batch:
                time.sleep(0.01)
                continue
            for sample in batch:
                if int(sample["sequence"]) <= last_sequence:
                    continue
                last_sequence = int(sample["sequence"])
                capture_unix = float(sample["capture_time_unix_sec"])
                capture_monotonic = capture_unix - wall_minus_monotonic
                try:
                    base_to_tcp = history.transform_at(capture_monotonic, wait_sec=0.10)
                except RuntimeError as exc:
                    key = str(exc)
                    report["rejections"][key] = report["rejections"].get(key, 0) + 1
                    continue
                base_to_camera = base_to_tcp @ handeye
                raw_base_to_button = (
                    base_to_camera
                    @ np.asarray(sample["camera_to_button"], dtype=float)
                )
                if initial_live_estimate is None:
                    raise RuntimeError(
                        "online contact requires a fresh initialized panel plane"
                    )
                base_to_button, frozen_plane_correction_m = (
                    constrain_button_ray_to_reference_plane(
                        base_to_camera,
                        raw_base_to_button,
                        coarse_button,
                    )
                )
                try:
                    accepted=window.add(timed_target(sample,base_to_button,target_label=args.target_label))
                except ValueError as exc:
                    if sample.get('position_source')!='panel_layout_inferred':raise
                    key=str(exc)
                    report['rejections'][key]=report['rejections'].get(key,0)+1
                    continue
                samples_audit.append(
                    {
                        "sequence": last_sequence,
                        "capture_time_unix_sec": capture_unix,
                        "capture_age_sec": time.time() - capture_unix,
                        "accepted": bool(accepted),
                        "raw_depth_position_m": raw_base_to_button[:3, 3].tolist(),
                        "base_to_button_position_m": base_to_button[:3, 3].tolist(),
                        "frozen_plane_depth_correction_m": frozen_plane_correction_m,
                        "bbox_xyxy": list(sample.get("bbox_xyxy", [])),
                        "confidence": (None if sample.get('position_source')=='panel_layout_inferred'
                                       else float(sample.get("confidence", 0.0))),
                        "position_source": sample.get('position_source','yolo'),
                        "plane_rms_m": float(sample.get("plane_rms_m", float("nan"))),
                        "panel_support_detections": int(
                            sample.get("panel_support_detections", 0)
                        ),
                        "local_global_depth_delta_m": float(
                            sample.get("local_global_depth_delta_m", float("nan"))
                        ),
                    }
                )
                if not accepted:
                    report["rejections"]["input_gate"] = report["rejections"].get("input_gate", 0) + 1
                    continue
                estimate = window.estimate()
                if estimate.stable:
                    if args.final_depth_degraded_commit:
                        with degraded_commit_lock:
                            degraded_commit_target = (
                                estimate.base_to_button.copy()
                            )
                        degraded_commit_allowed.set()
                    break
                for reason in estimate.reasons:
                    report["rejections"][reason] = report["rejections"].get(reason, 0) + 1

            if estimate is not None and estimate.stable:
                break

        report["online_capture_timing"]["collection_sec"] = time.monotonic() - online_started
        report["online_samples"] = samples_audit
        report["second_capture"].update(
            status=("stable_six_frame_window" if estimate is not None and estimate.stable else "unstable_or_timed_out"),
            sample_count=len(samples_audit),
            accepted_sample_count=sum(1 for item in samples_audit if item["accepted"]),
            final_window=(None if estimate is None else {
                "stable": bool(estimate.stable),
                "sequence_range": list(estimate.sequence_range),
                "span_sec": float(estimate.span_sec),
                "position_rms_m": float(estimate.position_rms_m),
                "normal_rms_deg": float(estimate.normal_rms_deg),
                "max_position_jump_m": float(estimate.max_position_jump_m),
                "reasons": list(estimate.reasons),
            }),
        )
        persist_live_audit("second_capture_complete")
        if args.dynamic_fallback_only:
            report["online_plan_status"] = "validated_dynamic_fallback_only"
            if estimate is not None:
                report["fallback_only_window_stable"] = bool(estimate.stable)
        elif estimate is None or not estimate.stable:
            report["online_plan_status"] = "rejected_dynamic_fallback"
            report["second_capture"]["trajectory_replacement_scheduled"] = False
            persist_live_audit("second_capture_gate_rejected")
            # A pre-contact/press request is meaningful only after the live
            # six-frame measurement has actually replaced the coarse path.
            # Do not continue along a distant fallback leg and present it as a
            # 10 mm standoff test.
            raise RuntimeError(
                "second live capture was not stable; near-stage motion is inhibited"
            )
        else:
            estimated_active_elapsed_sec = max(
                0.0,
                time.monotonic() - started_local - 0.30
                - (
                    safety_supervisor.total_pause_sec()
                    if safety_supervisor is not None else 0.0
                ),
            )
            minimum_handoff = select_future_handoff(
                active,
                active_elapsed_sec=estimated_active_elapsed_sec,
                lead_sec=float(args.minimum_handoff_lead_sec),
            )
            handoff = (
                desired_handoff
                if desired_handoff.active_time_sec >= minimum_handoff.active_time_sec
                else minimum_handoff
            )
            report["handoff_active_time_sec"] = float(handoff.active_time_sec)
            report["handoff_selection"] = {
                "mode": (
                    "desired"
                    if handoff.active_time_sec == desired_handoff.active_time_sec
                    else "adapted_later_for_compute_margin"
                ),
                "estimated_active_elapsed_sec": estimated_active_elapsed_sec,
                "minimum_compute_lead_sec": float(
                    args.minimum_handoff_lead_sec
                ),
            }
            press_endpoint_distance = press_endpoint_tcp_distance_m
            base_x_button = base_x_distance_frame(estimate.base_to_button)
            prepress_ik_target = estimate.base_to_button.copy()
            prepress_ik_target[0, 3] -= prepress_tcp_distance_m
            if (
                initial_prepress_seed_joints is not None
                and initial_prepress_desired_tcp is not None
            ):
                prepress_ik = refine_precomputed_ik(
                    initial_prepress_desired_tcp,
                    prepress_ik_target[:3, 3],
                    initial_prepress_seed_joints,
                )
            else:
                prepress_ik = FixedRollFastIkPlanner(
                    roll_deg=roll_deg,
                    stand_off_m=0.0,
                    max_joint_delta_rad=ONLINE_IK_MAX_JOINT_DELTA_RAD,
                    max_perpendicular_deviation_deg=(
                        MAX_PERPENDICULAR_DEVIATION_DEG
                    ),
                ).solve(prepress_ik_target, handoff.positions)
            target_joints = prepress_ik.joints.copy()
            target_tcp = forward_kinematics(target_joints)[:3, 3]
            expected_prepress = estimate.base_to_button[:3, 3].copy()
            expected_prepress[0] -= prepress_tcp_distance_m
            if np.linalg.norm(target_tcp - expected_prepress) > 0.001:
                raise RuntimeError("prepress IK does not satisfy base-X/Y/Z target")
            actual_distance = float(
                (target_tcp - estimate.base_to_button[:3, 3])
                @ base_x_button[:3, 2]
            )
            fraction = 1.0
            ik = prepress_ik
            if args.press_travel_m > 0.0:
                press_ik_target = estimate.base_to_button.copy()
                press_ik_target[0, 3] -= press_endpoint_distance
                # The press is a separate 7 mm translation from an already
                # verified prepress configuration. Re-running the global
                # orientation search here can consume the entire fallback
                # hold on high buttons such as 16. Preserve the accepted
                # prepress orientation and solve only this local translation.
                press_ik = solve_independent_press_ik(
                    target_joints,
                    press_ik_target[:3, 3],
                )
                pressed_joints = press_ik.joints.copy()
                pressed_distance = float(
                    (
                        forward_kinematics(pressed_joints)[:3, 3]
                        - estimate.base_to_button[:3, 3]
                    )
                    @ base_x_button[:3, 2]
                )
                if abs(pressed_distance - press_endpoint_distance) > 0.001:
                    raise RuntimeError(
                        "press IK panel distance is "
                        f"{pressed_distance:.4f} m; expected "
                        f"{press_endpoint_distance:.4f} m"
                    )
                pressed_tcp = forward_kinematics(pressed_joints)[:3, 3]
                expected_pressed = estimate.base_to_button[:3, 3].copy()
                expected_pressed[0] -= press_endpoint_distance
                if np.linalg.norm(pressed_tcp - expected_pressed) > 0.001:
                    raise RuntimeError("press IK does not satisfy base-X/Y/Z target")
                ik = press_ik
            post_ik_active_elapsed_sec = max(
                0.0,
                time.monotonic() - started_local - 0.30
                - (
                    safety_supervisor.total_pause_sec()
                    if safety_supervisor is not None else 0.0
                ),
            )
            handoff, handoff_refreshed, refreshed_remaining_sec = (
                refresh_handoff_after_compute(
                    active,
                    handoff,
                    active_elapsed_sec=post_ik_active_elapsed_sec,
                )
            )
            if handoff_refreshed:
                report["handoff_selection"].update(
                    mode="refreshed_after_slow_ik",
                    post_ik_active_elapsed_sec=post_ik_active_elapsed_sec,
                    refreshed_remaining_sec=refreshed_remaining_sec,
                    refreshed_handoff_active_time_sec=float(
                        handoff.active_time_sec
                    ),
                )
                report["handoff_active_time_sec"] = float(
                    handoff.active_time_sec
                )
            replacement, replacement_audit = generate_splice_trajectory(
                handoff, target_joints,
                cartesian_speed_m_s=float(args.near_speed_m_s),
                joint_velocity_scale=float(args.joint_velocity_scale),
                point_period_sec=0.10,
                end_zone_distance_m=(
                    float(args.final_depth_degraded_distance_m)
                    if args.final_depth_degraded_commit else None
                ),
                end_zone_speed_m_s=(
                    effective_near_speed_m_s
                    if args.final_depth_degraded_commit else None
                ),
            )
            replacement_id = f"online-target-{suffix}"
            prepared_replacement = splice.prepare(
                replacement, trajectory_id=replacement_id, token=token
            )
            scheduled = splice.schedule_replacement(
                active_id=active_id, replacement_id=replacement_id,
                handoff_active_time_sec=handoff.active_time_sec, token=token,
            )
            online_schedule_compute_sec = time.monotonic() - online_started
            if args.inverse_return:
                executed_approach = compose_executed_approach(
                    active, handoff, replacement
                )
                if smooth_carried_return:
                    inverse_return, inverse_return_audit = (
                        generate_continuous_waypoint_trajectory(
                            target_joints,
                            pointing_joints,
                            origin,
                            first_cartesian_speed_m_s=float(
                                args.retreat_speed_m_s
                            ),
                            second_cartesian_speed_m_s=float(
                                args.return_speed_m_s
                            ),
                            first_joint_velocity_scale=float(
                                args.joint_velocity_scale
                            ),
                            second_joint_velocity_scale=float(
                                args.return_joint_velocity_scale
                            ),
                            point_period_sec=0.10,
                            duration_margin=float(args.duration_margin),
                        )
                    )
                    inverse_return_includes_carried_pointing = True
                else:
                    inverse_return = reverse_timed_trajectory(
                        executed_approach
                    )
                    inverse_return_audit = audit_sampled_trajectory(
                        inverse_return
                    )
                inverse_return_validation = standard.execute(
                    inverse_return, token=token,
                    task_id=f"online-inverse-return-prevalidate-{time.time_ns()}",
                    dry_run=True,
                )
            else:
                continuous_return, continuous_return_audit = (
                    generate_continuous_waypoint_trajectory(
                        target_joints, preapproach, origin,
                        first_cartesian_speed_m_s=float(args.retreat_speed_m_s),
                        second_cartesian_speed_m_s=float(args.return_speed_m_s),
                        first_joint_velocity_scale=float(args.joint_velocity_scale),
                        second_joint_velocity_scale=float(
                            args.return_joint_velocity_scale
                        ),
                        point_period_sec=0.10,
                        duration_margin=float(args.duration_margin),
                    )
                )
                continuous_return_validation = standard.execute(
                    continuous_return, token=token,
                    task_id=(
                        f"online-continuous-return-prevalidate-{time.time_ns()}"
                    ),
                    dry_run=True,
                )
            if (
                pressed_joints is not None
                and inverse_return is not None
            ):
                press_return, press_return_audit = (
                    generate_blended_press_inverse_return(
                        target_joints,
                        pressed_joints,
                        inverse_return,
                        press_speed_m_s=float(args.press_speed_m_s),
                        release_speed_m_s=float(args.return_speed_m_s),
                        press_joint_velocity_scale=float(
                            args.press_joint_velocity_scale
                        ),
                        return_joint_velocity_scale=float(
                            args.return_joint_velocity_scale
                        ),
                        point_period_sec=0.05,
                        duration_margin=float(args.duration_margin),
                    )
                )
                press_return_xyz = forward_kinematics_batch(
                    np.asarray([point["positions"] for point in press_return])
                )[:, :3, 3]
                press_endpoint_index = int(
                    np.argmin(
                        np.abs(
                            np.asarray(
                                [p["time_from_start_sec"] for p in press_return]
                            )
                            - press_return_audit.press_endpoint_time_sec
                        )
                    )
                )
                outbound_x = press_return_xyz[: press_endpoint_index + 1, 0]
                if np.any(np.diff(outbound_x) < -0.0005):
                    raise RuntimeError("press trajectory is not monotonic on base-X")
                press_lateral_error_m = float(
                    np.max(
                        np.linalg.norm(
                            press_return_xyz[: press_endpoint_index + 1, 1:3]
                            - estimate.base_to_button[None, 1:3, 3],
                            axis=1,
                        )
                    )
                )
                if press_lateral_error_m > 0.003:
                    raise RuntimeError(
                        "press trajectory lateral Y/Z error is "
                        f"{press_lateral_error_m * 1000.0:.2f} mm"
                    )
                if press_return_audit.prepress_joint_speed_rad_s < 0.01:
                    raise RuntimeError("blended return still stops at prepress")
                if press_return_audit.prepress_cartesian_speed_m_s < 0.02:
                    raise RuntimeError(
                        "blended return Cartesian speed is too low at prepress"
                    )
                press_return_validation = standard.execute(
                    press_return,
                    token=token,
                    task_id=f"online-press-return-prevalidate-{time.time_ns()}",
                    dry_run=True,
                )
            elif pressed_joints is not None:
                press_cycle, press_cycle_audit = generate_press_cycle_trajectory(
                    target_joints,
                    pressed_joints,
                    cartesian_speed_m_s=float(args.press_speed_m_s),
                    joint_velocity_scale=float(args.press_joint_velocity_scale),
                    point_period_sec=0.05,
                    duration_margin=float(args.duration_margin),
                )
                press_cycle_validation = standard.execute(
                    press_cycle,
                    token=token,
                    task_id=f"online-press-cycle-prevalidate-{time.time_ns()}",
                    dry_run=True,
                )
            report.update(
                online_plan_status="scheduled",
                online_compute_sec=online_schedule_compute_sec,
                continuous_return_prepare_sec=(
                    time.monotonic() - online_started - online_schedule_compute_sec
                ),
                window={
                    "sequence_range": list(estimate.sequence_range),
                    "span_sec": estimate.span_sec,
                    "latest_age_sec": estimate.latest_age_sec,
                    "position_rms_m": estimate.position_rms_m,
                    "normal_rms_deg": estimate.normal_rms_deg,
                    "max_position_jump_m": estimate.max_position_jump_m,
                },
                fast_ik_solve_ms=ik.solve_sec * 1000.0,
                fast_ik_position_error_m=ik.position_error_m,
                fast_ik_continuation_steps=ik.continuation_steps,
                fast_ik_max_continuation_joint_step_rad=(
                    ik.max_continuation_joint_step_rad
                ),
                prepress_ik_solve_ms=prepress_ik.solve_sec * 1000.0,
                prepress_ik_continuation_steps=prepress_ik.continuation_steps,
                press_ik_solve_ms=(
                    None
                    if pressed_joints is None
                    else press_ik.solve_sec * 1000.0
                ),
                target_fraction=float(fraction), actual_target_distance_m=float(actual_distance),
                replacement_duration_sec=replacement_audit.duration_sec,
                replacement_peak_speed_m_s=replacement_audit.cartesian_peak_speed_m_s,
                replacement_end_zone_distance_m=(
                    replacement_audit.end_zone_distance_m
                ),
                replacement_end_zone_speed_limit_m_s=(
                    replacement_audit.end_zone_speed_limit_m_s
                ),
                replacement_end_zone_peak_speed_m_s=(
                    replacement_audit.end_zone_peak_speed_m_s
                ),
                prepared_replacement=prepared_replacement, scheduled=scheduled,
                press_cycle_precomputed=(
                    None
                    if press_cycle_audit is None
                    else {
                        "single_bridge_command": True,
                        "returns_to_prepress": True,
                        "duration_sec": press_cycle_audit.duration_sec,
                        "point_count": press_cycle_audit.point_count,
                        "peak_speed_m_s": (
                            press_cycle_audit.cartesian_peak_speed_m_s
                        ),
                        "max_lateral_yz_error_m": press_lateral_error_m,
                        "validation": press_cycle_validation,
                    }
                ),
                press_return_precomputed=(
                    None
                    if press_return_audit is None
                    else {
                        "single_bridge_command": True,
                        "stops_only_at_press_endpoint": True,
                        "passes_prepress_without_stopping": True,
                        "duration_sec": press_return_audit.duration_sec,
                        "point_count": press_return_audit.point_count,
                        "peak_speed_m_s": (
                            press_return_audit.cartesian_peak_speed_m_s
                        ),
                        "press_endpoint_time_sec": (
                            press_return_audit.press_endpoint_time_sec
                        ),
                        "prepress_pass_time_sec": (
                            press_return_audit.prepress_pass_time_sec
                        ),
                        "prepress_joint_speed_rad_s": (
                            press_return_audit.prepress_joint_speed_rad_s
                        ),
                        "prepress_cartesian_speed_m_s": (
                            press_return_audit.prepress_cartesian_speed_m_s
                        ),
                        "inverse_join_index": (
                            press_return_audit.inverse_join_index
                        ),
                        "inverse_join_time_sec": (
                            press_return_audit.inverse_join_time_sec
                        ),
                        "max_press_lateral_yz_error_m": press_lateral_error_m,
                        "validation": press_return_validation,
                    }
                ),
                continuous_return_precomputed=(
                    None
                    if continuous_return_audit is None
                    else {
                        "duration_sec": continuous_return_audit.duration_sec,
                        "waypoint_time_sec": continuous_return_audit.waypoint_time_sec,
                        "retreat_peak_speed_m_s": (
                            continuous_return_audit.first_zone_peak_speed_m_s
                        ),
                        "return_peak_speed_m_s": (
                            continuous_return_audit.second_zone_peak_speed_m_s
                        ),
                        "waypoint_joint_speed_rad_s": (
                            continuous_return_audit.waypoint_joint_speed_rad_s
                        ),
                        "validation": continuous_return_validation,
                    }
                ),
                inverse_return_precomputed=(
                    None
                    if inverse_return_audit is None
                    else {
                        "mode": (
                            "continuous_through_pointing_to_origin"
                            if inverse_return_includes_carried_pointing
                            else "exact_inverse_executed_approach"
                        ),
                        "exact_joint_path_reversal": bool(
                            not inverse_return_includes_carried_pointing
                        ),
                        "passes_pointing_without_stopping": bool(
                            inverse_return_includes_carried_pointing
                        ),
                        "duration_sec": inverse_return_audit.duration_sec,
                        "point_count": inverse_return_audit.point_count,
                        "peak_speed_m_s": (
                            inverse_return_audit.cartesian_peak_speed_m_s
                        ),
                        "validation": inverse_return_validation,
                    }
                ),
            )
            report["second_capture"]["trajectory_replacement_scheduled"] = True
            persist_live_audit("trajectory_replacement_scheduled")
        terminal, transitions = wait_terminal(
            splice, token, active_duration_sec + 12.0
        )
        report["terminal"] = terminal
        report["execution_transitions"] = transitions
        if terminal.get("execution_status") == "failed":
            if safety_supervisor is not None and safety_supervisor.stop_active.is_set():
                raise RuntimeError("visual safety stop interrupted online approach")
            raise RuntimeError(f"asynchronous execution failed: {terminal}")
        reached = robot.state(max_age=2.0)
        if reached.raw.get("hardware_ok") != 1:
            raise RuntimeError("hardware state is not OK after online approach")
        if args.hold_at_fallback:
            if terminal.get("execution_status") != "fallback_complete":
                raise RuntimeError(
                    "diagnostic hold expected fallback_complete, got "
                    f"{terminal.get('execution_status')!r}"
                )
            held_distance_m = float(
                (
                    reached.base_to_gripper[:3, 3]
                    - coarse_button[:3, 3]
                ) @ coarse_button[:3, 2]
            )
            expected_hold_m = float(receding_plan.dynamic_guard_distance_m)
            if abs(held_distance_m - expected_hold_m) > 0.005:
                raise RuntimeError(
                    "held fallback distance is "
                    f"{held_distance_m:.4f} m; expected "
                    f"{expected_hold_m:.4f} +/- 0.005 m"
                )
            if safety_supervisor is not None:
                safety_supervisor.disarm()
                vision.request("safety_motion_state", paused=True)
                vision.request("safety_disable_matrix_reference")
                report["visual_safety_events"] = list(
                    safety_supervisor.events
                )
            report.update(
                status="held_at_closest_reachable_fallback",
                held_position=True,
                held_panel_distance_m=held_distance_m,
                held_joints_rad=reached.joints.tolist(),
                returned_to_origin=False,
            )
            persist_live_audit("held_at_closest_reachable_fallback")
            return 0
        if terminal.get("execution_status") == "replacement_complete":
            measured_start_distance = float(
                (
                    initial.base_to_gripper[:3, 3]
                    - estimate.base_to_button[:3, 3]
                ) @ base_x_button[:3, 2]
            )
            measured_reached_distance = float(
                (
                    reached.base_to_gripper[:3, 3]
                    - estimate.base_to_button[:3, 3]
                ) @ base_x_button[:3, 2]
            )
            measured_forward_progress = (
                measured_start_distance - measured_reached_distance
            )
            report["encoder_verified_reaching"] = {
                "start_distance_m": measured_start_distance,
                "reached_distance_m": measured_reached_distance,
                "forward_progress_m": measured_forward_progress,
                "requested_material_surface_distance_m": float(
                    args.target_distance_m
                ),
                "material_tip_offset_m": float(args.material_tip_offset_m),
                "requested_tcp_distance_m": prepress_tcp_distance_m,
            }
            if args.snapshot_at_prepress:
                time.sleep(0.15)
                report["prepress_camera_snapshot"] = vision.request("snapshot")
            persist_live_audit("encoder_reaching_verified")
            if measured_forward_progress < 0.03:
                raise RuntimeError(
                    "encoder verification found no meaningful reaching motion: "
                    f"{measured_forward_progress:.4f} m"
                )
            if abs(
                measured_reached_distance - prepress_tcp_distance_m
            ) > 0.005:
                raise RuntimeError(
                    "encoder-verified precontact distance is "
                    f"{measured_reached_distance:.4f} m; expected "
                    f"{prepress_tcp_distance_m:.4f} +/- 0.005 m"
                )
            contact_start_error = float(
                np.max(np.abs(reached.joints - target_joints))
            )
            if contact_start_error <= 0.03 and press_return is not None:
                report["contact_attempted"] = True
                return_points = press_return
                return_audit = press_return_audit
                return_validation = press_return_validation
                return_result = execute_blended_press_return(
                    return_points,
                    press_return_audit,
                    task_id=f"online-press-return-execute-{time.time_ns()}",
                )
                return_mode = "blended_press_release_inverse_return"
                report["press_execution"] = {
                    "result": return_result,
                    "continuous_through_prepress": True,
                    "prepress_joint_speed_rad_s": (
                        press_return_audit.prepress_joint_speed_rad_s
                    ),
                    "prepress_cartesian_speed_m_s": (
                        press_return_audit.prepress_cartesian_speed_m_s
                    ),
                }
            elif contact_start_error <= 0.03 and press_cycle is not None:
                report["contact_attempted"] = True
                press_result = standard.execute(
                    press_cycle,
                    token=token,
                    task_id=f"online-press-cycle-execute-{time.time_ns()}",
                )
                after_press = robot.state(max_age=2.0)
                press_return_error = float(
                    np.max(np.abs(after_press.joints - target_joints))
                )
                if after_press.raw.get("hardware_ok") != 1 or press_return_error > 0.03:
                    raise RuntimeError(
                        "press cycle did not return to prepress: "
                        f"{press_return_error:.4f} rad"
                    )
                report["press_execution"] = {
                    "result": press_result,
                    "returned_to_prepress": True,
                    "prepress_error_rad": press_return_error,
                }
                if inverse_return is not None:
                    return_points = inverse_return
                    return_audit = inverse_return_audit
                    return_validation = inverse_return_validation
                    return_result = execute_return(
                        return_points,
                        task_id=f"online-inverse-return-execute-{time.time_ns()}",
                    )
                    return_mode = (
                        "continuous_through_pointing_to_origin"
                        if inverse_return_includes_carried_pointing
                        else "exact_inverse_executed_approach"
                    )
                elif continuous_return is not None:
                    return_points = continuous_return
                    return_audit = continuous_return_audit
                    return_validation = continuous_return_validation
                    return_result = execute_return(
                        return_points,
                        task_id=f"online-continuous-return-execute-{time.time_ns()}",
                    )
                    return_mode = "continuous_precomputed_via_10cm"
                else:
                    raise RuntimeError("no return trajectory after press cycle")
            else:
                if contact_start_error <= 0.03 and inverse_return is not None:
                    return_points = inverse_return
                    return_audit = inverse_return_audit
                    return_validation = inverse_return_validation
                    return_result = execute_return(
                        return_points,
                        task_id=f"online-inverse-return-execute-{time.time_ns()}",
                    )
                    return_mode = (
                        "continuous_through_pointing_to_origin"
                        if inverse_return_includes_carried_pointing
                        else "exact_inverse_executed_approach"
                    )
                elif contact_start_error <= 0.03 and continuous_return is not None:
                    return_points = continuous_return
                    return_audit = continuous_return_audit
                    return_validation = continuous_return_validation
                    return_result = execute_return(
                        return_points,
                        task_id=f"online-continuous-return-execute-{time.time_ns()}",
                    )
                    return_mode = "continuous_precomputed_via_10cm"
                else:
                    return_points, return_audit = generate_continuous_waypoint_trajectory(
                        reached.joints, preapproach, origin,
                        first_cartesian_speed_m_s=float(args.retreat_speed_m_s),
                        second_cartesian_speed_m_s=float(args.return_speed_m_s),
                        first_joint_velocity_scale=float(args.joint_velocity_scale),
                        second_joint_velocity_scale=float(args.return_joint_velocity_scale),
                        point_period_sec=0.10,
                        duration_margin=float(args.duration_margin),
                    )
                    return_validation = standard.execute(
                        return_points,
                        token=token,
                        task_id=f"online-continuous-return-regenerated-validate-{time.time_ns()}",
                        dry_run=True,
                    )
                    return_result = execute_return(
                        return_points,
                        task_id=f"online-continuous-return-regenerated-{time.time_ns()}",
                    )
                    return_mode = "continuous_regenerated_via_10cm"
            precomputed_start_error = contact_start_error
            if return_mode == "blended_press_release_inverse_return":
                report["retreat"] = {
                    "continuous_with_return": True,
                    "passes_prepress_without_stopping": True,
                    "prepress_joint_speed_rad_s": (
                        return_audit.prepress_joint_speed_rad_s
                    ),
                    "prepress_cartesian_speed_m_s": (
                        return_audit.prepress_cartesian_speed_m_s
                    ),
                    "inverse_rejoined": True,
                }
            elif return_mode == "exact_inverse_executed_approach":
                report["retreat"] = {
                    "continuous_with_return": True,
                    "exact_inverse": True,
                    "duration_sec": float(replacement_audit.duration_sec),
                    "peak_speed_m_s": float(
                        replacement_audit.cartesian_peak_speed_m_s
                    ),
                }
            elif return_mode == "continuous_through_pointing_to_origin":
                report["retreat"] = {
                    "continuous_with_return": True,
                    "single_command_to_origin": True,
                    "passes_pointing_without_stopping": True,
                    "pointing_time_sec": return_audit.waypoint_time_sec,
                    "pointing_joint_speed_rad_s": (
                        return_audit.waypoint_joint_speed_rad_s
                    ),
                    "first_zone_peak_speed_m_s": (
                        return_audit.first_zone_peak_speed_m_s
                    ),
                    "second_zone_peak_speed_m_s": (
                        return_audit.second_zone_peak_speed_m_s
                    ),
                }
            else:
                report["retreat"] = {
                    "continuous_with_return": True,
                    "duration_sec": return_audit.waypoint_time_sec,
                    "peak_speed_m_s": return_audit.first_zone_peak_speed_m_s,
                    "waypoint_joint_speed_rad_s": (
                        return_audit.waypoint_joint_speed_rad_s
                    ),
                }
        else:
            precomputed_start_error = float(
                np.max(np.abs(reached.joints - active_endpoint))
            )
            return_audit = precomputed_return_audit
            return_validation = return_prevalidation
            return_result = execute_return(
                precomputed_return,
                task_id=f"online-return-prevalidated-execute-{time.time_ns()}",
            )
            return_mode = (
                "continuous_precomputed_through_pointing_to_origin"
                if smooth_carried_return
                else "precomputed_prevalidated"
            )
        carried_return_events = finish_carried_pointing_return("online-success")
        final = robot.state(max_age=2.0)
        if safety_supervisor is not None:
            # Unless the explicitly selected return policy already disarmed it,
            # keep the guard armed through press, retreat, and full return.
            safety_supervisor.disarm()
            vision.request("safety_disable_matrix_reference")
        origin_error = float(np.max(np.abs(final.joints - origin)))
        if final.raw.get("hardware_ok") != 1 or origin_error > 0.03:
            raise RuntimeError(f"origin return failed: {origin_error:.4f} rad")
        report.update(
            status=(
                "complete_online_press_returned_origin"
                if terminal.get("execution_status") == "replacement_complete"
                and args.press_travel_m > 0.0
                else "complete_online_touch_returned_origin"
                if terminal.get("execution_status") == "replacement_complete"
                and args.target_distance_m <= 0.001
                else "complete_online_replan_returned_origin"
                if terminal.get("execution_status") == "replacement_complete"
                else "complete_safe_fallback_returned_origin"
            ),
            returned_to_origin=True,
            origin_error_rad=origin_error,
            return_motion={
                "mode": return_mode,
                "precomputed_start_error_rad": precomputed_start_error,
                "duration_sec": return_audit.duration_sec,
                "peak_speed_m_s": return_audit.cartesian_peak_speed_m_s,
                "validation": return_validation, "result": return_result,
                "carried_pointing_return_events": carried_return_events,
            },
        )
        if safety_supervisor is not None:
            report["visual_safety_events"] = list(safety_supervisor.events)
        return 0
    except Exception as exc:
        safety_stopped = bool(
            safety_supervisor is not None and safety_supervisor.stop_active.is_set()
        )
        report.update(
            status="visual_safety_stopped" if safety_stopped else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        if safety_supervisor is not None:
            # A stop may already have cleared and resumed before an unrelated
            # trajectory error is raised. Preserve those completed events too.
            report["visual_safety_events"] = list(safety_supervisor.events)
        if safety_stopped:
            report["visual_safety_stop"] = {
                "guard": safety_supervisor.trigger_status,
                "stop_response": safety_supervisor.stop_response,
                "monitor_error": safety_supervisor.error,
                "automatic_recovery_inhibited": True,
            }
        # A stopped bridge can still own a queued press-return command.
        # Resetting its latch here would resume that command concurrently with
        # recovery. Preserve the latch and require explicit guarded recovery.
        if (
            robot_open and motion_started and not report["returned_to_origin"]
            and not safety_stopped and not isinstance(exc, TimeoutError)
        ):
            try:
                terminal, transitions = wait_terminal(
                    splice, token, active_duration_sec + 12.0
                )
                current = robot.state(max_age=2.0)
                pre_error = float(np.max(np.abs(current.joints - active_endpoint)))
                if pre_error <= 0.03:
                    recovery_audit = precomputed_return_audit
                    validation = return_prevalidation
                    result = execute_return(
                        precomputed_return,
                        task_id=f"online-failure-prevalidated-return-{time.time_ns()}",
                    )
                    recovery_mode = "precomputed_prevalidated"
                else:
                    recovery, recovery_audit = generate_timed_joint_trajectory(
                        current.joints, origin,
                        cartesian_speed_m_s=float(args.return_speed_m_s),
                        joint_velocity_scale=float(args.return_joint_velocity_scale),
                        point_period_sec=0.10,
                        duration_margin=float(args.duration_margin),
                        cartesian_speed_cap_m_s=PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
                    )
                    validation = standard.execute(
                        recovery,
                        token=token,
                        task_id=f"online-failure-return-origin-validate-{time.time_ns()}",
                        dry_run=True,
                    )
                    result = execute_return(
                        recovery,
                        task_id=f"online-failure-return-origin-{time.time_ns()}",
                    )
                    recovery_mode = "regenerated"
                carried_return_events = finish_carried_pointing_return(
                    "online-failure"
                )
                final = robot.state(max_age=2.0)
                error = float(np.max(np.abs(final.joints - origin)))
                report["failure_recovery"] = {
                    "terminal": terminal, "transitions": transitions,
                    "mode": recovery_mode,
                    "duration_sec": recovery_audit.duration_sec,
                    "origin_error_rad": error,
                    "validation": validation, "result": result,
                    "carried_pointing_return_events": carried_return_events,
                }
                report["returned_to_origin"] = error <= 0.03
            except Exception as recovery_exc:
                report["failure_recovery_error"] = (
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                )
        raise
    finally:
        stop_monitor.set()
        degraded_commit_allowed.clear()
        if degraded_commit_active.is_set():
            try:
                vision.request(
                    "safety_set_depth_degraded_commit", enabled=False
                )
            except Exception:
                pass
            degraded_commit_active.clear()
            report["final_depth_degraded_commit"]["active"] = False
        if (
            safety_supervisor is not None
            and (report.get("returned_to_origin") or not motion_started)
        ):
            try:
                vision.request("safety_disable_matrix_reference")
            except Exception:
                pass
        if safety_supervisor is not None:
            safety_supervisor.close()
        robot.close()
        persist_live_audit("finalized")
        path = output / "online_replan_report.yaml"
        path.write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
        print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), end="")


if __name__ == "__main__":
    raise SystemExit(main())
