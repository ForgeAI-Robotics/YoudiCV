#!/usr/bin/env python3
"""Run the test-ready light-verified elevator button workflow."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import yaml
from piper_handeye.numeric_layout_handoff import (
    task_vision, require_layout_server_support, LayoutTaskVision,
    SourceAwareButtonWindow, target_batch, timed_target,
    localization_provenance,
)
from piper_handeye.cycle_recovery import cycle_parent_recovery_required
from piper_handeye.retry_panel_localization import (
    RetryPanelReference, capture_camera_matrix, collect_retry_panel_target,
)
from piper_handeye.retry_observation import checked_retry_distance, RETRY_PREPARATION_DISTANCE_M

from piper_handeye.button_light_flow import (
    PRODUCTION_RETRY_STANDOFF_M,
    ButtonFlowConfig,
    ButtonLightFlow,
    CALL_OFF_TIMEOUT_FLOW,
    NUMERIC_CYCLE_TIMEOUT_FLOWS,
    production_flow_config,
)
from piper_handeye.mission_status import (
    DEFAULT_OUTBOX as DEFAULT_MISSION_STATUS_OUTBOX,
    append_status as append_mission_status,
    normalize_flow_event,
    normalize_light_observation,
    new_task_id,
)
from piper_handeye.button_light_hardware import (
    DryValidatedSegmentExecutor,
    FreshLightObserver,
    POST_PRESS_ON_CONFIRMATION_FRAMES,
    POST_PRESS_RECHECK_BUDGET_SEC,
    POST_PRESS_RECHECK_MAX_FRAMES,
    POST_PRESS_RECHECK_OFF_FRAMES,
    POST_PRESS_VISIBILITY_EDGE_MARGIN_PX,
    VisualSegmentMotionSafety,
    assemble_button_light_runtime,
    target_is_safely_visible,
)
from piper_handeye.near_button_depth_pause import NEAR_BUTTON_DEPTH_LEASE_SEC
from piper_handeye.button_light_process import OpenCvButtonLightProcess
from piper_handeye.arm_report import (
    capture_failure, finalize_report, report_yaml, update_report_checkpoint, write_report_atomic,
)
from piper_handeye.guarded_press import GuardedPressUdpClient
from piper_handeye.press_preparation import BackgroundPressPreparation
from piper_handeye.failure_image_archive import FailureImageArchive, remove_success_images
from piper_handeye.observation_pose import (
    OBSERVATION_JOINT_TOLERANCE_RAD, OBSERVATION_STABLE_SEC,
    OBSERVATION_CONFIRM_TIMEOUT_SEC, OBSERVATION_STABILITY_DELTA_RAD,
    OBSERVATION_CAPTURE_DRIFT_RAD, OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD,
    OBSERVATION_FEEDBACK_MAX_AGE_SEC, wait_for_stable_observation_pose,
)
from piper_handeye.pipeline_timing import configure_timing, finish_timing_with_analysis, traced, record_interval
from piper_handeye.kinematics import forward_kinematics
from piper_handeye.realtime_planner import (
    FixedRollFastIkPlanner,
    PRESS_SEGMENT_MAX_JOINT_DELTA_RAD,
    RobotTransformHistory,
    TimedButtonTransform,
)
from piper_handeye.robot import PiperUdpRobot
from piper_handeye.rolling_speed import RollingSpeedConfig, plan_receding_approach
from piper_handeye.safety_stop import SafetyStopUdpClient, VisualSafetySupervisor
from piper_handeye.segmented_button_motion import (
    plan_segmented_press_attempt,
    reusable_near_observation_joints,
    solve_camera_centered_observation_joints,
)
from piper_handeye.timed_trajectory import (
    TimedTrajectoryUdpClient,
    generate_timed_joint_trajectory,
)
from piper_handeye.trajectory_splice import (
    PreparedSpliceUdpClient,
    generate_first_feasible_splice,
    select_future_handoff,
)
from piper_handeye.vision_rpc import VisionRpcClient
from piper_handeye.observation_visibility import (
    complete_button_box, stable_target_visibility, OBSERVATION_EDGE_MARGIN_PX)
from run_online_replan_no_contact import (
    collect_stationary_target,
    constrain_button_ray_to_reference_plane,
    load_handeye,
    make_button_window,
    resolve_path,
    solve_independent_press_ik,
    visible_target,
    complete_visible_target,
    wait_for_stable_visibility_after_target_switch,
    wait_terminal,
)


ROOT = Path(__file__).resolve().parent


def preload_light_processing_dependencies(audit: dict) -> tuple:
    """Finish first imports before pointing or starting arm feedback threads.

    This only loads code in this task process. It neither starts the light
    worker nor creates reusable lamp references or changes any safety gate.
    """
    audit.update({
        "policy": "before_pointing_and_arm_feedback",
        "status": "loading",
        "started_monotonic_sec": time.monotonic(),
        "started_unix_sec": time.time(),
        "modules": [],
    })
    modules = []
    try:
        for name in (
            "cv2",
            "piper_handeye.paired_button_light",
            "piper_handeye.initial_light_guard",
            "piper_handeye.initial_observation_recovery",
        ):
            began = time.monotonic(), time.time()
            entry = {"name": name, "already_loaded": name in sys.modules,
                     "started_monotonic_sec": began[0]}
            audit["modules"].append(entry)
            status = "error"
            try:
                modules.append(importlib.import_module(name))
                status = "ok"
            finally:
                entry.update(status=status, ended_monotonic_sec=time.monotonic())
                entry["elapsed_sec"] = entry["ended_monotonic_sec"] - began[0]
                record_interval("startup.import." + name, "startup", began, status=status)
        dependencies = (
            modules[1].PairedButtonLight,
            modules[2].InitialLightGuard,
            modules[3].InitialObservationRecovery,
            modules[3].recheck_initial_call_view,
        )
        audit["status"] = "ready"
        return dependencies
    except BaseException:
        audit["status"] = "failed"
        raise
    finally:
        audit["ended_monotonic_sec"] = time.monotonic()
        audit["elapsed_sec"] = audit["ended_monotonic_sec"] - audit["started_monotonic_sec"]


def requires_fixed_call_reference(target_label):
    return str(target_label).casefold() in {"up", "down"}


def is_numeric_button(target_label):
    label = str(target_label)
    return label.isdecimal() and 1 <= int(label) <= 16


def requires_stationary_press_reference(target_label):
    return requires_fixed_call_reference(target_label) or is_numeric_button(target_label)


LIGHT_EVIDENCE_POLICY = {
    "image_box_pairing": "same_inference_frame_only",
    "light_rgb_publication": {
        "ordinary_yolo": "yolo_2d_before_depth",
        "motion_inputs": "completed_rgbd_only",
        "inferred_roi": "completed_same_frame_fallback",
        "max_rgb_age_sec": .35,
        "stationary_numeric_max_rgb_age_sec": .7,
        "stationary_numeric_local_retry_max_rgb_age_sec": 1.0,
        "stationary_call_max_rgb_age_sec": .7,
        "stationary_call_local_retry_max_rgb_age_sec": 1.0,
        "stationary_retry_admission": "same_frame_region_and_continuous_healthy_encoder_hold",
        "stale_retry": "existing_budget_only_after_continuity_already_broken",
        "decisive_to_stale": "uncertain_break_preserved",
    },
    "post_press": "opencv_on_priority_unless_paired_off_otherwise_original_comparison",
    "post_press_stationary_confirmation": {
        "paired_on_frames": 1,
        "paired_off_observations_before_retry": 2,
        "on_source": "paired_single_stationary_observation",
        "moving_monitor_on_frames": POST_PRESS_ON_CONFIRMATION_FRAMES,
        "image_and_comparison_quality": "unchanged",
    },
    "post_press_uncertain_adoption": {
        "policy": "opencv_on_priority_unless_paired_off_v1",
        "single_image_on_priority": "adopt_opencv_on_immediately_unless_comparison_off",
        "comparison_exception": "preserve_absolute_result_and_check_independent_motion_health",
        "three_uncertain_then": "only_when_no_adoptable_opencv_on",
        "stationary_recheck_paired_on_frames": 1,
        "bounded_recheck_unresolved": "adopt_latest_valid_opencv_on_or_off",
        "moving_monitor_on_frames": 2,
        "initial_and_arrival_confirmation": "unchanged",
        "invalid_or_stale_frame": "never_adopt",
    },
    "reference_collection": "fixed_5cm_observation_before_press",
    "post_press_off_crop_alignment": {
        "positions": ["near", "near_confirmation", "near_recheck"],
        "authorization": "frozen_verified_reference_and_fresh_stationary_pose_and_absolute_off",
        "fallback_only": "original_comparison_uncertain",
        "method": "common_integer_union_crop_original_off_thresholds",
        "min_button_size_px": 48., "max_center_shift_fraction": .06,
        "max_log_size_change": .06, "max_pair_span_sec": 30.,
        "processing_budget_sec": .20,
        "reference_selection": "original_background_score_no_reselection",
        "motion_unknowns": "not_used_in_stationary_off_streak",
    },
    "call_off_reference_alignment": {
        "trigger": "stationary_confirmed_off_pair_original_comparison_failed",
        "methods": ["common_pixel_crop", "background_registration"],
        "background_limits": "same_existing_aligned_off_checks_with_mean15_fallback",
        "arrow_masks_and_absolute_on_threshold": "unchanged",
        "moving_and_depth_max_age_sec": .35,
    },
    "call_first_approach": "moving_ik_direct_to_reference_no_prepress_retreat",
    "numeric_first_approach": "moving_ik_direct_to_reference_no_prepress_retreat",
    "initial_and_waiting": "absolute_with_existing_temporal_confirmation",
    "arrival_off_during_return": "three_fresh_absolute_off_frames_after_verified_press_on",
    "comparison_parameters": "provisional_pending_live_validation",
    "background_normalization": "robust_ring_affine_v1",
    "numeric_post_press_registration": {
        "mode": "stationary_background_registration_v1",
        "targets": "numeric_1_to_16",
        "trigger": "existing_comparison_uncertain_and_whole_local_strong_on",
        "position": "fresh_encoder_verified_fixed_observation_only",
        "fit_region": "surroundings_excluding_button",
        "max_center_shift_fraction": .10,
        "max_log_size_change": .08,
        "previous_strong_on_view_limits": [.06, .06],
        "background_and_brightness_thresholds": "unchanged",
        "reference_selection": "original_background_score_before_registration",
        "initial_off_reference_arrows_moving_monitor": "unchanged",
        "original_images_and_motion_geometry": "unchanged",
    },
    "numeric_post_press_changed_area_relaxation": {
        "policy": "numeric_stationary_strong_on_area99_v1",
        "targets": "numeric_1_to_16_only",
        "position": "fresh_encoder_verified_stationary_post_press_only",
        "max_changed_fraction": .99,
        "denominator": "button_center_analysis_mask_pixels",
        "numerator": "largest_connected_area_with_background_corrected_rise_at_least_25_gray",
        "absolute_requirement": "whole_and_local_on",
        "min_original_center_mean_rise_gray": 25.,
        "base_max_changed_fraction": .70,
        "moving_monitor_arrows_initial_and_off_references": "unchanged",
    },
    "numeric_post_press_background_fallback": {
        "mode": "numeric_stationary_strong_on_mean20_v1",
        "targets": "numeric_1_to_16",
        "position": "verified_fixed_5cm_observation_only",
        "absolute_requirement": "whole_and_local_on",
        "background_max_mean_error": 20.,
        "background_max_p90_error": 50.,
        "background_inlier_max_error": 30.,
        "background_min_inlier_fraction": .75,
        "changed_fraction_range": [.08, .70],
        "changed_source_pixels_min": 12,
        "initial_and_off_reference_policy": "unchanged",
        "moving_monitor_enabled": False,
    },
    "clipped_background_fallback": {
        "mode": "clipped_background_70_80_v1",
        "position": "verified_fixed_5cm_observation_only",
        "moving_monitor_enabled": False,
        "background_and_light_checks": "unchanged",
    },
    "down_initial": "reject_local_only_on_with_bounded_view_recovery_and_stationary_wait",
    "call_initial_recovery": {
        "labels": ["up", "down"], "view_budget_sec": 6.0,
        "missing_frames_trigger_reacquisition": 3, "max_view_attempts": 2,
        "j5_step_limit_rad": .025, "j5_total_offset_limit_rad": .05,
        "stationary_wait_interval_sec": 1.0,
        "confirmation": "three_consecutive_valid_new_frames_after_view_and_depth_reacquisition",
        "numeric_initial_policy": "unchanged",
    },
    "post_press_uncertain_recheck": {
        "mode": "bounded_stationary_post_press_recheck_v1",
        "position": "fixed_5cm_observation_only",
        "budget_sec": POST_PRESS_RECHECK_BUDGET_SEC,
        "max_new_frames": POST_PRESS_RECHECK_MAX_FRAMES,
        "off_confirmation_frames": POST_PRESS_RECHECK_OFF_FRAMES,
        "on_confirmation_frames": POST_PRESS_ON_CONFIRMATION_FRAMES,
        "baseline": "original_frozen_prepress_off_bank",
    },
}
POST_PRESS_RETREAT_SPEED_M_S = 0.25
POST_PRESS_RETREAT_JOINT_VELOCITY_SCALE = 0.75
PREPRESS_RETURN_SPEED_M_S = 0.25


class TerminationRequested(RuntimeError):
    """Convert service termination into normal guarded cleanup."""


SUCCESS_FLOW_STATUSES = {
    "arrived_after_initial_on",
    "arrived_after_verified_press",
    CALL_OFF_TIMEOUT_FLOW,
    *NUMERIC_CYCLE_TIMEOUT_FLOWS,
}


def flow_status_exit_code(status: str) -> int:
    """Expose business-flow failure to process supervisors."""
    return 0 if str(status) in SUCCESS_FLOW_STATUSES else 2


def retain_button_run_images(report, archive, output, task_id, target_label):
    """Run after safety/robot cleanup; a skipped ON timeout is not full success."""
    image_success = (int(report.get('process_exit_code', 1)) == 0
                     and report.get('flow_status') not in NUMERIC_CYCLE_TIMEOUT_FLOWS)
    try:
        report['image_retention'] = archive.finalize(success=image_success, context={
            'run': output.name, 'task_id': task_id, 'target_label': str(target_label),
            'process_exit_code': report.get('process_exit_code'),
            'failure': report.get('failure'), 'flow_status': report.get('flow_status'),
            'failure_stage': report.get('failure_stage'),
            'numeric_cycle_on_wait_timeout': report.get('numeric_cycle_on_wait_timeout'),
            'observation_policy': report.get('observation_pose_policy')})
        if image_success:
            report['image_retention']['removed_success_images'] = remove_success_images(output)
    except Exception as image_error:
        report['image_retention'] = {'policy': 'failure_only', 'error': str(image_error)}


def journal_phase(kind: str, value: dict) -> str:
    if str(kind) == "flow_event" and isinstance(value, dict) and value.get("event"):
        return f"flow:{value['event']}"
    return str(kind)


def append_journal_checkpoint(
    report: dict,
    kind: str,
    value: dict,
    *,
    monotonic_sec: float | None = None,
    time_unix_sec: float | None = None,
) -> dict:
    """Append an ordered checkpoint without erasing a terminal report status."""
    journal = report.setdefault("journal", [])
    row = {
        "sequence": len(journal) + 1,
        "kind": str(kind),
        "monotonic_sec": time.monotonic() if monotonic_sec is None else monotonic_sec,
        "time_unix_sec": time.time() if time_unix_sec is None else time_unix_sec,
        "value": value,
    }
    journal.append(row)
    if report.get("lifecycle_status") in {None, "initializing", "running"}:
        report["lifecycle_status"] = "running"
    if report.get("status") in {None, "initializing", "running"}:
        report["status"] = "running"
    report["phase"] = journal_phase(kind, value)
    return row


@traced('report.journal_append', 'journal')
def persist_journal_row(path: Path, row: dict) -> None:
    """Durably append one checkpoint without serialising the full report."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(
            row, ensure_ascii=False, separators=(",", ":"), default=str
        ))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def origin_verification(state, origin: np.ndarray, tolerance_rad: float = 0.03) -> dict:
    error = float(np.max(np.abs(np.asarray(state.joints, dtype=float) - origin)))
    hardware_ok = state.raw.get("hardware_ok")
    return {
        "verified": bool(hardware_ok == 1 and error <= float(tolerance_rad)),
        "hardware_ok": hardware_ok,
        "encoder_error_rad": error,
        "tolerance_rad": float(tolerance_rad),
        "joints_rad": np.asarray(state.joints, dtype=float).tolist(),
    }


def collect_cleanup_errors(callbacks) -> list[dict]:
    """Run every cleanup callback and retain all failures without masking peers."""
    errors = []
    for name, callback in callbacks:
        try:
            callback()
        except Exception as exc:
            errors.append({
                "component": str(name),
                "error": f"{type(exc).__name__}: {exc}",
            })
    return errors


def _raise_termination(signum, _frame):
    raise TerminationRequested(f"received signal {signum}")


def press_attempt_mode(attempt: int) -> str:
    """Select the motion contract from the state machine's attempt number."""
    value = int(attempt)
    if value < 1:
        raise ValueError("press attempt must be positive")
    return (
        "moving_rgbd_receding_approach"
        if value == 1
        else "current_observation_to_prepress"
    )


def validate_production_tactics(config: dict) -> dict:
    """Fail before robot I/O if an obsolete press tactic is configured."""
    guarded = config["guarded_press"]
    force = guarded["current_guard"]
    expected = {
        "retry_standoff_m": PRODUCTION_RETRY_STANDOFF_M,
        "retry_camera_alignment": "optical_axis",
        "press_speed_m_s": 0.01,
        "contact_persistence_sec": 0.80,
        "max_forward_travel_m": 0.022,
        "contact_arm_travel_m": 0.008,
        "relative_current_hard_stop_enabled": False,
        "absolute_current_hard_stop_enabled": False,
        "contact_estimator": "cartesian_normal_force_kalman",
        "tangential_force_threshold_n": 0.60,
        "tangential_force_sigma_multiplier": 4.0,
        "tangential_force_persistence_sec": 0.40,
        "post_press_continuous_rgb_monitor": True,
        "post_press_on_confirmation_frames": POST_PRESS_ON_CONFIRMATION_FRAMES,
        "initial_confirmation_frames": 3,
        "arrival_off_confirmation_frames": 3,
        "waiting_visibility_edge_margin_px": OBSERVATION_EDGE_MARGIN_PX,
        "post_press_visibility_edge_margin_px": POST_PRESS_VISIBILITY_EDGE_MARGIN_PX,
        "post_press_retreat_speed_m_s": POST_PRESS_RETREAT_SPEED_M_S,
        "post_press_retreat_joint_velocity_scale": POST_PRESS_RETREAT_JOINT_VELOCITY_SCALE,
        "prepress_return_prevalidated": True,
        "prepress_return_speed_m_s": PREPRESS_RETURN_SPEED_M_S,
    }
    actual = {
        "retry_standoff_m": ButtonFlowConfig().near_observation_distance_m,
        "retry_camera_alignment": "optical_axis",
        "press_speed_m_s": 0.01,
        "normal_force_total_floor_n": float(force.get("normal_force_total_floor_n", 0.0)),
        "contact_persistence_sec": float(force["persistence_sec"]),
        "max_forward_travel_m": float(guarded["max_forward_travel_m"]),
        "press_segment_max_joint_delta_rad": PRESS_SEGMENT_MAX_JOINT_DELTA_RAD,
        "contact_arm_travel_m": float(force["contact_arm_travel_m"]),
        "relative_current_hard_stop_enabled": bool(
            force["relative_current_hard_stop_enabled"]
        ),
        "absolute_current_hard_stop_enabled": bool(
            force["absolute_current_hard_stop_enabled"]
        ),
        "contact_estimator": str(force["contact_estimator"]),
        "tangential_force_threshold_n": float(
            force["tangential_force_threshold_n"]
        ),
        "tangential_force_sigma_multiplier": float(
            force["tangential_force_sigma_multiplier"]
        ),
        "tangential_force_persistence_sec": float(
            force["tangential_force_persistence_sec"]
        ),
        "post_press_continuous_rgb_monitor": True,
        "post_press_on_confirmation_frames": POST_PRESS_ON_CONFIRMATION_FRAMES,
        "initial_confirmation_frames": ButtonFlowConfig().initial_confirmation_frames,
        "arrival_off_confirmation_frames": ButtonFlowConfig().off_confirmation_frames,
        "waiting_visibility_edge_margin_px": OBSERVATION_EDGE_MARGIN_PX,
        "post_press_visibility_edge_margin_px": POST_PRESS_VISIBILITY_EDGE_MARGIN_PX,
        "post_press_retreat_speed_m_s": POST_PRESS_RETREAT_SPEED_M_S,
        "post_press_retreat_joint_velocity_scale": POST_PRESS_RETREAT_JOINT_VELOCITY_SCALE,
        "prepress_return_prevalidated": True,
        "prepress_return_speed_m_s": PREPRESS_RETURN_SPEED_M_S,
    }
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if isinstance(expected_value, float):
            matches = bool(np.isclose(actual_value, expected_value, atol=1e-9))
        else:
            matches = actual_value == expected_value
        if not matches:
            raise RuntimeError(
                f"production tactic mismatch for {name}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    return actual


@traced('approach.receding_total', 'approach')
def execute_receding_prepress(
    *, vision, robot, safety, config, handeye, target_label, roll_deg,
    initial_target, token, event_sink=None, replacement_ready=None,
    stop_at_reference=False,
):
    """Move toward a safe guard while RGB-D refreshes the final prepress IK.

    The active command ends at a speed-dependent guard if perception or IK fails.
    A stable, timestamp-synchronised moving-camera estimate schedules a C2
    splice before that endpoint, so the successful path has no 15 cm stop.
    Fixed-reference approaches end directly at the saved photo pose while
    retaining the moving RGB-D/IK correction.
    """
    function_started = time.monotonic()
    current_state = robot.state(max_age=2.0)
    start_panel_distance_m = float(
        (
            current_state.base_to_gripper[:3, 3]
            - initial_target.base_to_button[:3, 3]
        ) @ initial_target.base_to_button[:3, 2]
    )
    receding_plan = plan_receding_approach(
        start_panel_distance_m,
        predicted_ready_sec=0.8,
        config=RollingSpeedConfig(
            max_speed_m_s=0.18,
            desired_schedule_margin_sec=0.40,
            coarse_target_uncertainty_m=0.020,
            control_latency_sec=0.12,
        ),
    )
    guard_ik = FixedRollFastIkPlanner(
        roll_deg=roll_deg,
        stand_off_m=receding_plan.dynamic_guard_distance_m,
        max_joint_delta_rad=1.0,
        max_perpendicular_deviation_deg=35.0,
        downstream_base_x_distances_m=(0.01, -0.012),
        downstream_max_segment_joint_deltas_rad=(None, PRESS_SEGMENT_MAX_JOINT_DELTA_RAD),
    ).solve(initial_target.base_to_button, current_state.joints)
    guard_joints = guard_ik.joints.copy()
    actual_guard_m = float(
        (
            forward_kinematics(guard_joints)[:3, 3]
            - initial_target.base_to_button[:3, 3]
        ) @ initial_target.base_to_button[:3, 2]
    )
    if abs(actual_guard_m - receding_plan.dynamic_guard_distance_m) > 0.002:
        raise RuntimeError(
            "dynamic guard IK does not satisfy the planned panel distance"
        )
    active, active_audit = generate_timed_joint_trajectory(
        current_state.joints, guard_joints,
        cartesian_speed_m_s=receding_plan.speed_cap_m_s,
        cartesian_speed_cap_m_s=0.20,
        joint_velocity_scale=0.75,
        point_period_sec=0.10, duration_margin=1.02,
    )
    # Keep the fail-safe endpoint available long enough for four real camera
    # frames and IK/upload latency.  A successful replacement splices before
    # this zero-speed guard hold; a failed refresh remains 5 cm from contact.
    active_motion_duration_sec = float(active[-1]["time_from_start_sec"])
    guard_positions = list(active[-1]["positions"])
    for index in range(1, 21):
        active.append({
            "time_from_start_sec": active_motion_duration_sec + index * 0.10,
            "positions": guard_positions,
            "velocities": [0.0] * 6,
            "accelerations": [0.0] * 6,
        })
    active_total_duration_sec = float(active[-1]["time_from_start_sec"])
    splice = PreparedSpliceUdpClient(
        config["timed_trajectory"]["host"], 5011, timeout_sec=40.0
    )
    suffix = str(time.time_ns())
    active_id = f"button-light-receding-guard-{suffix}"
    prepared_active = splice.prepare(active, trajectory_id=active_id, token=token)
    history = RobotTransformHistory(max_gap_sec=0.12)
    stop_history = threading.Event()

    def record_history():
        while not stop_history.is_set():
            try:
                state = robot.state(max_age=0.25)
                history.add(state.timestamp, state.base_to_gripper)
            except Exception:
                pass
            stop_history.wait(0.01)

    history_thread = threading.Thread(target=record_history, daemon=True)
    history_thread.start()
    history.add(current_state.timestamp, current_state.base_to_gripper)
    started_local = None
    execution_timing_started = None
    terminal_reached = False
    failed = False
    samples_audit = []
    try:
        safety.before_segment("receding_approach")
        started_response = splice.start_active(active_id=active_id, token=token)
        started_local = time.monotonic()
        execution_timing_started = (started_local, time.time())
        window = make_button_window(
            config, size=2, max_position_rms_m=0.0035,
            max_position_jump_m=0.0070, max_sample_age_sec=0.35,
            max_span_sec=1.05,
        )
        if isinstance(vision,LayoutTaskVision):
            window=SourceAwareButtonWindow.from_window(window)
        wall_minus_monotonic = time.time() - time.monotonic()
        last_sequence = 0
        estimate = None
        deadline = time.monotonic() + min(
            3.0, max(1.5, active_total_duration_sec - 0.45)
        )
        while time.monotonic() < deadline:
            response = vision.request(
                "latest", target_label=str(target_label), window_size=6,
                after_sequence=last_sequence, include_realtime=False,
                max_batch_age_sec=0.35,
            )
            batch = target_batch(response,after_sequence=last_sequence)
            for sample in sorted(batch, key=lambda value: int(value["sequence"])):
                sequence = int(sample["sequence"])
                if sequence <= last_sequence:
                    continue
                last_sequence = sequence
                capture_unix = float(sample["capture_time_unix_sec"])
                try:
                    base_to_tcp = history.transform_at(
                        capture_unix - wall_minus_monotonic, wait_sec=0.10
                    )
                except RuntimeError as exc:
                    samples_audit.append({
                        "sequence": sequence, "accepted": False,
                        "reason": str(exc),
                    })
                    continue
                base_to_camera = base_to_tcp @ handeye
                raw_target = (
                    base_to_camera
                    @ np.asarray(sample["camera_to_button"], dtype=float)
                )
                target, correction = constrain_button_ray_to_reference_plane(
                    base_to_camera, raw_target, initial_target.base_to_button
                )
                try:
                    accepted = window.add(timed_target(sample,target,target_label=target_label))
                except ValueError as exc:
                    if sample.get('position_source')!='panel_layout_inferred':raise
                    samples_audit.append(dict(sequence=sequence,accepted=False,reason=str(exc),
                                              position_source='panel_layout_inferred'))
                    continue
                samples_audit.append({
                    "sequence": sequence, "accepted": bool(accepted),
                    "capture_age_sec": time.time() - capture_unix,
                    "plane_correction_m": correction,
                    "position_source": sample.get('position_source','yolo'),
                    "target_detected": sample.get('position_source')!='panel_layout_inferred',
                    "confidence": sample.get('confidence'),
                })
                if accepted:
                    estimate = window.estimate()
                    if estimate.stable:
                        break
            if estimate is not None and estimate.stable:
                break
            time.sleep(0.01)
        if estimate is None or not estimate.stable:
            terminal, transitions = wait_terminal(
                splice, token, active_total_duration_sec + 5.0
            )
            terminal_reached = True
            raise RuntimeError(
                "moving RGB-D target was not stable; stopped at the dynamic guard "
                "and inhibited pressing"
            )

        moving_target_ready_sec = time.monotonic() - started_local
        elapsed = moving_target_ready_sec
        # Keep the coarse trajectory at its free-space speed cap for nearly the entire
        # reach.  The earlier implementation selected elapsed+450 ms and
        # replaced almost the whole reach with the slow near-panel quintic
        # (10.83 s in the 2026-09-11 button-10 trial).  The target is already
        # computed, so schedule a future handoff just before the dynamic guard.
        # Begin the correction while useful forward velocity remains. The old
        # near-end handoff made the arm almost reach the stale guard point and
        # then visibly bend back toward the refreshed target.
        desired_handoff_start = max(0.0, active_motion_duration_sec - 0.90)
        ik_seed_handoff = select_future_handoff(
            active, active_elapsed_sec=desired_handoff_start, lead_sec=0.20
        )
        target_frame = estimate.base_to_button.copy()
        target_frame[0, 3] -= 0.01
        prepress = FixedRollFastIkPlanner(
            roll_deg=roll_deg, stand_off_m=0.0, max_joint_delta_rad=1.0,
            max_perpendicular_deviation_deg=35.0,
            # ``target_frame`` is already 10 mm before the button. Preview
            # the complete path to the +12 mm hard press endpoint (22 mm
            # from prepress), and reject wrist configurations that would
            # exceed the shared short-press total-joint bound.
            downstream_base_x_distances_m=(-0.022,),
            downstream_max_segment_joint_deltas_rad=(PRESS_SEGMENT_MAX_JOINT_DELTA_RAD,),
        ).solve(target_frame, ik_seed_handoff.positions)
        def approach_endpoint(solved):
            if not stop_at_reference:
                return solved.joints, None
            panel = estimate.base_to_button.copy()
            panel[:3, 2] = [-1.0, 0.0, 0.0]
            joints, alignment = solve_camera_centered_observation_joints(
                solved.joints, panel, handeye, PRODUCTION_RETRY_STANDOFF_M)
            return joints, {"mode": "direct_camera_aimed_standoff", **alignment}

        endpoint_joints, reference_alignment = approach_endpoint(prepress)
        # IK and upload consume part of the lead.  A single fixed handoff can
        # also be incompatible with the C2/end-zone limits when its inherited
        # acceleration is high. Search forward through the already validated
        # active trajectory; its zero-speed guard hold is the final candidate.
        elapsed_after_ik = time.monotonic() - started_local
        earliest_handoff = max(
            elapsed_after_ik + 0.40,
            active_motion_duration_sec - 0.75,
        )
        handoff, replacement, replacement_audit, candidate_failures = (
            generate_first_feasible_splice(
                active, endpoint_joints,
                earliest_handoff_time_sec=earliest_handoff,
                cartesian_speed_m_s=0.10, joint_velocity_scale=0.75,
                point_period_sec=0.10, end_zone_distance_m=0.04,
                end_zone_speed_m_s=0.06,
            )
        )
        # The successful handoff can be later than the original IK seed. Solve
        # once more from the state that will actually be reached, then repeat
        # the feasibility search so the endpoint and audit agree exactly.
        if handoff.active_index != ik_seed_handoff.active_index:
            prepress = FixedRollFastIkPlanner(
                roll_deg=roll_deg, stand_off_m=0.0, max_joint_delta_rad=1.0,
                max_perpendicular_deviation_deg=35.0,
                downstream_base_x_distances_m=(-0.022,),
                downstream_max_segment_joint_deltas_rad=(PRESS_SEGMENT_MAX_JOINT_DELTA_RAD,),
            ).solve(target_frame, handoff.positions)
            endpoint_joints, reference_alignment = approach_endpoint(prepress)
            elapsed_after_resolve = time.monotonic() - started_local
            handoff, replacement, replacement_audit, second_failures = (
                generate_first_feasible_splice(
                    active, endpoint_joints,
                    earliest_handoff_time_sec=max(
                        handoff.active_time_sec, elapsed_after_resolve + 0.40
                    ),
                    cartesian_speed_m_s=0.10, joint_velocity_scale=0.75,
                    point_period_sec=0.10, end_zone_distance_m=0.04,
                    end_zone_speed_m_s=0.06,
                )
            )
            candidate_failures.extend(second_failures)
        endpoint_name = "reference" if stop_at_reference else "prepress"
        replacement_id = f"button-light-receding-{endpoint_name}-{suffix}"
        prepared_replacement = splice.prepare(
            replacement, trajectory_id=replacement_id, token=token
        )
        scheduled = splice.schedule_replacement(
            active_id=active_id, replacement_id=replacement_id,
            handoff_active_time_sec=handoff.active_time_sec, token=token,
        )
        if replacement_ready is not None:
            if stop_at_reference:
                replacement_ready(estimate, prepress,
                    reference_joints=endpoint_joints.copy(),
                    reference_alignment=reference_alignment)
            else:
                replacement_ready(estimate, prepress)
        terminal, transitions = wait_terminal(
            splice, token, active_total_duration_sec + replacement_audit.duration_sec + 8.0
        )
        terminal_reached = True
        if terminal.get("execution_status") != "replacement_complete":
            raise RuntimeError(f"receding approach did not complete replacement: {terminal}")
        observation_pose_audit = None
        if stop_at_reference:
            try:
                reached, observation_pose_audit = wait_for_stable_observation_pose(
                    endpoint_joints,
                    lambda: robot.latest_state(max_age=OBSERVATION_FEEDBACK_MAX_AGE_SEC),
                    safety._assert_clear)
            except BaseException as exc:
                if event_sink is not None:
                    event_sink("stationary_reference_pose_confirmation", {
                        "stage": "receding_reference", "status": "failed",
                        "audit": getattr(exc, "audit", getattr(exc, "observation_pose_audit", {})),
                        "error": str(exc)})
                raise
            if event_sink is not None:
                event_sink("stationary_reference_pose_confirmation", {
                    "stage": "receding_reference", "status": "stable", "audit": observation_pose_audit})
        else:
            reached = robot.state(max_age=2.0)
        error = float(np.max(np.abs(reached.joints - endpoint_joints)))
        if (reached.raw.get("hardware_ok") != 1 or not np.isfinite(error)
                or error > (OBSERVATION_JOINT_TOLERANCE_RAD if stop_at_reference else .03)):
            raise RuntimeError(f"receding {endpoint_name} encoder error is {error:.4f} rad")
        audit = {
            "mode": "moving_rgbd_receding_ik_splice",
            "endpoint": endpoint_name,
            "endpoint_joints_rad": endpoint_joints.tolist(),
            "reference_alignment": reference_alignment,
            "prepress_joints_rad": prepress.joints.tolist(),
            "fallback_guard_distance_m": float(actual_guard_m),
            "rolling_speed_plan": receding_plan.__dict__,
            "guard_ik_solve_ms": guard_ik.solve_sec * 1000.0,
            "active_duration_sec": active_audit.duration_sec,
            "active_guard_hold_sec": active_total_duration_sec - active_motion_duration_sec,
            "prepared_active": prepared_active,
            "started": started_response,
            "moving_samples": samples_audit,
            "localization_provenance": localization_provenance(estimate,samples_audit),
            "window_sequence_range": list(estimate.sequence_range),
            "moving_target_ready_sec": moving_target_ready_sec,
            "ik_solve_ms": prepress.solve_sec * 1000.0,
            "handoff_active_time_sec": handoff.active_time_sec,
            "splice_candidate_failures": candidate_failures,
            "replacement_duration_sec": replacement_audit.duration_sec,
            "replacement_geometry": {
                "cartesian_chord_m": replacement_audit.cartesian_chord_m,
                "cartesian_max_overshoot_m": (
                    replacement_audit.cartesian_max_overshoot_m
                ),
                "cartesian_max_backtrack_step_m": (
                    replacement_audit.cartesian_max_backtrack_step_m
                ),
                "cartesian_max_cross_track_m": (
                    replacement_audit.cartesian_max_cross_track_m
                ),
                "joint_max_overshoot_rad": replacement_audit.joint_max_overshoot_rad,
            },
            "prepared_replacement": prepared_replacement,
            "scheduled": scheduled,
            "terminal": terminal,
            "transitions": transitions,
            "encoder_error_rad": error,
            "observation_pose_confirmation": observation_pose_audit,
            "total_receding_wall_sec": time.monotonic() - function_started,
        }
        if event_sink is not None:
            event_sink("receding_approach_completed", audit)
        return estimate, prepress, audit
    except BaseException:
        failed = True
        # Never leave a normally running asynchronous fallback cached behind
        # an exception.  Wait until its bounded dynamic endpoint completes before
        # the caller attempts a return.  A visual stop remains latched and is
        # deliberately left for guarded recovery.
        if started_local is not None and not terminal_reached:
            # Never use wait_terminal here: it deliberately extends its
            # deadline forever while safety_paused. During unwinding that
            # behavior masks the original error and strands the robot. Wait a
            # bounded interval for the safe guard, or abort immediately once
            # the independent safety bridge has paused/cancelled the goal.
            cleanup_deadline = time.monotonic() + active_total_duration_sec + 2.0
            while time.monotonic() < cleanup_deadline:
                try:
                    status = splice.execution_status(token=token)
                    state = status.get("execution_status")
                    if state in {
                        "idle", "replacement_complete", "fallback_complete",
                        "failed", "aborted",
                    }:
                        break
                    if state == "safety_paused":
                        splice.abort_active(
                            token=token, active_id=status.get("active_id")
                        )
                        break
                except Exception:
                    break
                time.sleep(0.05)
        raise
    finally:
        if execution_timing_started is not None:
            record_interval('motion.receding_execution_window', 'motion_window',
                execution_timing_started, status='error' if failed else 'ok',
                scope='start acknowledgement through terminal observation/cleanup; includes pauses and polling')
        stop_history.set()
        history_thread.join(timeout=1.0)
        if started_local is not None and not failed:
            safety.after_segment("receding_approach")


def pointing_command(args) -> list[str]:
    # This subprocess only raises the camera to a distant observation pose.  The
    # parent process owns the subsequent near-panel approach and RGB
    # fallback.  Forwarding that fallback here both widens the safety policy too
    # early and can exceed the pointing runner's 10 cm validation limit.
    command = [
        sys.executable,
        str(ROOT / "run_online_replan_no_contact.py"),
        "--hold-report", str(args.hold_report),
        "--config", str(args.config),
        "--target-label", str(args.target_label),
        "--target-distance-m", "0.01",
        "--press-travel-m", "0",
        "--receding-schedule",
        "--pointing-only",
        "--hold-at-pointing",
        "--execute",
        "--acknowledge-clear-workspace",
        "--visual-safety",
        "--inverse-return",
        "--return-without-visual-safety",
    ]
    return command


def wait_for_visual_clear(vision, *, timeout_sec: float = 30.0,
                          required_clear_frames: int = 3) -> dict:
    """Wait for a stable clear view before restarting a stopped pointing task."""
    deadline = time.monotonic() + float(timeout_sec)
    streak = 0
    last = None
    while time.monotonic() < deadline:
        last = vision.request("safety")["visual_safety"]
        if (
            last.get("ready")
            and not last.get("blocked")
            and last.get("monitoring_available", True)
        ):
            streak += 1
            if streak >= int(required_clear_frames):
                return last
        else:
            streak = 0
        time.sleep(0.05)
    raise TimeoutError(f"visual safety did not become stably clear: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_label")
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument(
        "--hold-report", type=Path,
        default=ROOT / "button_fast_chain_runs/20260826_150725/chain_report.yaml",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-clear-workspace", action="store_true")
    parser.add_argument("--pause-depth-near-button", action="store_true")
    parser.add_argument(
        "--max-press-attempts", type=int, default=3,
        help="maximum physical attempts; use 1 for guarded-current calibration",
    )
    parser.add_argument(
        "--depth-degraded-entry-distance-m", type=float, default=0.15,
        help="when the explicit near-panel depth policy is enabled, arm its coverage-only exception at this panel distance (0.01..0.15 m)",
    )
    parser.add_argument("--check", action="store_true", help="validate setup without motion")
    parser.add_argument("--task-id", default=os.environ.get("YOUXIAOMEI_TASK_ID"))
    parser.add_argument("--mission-id", default=os.environ.get("YOUXIAOMEI_MISSION_ID"))
    parser.add_argument("--segment-id", default=os.environ.get("YOUXIAOMEI_SEGMENT_ID"))
    parser.add_argument(
        "--observation-only", action="store_true",
        help="execute only the validated observation leg and its return; never press",
    )
    parser.add_argument(
        "--validate-plan-only", action="store_true",
        help="build and dry-validate the live segmented and guarded plan, then exit without motion",
    )
    args = parser.parse_args()
    task_id = str(args.task_id or new_task_id())
    mission_id = str(args.mission_id or task_id)
    def optional_floor(name: str) -> int | None:
        raw = os.environ.get(name, "").strip()
        try:
            floor = int(raw)
        except ValueError:
            return None
        return floor if 1 <= floor <= 16 else None

    current_floor = optional_floor("YOUXIAOMEI_CURRENT_FLOOR")
    destination_floor = optional_floor("YOUXIAOMEI_DESTINATION_FLOOR")
    status_outbox = Path(os.environ.get(
        "YOUXIAOMEI_MISSION_STATUS_OUTBOX", str(DEFAULT_MISSION_STATUS_OUTBOX)
    ))
    target_label = str(args.target_label)
    if not (
        target_label.casefold() in {"up", "down"}
        or (target_label.isdigit() and 1 <= int(target_label) <= 16)
    ):
        raise ValueError("target label must be in 1..16, up, or down")
    if args.execute == args.check:
        raise ValueError("select exactly one of --check or --execute")
    if args.observation_only and not args.execute:
        raise ValueError("--observation-only requires --execute")
    if args.execute and not args.acknowledge_clear_workspace:
        raise RuntimeError("--execute requires --acknowledge-clear-workspace")
    if not 0.01 <= args.depth_degraded_entry_distance_m <= 0.15:
        raise ValueError("--depth-degraded-entry-distance-m must be in 0.01..0.15")
    if not 1 <= args.max_press_attempts <= 3:
        raise ValueError("--max-press-attempts must be in 1..3")
    if args.execute:
        signal.signal(signal.SIGTERM, _raise_termination)

    config = yaml.safe_load(args.config.read_text())
    production_tactics = validate_production_tactics(config)
    hold = yaml.safe_load(args.hold_report.read_text())
    origin = np.asarray(hold["stage_1_states"][0]["joints_rad"], dtype=float)
    roll_deg = float(hold["stage_1"]["selected_approach_roll_deg"])
    handeye, full_camera_matrix, _ = load_handeye(
        resolve_path(config["calibration"]["handeye_yaml"], args.config)
    )
    vision_cfg = config["vision_service"]
    trajectory_cfg = config["timed_trajectory"]
    safety_cfg = config["safety_stop"]
    vision = VisionRpcClient(
        vision_cfg["host"], int(vision_cfg["port"]), vision_cfg["token"], timeout=2.0
    )
    vision=task_vision(vision,config,args.target_label)
    health = vision.request("health")
    require_layout_server_support(health,config,args.target_label)
    stop_health = SafetyStopUdpClient(
        safety_cfg["host"], int(safety_cfg["port"]), timeout_sec=2.0
    ).health(token=safety_cfg["token"])
    if args.check:
        # Check mode intentionally avoids preflight because preflight resets the
        # independent latch. It also never starts a robot adapter.
        report = {
            "status": "ready_for_guarded_test",
            "target_label": str(args.target_label),
            "vision_status": health.get("status"),
            "camera_sequence": health.get("camera_sequence"),
            "stop_bridge_status": stop_health.get("status"),
            "stop_bridge_latched": stop_health.get("latched"),
            "active_task_id": stop_health.get("active_task_id"),
            "pointing_command": pointing_command(args),
            "physical_execution_enabled": False,
            "production_tactics": production_tactics,
            "observation_pose_policy": {
                "joint_tolerance_rad": OBSERVATION_JOINT_TOLERANCE_RAD,
                "stable_sec": OBSERVATION_STABLE_SEC,
                "confirmation_timeout_sec": OBSERVATION_CONFIRM_TIMEOUT_SEC,
                "stability_delta_rad": OBSERVATION_STABILITY_DELTA_RAD,
                "capture_drift_tolerance_rad": OBSERVATION_CAPTURE_DRIFT_RAD,
                "capture_drift_numeric_epsilon_rad": OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD,
                "feedback_max_age_sec": OBSERVATION_FEEDBACK_MAX_AGE_SEC,
            },
            "light_evidence_policy": {**LIGHT_EVIDENCE_POLICY,
                "reference_collection": ("fixed_5cm_observation_before_press"
                    if requires_stationary_press_reference(args.target_label)
                    else LIGHT_EVIDENCE_POLICY["reference_collection"]),
                "stationary_reference_enabled": requires_stationary_press_reference(args.target_label)},
            "robot_commands_sent": False,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    recognizer = None
    observer = None
    safety = None
    supervisor = None
    executor = None
    flow = None
    model_loader = None
    recognizer_future = None
    robot = None
    output = ROOT / "button_light_closed_loop_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    configure_timing(output / 'pipeline_timing.jsonl', task_id)
    report = {
        "schema_version": 2,
        "status": "initializing",
        "lifecycle_status": "initializing",
        "task_id": task_id,
        "mission_id": mission_id,
        "segment_id": args.segment_id,
        "current_floor": current_floor,
        "destination_floor": destination_floor,
        "target_label": str(args.target_label),
        "physical_execution_enabled": True,
        "production_tactics": production_tactics,
        "observation_pose_policy": {
            "joint_tolerance_rad": OBSERVATION_JOINT_TOLERANCE_RAD,
            "stable_sec": OBSERVATION_STABLE_SEC,
            "confirmation_timeout_sec": OBSERVATION_CONFIRM_TIMEOUT_SEC,
            "stability_delta_rad": OBSERVATION_STABILITY_DELTA_RAD,
            "capture_drift_tolerance_rad": OBSERVATION_CAPTURE_DRIFT_RAD,
            "capture_drift_numeric_epsilon_rad": OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD,
            "feedback_max_age_sec": OBSERVATION_FEEDBACK_MAX_AGE_SEC,
        },
        "light_evidence_policy": {**LIGHT_EVIDENCE_POLICY,
                "reference_collection": ("fixed_5cm_observation_before_press"
                    if requires_stationary_press_reference(args.target_label)
                    else LIGHT_EVIDENCE_POLICY["reference_collection"]),
                "stationary_reference_enabled": requires_stationary_press_reference(args.target_label)},
        "depth_degraded_policy": {
            "enabled": bool(args.pause_depth_near_button),
            "entry_distance_m": float(args.depth_degraded_entry_distance_m),
            "lease_max_duration_sec": NEAR_BUTTON_DEPTH_LEASE_SEC,
        },
        "returned_to_origin": False,
    }
    failure_images = FailureImageArchive(ROOT / "artifacts/runs/button_failure_records" / output.name)
    report_path = output / "closed_loop_report.yaml"
    journal_path = output / "journal.jsonl"
    checkpoint_lock = threading.RLock()

    def checkpoint(kind: str, value: dict) -> None:
        """Persist evidence before the next physical action can begin."""
        with checkpoint_lock:
            row = append_journal_checkpoint(report, kind, value)
            update_report_checkpoint(report, kind, value, journal_phase(kind, value))
            if observer is not None:
                report["light_observations"] = list(observer.audit)
            if executor is not None:
                report["motion_executions"] = list(executor.executions)
                report["motion_validations"] = dict(executor.validations)
            # The old implementation rewrote an increasingly large YAML file
            # at every state transition. After a guarded press, five such
            # writes delayed the retreat command by several seconds. An
            # append-only fsynced journal keeps the same crash evidence in a
            # few milliseconds; the complete YAML is assembled in ``finally``.
            persist_journal_row(journal_path, row)
            status_value = None
            if kind == "flow_event" and isinstance(value, dict):
                status_value = normalize_flow_event(value)
            elif kind == "light_observation" and isinstance(value, dict):
                status_value = normalize_light_observation(value)
            else:
                phase_map = {
                    "run_created": "arm_task_started",
                    "pointing_phase_started": "preflight",
                    "pointing_phase_completed": "button_observation_ready",
                    "pointing_attempt_started": "approaching_button",
                    "pointing_attempt_finished": "approach_result",
                    "receding_target_staged": "approach_target_staged",
                    "receding_approach_completed": "approach_completed",
                    "press_attempt_started": "pressing",
                    "prepress_visibility_probe": "press_readiness_checked",
                    "origin_return_requested": "returning_origin",
                    "origin_return_started": "returning_origin",
                    "origin_return_completed": "origin_verified",
                    "origin_return_verified": "origin_verified",
                    "origin_return_failed": "origin_return_failed",
                    "guarded_press_failed": "press_failed",
                }
                phase = phase_map.get(kind)
                if kind == "visual_safety_paused":
                    phase = "safety_paused"
                elif kind == "visual_safety_resumed":
                    phase = "safety_resumed"
                elif kind == "motion_segment_started":
                    phase = "motion_segment_started"
                elif kind == "motion_segment_completed":
                    phase = "motion_segment_completed"
                elif kind == "motion_plan_validated":
                    phase = "motion_plan_validated"
                elif kind == "motion_segment_feedback_unconfirmed":
                    phase = "press_feedback_unconfirmed"
                elif kind in {"motion_segment_failed", "motion_segment_cleanup_failed"}:
                    phase = kind
                if phase is not None:
                    failed_phase = phase.endswith("failed") or phase.endswith("unconfirmed")
                    status_value = {
                        "phase": phase,
                        "outcome": (
                            "paused" if phase == "safety_paused"
                            else "active" if phase == "safety_resumed"
                            else "failed" if failed_phase else "active"
                        ),
                        "reason_code": (
                            str(value.get("reason") or kind).upper()
                            if phase == "safety_paused" or failed_phase else None
                        ),
                        "evidence": value,
                        "raw_event": kind,
                    }
            if status_value is not None:
                append_mission_status(status_outbox, {
                    "task_id": task_id,
                    "mission_id": mission_id,
                    "segment_id": args.segment_id,
                    "target": str(args.target_label),
                    "current_floor": current_floor,
                    "destination_floor": destination_floor,
                    "producer": "arm_pressing",
                    **status_value,
                }, durable=False)
            if row["sequence"] == 1:
                write_report_atomic(report_path, report)
            print(json.dumps({
                "run": output.name,
                "phase": journal_phase(kind, value),
                "value": value,
            }, ensure_ascii=False, separators=(",", ":"), default=str), flush=True)

    checkpoint("run_created", {"target_label": str(args.target_label)})
    # Create the authoritative report before pointing. This phase can move the
    # arm and therefore must never fail without a parent-run journal.
    try:
        report["light_processing_preload"] = {}
        checkpoint("light_processing_preload_started", {
            "policy": "before_pointing_and_arm_feedback",
        })
        (PairedButtonLight, InitialLightGuard, InitialObservationRecovery,
         recheck_initial_call_view) = preload_light_processing_dependencies(
             report["light_processing_preload"])
        checkpoint("light_processing_preload_completed", report["light_processing_preload"])
        checkpoint("pointing_phase_started", {
            "target_label": str(args.target_label),
            "origin_visibility_checked": False,
        })
        # The pointing subprocess also switches detector labels, but the safety
        # service must receive the requested label before any motion.
        switched=vision.request("set_target", target_label=str(args.target_label))
        visible,target_stable = wait_for_stable_visibility_after_target_switch(
            vision,target_label=args.target_label,min_confidence=.70,
            sequence_barrier=switched['barrier_sequence'])
        waiting_is_origin = (target_stable and
            complete_visible_target(visible, args.target_label, 0.70) is not None)
        checkpoint("origin_visibility_resolved", {
            "waiting_position_is_origin": waiting_is_origin,
            "visibility_resolution": visible.get("visibility_resolution", "existing_visibility_policy"),
        })
        model_loader = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="button-light"
        )
        model_load_requested = time.perf_counter()
        recognizer_future = model_loader.submit(OpenCvButtonLightProcess)
        if not waiting_is_origin:
            stop_client_for_pointing = SafetyStopUdpClient(
                safety_cfg["host"], int(safety_cfg["port"]), timeout_sec=2.0
            )
            generation = int(stop_health.get("generation", 0))
            for pointing_attempt in range(1, 4):
                checkpoint("pointing_attempt_started", {
                    "attempt": pointing_attempt,
                    "safety_generation": generation,
                })
                result = subprocess.run(
                    pointing_command(args), cwd=ROOT, check=False,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True,
                )
                pointing_result = {
                    "attempt": pointing_attempt,
                    "return_code": result.returncode,
                }
                if result.returncode != 0:
                    # The pointing runner prints its detailed failure report to
                    # stdout.  The GUI only consumes JSON lines, so inherited
                    # output used to disappear and the parent report retained
                    # only return_code=1. Keep the diagnostic tail in the
                    # authoritative parent journal for root-cause analysis.
                    output_text = result.stdout or ""
                    pointing_result["diagnostic_tail"] = output_text[-16000:]
                    pointing_result["diagnostic_truncated"] = (
                        len(output_text) > 16000
                    )
                checkpoint("pointing_attempt_finished", pointing_result)
                if result.returncode == 0:
                    break
                after = stop_client_for_pointing.health(token=safety_cfg["token"])
                new_generation = int(after.get("generation", 0))
                if new_generation <= generation or pointing_attempt >= 3:
                    raise RuntimeError("safe pointing phase failed")
                wait_for_visual_clear(vision)
                generation = new_generation
        checkpoint("pointing_phase_completed", {
            "waiting_position_is_origin": waiting_is_origin,
        })
    except BaseException as exc:
        capture_failure(report, exc, traceback.format_exc())
        report['phase'] = (
            'light_processing_preload_failed'
            if report.get('light_processing_preload', {}).get('status') != 'ready'
            else 'pointing_phase_failed'
        )
        pointing_cleanup_errors = []
        if model_loader is not None:
            try:
                if recognizer_future is not None:
                    worker = recognizer_future.result(timeout=35.0)
                    worker.close()
            except Exception as cleanup_exc:
                pointing_cleanup_errors.append({
                    "component": "pointing_light_recognizer",
                    "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                })
            finally:
                try:
                    model_loader.shutdown(wait=True)
                except Exception as cleanup_exc:
                    pointing_cleanup_errors.append({
                        "component": "pointing_model_loader",
                        "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                    })
        if pointing_cleanup_errors:
            report["cleanup_errors"] = pointing_cleanup_errors
        try:
            report['image_retention'] = failure_images.finalize(success=False, context={
                'run': output.name, 'task_id': task_id, 'target_label': str(args.target_label),
                'process_exit_code': report.get('process_exit_code'),
                'failure': report.get('failure'), 'flow_status': report.get('flow_status'),
                'failure_stage': report.get('failure_stage'),
                'observation_policy': report.get('observation_pose_policy')})
        except Exception as image_error:
            report['image_retention'] = {'policy':'failure_only','error':str(image_error)}
        report['pipeline_timing'] = finish_timing_with_analysis()
        finalize_report(report, report_path,
                        lambda event: append_mission_status(status_outbox, event))
        raise

    robot = PiperUdpRobot(config["robot"])
    try:
        robot.open()
        waiting_state = robot.state(max_age=2.0)
        origin_error = float(np.max(np.abs(waiting_state.joints - origin)))
        if waiting_state.raw.get("hardware_ok") != 1:
            raise RuntimeError("robot hardware is not OK at waiting observation")
        if waiting_is_origin and origin_error > 0.03:
            raise RuntimeError(
                f"target is visible but arm is not at recorded origin: {origin_error:.4f} rad"
            )
        # Keep the pose at which the requested button became visible as the
        # task's watch position. The initial lamp read happens here, and the
        # successful-press return trajectory is planned back to these joints.
        watch_position_joints = waiting_state.joints.copy()
        waiting_joints = watch_position_joints.copy()
        report["watch_position"] = {
            "source": "live_encoder_after_pointing",
            "joints_rad": watch_position_joints.tolist(),
            "j5_rad": float(watch_position_joints[4]),
            "waiting_position_is_origin": waiting_is_origin,
        }
        checkpoint("watch_position_saved", report["watch_position"])
        vision.request("set_target", target_label=str(args.target_label))
        estimate, initial_target_audit = collect_stationary_target(
            vision, robot, handeye, args.target_label, config
        )
        report["initial_target_audit"] = initial_target_audit
        visibility = stable_target_visibility(
            estimate, initial_target_audit, config['depth'].get('capture_size', [640, 360]))
        report['initial_target_visibility'] = visibility
        checkpoint('initial_target_visibility_checked', visibility)
        if not visibility['complete']:
            if visibility['reason'] == 'stable_target_box_incomplete':
                raise RuntimeError('button is not fully visible at the saved waiting pose')
            raise RuntimeError('stable target visibility evidence is invalid: ' + visibility['reason'])
        stop_client = SafetyStopUdpClient(
            safety_cfg["host"], int(safety_cfg["port"]), timeout_sec=2.0
        )
        supervisor = VisualSafetySupervisor(
            vision, stop_client, safety_cfg["token"], auto_resume=True,
            event_sink=checkpoint,
        )
        safety = VisualSegmentMotionSafety(
            supervisor, vision, robot, handeye,
            pause_depth_near_button=args.pause_depth_near_button,
            depth_degraded_entry_distance_m=args.depth_degraded_entry_distance_m,
            stationary_reference_handoff=requires_stationary_press_reference(args.target_label),
        )
        safety.set_target(estimate.base_to_button)
        safety_preparation = safety.prepare()
        report["safety_preparation"] = safety_preparation
        bridge = TimedTrajectoryUdpClient(
            trajectory_cfg["host"], int(trajectory_cfg["port"]), timeout_sec=30.0
        )
        trajectory_bridge_health = bridge.health(token=trajectory_cfg["token"])
        report["trajectory_bridge_health"] = trajectory_bridge_health
        guarded_cfg = config["guarded_press"]
        guarded_bridge = GuardedPressUdpClient(
            guarded_cfg["host"], int(guarded_cfg["port"]), timeout_sec=20.0
        )
        report["guarded_press_bridge_health"] = guarded_bridge.health(
            token=guarded_cfg["token"]
        )
        executor = DryValidatedSegmentExecutor(
            bridge, robot, trajectory_cfg["token"], execute_enabled=True,
            motion_safety=safety, event_sink=checkpoint,
            guarded_bridge=guarded_bridge,
            guarded_press_config=guarded_cfg,
        )
        latest_target = {"estimate": estimate}

        @traced('press.build_plan', 'planning')
        def build_plan(refreshed, audit, *, solved_prepress=None, current_override=None,
                       reference_joints=None, reference_alignment=None, reference_distance_m=None):
            latest_target["estimate"] = refreshed
            current = (robot.state(max_age=2.0).joints if current_override is None
                       else np.asarray(current_override,dtype=float).copy())
            if solved_prepress is None:
                prepress_target = refreshed.base_to_button.copy()
                prepress_target[0, 3] -= 0.01
                prepress = FixedRollFastIkPlanner(
                    roll_deg=roll_deg, stand_off_m=0.0,
                    max_joint_delta_rad=1.0,
                    max_perpendicular_deviation_deg=35.0,
                    downstream_base_x_distances_m=(-0.022,),
                    downstream_max_segment_joint_deltas_rad=(PRESS_SEGMENT_MAX_JOINT_DELTA_RAD,),
                ).solve(prepress_target, current)
            else:
                prepress = solved_prepress
            pressed_target = refreshed.base_to_button.copy()
            # The 22 mm endpoint from prepress is only a hard geometric bound;
            # sustained current feedback normally cancels the forward motion.
            pressed_target[0, 3] -= -0.012
            pressed = solve_independent_press_ik(
                prepress.joints, pressed_target[:3, 3]
            )
            frame = refreshed.base_to_button.copy()
            frame[:3, 2] = [-1.0, 0.0, 0.0]
            # The post-press standoff is the retry's current pose. Reuse its
            # exact validated joint target when the encoders confirm we are
            # still there; re-solving this pose from a refreshed panel estimate
            # can produce a different joint solution without moving the arm.
            prior_plan = executor.plan
            previous_observation, reuse_error = reusable_near_observation_joints(
                current,
                None if prior_plan is None else prior_plan.observation_joints,
            )
            plan_observation_distance_m = PRODUCTION_RETRY_STANDOFF_M
            if reference_joints is not None:
                # Reuse the exact endpoint submitted to the splice bridge.
                # Re-solving here could select a different pose and invalidate
                # the before/after image comparison or require another move.
                observation_joints = np.asarray(reference_joints, dtype=float).copy()
                observation_alignment = dict(reference_alignment or {})
                if reference_distance_m is not None:
                    plan_observation_distance_m=checked_retry_distance(
                        observation_joints,frame,reference_distance_m)
            elif previous_observation is not None:
                # collect_stationary_target() has just seen the button from the
                # current encoder pose.  Millimetre-scale plane refinement can
                # make that same hold 4.7 cm instead of exactly 5.0 cm; asking
                # a path that only moves toward the panel to go back to 5.0 cm
                # is then mathematically impossible and used to abort retries.
                # Measure the distance from the saved joint pose, so the plan's
                # requested standoff and exact retry endpoint are consistent.
                current_distance_m = float(
                    (forward_kinematics(previous_observation)[:3, 3] - frame[:3, 3])
                    @ frame[:3, 2]
                )
                if not 0.03 <= current_distance_m <= 0.07:
                    raise RuntimeError(
                        "retry holding pose is outside the 5 cm standoff band: "
                        f"{current_distance_m:.4f} m"
                    )
                plan_observation_distance_m = current_distance_m
                observation_joints = previous_observation.copy()
                observation_alignment = {
                    "mode": "retry_reused_validated_standoff",
                    "prior_observation_encoder_error_rad": reuse_error,
                    "refreshed_panel_distance_m": current_distance_m,
                    "reused_joints_rad": observation_joints.tolist(),
                }
            else:
                observation_joints, centered = solve_camera_centered_observation_joints(
                    prepress.joints, frame, handeye,
                    PRODUCTION_RETRY_STANDOFF_M,
                )
                observation_alignment = {
                    "mode": "camera_aimed_standoff",
                    "prior_observation_encoder_error_rad": reuse_error,
                    **centered,
                }
            observation_alignment.update(localization_provenance(refreshed,audit))
            plan = plan_segmented_press_attempt(
                current, waiting_joints, prepress.joints, pressed.joints, frame,
                observation_distance_m=plan_observation_distance_m,
                approach_speed_m_s=0.10, press_speed_m_s=0.01,
                retreat_speed_m_s=POST_PRESS_RETREAT_SPEED_M_S,
                retreat_joint_velocity_scale=POST_PRESS_RETREAT_JOINT_VELOCITY_SCALE,
                return_speed_m_s=0.20,
                prepress_return_speed_m_s=PREPRESS_RETURN_SPEED_M_S,
                observation_joints=observation_joints,
            )
            latest_target["audit"] = audit
            latest_target["observation_alignment"] = observation_alignment
            return plan

        refresh_count = 0
        retry_panel_reference = None

        def refreshed_plan():
            nonlocal refresh_count, retry_panel_reference
            refresh_count += 1
            # Initial localization/search references already used four frames.
            # This refresh is stationary and retains all freshness, encoder,
            # plane and inter-frame geometry checks with two new samples.
            if refresh_count == 1:
                refreshed, audit = collect_stationary_target(
                    vision, robot, handeye, args.target_label, config,
                    frame_count=2,
                )
                # Freeze the exact verified plane that initializes the first
                # moving approach. Neither moving near-depth nor retry samples
                # may replace this task-local reference.
                retry_panel_reference = RetryPanelReference.freeze(
                    refreshed, audit, target_label=args.target_label, task_id=task_id)
                report['retry_panel_reference'] = retry_panel_reference.audit()
                checkpoint('retry_panel_reference_saved', report['retry_panel_reference'])
            else:
                if executor.plan is None or retry_panel_reference is None:
                    raise RuntimeError('retry requires the current task verified panel and observation plan')
                checkpoint('retry_panel_target_refresh_started', {
                    'attempt': refresh_count, 'reference': retry_panel_reference.audit()})
                try:
                    # OFF has already been confirmed. Only now may the first
                    # 5cm photo pose be replaced by the dry-validated 8cm pose.
                    retry_observation=executor.prepare_retry_observation(handeye)
                    checkpoint('retry_preparation_observation_reached',retry_observation)
                    refreshed, audit = collect_retry_panel_target(
                        vision, robot, handeye, retry_panel_reference, config,
                        capture_camera_matrix(full_camera_matrix, config), retry_observation['observation_joints'],
                        target_label=args.target_label, task_id=task_id, check_hold=safety._assert_clear)
                except Exception as exc:
                    report['retry_target_localization'] = {
                        'status': 'failed', 'attempt': refresh_count,
                        'error': f'{type(exc).__name__}: {exc}',
                        'sampling': getattr(exc, 'stationary_target_failure', None)}
                    checkpoint('retry_panel_target_refresh_failed', report['retry_target_localization'])
                    raise
                report['retry_target_localization'] = {
                    'status': 'ready', 'attempt': refresh_count, 'audit': audit}
                checkpoint('retry_panel_target_refresh_ready', report['retry_target_localization'])
            # The first long approach generates and remotely validates its own
            # active and replacement trajectories. Building and validating the
            # unused five-segment stationary plan here cost 1.7--2.0 seconds.
            # Retried short approaches still require the complete plan below.
            if (
                refresh_count == 1
                and not args.validate_plan_only
                and not args.observation_only
            ):
                latest_target["estimate"] = refreshed
                latest_target["audit"] = audit
                checkpoint("receding_target_staged", {
                    "attempt": 1,
                    "validation": "deferred_to_active_and_replacement_trajectories",
                })
                return None
            if refresh_count>1:
                return build_plan(refreshed,audit,
                    reference_joints=retry_observation['observation_joints'],
                    reference_alignment=retry_observation,
                    reference_distance_m=RETRY_PREPARATION_DISTANCE_M)
            return build_plan(refreshed, audit)

        @traced('press.attempt', 'attempt')
        def press_attempt(attempt):
            failure_images.begin_attempt(attempt)
            mode = press_attempt_mode(attempt)
            checkpoint("press_attempt_started", {
                "attempt": int(attempt),
                "mode": mode,
            })
            executor.begin_press_attempt()
            if mode == "current_observation_to_prepress":
                # A retry already starts at the freshly observed ~5 cm pose.
                # Receding control is exclusively for the first long approach;
                # starting another asynchronous far-approach transaction here
                # gives it the wrong initial-state contract and can strand it
                # behind a safety pause.
                checkpoint("retry_direct_approach_started", {
                    "attempt": int(attempt),
                    "mode": "current_observation_to_prepress",
                })
                if requires_stationary_press_reference(args.target_label):
                    result = executor.press_with_stationary_reference(
                        observer.capture_stationary_off_reference, already_at_observation=True)
                else:
                    result = executor.press_from_observation()
                checkpoint("retry_direct_approach_completed", {
                    "attempt": int(attempt),
                })
                return result
            initial_estimate = latest_target["estimate"]
            direct_reference = requires_stationary_press_reference(args.target_label)
            with BackgroundPressPreparation() as preparation:
                def prepare_after_schedule(refreshed, solved, *, reference_joints=None,
                                           reference_alignment=None):
                    preparation.start(lambda: build_plan(
                        refreshed, {'mode':'moving_rgbd_receding_ik','attempt':int(attempt)},
                        solved_prepress=solved,
                        current_override=(solved.joints if reference_joints is None else reference_joints),
                        reference_joints=reference_joints,
                        reference_alignment=reference_alignment))
                moving_estimate, moving_prepress, receding_audit = execute_receding_prepress(
                    vision=vision, robot=robot, safety=safety, config=config,
                    handeye=handeye, target_label=args.target_label, roll_deg=roll_deg,
                    initial_target=initial_estimate,
                    token=trajectory_cfg['token'], event_sink=checkpoint,
                    replacement_ready=prepare_after_schedule,
                    stop_at_reference=direct_reference,
                )
                latest_target['receding_approach']=receding_audit
                updated,preparation_audit=preparation.finish()
            checkpoint('post_approach_plan_prepared',preparation_audit)
            executor.load_plan(updated, already_at_prepress=not direct_reference,
                               already_at_observation=direct_reference)
            try:
                if requires_stationary_press_reference(args.target_label):
                    return executor.press_with_stationary_reference(
                        observer.capture_stationary_off_reference, already_at_observation=True,
                        keep_approach_lease=True)
                return executor.press_from_prepress()
            except Exception as press_exc:
                failure_kind = ("press_reference_preparation_failed"
                    if requires_stationary_press_reference(args.target_label) and not observer.paired_light.frozen
                    else "guarded_press_failed")
                checkpoint(failure_kind, {
                    "attempt": int(attempt),
                    "type": type(press_exc).__name__,
                    "message": str(press_exc),
                    "bridge_failure": getattr(press_exc, "response", None),
                    "automatic_retreat_requested": True,
                    "automatic_retreat_started": False,
                })
                if cycle_parent_recovery_required(press_exc, os.environ.get('YOUXIAOMEI_CYCLE_PARENT_RECOVERY') == '1'):
                    # Do not launch a second old-plan retreat from an endpoint
                    # that failed encoder confirmation. The cycle owner waits
                    # for this process to exit and plans the proven dedicated
                    # return from fresh actual joints, without clearing a stop.
                    checkpoint('guarded_press_failure_recovery_deferred', {
                        'reason':'cycle_parent_controls_fresh_state_return',
                        'automatic_retreat_started':False})
                    raise
                # The guarded bridge always retracts to prepress before it
                # reports an unconfirmed contact. If no independent safety
                # latch is active, complete the ordinary retreat chain so a
                # missed button can never strand the arm near the panel.
                stop_state = stop_client.health(token=safety_cfg["token"])
                if not stop_state.get("latched"):
                    checkpoint("guarded_press_failure_recovery_started", {
                        "attempt": int(attempt), "automatic_retreat_started": True})
                    executor.disable_visual_safety_for_return()
                    executor.retreat_to_observation(PRODUCTION_RETRY_STANDOFF_M)
                    executor.return_to_waiting()
                    recovery = move_origin()
                    report["returned_to_origin"] = bool(recovery["verified"])
                    checkpoint("guarded_press_failure_recovered", {
                        "attempt": int(attempt),
                        "origin_verification": recovery,
                    })
                else:
                    checkpoint("guarded_press_failure_recovery_skipped", {
                        "attempt": int(attempt), "automatic_retreat_started": False,
                        "reason": "independent_stop_latched"})
                raise

        def move_origin():
            current = robot.state(max_age=2.0)
            initial = origin_verification(current, origin)
            if initial["hardware_ok"] != 1:
                checkpoint("origin_return_failed", {
                    "phase": "initial_hardware_verification", **initial,
                })
                raise RuntimeError("robot hardware is not OK before origin return")
            if initial["verified"]:
                checkpoint("origin_return_verified", {
                    "motion_required": False, **initial,
                })
                return {"motion_required": False, **initial}
            checkpoint("origin_return_requested", initial)
            points, _ = generate_timed_joint_trajectory(
                current.joints, origin, cartesian_speed_m_s=0.15,
                joint_velocity_scale=0.75, point_period_sec=0.10,
                duration_margin=1.02,
            )
            suffix = time.time_ns()
            validation = bridge.execute(
                points, token=trajectory_cfg["token"],
                task_id=f"button-light-origin-validate-{suffix}", dry_run=True,
            )
            checkpoint("origin_return_validated", {"validation": validation})
            # Exception entry may have restored outbound flags before its
            # near handshake failed. Always prepare the validated return now.
            executor.prepare_origin_return()
            checkpoint("origin_return_started", {
                "visual_safety_disabled": executor.visual_safety_disabled_for_return,
            })
            if executor.visual_safety_disabled_for_return:
                execution = bridge.execute(
                    points, token=trajectory_cfg["token"],
                    task_id=f"button-light-origin-execute-{suffix}",
                )
            else:
                safety.before_segment("observation_to_waiting")
                try:
                    execution = bridge.execute(
                        points, token=trajectory_cfg["token"],
                        task_id=f"button-light-origin-execute-{suffix}",
                    )
                finally:
                    safety.after_segment("observation_to_waiting")
            final = robot.state(max_age=2.0)
            verification = origin_verification(final, origin)
            if not verification["verified"]:
                checkpoint("origin_return_failed", {
                    "phase": "terminal_encoder_verification",
                    "execution": execution,
                    **verification,
                })
                raise RuntimeError(
                    "origin return failed: "
                    f"{verification['encoder_error_rad']:.4f} rad, "
                    f"hardware_ok={verification['hardware_ok']}"
                )
            result = {
                "motion_required": True,
                "execution": execution,
                **verification,
            }
            checkpoint("origin_return_completed", result)
            return result

        if args.validate_plan_only:
            plan = refreshed_plan()
            executor.load_plan(plan)
            # Targets outside the origin view require the pointing subprocess,
            # which physically raises the camera before this dry validation.
            # Finish that bounded motion transaction instead of claiming that
            # no command was sent while leaving the arm at the pointing pose.
            if not waiting_is_origin:
                executor.disable_visual_safety_for_return()
                move_origin()
            report.update({
                "status": (
                    "guarded_plan_validated_no_motion"
                    if waiting_is_origin
                    else "guarded_plan_validated_returned_origin"
                ),
                "physical_execution_enabled": False,
                "robot_commands_sent": not waiting_is_origin,
                "motion_validations": executor.validations,
                "guarded_press_bridge_health": report["guarded_press_bridge_health"],
                "returned_to_origin": True,
                "lifecycle_status": "completed",
                "process_exit_code": 0,
            })
            print(report_yaml(report))
            return 0

        if args.observation_only:
            plan = refreshed_plan()
            executor.load_plan(plan)
            approach = executor.approach_to_observation()
            observation_safety = vision.request("safety")["visual_safety"]
            returned = executor.return_to_waiting()
            final = robot.state(max_age=2.0)
            origin_error = float(np.max(np.abs(final.joints - origin)))
            if final.raw.get("hardware_ok") != 1 or origin_error > 0.03:
                raise RuntimeError(
                    f"observation-only origin return failed: {origin_error:.4f} rad"
                )
            report.update({
                "status": "observation_only_returned_origin",
                "lifecycle_status": "completed",
                "process_exit_code": 0,
                "robot_commands_sent": True,
                "observation_alignment": latest_target.get("observation_alignment"),
                "observation_safety": observation_safety,
                "observation_approach": approach,
                "observation_return": returned,
                "origin_error_rad": origin_error,
                "returned_to_origin": True,
            })
            print(report_yaml(report))
            return 0

        recognizer = recognizer_future.result(timeout=35.0)
        model_ready_after_start_sec = time.perf_counter() - model_load_requested
        initial_guard = InitialLightGuard(str(args.target_label), output / "initial_light_evidence")
        observer = FreshLightObserver(
            vision, recognizer, str(args.target_label), event_sink=checkpoint,
            paired_light=PairedButtonLight(
                output / "paired_light_evidence",
                require_stationary_reference=requires_stationary_press_reference(args.target_label),
                stationary_check=executor.stationary_observation_verified),
            initial_light_guard=initial_guard,
            failure_image_archive=failure_images,
        )

        from piper_handeye.stationary_light_freshness import StationaryRetryLightGuard
        observer.stationary_retry_light_guard = StationaryRetryLightGuard(
            lambda: robot.latest_state(max_age=.25), safety._assert_clear, str(args.target_label))
        observer.stationary_budget_remaining = safety.stationary_observation_remaining
        observer.post_press_health_check = executor.post_press_motion_health

        original_watch_joints = waiting_joints.copy()
        initial_geometry_pending = False
        initial_geometry_barrier = None
        initial_hold_last_health = float('-inf')

        def check_initial_hold():
            nonlocal initial_hold_last_health
            safety._assert_clear()
            current = robot.state(max_age=.25)
            if (current.raw.get('hardware_ok') != 1
                    or np.max(np.abs(current.joints-waiting_joints)) > .005):
                raise RuntimeError('initial observation hold lost its healthy stationary pose')
            now = time.monotonic()
            if now-initial_hold_last_health >= 1.0:
                health = stop_client.health(token=safety_cfg['token'])
                if health.get('status') != 'ok' or health.get('latched') is not False:
                    raise RuntimeError('independent safety stop is unavailable or latched during initial wait')
                initial_hold_last_health = now

        def commit_initial_view(execution):
            nonlocal waiting_joints, waiting_is_origin, watch_position_joints
            nonlocal initial_geometry_pending, initial_geometry_barrier
            waiting_joints = np.asarray(execution['joints_rad'], dtype=float)
            watch_position_joints = waiting_joints.copy()
            waiting_is_origin = False
            runtime.waiting_position_is_origin = False
            report['watch_position'] = {
                "source": "initial_light_recheck_encoder", "joints_rad": waiting_joints.tolist(),
                "j5_rad": float(waiting_joints[4]), "waiting_position_is_origin": False}
            initial_geometry_pending = True
            initial_geometry_barrier = time.time()
            observer.initial_not_before_unix = initial_geometry_barrier
            if abs(float(waiting_joints[4]-original_watch_joints[4])) > .055:
                raise RuntimeError('initial J5 recovery exceeded actual encoder travel budget')

        def recheck_initial_view(attempt):
            check_initial_hold()
            report['active_stage'] = 'initial_light_recheck_pre_motion_target'
            return recheck_initial_call_view(
                vision=vision, robot=robot, handeye=handeye, config=config,
                matrix=full_camera_matrix, executor=executor, safety=safety,
                origin=origin, original_watch=original_watch_joints,
                target_label=str(args.target_label).casefold(),
                reason=observer.initial_observation_reason(), attempt=attempt,
                collect_target=collect_stationary_target, wait_clear=wait_for_visual_clear,
                check_hold=check_initial_hold, commit_view=commit_initial_view, checkpoint=checkpoint)

        def confirm_initial_geometry():
            nonlocal initial_geometry_pending
            if not initial_geometry_pending:
                return True
            report['active_stage'] = 'initial_light_recheck_post_motion_target'
            try:
                refreshed, audit = collect_stationary_target(
                    vision, robot, handeye, args.target_label, config,
                    min_capture_unix_sec=initial_geometry_barrier)
            except RuntimeError as exc:
                if not hasattr(exc, 'stationary_target_failure'):
                    raise
                checkpoint('initial_light_geometry_pending', {
                    'reason': 'target_localization_pending',
                    'sampling': exc.stationary_target_failure})
                return False
            check_initial_hold()
            visibility = stable_target_visibility(refreshed, audit, config['depth']['capture_size'])
            report['initial_light_geometry_visibility'] = visibility
            if not visibility['complete']:
                checkpoint('initial_light_geometry_pending', {
                    'reason': visibility['reason'], 'visibility': visibility})
                return False
            latest_target['estimate'] = refreshed
            safety.set_target(refreshed.base_to_button)
            initial_geometry_pending = False
            observer.initial_not_before_unix = time.time()
            checkpoint("initial_light_view_ready", {"target_audit": audit})
            return 'resample'

        observer.initial_view_recheck = recheck_initial_view

        def hold_at_prepress_if_visible() -> bool:
            """Observe at prepress when usable; otherwise retreat to 5 cm."""
            if is_numeric_button(args.target_label):
                # Numeric OFF references belong to plan.observation_joints.
                # Keep early paired ON capture, but make the stationary check
                # use that same endpoint rather than an arbitrary visible pose.
                return False
            try:
                snapshot = vision.request(
                    "light_snapshot",
                    after_sequence=observer.last_sequence,
                    target_label=str(args.target_label),
                )
                visible, visibility_audit = target_is_safely_visible(snapshot)
            except Exception as exc:
                visible = False
                visibility_audit = {
                    "visible": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            checkpoint("prepress_visibility_probe", visibility_audit)
            return visible

        runtime = assemble_button_light_runtime(
            observer, executor, refreshed_plan, move_origin,
            waiting_position_is_origin=waiting_is_origin,
            press_attempt=press_attempt,
            hold_at_prepress_if_visible=hold_at_prepress_if_visible,
            retreat_off_evidence_directory=output / "retreat_off_evidence",
        )
        if requires_fixed_call_reference(args.target_label):
            observer.initial_recovery_active = True
            initial_guard.extended_observation = True
            initial_recovery = InitialObservationRecovery(
                observe=lambda: observer.observe('waiting', emit_event=False),
                check_hold=check_initial_hold, recheck_view=recheck_initial_view,
                reason=lambda: ('target_localization_pending' if initial_geometry_pending
                                else observer.initial_observation_reason()),
                confirm_target=confirm_initial_geometry,
            )
            runtime.initial_observation = initial_recovery.run
        flow = ButtonLightFlow(
            runtime,
            production_flow_config(args.target_label, config,
                                   max_press_attempts=args.max_press_attempts,
                                   cycle_numeric_wait=os.environ.get("YOUXIAOMEI_CYCLE_NUMERIC_ON_TIMEOUT") == "1"),
            event_sink=checkpoint,
        )
        result = flow.run()
        terminal_origin = origin_verification(robot.state(max_age=2.0), origin)
        checkpoint("final_origin_verification", terminal_origin)
        if not terminal_origin["verified"]:
            raise RuntimeError(
                "flow ended without verified origin: "
                f"{terminal_origin['encoder_error_rad']:.4f} rad, "
                f"hardware_ok={terminal_origin['hardware_ok']}"
            )
        exit_code = flow_status_exit_code(result.status)
        report.update({
            "status": result.status,
            "flow_status": result.status,
            "numeric_cycle_on_wait_timeout": next(
                (e for e in result.events if e.get("event") == "numeric_cycle_on_wait_timeout"), None),
            "external_call_off_wait_timeout": next(
                (e for e in result.events if e.get("event") == "external_call_off_wait_timeout"), None),
            "lifecycle_status": "completed",
            "process_exit_code": exit_code,
            "press_attempts": result.press_attempts,
            "target_label": str(args.target_label),
            "waiting_position_is_origin": waiting_is_origin,
            "watch_position": report.get("watch_position"),
            "events": result.events,
            "light_observations": observer.audit,
            "motion_validations": executor.validations,
            "motion_executions": executor.executions,
            "receding_approach": latest_target.get("receding_approach"),
            "initial_target_audit": initial_target_audit,
            "safety_preparation": safety_preparation,
            "trajectory_bridge_health": trajectory_bridge_health,
            "safety_events": supervisor.events,
            "near_depth_pause_events": safety.near_pause.events,
            "depth_degraded_commit_events": safety.depth_degraded_commit_events,
            "recognizer_startup": {
                "overlapped_with_pointing": True,
                "ready_after_start_sec": model_ready_after_start_sec,
                "worker": recognizer.ready,
            },
            "origin_verification": terminal_origin,
            "returned_to_origin": terminal_origin["verified"],
        })
        print(report_yaml(report))
        return exit_code
    except BaseException as exc:
        capture_failure(report, exc, traceback.format_exc())
        state = None
        try:
            state = robot.state(max_age=2.0)
        except Exception as state_exc:
            report["failure_robot_state_error"] = (
                f"{type(state_exc).__name__}: {state_exc}"
            )
        if state is not None:
            verification = origin_verification(state, origin)
            report["failure_robot_state"] = {
                **verification,
            }
            report['origin_verification'] = verification
            report["returned_to_origin"] = verification["verified"]
            recovery_blockers = []
            if cycle_parent_recovery_required(exc, os.environ.get('YOUXIAOMEI_CYCLE_PARENT_RECOVERY') == '1'):
                recovery_blockers.append('cycle_parent_controlled_recovery_pending')
            if safety is None or supervisor is None:
                recovery_blockers.append("safety_not_initialized")
            else:
                if supervisor.stop_active.is_set():
                    recovery_blockers.append("independent_stop_active")
                if supervisor.triggered.is_set():
                    recovery_blockers.append("visual_safety_triggered")
                if supervisor.error:
                    recovery_blockers.append("supervisor_error")
                if safety.publisher_error:
                    recovery_blockers.append("safety_pose_publisher_error")
            if verification["hardware_ok"] != 1:
                recovery_blockers.append("hardware_not_ok")
            if not report["returned_to_origin"] and not recovery_blockers:
                try:
                    recovery = move_origin()
                    report["failure_recovery"] = {
                        "mode": "guarded_origin_return",
                        "origin_verification": recovery,
                    }
                    report["returned_to_origin"] = bool(recovery["verified"])
                    report['origin_verification'] = recovery
                except Exception as recovery_exc:
                    report["failure_recovery_error"] = (
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    )
            elif not report["returned_to_origin"]:
                report["failure_recovery_skipped"] = {
                    "reasons": recovery_blockers,
                }
        report["status"] = "failed"
        report["lifecycle_status"] = "failed"
        report["phase"] = f"failed:{type(exc).__name__}"
        raise
    finally:
        primary_exception_active = sys.exc_info()[0] is not None
        if observer is not None:
            report["light_observations"] = list(observer.audit)
        if flow is not None:
            report["events"] = list(flow.events)
        if executor is not None:
            report["motion_validations"] = dict(executor.validations)
            report["motion_executions"] = list(executor.executions)
        if safety is not None:
            report["near_depth_pause_events"] = list(safety.near_pause.events)
            report["depth_degraded_commit_events"] = list(safety.depth_degraded_commit_events)
        if supervisor is not None:
            report["safety_events"] = list(supervisor.events)
            report["safety_trigger_status"] = supervisor.trigger_status
            report["safety_error"] = supervisor.error
        cleanup_callbacks = []
        if recognizer is not None:
            cleanup_callbacks.append(("light_recognizer", recognizer.close))
        if model_loader is not None:
            cleanup_callbacks.append(
                ("model_loader", lambda: model_loader.shutdown(wait=True))
            )
        if (
            recognizer is None
            and recognizer_future is not None
            and recognizer_future.done()
            and recognizer_future.exception() is None
        ):
            cleanup_callbacks.append((
                "deferred_light_recognizer",
                lambda: recognizer_future.result().close(),
            ))
        if safety is not None:
            cleanup_callbacks.append(("motion_safety", safety.close))
        if robot is not None:
            cleanup_callbacks.append(("robot_adapter", robot.close))
        cleanup_errors = collect_cleanup_errors(cleanup_callbacks)
        # STEP 1 evidence only: collected after existing safety/robot cleanup,
        # with no extra requests, gates or latch changes in the motion path.
        from piper_handeye.pose_timing import collect_pose_timing
        report['camera_pose_timing'] = collect_pose_timing(safety, robot)
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
            if not primary_exception_active:
                report.update({
                    "status": "failed",
                    "lifecycle_status": "failed",
                    "process_exit_code": 1,
                    "phase": "cleanup_failed",
                })
        was_success = int(report.get('process_exit_code', 1)) == 0
        retain_button_run_images(report, failure_images, output, task_id, args.target_label)
        report['pipeline_timing'] = finish_timing_with_analysis()
        reporting_errors = finalize_report(
            report, report_path,
            lambda event: append_mission_status(status_outbox, event))
        if reporting_errors and was_success and not primary_exception_active:
            try:
                failure_images.finalize(success=False, context={
                    'run':output.name, 'task_id':task_id, 'target_label':str(args.target_label),
                    'reporting_errors':reporting_errors, 'motion_result_previously_successful':True})
            except Exception:
                pass  # Keep the original reporting failure, never invent image evidence.
            raise RuntimeError(f'run reporting failed: {reporting_errors}')
        if cleanup_errors and not primary_exception_active:
            raise RuntimeError(f"run cleanup failed: {cleanup_errors}")


if __name__ == "__main__":
    raise SystemExit(main())
