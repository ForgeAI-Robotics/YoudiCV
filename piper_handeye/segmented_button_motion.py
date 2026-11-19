"""Plan press attempts that stop at a camera observation pose after release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kinematics import forward_kinematics, forward_kinematics_batch, inverse_kinematics
from .timed_trajectory import (
    TrajectoryAudit,
    generate_press_cycle_trajectory,
    generate_timed_joint_trajectory,
    joint_target_at_panel_distance,
)


@dataclass(frozen=True)
class SegmentedPressPlan:
    """Independently validatable commands with zero-speed observation stops."""

    observation_joints: np.ndarray
    observation_distance_m: float
    # The reaching path deliberately stops at the camera observation pose
    # before continuing to the pre-press pose.
    approach: list[dict[str, Any]]
    observation_to_prepress: list[dict[str, Any]]
    press_cycle: list[dict[str, Any]]
    retreat_to_observation: list[dict[str, Any]]
    observation_to_waiting: list[dict[str, Any]]
    approach_audit: TrajectoryAudit
    observation_to_prepress_audit: TrajectoryAudit
    press_cycle_audit: TrajectoryAudit
    retreat_audit: TrajectoryAudit
    waiting_return_audit: TrajectoryAudit
    base_to_button: np.ndarray | None = None
    # Precomputed direct return used when moving RGB already proved ON before
    # the ordinary observation retreat. Keeping it in the immutable plan
    # removes all trajectory generation/MoveIt validation after the press.
    prepress_to_waiting: list[dict[str, Any]] | None = None
    prepress_return_audit: TrajectoryAudit | None = None


def camera_target_visibility_audit(
    observation_joints: Any,
    base_to_button: Any,
    gripper_to_camera: Any,
    *,
    max_lateral_error_m: float = 0.08,
) -> dict[str, float | bool]:
    """Check that an interpolated observation pose still aims at the button.

    The tolerance is expressed at the target depth, rather than requiring the
    button to be at the image centre.  This preserves the reaching trajectory
    whenever the button remains comfortably within the usable camera view.
    """
    joints = np.asarray(observation_joints, dtype=float)
    button = np.asarray(base_to_button, dtype=float)
    handeye = np.asarray(gripper_to_camera, dtype=float)
    if joints.shape != (6,) or button.shape != (4, 4) or handeye.shape != (4, 4):
        raise ValueError("invalid camera-visibility inputs")
    if not 0.005 <= float(max_lateral_error_m) <= 0.20:
        raise ValueError("max_camera_lateral_error_m must be in 0.005..0.20 m")
    base_to_camera = forward_kinematics(joints) @ handeye
    point_camera = (
        np.linalg.inv(base_to_camera) @ np.r_[button[:3, 3], 1.0]
    )[:3]
    depth = float(point_camera[2])
    lateral = float(np.linalg.norm(point_camera[:2]))
    return {
        "camera_depth_m": depth,
        "camera_lateral_error_m": lateral,
        "camera_lateral_tolerance_m": float(max_lateral_error_m),
        "camera_points_at_button": bool(depth > 0.08 and lateral <= max_lateral_error_m),
    }


def reusable_near_observation_joints(
    current_joints: Any,
    previous_observation_joints: Any | None,
    *,
    tolerance_rad: float = 0.03,
) -> tuple[np.ndarray | None, float | None]:
    """Reuse a prior reaching-path hold pose only when the arm is still at it."""
    if previous_observation_joints is None:
        return None, None
    current = np.asarray(current_joints, dtype=float)
    previous = np.asarray(previous_observation_joints, dtype=float)
    if current.shape != (6,) or previous.shape != (6,):
        raise ValueError("observation joints must contain six values")
    if not 0.001 <= float(tolerance_rad) <= 0.10:
        raise ValueError("observation reuse tolerance must be in 0.001..0.10 rad")
    error = float(np.max(np.abs(current - previous)))
    if error > float(tolerance_rad):
        return None, error
    return previous.copy(), error


def solve_camera_centered_observation_joints(
    prepress_joints: Any,
    base_to_button: Any,
    gripper_to_camera: Any,
    observation_distance_m: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve a stopped observation pose with the button on the camera axis."""
    prepress = np.asarray(prepress_joints, dtype=float)
    button = np.asarray(base_to_button, dtype=float)
    handeye = np.asarray(gripper_to_camera, dtype=float)
    if prepress.shape != (6,) or button.shape != (4, 4) or handeye.shape != (4, 4):
        raise ValueError("invalid camera-centered observation inputs")
    tcp = forward_kinematics(prepress)
    camera_rotation = tcp[:3, :3] @ handeye[:3, :3]
    camera_offset = tcp[:3, :3] @ handeye[:3, 3]
    optical_axis = camera_rotation[:, 2]
    normal = button[:3, 2] / np.linalg.norm(button[:3, 2])
    denominator = float(optical_axis @ normal)
    if denominator >= -0.5:
        raise RuntimeError("camera optical axis is not facing the button panel")
    camera_depth = (
        -float(observation_distance_m) - float(camera_offset @ normal)
    ) / denominator
    if camera_depth <= 0.08:
        raise RuntimeError("camera-centered observation depth is invalid")
    desired = tcp.copy()
    desired[:3, 3] = button[:3, 3] - camera_depth * optical_axis - camera_offset
    joints, position_error, angle_error = inverse_kinematics(
        desired, prepress, regularization=0.005
    )
    actual_tcp = forward_kinematics(joints)
    actual_camera = actual_tcp @ handeye
    point_camera = (
        np.linalg.inv(actual_camera) @ np.r_[button[:3, 3], 1.0]
    )[:3]
    actual_distance = float((actual_tcp[:3, 3] - button[:3, 3]) @ normal)
    lateral = float(np.linalg.norm(point_camera[:2]))
    if position_error > 0.001 or np.degrees(angle_error) > 0.5:
        raise RuntimeError("camera-centered observation IK did not converge")
    if abs(actual_distance - float(observation_distance_m)) > 0.001:
        raise RuntimeError("camera-centered observation distance is inaccurate")
    if lateral > 0.002:
        raise RuntimeError("camera-centered observation misses the optical axis")
    return joints, {
        "panel_distance_m": actual_distance,
        "camera_depth_m": float(point_camera[2]),
        "camera_lateral_error_m": lateral,
        "ik_position_error_m": float(position_error),
        "ik_orientation_error_deg": float(np.degrees(angle_error)),
    }


def plan_segmented_press_attempt(
    current_joints: Any,
    waiting_joints: Any,
    prepress_joints: Any,
    pressed_joints: Any,
    base_to_button: Any,
    *,
    observation_distance_m: float = 0.15,
    approach_speed_m_s: float = 0.10,
    press_speed_m_s: float = 0.01,
    retreat_speed_m_s: float = 0.20,
    retreat_joint_velocity_scale: float = 0.75,
    return_speed_m_s: float = 0.20,
    prepress_return_speed_m_s: float = 0.25,
    joint_velocity_scale: float = 0.65,
    press_joint_velocity_scale: float = 0.75,
    return_joint_velocity_scale: float = 0.75,
    duration_margin: float = 1.02,
    observation_joints: Any | None = None,
) -> SegmentedPressPlan:
    """Build a reaching-path stop, pre-press, press, and verification retreat.

    ``base_to_button`` and both contact joint targets are expected to come from
    the newest RGB-D target refresh. No previous attempt's pixel box or 3-D
    target is reused by this function.
    """
    if not 0.04 <= float(observation_distance_m) <= 0.20:
        raise ValueError("observation distance must be in 0.04..0.20 m")

    if observation_joints is None:
        observation_joints, _, actual_distance = joint_target_at_panel_distance(
            waiting_joints,
            prepress_joints,
            base_to_button,
            float(observation_distance_m),
        )
    else:
        observation_joints = np.asarray(observation_joints, dtype=float)
        panel = np.asarray(base_to_button, dtype=float)
        normal = panel[:3, 2] / np.linalg.norm(panel[:3, 2])
        actual_distance = float(
            (forward_kinematics(observation_joints)[:3, 3] - panel[:3, 3])
            @ normal
        )
    if abs(actual_distance - float(observation_distance_m)) > 0.001:
        raise RuntimeError("observation target does not satisfy panel distance")

    # Make the camera-visible standoff a true zero-velocity intermediate
    # endpoint on the reaching path.  It is no longer only a post-press retreat.
    approach, approach_audit = generate_timed_joint_trajectory(
        current_joints,
        observation_joints,
        cartesian_speed_m_s=float(approach_speed_m_s),
        joint_velocity_scale=float(joint_velocity_scale),
        point_period_sec=0.10,
        duration_margin=float(duration_margin),
    )
    observation_to_prepress, observation_to_prepress_audit = generate_timed_joint_trajectory(
        observation_joints,
        prepress_joints,
        cartesian_speed_m_s=float(approach_speed_m_s),
        joint_velocity_scale=float(joint_velocity_scale),
        point_period_sec=0.10,
        duration_margin=float(duration_margin),
    )
    press_cycle, press_cycle_audit = generate_press_cycle_trajectory(
        prepress_joints,
        pressed_joints,
        cartesian_speed_m_s=float(press_speed_m_s),
        joint_velocity_scale=float(press_joint_velocity_scale),
        point_period_sec=0.05,
        duration_margin=float(duration_margin),
    )
    retreat, retreat_audit = generate_timed_joint_trajectory(
        prepress_joints,
        observation_joints,
        cartesian_speed_m_s=float(retreat_speed_m_s),
        cartesian_speed_cap_m_s=0.25,
        # The immediate retreat is a short, prevalidated path away from the
        # panel. Use the bridge's allowed velocity ceiling here so the arm
        # reaches the 5 cm observation pose promptly after contact.
        joint_velocity_scale=float(retreat_joint_velocity_scale),
        point_period_sec=0.10,
        duration_margin=float(duration_margin),
    )
    retreat_joints = np.asarray([point["positions"] for point in retreat])
    panel = np.asarray(base_to_button, dtype=float)
    normal = panel[:3, 2] / np.linalg.norm(panel[:3, 2])
    retreat_distance = (
        forward_kinematics_batch(retreat_joints)[:, :3, 3] - panel[:3, 3]
    ) @ normal
    if float(np.min(np.diff(retreat_distance))) < -0.002:
        raise RuntimeError("observation retreat is not monotonic away from panel")
    waiting_return, waiting_return_audit = generate_timed_joint_trajectory(
        observation_joints,
        waiting_joints,
        cartesian_speed_m_s=float(return_speed_m_s),
        joint_velocity_scale=float(return_joint_velocity_scale),
        point_period_sec=0.10,
        duration_margin=float(duration_margin),
        cartesian_speed_cap_m_s=0.25,
    )
    prepress_return, prepress_return_audit = generate_timed_joint_trajectory(
        prepress_joints,
        waiting_joints,
        cartesian_speed_m_s=float(prepress_return_speed_m_s),
        joint_velocity_scale=float(return_joint_velocity_scale),
        point_period_sec=0.10,
        duration_margin=float(duration_margin),
        cartesian_speed_cap_m_s=0.25,
    )
    return SegmentedPressPlan(
        observation_joints=observation_joints,
        observation_distance_m=float(actual_distance),
        approach=approach,
        observation_to_prepress=observation_to_prepress,
        press_cycle=press_cycle,
        retreat_to_observation=retreat,
        observation_to_waiting=waiting_return,
        approach_audit=approach_audit,
        observation_to_prepress_audit=observation_to_prepress_audit,
        press_cycle_audit=press_cycle_audit,
        retreat_audit=retreat_audit,
        waiting_return_audit=waiting_return_audit,
        base_to_button=np.asarray(base_to_button, dtype=float).copy(),
        prepress_to_waiting=prepress_return,
        prepress_return_audit=prepress_return_audit,
    )
