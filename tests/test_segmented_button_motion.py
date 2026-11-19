import numpy as np

from piper_handeye.kinematics import forward_kinematics
from piper_handeye.segmented_button_motion import (
    camera_target_visibility_audit,
    plan_segmented_press_attempt,
    reusable_near_observation_joints,
    solve_camera_centered_observation_joints,
)
from piper_handeye.timed_trajectory import PIPER_JOINT_NAMES, validate_trajectory_payload


def _payload(points):
    return {
        "protocol": "piper_timed_trajectory_v1",
        "command": "validate",
        "joint_names": PIPER_JOINT_NAMES,
        "points": points,
    }


def test_segmented_press_stops_at_exact_observation_distance():
    waiting = np.array([0.23, 1.63, -0.98, 0.39, -0.55, -0.03])
    prepress = np.array([0.19, 1.99, -1.46, 0.37, -0.48, -0.04])
    pressed = prepress + np.array([-0.003, 0.006, -0.006, 0.0, 0.0, 0.0])
    panel = np.eye(4)
    panel[:3, 3] = [0.616, 0.077, 0.277]
    panel[:3, 2] = [-0.9998, -0.0172, -0.0071]
    # Move the synthetic panel origin away so waiting/prepress bracket 15 cm.
    panel[:3, 3] -= 0.10 * panel[:3, 2]

    plan = plan_segmented_press_attempt(
        waiting,
        waiting,
        prepress,
        pressed,
        panel,
        observation_distance_m=0.15,
    )

    for points in (
        plan.approach,
        plan.observation_to_prepress,
        plan.press_cycle,
        plan.retreat_to_observation,
        plan.observation_to_waiting,
        plan.prepress_to_waiting,
    ):
        validate_trajectory_payload(_payload(points))
        np.testing.assert_allclose(points[0]["velocities"], 0.0, atol=1e-12)
        np.testing.assert_allclose(points[-1]["velocities"], 0.0, atol=1e-12)

    np.testing.assert_allclose(plan.approach[-1]["positions"], plan.observation_joints)
    np.testing.assert_allclose(plan.observation_to_prepress[0]["positions"], plan.observation_joints)
    np.testing.assert_allclose(plan.observation_to_prepress[-1]["positions"], prepress)
    np.testing.assert_allclose(plan.press_cycle[0]["positions"], prepress)
    np.testing.assert_allclose(plan.press_cycle[-1]["positions"], prepress)
    np.testing.assert_allclose(
        plan.retreat_to_observation[-1]["positions"], plan.observation_joints
    )
    np.testing.assert_allclose(
        plan.observation_to_waiting[0]["positions"], plan.observation_joints
    )
    np.testing.assert_allclose(plan.observation_to_waiting[-1]["positions"], waiting)
    np.testing.assert_allclose(plan.prepress_to_waiting[0]["positions"], prepress)
    np.testing.assert_allclose(plan.prepress_to_waiting[-1]["positions"], waiting)
    assert plan.prepress_return_audit is not None
    normal = panel[:3, 2]
    distance = float(
        (forward_kinematics(plan.observation_joints)[:3, 3] - panel[:3, 3])
        @ normal
    )
    assert abs(distance - 0.15) < 1e-9


def test_retry_approach_starts_from_near_observation():
    waiting = np.array([0.23, 1.63, -0.98, 0.39, -0.55, -0.03])
    prepress = np.array([0.19, 1.99, -1.46, 0.37, -0.48, -0.04])
    pressed = prepress + np.array([-0.003, 0.006, -0.006, 0.0, 0.0, 0.0])
    panel = np.eye(4)
    panel[:3, 3] = [0.616, 0.077, 0.277]
    panel[:3, 2] = [-0.9998, -0.0172, -0.0071]
    panel[:3, 3] -= 0.10 * panel[:3, 2]
    first = plan_segmented_press_attempt(waiting, waiting, prepress, pressed, panel)

    # A retry is regenerated from the encoder-confirmed near pose and newest
    # target rather than replaying the first approach command.
    second = plan_segmented_press_attempt(
        first.observation_joints, waiting, prepress, pressed, panel
    )
    np.testing.assert_allclose(
        second.approach[0]["positions"], first.observation_joints
    )
    assert second.approach_audit.duration_sec < first.approach_audit.duration_sec


def test_post_press_retreat_uses_max_allowed_joint_speed_and_is_faster():
    waiting = np.array([0.23, 1.63, -0.98, 0.39, -0.55, -0.03])
    prepress = np.array([0.19, 1.99, -1.46, 0.37, -0.48, -0.04])
    pressed = prepress + np.array([-0.003, 0.006, -0.006, 0.0, 0.0, 0.0])
    panel = np.eye(4)
    panel[:3, 3] = [0.616, 0.077, 0.277]
    panel[:3, 2] = [-0.9998, -0.0172, -0.0071]
    panel[:3, 3] -= 0.10 * panel[:3, 2]

    fast = plan_segmented_press_attempt(
        waiting, waiting, prepress, pressed, panel,
        retreat_joint_velocity_scale=0.75,
    )
    conservative = plan_segmented_press_attempt(
        waiting, waiting, prepress, pressed, panel,
        retreat_joint_velocity_scale=0.65,
    )

    assert fast.retreat_audit.duration_sec <= conservative.retreat_audit.duration_sec
    assert fast.retreat_audit.joint_velocity_fraction.max() <= 0.75 + 1e-9
    validate_trajectory_payload(_payload(fast.retreat_to_observation))


def test_camera_visibility_audit_allows_tolerated_off_centre_target():
    joints = np.array([0.23, 1.63, -0.98, 0.39, -0.55, -0.03])
    camera = forward_kinematics(joints)
    button = camera.copy()
    button[:3, 3] += camera[:3, :3] @ np.array([0.04, -0.02, 0.20])
    accepted = camera_target_visibility_audit(
        joints, button, np.eye(4), max_lateral_error_m=0.05
    )
    assert accepted["camera_points_at_button"]
    rejected = camera_target_visibility_audit(
        joints, button, np.eye(4), max_lateral_error_m=0.03
    )
    assert not rejected["camera_points_at_button"]


def test_five_centimetre_observation_aims_camera_at_button():
    seed = np.array([0.23, 1.63, -0.98, 0.39, -0.55, -0.03])
    tcp = forward_kinematics(seed)
    handeye = np.eye(4)
    handeye[:3, 3] = [0.02, 0.01, -0.15]
    camera = tcp @ handeye
    optical_axis = camera[:3, 2]
    button = np.eye(4)
    button[:3, :3] = tcp[:3, :3]
    button[:3, 2] = -optical_axis
    button[:3, 3] = (
        tcp[:3, 3] + 0.05 * optical_axis
        + tcp[:3, :3] @ np.array([0.015, -0.010, 0.0])
    )

    aimed, audit = solve_camera_centered_observation_joints(
        seed, button, handeye, 0.05
    )

    assert abs(audit["panel_distance_m"] - 0.05) < 0.001
    assert audit["camera_lateral_error_m"] < 0.002
    visibility = camera_target_visibility_audit(
        aimed, button, handeye, max_lateral_error_m=0.005
    )
    assert visibility["camera_points_at_button"]


def test_retry_reuses_only_the_verified_near_observation_pose():
    current = np.array([0.1, 0.2, -0.3, 0.4, 0.5, -0.6])
    reused, error = reusable_near_observation_joints(current, current + 0.002)
    np.testing.assert_allclose(reused, current + 0.002)
    assert np.isclose(error, 0.002)
    missing, mismatch = reusable_near_observation_joints(current, current + 0.04)
    assert missing is None
    assert np.isclose(mismatch, 0.04)
