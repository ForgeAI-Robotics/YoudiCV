import unittest
from unittest.mock import patch

import numpy as np
import yaml

from piper_handeye.timed_trajectory import TrajectoryAudit
from piper_handeye.trajectory_splice import HandoffState
from piper_handeye.kinematics import forward_kinematics
from run_online_replan_no_contact import (
    DEFAULT_PRESS_TRAVEL_M,
    base_x_distance_frame,
    concatenate_stopped_trajectories,
    constrain_button_ray_to_reference_plane,
    interpolate_trajectory_joints,
    motion_stage_sequence,
    refine_precomputed_ik,
    refresh_handoff_after_compute,
    should_center_visible_target,
    solve_independent_press_ik,
    tcp_panel_distances,
    trajectory_audit_payload,
    wait_for_stable_visibility_after_target_switch,
)


class OnlineReplanGeometryTests(unittest.TestCase):
    def test_target_switch_waits_for_fresh_consistent_inference_frames(self):
        class FakeVision:
            def __init__(self):
                self.responses = iter([
                    {"frame_sequence": 10, "detections": []},
                    {"frame_sequence": 11, "detections": []},
                    {
                        "frame_sequence": 12,
                        "detections": [{
                            "label": "16", "confidence": 0.95,
                            "bbox_xyxy": [100, 100, 120, 120],
                        }],
                    },
                    {
                        "frame_sequence": 13,
                        "detections": [{
                            "label": "16", "confidence": 0.96,
                            "bbox_xyxy": [101, 100, 121, 120],
                        }],
                    },
                ])

            def request(self, *_args, **_kwargs):
                response = next(self.responses)
                response['capture_time_unix_sec'] = __import__('time').time()
                response['image_size'] = [640, 360]
                return response

        response, target_visible = wait_for_stable_visibility_after_target_switch(
            FakeVision(), target_label="16", min_confidence=0.7,
            sequence_barrier=10, timeout_sec=0.5,
        )
        self.assertTrue(target_visible)
        self.assertEqual(response["frame_sequence"], 13)

    def test_missing_up_returns_stable_visible_down_anchor(self):
        class FakeVision:
            def __init__(self):
                self.sequence = 20

            def request(self, *_args, **_kwargs):
                self.sequence += 1
                return {
                    "frame_sequence": self.sequence,
                    "capture_time_unix_sec": __import__("time").time(),
                    "image_size": [640, 360],
                    "detections": [{
                        "label": "down", "confidence": 0.91,
                        "bbox_xyxy": [100, 100, 120, 120],
                    }],
                }

        response, target_visible = wait_for_stable_visibility_after_target_switch(
            FakeVision(), target_label="up", min_confidence=0.7,
            sequence_barrier=20, timeout_sec=0.5,
        )
        self.assertFalse(target_visible)
        self.assertEqual(response["detections"][0]["label"], "down")

    def test_second_capture_rgb_ray_uses_frozen_initialized_panel_depth(self):
        camera = np.eye(4)
        measured = np.eye(4)
        measured[:3, 3] = [0.10, -0.05, 0.60]
        reference = np.eye(4)
        reference[:3, 3] = [0.0, 0.0, 0.50]
        reference[:3, 2] = [0.0, 0.0, -1.0]
        constrained, correction = constrain_button_ray_to_reference_plane(
            camera, measured, reference
        )
        np.testing.assert_allclose(
            constrained[:3, 3], [0.10 / 1.2, -0.05 / 1.2, 0.50]
        )
        self.assertGreater(correction, 0.09)
        np.testing.assert_allclose(constrained[:3, 2], [0.0, 0.0, -1.0])

    def test_material_offset_converts_surface_gap_without_changing_tcp_model(self):
        prepress_tcp, pressed_tcp = tcp_panel_distances(0.010, 0.007, 0.004)
        self.assertAlmostEqual(prepress_tcp, 0.014)
        self.assertAlmostEqual(pressed_tcp, 0.007)

    def test_zero_material_offset_preserves_existing_motion_geometry(self):
        prepress_tcp, pressed_tcp = tcp_panel_distances(0.010, 0.007, 0.0)
        self.assertAlmostEqual(prepress_tcp, 0.010)
        self.assertAlmostEqual(pressed_tcp, 0.003)

    def test_origin_visible_target_keeps_direct_box_without_legacy_centering(self):
        self.assertFalse(should_center_visible_target(
            auto_pointing_enabled=True,
            legacy_center_requested=False,
            target_visible_at_origin=True,
            target_was_pointed=False,
        ))
        self.assertFalse(should_center_visible_target(
            auto_pointing_enabled=True,
            legacy_center_requested=False,
            target_visible_at_origin=False,
            target_was_pointed=True,
        ))

    def test_independent_seven_mm_press_uses_local_prepress_branch(self):
        prepress = np.asarray(
            [-0.3503, 1.6016, -2.1540, 0.6018, 0.5180, -0.1737]
        )
        prepress_pose = forward_kinematics(prepress)
        pressed_position = prepress_pose[:3, 3].copy()
        pressed_position[0] += DEFAULT_PRESS_TRAVEL_M
        result = solve_independent_press_ik(prepress, pressed_position)
        pressed_pose = forward_kinematics(result.joints)
        self.assertLessEqual(result.position_error_m, 0.001)
        self.assertLessEqual(result.orientation_error_deg, 0.5)
        self.assertLessEqual(result.max_joint_delta_rad, 0.25)
        np.testing.assert_allclose(
            pressed_pose[:3, 3], pressed_position, atol=0.001
        )

    def test_default_press_is_independent_seven_millimeter_stage(self):
        self.assertAlmostEqual(DEFAULT_PRESS_TRAVEL_M, 0.007)
        stages = motion_stage_sequence(DEFAULT_PRESS_TRAVEL_M)
        self.assertEqual(
            stages,
            [
                "approach_to_second_capture_prepress",
                "encoder_verify_prepress",
                "independent_press_forward",
                "independent_press_reverse_to_prepress",
                "continuous_return_to_origin",
            ],
        )

    def test_zero_press_disables_only_the_independent_press_stage(self):
        self.assertEqual(
            motion_stage_sequence(0.0),
            [
                "approach_to_second_capture_prepress",
                "encoder_verify_prepress",
                "continuous_return_to_origin",
            ],
        )

    def test_second_capture_local_ik_refines_precomputed_solution(self):
        seed = np.zeros(6)
        desired = forward_kinematics(seed)
        target_position = desired[:3, 3].copy()
        target_position[0] += 0.001
        result = refine_precomputed_ik(
            desired,
            target_position,
            seed,
        )
        self.assertLessEqual(result.position_error_m, 0.001)
        self.assertLessEqual(result.orientation_error_deg, 0.5)
        self.assertLessEqual(result.max_joint_delta_rad, 0.25)

    def test_local_ik_uses_fixed_orientation_continuation_after_direct_failure(self):
        seed = np.zeros(6)
        desired = forward_kinematics(seed)
        target_position = desired[:3, 3].copy()
        target_position[0] += 0.03
        calls = []

        def fake_inverse(target, joints, **_):
            calls.append(np.asarray(target).copy())
            if len(calls) == 1:
                return np.asarray(joints).copy(), 0.01, 0.0
            return np.asarray(joints).copy(), 0.0, 0.0

        with patch(
            "run_online_replan_no_contact.inverse_kinematics",
            side_effect=fake_inverse,
        ):
            result = refine_precomputed_ik(
                desired,
                target_position,
                seed,
                max_joint_delta_rad=1.0,
            )
        self.assertEqual(result.continuation_steps, 3)
        self.assertEqual(len(calls), 4)
        for target in calls[1:]:
            np.testing.assert_allclose(target[:3, :3], desired[:3, :3])

    def test_slow_compute_refreshes_handoff_into_future_hold(self):
        active = [
            {
                "time_from_start_sec": float(stamp),
                "positions": [stamp / 10.0] * 6,
                "velocities": [0.0] * 6,
                "accelerations": [0.0] * 6,
            }
            for stamp in range(9)
        ]
        original = HandoffState(
            active_index=3,
            active_time_sec=3.0,
            positions=np.full(6, 0.3),
            velocities=np.zeros(6),
            accelerations=np.zeros(6),
        )
        refreshed, changed, remaining = refresh_handoff_after_compute(
            active,
            original,
            active_elapsed_sec=5.0,
            scheduling_lead_sec=1.0,
        )
        self.assertTrue(changed)
        self.assertEqual(refreshed.active_time_sec, 6.0)
        self.assertEqual(remaining, 1.0)

        unchanged, changed, remaining = refresh_handoff_after_compute(
            active,
            original,
            active_elapsed_sec=1.0,
            scheduling_lead_sec=1.0,
        )
        self.assertFalse(changed)
        self.assertIs(unchanged, original)
        self.assertEqual(remaining, 2.0)

    def test_base_x_distance_ignores_panel_tilt(self):
        button = np.eye(4)
        button[:3, 3] = [0.56, -0.07, 0.27]
        button[:3, 2] = [-0.98, 0.12, 0.10]
        distance_frame = base_x_distance_frame(button)

        prepress = np.asarray([0.55, -0.07, 0.27])
        pressed = np.asarray([0.575, -0.07, 0.27])
        self.assertAlmostEqual(
            (prepress - button[:3, 3]) @ distance_frame[:3, 2], 0.01
        )
        self.assertAlmostEqual(
            (pressed - button[:3, 3]) @ distance_frame[:3, 2], -0.015
        )
        np.testing.assert_allclose(distance_frame[:3, 2], [-1.0, 0.0, 0.0])

    def test_pointing_hold_offsets_the_approach(self):
        point = lambda stamp, value: {
            "time_from_start_sec": stamp,
            "positions": [value] * 6,
            "velocities": [0.0] * 6,
            "accelerations": [0.0] * 6,
        }
        first = [point(0.0, 0.0), point(1.0, 1.0)]
        second = [point(0.0, 1.0), point(2.0, 2.0)]
        joined, approach_offset = concatenate_stopped_trajectories(
            first, second, hold_sec=1.0
        )
        self.assertAlmostEqual(approach_offset, 2.0)
        self.assertAlmostEqual(joined[-1]["time_from_start_sec"], 4.0)
        self.assertTrue(
            all(
                current["time_from_start_sec"]
                < following["time_from_start_sec"]
                for current, following in zip(joined, joined[1:])
            )
        )

    def test_pointing_audit_is_yaml_safe(self):
        audit = TrajectoryAudit(
            duration_sec=1.0,
            point_count=2,
            cartesian_path_m=0.02,
            cartesian_peak_speed_m_s=0.02,
            joint_peak_velocity_rad_s=np.zeros(6),
            joint_velocity_fraction=np.zeros(6),
        )
        payload = trajectory_audit_payload(audit)
        self.assertIsInstance(payload["joint_peak_velocity_rad_s"], list)
        yaml.safe_dump(payload)

    def test_commanded_pose_interpolation_tracks_trajectory_time(self):
        points = [
            {"time_from_start_sec": 0.0, "positions": [0.0] * 6},
            {"time_from_start_sec": 2.0, "positions": [2.0] * 6},
        ]
        np.testing.assert_allclose(
            interpolate_trajectory_joints(points, 0.5), [0.5] * 6
        )
        np.testing.assert_allclose(
            interpolate_trajectory_joints(points, 3.0), [2.0] * 6
        )


if __name__ == "__main__":
    unittest.main()
