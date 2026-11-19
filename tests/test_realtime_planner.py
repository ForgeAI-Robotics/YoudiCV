import unittest
import time

import numpy as np
from scipy.spatial.transform import Rotation

from piper_handeye.geometry import transform
from piper_handeye.kinematics import forward_kinematics
from piper_handeye.realtime_planner import (
    FastIkTimeoutError,
    FixedRollFastIkPlanner,
    RobotTransformHistory,
    SlidingButtonWindow,
    TimedButtonTransform,
)


class SlidingButtonWindowTests(unittest.TestCase):
    def test_stable_six_sample_window_and_deduplication(self):
        window = SlidingButtonWindow(size=6)
        for index in range(6):
            capture = 100.0 + 0.05 * index
            pose = np.eye(4)
            pose[:3, 3] = [0.5 + 0.00005 * index, 0.1, 0.3]
            sample = TimedButtonTransform(index + 1, capture, pose, 0.9, 0.001)
            self.assertTrue(window.add(sample, now=capture + 0.05))
        self.assertFalse(window.add(sample, now=capture + 0.05))
        estimate = window.estimate(now=capture + 0.05)
        self.assertTrue(estimate.stable, estimate.reasons)
        self.assertEqual(estimate.sequence_range, (1, 6))
        self.assertLess(estimate.position_rms_m, 0.001)

    def test_rejects_stale_input(self):
        window = SlidingButtonWindow()
        self.assertFalse(
            window.add(
                TimedButtonTransform(1, 10.0, np.eye(4), 0.9, 0.001),
                now=10.5,
            )
        )


class FastIkTests(unittest.TestCase):
    def test_solve_timeout_is_validated(self):
        with self.assertRaises(ValueError):
            FixedRollFastIkPlanner(roll_deg=0.0, max_solve_sec=0.0)

    def test_downstream_segment_limits_match_preview_segments(self):
        with self.assertRaises(ValueError):
            FixedRollFastIkPlanner(
                roll_deg=0.0,
                downstream_base_x_distances_m=(0.01, -0.012),
                downstream_max_segment_joint_deltas_rad=(0.25,),
            )

    def test_button_seven_press_branch_is_rejected_by_total_segment_delta(self):
        # Encoder state captured at the failed 2026-09-17 button-7 prepress.
        seed = np.asarray([-0.39, 2.143, -2.197, -0.084, -0.402, 0.127])
        roll_deg = -110.0
        prepress_pose = forward_kinematics(seed)
        nominal = (
            Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
            @ Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
        )
        target_frame = np.eye(4)
        target_frame[:3, :3] = prepress_pose[:3, :3] @ nominal.T
        target_frame[:3, 3] = prepress_pose[:3, 3]
        unrestricted = FixedRollFastIkPlanner(
            roll_deg=roll_deg,
            max_perpendicular_deviation_deg=0.0,
            downstream_base_x_distances_m=(-0.022,),
        ).solve(target_frame, seed)
        self.assertGreater(
            unrestricted.downstream_preview_max_segment_joint_delta_rad,
            0.25,
        )
        with self.assertRaisesRegex(RuntimeError, "no fast IK solution"):
            FixedRollFastIkPlanner(
                roll_deg=roll_deg,
                max_perpendicular_deviation_deg=0.0,
                downstream_base_x_distances_m=(-0.022,),
                downstream_max_segment_joint_deltas_rad=(0.25,),
            ).solve(target_frame, seed)

    def test_unreachable_search_respects_real_time_budget(self):
        button = np.eye(4)
        button[:3, 3] = [2.0, 2.0, 2.0]
        started = time.perf_counter()
        with self.assertRaisesRegex(FastIkTimeoutError, "IK search timed out"):
            FixedRollFastIkPlanner(
                roll_deg=-110.0,
                max_perpendicular_deviation_deg=35.0,
                max_solve_sec=0.05,
            ).solve(button, np.zeros(6))
        self.assertLess(time.perf_counter() - started, 0.5)

    def test_perpendicular_offset_accepts_35_degrees_but_not_more(self):
        planner = FixedRollFastIkPlanner(
            roll_deg=0.0, max_perpendicular_deviation_deg=35.0
        )
        self.assertEqual(planner.max_perpendicular_deviation_deg, 35.0)
        with self.assertRaises(ValueError):
            FixedRollFastIkPlanner(
                roll_deg=0.0, max_perpendicular_deviation_deg=35.1
            )

    def test_fixed_roll_warm_start_recovers_target(self):
        roll_deg = -110.0
        goal = np.asarray([0.2, 1.4, -0.8, 0.3, -0.4, 0.0])
        desired = forward_kinematics(goal)
        button_to_tcp = transform(
            Rotation.from_euler("z", np.deg2rad(roll_deg)).as_matrix()
            @ Rotation.from_euler("x", np.pi).as_matrix(),
            [0.0, 0.0, 0.0],
        )
        base_to_button = desired @ np.linalg.inv(button_to_tcp)
        seed = goal + np.asarray([0.01, -0.01, 0.01, 0.0, 0.0, 0.0])
        result = FixedRollFastIkPlanner(roll_deg=roll_deg).solve(
            base_to_button, seed
        )
        self.assertLess(result.position_error_m, 0.001)
        self.assertLess(result.orientation_error_deg, 0.5)
        self.assertLess(result.max_joint_delta_rad, 0.1)
        self.assertGreaterEqual(result.continuation_steps, 1)
        self.assertLessEqual(
            result.max_continuation_joint_step_rad,
            result.max_joint_delta_rad + 1e-12,
        )

    def test_virtual_continuation_keeps_each_step_below_joint_limit(self):
        roll_deg = -110.0
        goal = np.asarray([0.2, 1.4, -0.8, 0.3, -0.4, 0.0])
        desired = forward_kinematics(goal)
        button_to_tcp = transform(
            Rotation.from_euler("z", np.deg2rad(roll_deg)).as_matrix()
            @ Rotation.from_euler("x", np.pi).as_matrix(),
            [0.0, 0.0, 0.0],
        )
        base_to_button = desired @ np.linalg.inv(button_to_tcp)
        seed = goal + np.asarray([0.25, -0.25, 0.25, -0.25, 0.25, -0.25])
        result = FixedRollFastIkPlanner(
            roll_deg=roll_deg, max_joint_delta_rad=0.20
        ).solve(base_to_button, seed)
        self.assertGreater(result.max_joint_delta_rad, 0.20)
        self.assertGreater(result.continuation_steps, 1)
        self.assertLessEqual(result.max_continuation_joint_step_rad, 0.20)

    def test_guard_standoff_solves_the_actual_first_leg_endpoint(self):
        roll_deg = -110.0
        stand_off_m = 0.085
        goal = np.asarray([0.2, 1.4, -0.8, 0.3, -0.4, 0.0])
        desired = forward_kinematics(goal)
        button_to_tcp = transform(
            Rotation.from_euler("z", np.deg2rad(roll_deg)).as_matrix()
            @ Rotation.from_euler("x", np.pi).as_matrix(),
            [0.0, 0.0, stand_off_m],
        )
        base_to_button = desired @ np.linalg.inv(button_to_tcp)
        seed = goal + np.asarray([0.05, -0.05, 0.05, 0.0, 0.0, 0.0])
        result = FixedRollFastIkPlanner(
            roll_deg=roll_deg,
            stand_off_m=stand_off_m,
            max_joint_delta_rad=1.0,
            max_perpendicular_deviation_deg=35.0,
        ).solve(base_to_button, seed)
        actual = forward_kinematics(result.joints)
        panel_distance = float(
            (actual[:3, 3] - base_to_button[:3, 3])
            @ base_to_button[:3, 2]
        )
        self.assertAlmostEqual(panel_distance, stand_off_m, places=5)
        self.assertLess(result.position_error_m, 0.001)
        self.assertLess(result.orientation_error_deg, 0.5)

    def test_live_candidate_previews_prepress_and_press_reachability(self):
        roll_deg = -110.0
        goal = np.asarray([0.2, 1.4, -0.8, 0.3, -0.4, 0.0])
        desired = forward_kinematics(goal)
        button_to_tcp = transform(
            Rotation.from_euler("z", np.deg2rad(roll_deg)).as_matrix()
            @ Rotation.from_euler("x", np.pi).as_matrix(),
            [0.0, 0.0, 0.0],
        )
        base_to_button = desired @ np.linalg.inv(button_to_tcp)
        result = FixedRollFastIkPlanner(
            roll_deg=roll_deg,
            stand_off_m=0.0,
            max_joint_delta_rad=1.0,
            max_perpendicular_deviation_deg=35.0,
            downstream_base_x_distances_m=(0.0, -0.005),
            downstream_max_segment_joint_deltas_rad=(None, 0.25),
        ).solve(base_to_button, goal)
        self.assertEqual(
            result.downstream_preview_distances_m, (0.0, -0.005)
        )
        self.assertLessEqual(
            result.downstream_preview_max_joint_step_rad, 0.25
        )
        self.assertGreater(
            result.downstream_preview_min_joint_limit_margin_rad, 0.0
        )
        self.assertLessEqual(
            result.downstream_preview_max_segment_joint_delta_rad, 0.25
        )

    def test_mobile_base_uses_live_current_orientation_without_cache(self):
        roll_deg = -110.0
        stand_off_m = 0.09
        live_offset_deg = np.asarray([20.0, -15.0, 10.0])
        current_joints = np.asarray([0.2, 1.4, -0.8, 0.3, -0.4, 0.0])
        current_tcp = forward_kinematics(current_joints)
        nominal = (
            Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
            @ Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
        )
        button_to_tcp_rotation = (
            nominal
            @ Rotation.from_euler("xyz", live_offset_deg, degrees=True).as_matrix()
        )
        base_to_button = np.eye(4)
        base_to_button[:3, :3] = (
            current_tcp[:3, :3] @ button_to_tcp_rotation.T
        )
        base_to_button[:3, 3] = (
            current_tcp[:3, 3]
            - base_to_button[:3, 2] * stand_off_m
        )
        result = FixedRollFastIkPlanner(
            roll_deg=roll_deg,
            stand_off_m=stand_off_m,
            max_joint_delta_rad=1.0,
            max_perpendicular_deviation_deg=35.0,
        ).solve(base_to_button, current_joints)
        np.testing.assert_allclose(
            result.orientation_offset_deg, live_offset_deg, atol=1e-7
        )
        self.assertLess(result.solve_sec, 0.25)
        self.assertLess(result.max_joint_delta_rad, 1e-3)

    def test_mobile_xyz_perturbations_use_bounded_continuous_orientation(self):
        """A live-pose change must not fall through hundreds of lattice IKs."""
        roll_deg = -110.0
        seed = np.asarray([0.04, 0.105, 0.0, 0.051, 0.204, 0.0])
        projected_offset_deg = np.asarray(
            [-1.7022282238, 13.1300611360, -14.7478846825]
        )
        nominal = (
            Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
            @ Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
        )
        current_tcp = forward_kinematics(seed)
        button_rotation = (
            current_tcp[:3, :3]
            @ Rotation.from_euler(
                "xyz", projected_offset_deg, degrees=True
            ).as_matrix().T
            @ nominal.T
        )
        center = np.asarray([0.53497, -0.08411, 0.26699])
        for perturbation in (
            np.zeros(3),
            np.asarray([-0.03, -0.02, -0.02]),
        ):
            button = np.eye(4)
            button[:3, :3] = button_rotation
            button[:3, 3] = center + perturbation
            result = FixedRollFastIkPlanner(
                roll_deg=roll_deg,
                stand_off_m=0.0898488757,
                max_joint_delta_rad=1.0,
                max_perpendicular_deviation_deg=35.0,
                downstream_base_x_distances_m=(0.01, -0.005),
            ).solve(button, seed)
            self.assertLess(result.solve_sec, 1.5)
            self.assertEqual(result.direct_candidate_count, 0)
            self.assertEqual(result.continuation_attempt_count, 0)
            self.assertGreater(
                result.downstream_preview_min_joint_limit_margin_rad,
                0.5,
            )

    def test_reachability_is_not_clipped_by_arbitrary_cartesian_height_box(self):
        roll_deg = -110.0
        high_goal = np.asarray(
            [-0.15566, 1.75524, -1.74556, 0.10487, -0.52682, 0.24465]
        )
        desired = forward_kinematics(high_goal)
        self.assertGreater(desired[2, 3], 0.55)
        button_to_tcp = transform(
            Rotation.from_euler("z", np.deg2rad(roll_deg)).as_matrix()
            @ Rotation.from_euler("x", np.pi).as_matrix(),
            [0.0, 0.0, 0.0],
        )
        base_to_button = desired @ np.linalg.inv(button_to_tcp)
        seed = high_goal + np.asarray([0.01, -0.01, 0.01, 0.0, 0.0, 0.0])
        result = FixedRollFastIkPlanner(roll_deg=roll_deg).solve(
            base_to_button, seed
        )
        self.assertLess(result.position_error_m, 0.001)
        self.assertLess(result.orientation_error_deg, 0.5)

    def test_optional_workspace_box_is_explicit_and_validated(self):
        with self.assertRaises(ValueError):
            FixedRollFastIkPlanner(
                roll_deg=0.0,
                workspace_bounds_m=[[0.0, 1.0], [0.0, 1.0]],
            )


class RobotTransformHistoryTests(unittest.TestCase):
    def test_interpolates_translation_and_rotation(self):
        history = RobotTransformHistory(max_gap_sec=0.2)
        first = np.eye(4)
        second = np.eye(4)
        second[:3, 3] = [0.10, -0.04, 0.02]
        second[:3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
        history.add(10.0, first)
        history.add(10.1, second)
        middle = history.transform_at(10.05, wait_sec=0.0)
        np.testing.assert_allclose(middle[:3, 3], [0.05, -0.02, 0.01], atol=1e-10)
        angle = Rotation.from_matrix(middle[:3, :3]).as_euler("zyx", degrees=True)[0]
        self.assertAlmostEqual(angle, 10.0, places=8)

    def test_rejects_unbracketed_capture(self):
        history = RobotTransformHistory()
        history.add(10.0, np.eye(4))
        with self.assertRaises(RuntimeError):
            history.transform_at(10.1, wait_sec=0.0)


if __name__ == "__main__":
    unittest.main()
