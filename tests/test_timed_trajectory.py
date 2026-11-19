import copy
import unittest

import numpy as np

from piper_handeye.timed_trajectory import (
    PIPER_MAX_VELOCITIES,
    PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
    encode_trajectory_udp_payload,
    generate_continuous_waypoint_trajectory,
    generate_blended_press_inverse_return,
    generate_blended_guarded_retract,
    generate_press_cycle_trajectory,
    generate_timed_joint_trajectory,
    generate_two_zone_joint_trajectory,
    joint_target_at_panel_distance,
    reverse_timed_trajectory,
    validate_trajectory_payload,
)
from piper_handeye.trajectory_splice import (
    generate_first_feasible_splice,
    generate_splice_trajectory,
    select_future_handoff,
)


class TimedTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.start = np.array([0.23, 1.63, -0.98, 0.39, -0.55, -0.03])
        self.end = np.array([0.19, 1.99, -1.46, 0.37, -0.48, -0.04])

    def payload(self, points):
        return {
            "protocol": "piper_timed_trajectory_v1",
            "command": "validate",
            "joint_names": [f"piper_joint{i}" for i in range(1, 7)],
            "points": points,
        }

    def test_udp_encoding_compacts_float_tails_without_removing_samples(self):
        points = []
        base_positions = [0.1, 1.0, -1.0, 0.1, 0.1, 0.1]
        for index in range(140):
            stamp = index * 0.07123456789012345
            positions = [
                base_positions[joint] + index * (joint + 1) * 0.00001
                + np.sin(index + joint) * 1e-8
                for joint in range(6)
            ]
            points.append(
                {
                    "time_from_start_sec": stamp,
                    "positions": positions,
                    "velocities": [
                        0.01000000000012345 * (joint + 1)
                        for joint in range(6)
                    ],
                    "accelerations": [
                        0.02000000000012345 * (joint + 1)
                        for joint in range(6)
                    ],
                }
            )
        payload = self.payload(points)
        payload.update(task_id="wire-test", token="redacted")
        raw_size = len(
            __import__("json").dumps(
                payload, separators=(",", ":")
            ).encode("utf-8")
        )
        self.assertGreater(raw_size, 60000)

        encoded, wire_payload, decimals = encode_trajectory_udp_payload(payload)

        self.assertLessEqual(len(encoded), 60000)
        self.assertEqual(len(wire_payload["points"]), len(points))
        self.assertEqual(decimals, 14)
        validate_trajectory_payload(wire_payload)
        maximum_error = max(
            abs(
                wire_payload["points"][index]["positions"][joint]
                - points[index]["positions"][joint]
            )
            for index in range(len(points))
            for joint in range(6)
        )
        self.assertLessEqual(maximum_error, 5.1e-15)

    def test_generation_respects_speed_and_endpoints(self):
        points, audit = generate_timed_joint_trajectory(
            self.start, self.end, cartesian_speed_m_s=0.05
        )
        np.testing.assert_allclose(points[0]["positions"], self.start, atol=1e-12)
        np.testing.assert_allclose(points[-1]["positions"], self.end, atol=1e-12)
        self.assertLessEqual(audit.cartesian_peak_speed_m_s, 0.05)
        self.assertTrue(
            np.all(audit.joint_peak_velocity_rad_s <= PIPER_MAX_VELOCITIES * 0.25)
        )
        validate_trajectory_payload(self.payload(points))

    def test_high_speed_profile_respects_raised_caps(self):
        points, audit = generate_timed_joint_trajectory(
            self.start, self.end, cartesian_speed_m_s=0.15,
            joint_velocity_scale=0.75, duration_margin=1.02,
        )
        self.assertLessEqual(audit.cartesian_peak_speed_m_s, 0.15)
        self.assertTrue(
            np.all(audit.joint_peak_velocity_rad_s <= PIPER_MAX_VELOCITIES * 0.75)
        )
        validate_trajectory_payload(self.payload(points))

    def test_return_profile_can_exceed_contact_cartesian_cap(self):
        points, audit = generate_timed_joint_trajectory(
            self.end, self.start, cartesian_speed_m_s=0.20,
            cartesian_speed_cap_m_s=PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
            joint_velocity_scale=0.75, duration_margin=1.02,
        )
        self.assertGreater(audit.cartesian_peak_speed_m_s, 0.15)
        self.assertLessEqual(audit.cartesian_peak_speed_m_s, 0.20)
        self.assertTrue(
            np.all(audit.joint_peak_velocity_rad_s <= PIPER_MAX_VELOCITIES * 0.75)
        )
        validate_trajectory_payload(self.payload(points))

    def test_contact_profile_still_rejects_speed_above_standard_cap(self):
        with self.assertRaisesRegex(ValueError, "cartesian_speed_m_s"):
            generate_timed_joint_trajectory(
                self.start, self.end, cartesian_speed_m_s=0.20
            )

    def test_continuous_return_does_not_stop_at_safe_waypoint(self):
        contact = self.end + np.asarray([0.01, 0.03, -0.04, 0.0, 0.01, 0.0])
        points, audit = generate_continuous_waypoint_trajectory(
            contact, self.end, self.start,
            first_cartesian_speed_m_s=0.12,
            second_cartesian_speed_m_s=0.20,
            first_joint_velocity_scale=0.65,
            second_joint_velocity_scale=0.75,
            duration_margin=1.02,
        )
        waypoint = min(
            points,
            key=lambda point: abs(
                point["time_from_start_sec"] - audit.waypoint_time_sec
            ),
        )
        np.testing.assert_allclose(waypoint["positions"], self.end, atol=1e-9)
        self.assertGreater(np.linalg.norm(waypoint["velocities"]), 1e-3)
        self.assertLess(np.linalg.norm(waypoint["accelerations"]), 1e-7)
        self.assertLessEqual(audit.first_zone_peak_speed_m_s, 0.12 + 1e-9)
        self.assertLessEqual(audit.second_zone_peak_speed_m_s, 0.20 + 1e-9)
        waypoint_index = points.index(waypoint)
        local_joint_speeds = [
            np.linalg.norm(point["velocities"])
            for point in points[waypoint_index - 2 : waypoint_index + 3]
        ]
        self.assertTrue(
            all(
                current <= following + 1e-9
                for current, following in zip(
                    local_joint_speeds, local_joint_speeds[1:]
                )
            )
        )
        validate_trajectory_payload(self.payload(points))

    def test_continuous_return_accepts_maximum_homeward_speed(self):
        contact = self.end + np.asarray([0.01, 0.03, -0.04, 0.0, 0.01, 0.0])
        points, audit = generate_continuous_waypoint_trajectory(
            contact, self.end, self.start,
            first_cartesian_speed_m_s=0.15,
            second_cartesian_speed_m_s=PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
            first_joint_velocity_scale=0.65,
            second_joint_velocity_scale=0.75,
            duration_margin=1.02,
        )
        waypoint = min(
            points,
            key=lambda point: abs(
                point["time_from_start_sec"] - audit.waypoint_time_sec
            ),
        )
        self.assertGreater(np.linalg.norm(waypoint["velocities"]), 1e-3)
        self.assertLessEqual(audit.first_zone_peak_speed_m_s, 0.15 + 1e-9)
        self.assertLessEqual(
            audit.second_zone_peak_speed_m_s,
            PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S + 1e-9,
        )

    def test_continuous_return_adapts_velocity_near_joint_limit(self):
        start = np.asarray([0.04, 0.70, 0.0, 0.10, -0.30, 0.10])
        waypoint = np.asarray([0.04, 0.45, -0.019, -0.05, -0.55, 0.10])
        end = np.asarray([0.04, 0.105, 0.0, 0.051, 0.204, 0.0])
        points, audit = generate_continuous_waypoint_trajectory(
            start,
            waypoint,
            end,
            first_cartesian_speed_m_s=0.15,
            second_cartesian_speed_m_s=0.25,
            first_joint_velocity_scale=0.65,
            second_joint_velocity_scale=0.75,
        )
        waypoint_point = min(
            points,
            key=lambda point: abs(
                point["time_from_start_sec"] - audit.waypoint_time_sec
            ),
        )
        self.assertGreater(np.linalg.norm(waypoint_point["velocities"]), 1e-3)
        positions = np.asarray([point["positions"] for point in points])
        self.assertTrue(np.all(positions[:, 2] <= 1e-12))
        validate_trajectory_payload(self.payload(points))
        self.assertTrue(
            np.all(audit.joint_peak_velocity_rad_s <= PIPER_MAX_VELOCITIES * 0.75)
        )
        validate_trajectory_payload(self.payload(points))

    def test_two_zone_profile_is_continuous_and_bounded(self):
        points, audit = generate_two_zone_joint_trajectory(
            self.start, self.end, transition_fraction=0.70,
            fast_cartesian_speed_m_s=0.15,
            near_cartesian_speed_m_s=0.06,
            joint_velocity_scale=0.65,
            point_period_sec=0.05,
            duration_margin=1.02,
        )
        self.assertLessEqual(audit.fast_zone_peak_speed_m_s, 0.15 + 1e-9)
        self.assertLessEqual(audit.near_zone_peak_speed_m_s, 0.06 + 1e-9)
        transition = min(
            points,
            key=lambda point: abs(
                point["time_from_start_sec"] - audit.transition_time_sec
            ),
        )
        self.assertAlmostEqual(
            transition["time_from_start_sec"], audit.transition_time_sec, places=9
        )
        self.assertGreater(np.linalg.norm(transition["velocities"]), 1e-4)
        self.assertLess(np.linalg.norm(transition["accelerations"]), 1e-7)
        validate_trajectory_payload(self.payload(points))

    def test_validation_rejects_time_reversal(self):
        points, _ = generate_timed_joint_trajectory(self.start, self.end)
        broken = copy.deepcopy(points)
        broken[2]["time_from_start_sec"] = broken[1]["time_from_start_sec"]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_trajectory_payload(self.payload(broken))

    def test_validation_rejects_large_declared_velocity(self):
        points, _ = generate_timed_joint_trajectory(self.start, self.end)
        broken = copy.deepcopy(points)
        broken[1]["velocities"][0] = 1.0
        with self.assertRaisesRegex(ValueError, "declared velocity"):
            validate_trajectory_payload(self.payload(broken))

    def test_exact_time_reversal_preserves_path_and_derivatives(self):
        points, _ = generate_timed_joint_trajectory(
            self.start, self.end, cartesian_speed_m_s=0.08,
            point_period_sec=0.10,
        )
        reversed_points = reverse_timed_trajectory(points)
        validate_trajectory_payload(self.payload(reversed_points))
        np.testing.assert_allclose(
            [p["positions"] for p in reversed_points],
            [p["positions"] for p in reversed(points)],
        )
        np.testing.assert_allclose(
            [p["velocities"] for p in reversed_points],
            [-np.asarray(p["velocities"]) for p in reversed(points)],
        )
        np.testing.assert_allclose(
            [p["accelerations"] for p in reversed_points],
            [p["accelerations"] for p in reversed(points)],
        )
        restored = reverse_timed_trajectory(reversed_points)
        np.testing.assert_allclose(
            [p["positions"] for p in restored],
            [p["positions"] for p in points],
        )
        np.testing.assert_allclose(
            [p["velocities"] for p in restored],
            [p["velocities"] for p in points],
        )
        np.testing.assert_allclose(
            [p["accelerations"] for p in restored],
            [p["accelerations"] for p in points],
        )

    def test_press_cycle_is_one_fast_round_trip(self):
        points, audit = generate_press_cycle_trajectory(
            self.start,
            self.end,
            cartesian_speed_m_s=0.15,
            joint_velocity_scale=0.75,
        )
        midpoint = (len(points) - 1) // 2
        np.testing.assert_allclose(points[0]["positions"], self.start)
        np.testing.assert_allclose(points[midpoint]["positions"], self.end)
        np.testing.assert_allclose(points[-1]["positions"], self.start)
        self.assertLessEqual(audit.cartesian_peak_speed_m_s, 0.15 + 1e-9)
        self.assertAlmostEqual(
            points[midpoint]["time_from_start_sec"],
            audit.duration_sec / 2.0,
        )
        validate_trajectory_payload(self.payload(points))

    def test_blended_press_return_passes_prepress_without_stopping(self):
        pressed = self.end + np.asarray([0.01, 0.025, -0.03, 0.0, 0.01, 0.0])
        inverse, _ = generate_timed_joint_trajectory(
            self.end,
            self.start,
            cartesian_speed_m_s=0.15,
            cartesian_speed_cap_m_s=PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
            joint_velocity_scale=0.75,
            point_period_sec=0.05,
            duration_margin=1.02,
        )
        points, audit = generate_blended_press_inverse_return(
            self.end,
            pressed,
            inverse,
            press_speed_m_s=0.15,
            release_speed_m_s=0.20,
        )
        np.testing.assert_allclose(points[0]["positions"], self.end)
        np.testing.assert_allclose(points[-1]["positions"], self.start)
        self.assertGreater(audit.prepress_joint_speed_rad_s, 0.01)
        self.assertGreater(audit.prepress_cartesian_speed_m_s, 0.01)
        self.assertGreater(audit.inverse_join_index, 1)
        validate_trajectory_payload(self.payload(points))

    def test_guarded_retract_blends_through_prepress_to_observation(self):
        inverse, _ = generate_timed_joint_trajectory(
            self.end,
            self.start,
            cartesian_speed_m_s=0.20,
            cartesian_speed_cap_m_s=PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
            joint_velocity_scale=0.75,
            point_period_sec=0.05,
            duration_margin=1.02,
        )
        contact = self.end + np.asarray([0.007, 0.018, -0.021, 0.0, 0.007, 0.0])
        points, audit = generate_blended_guarded_retract(
            contact, self.end, inverse, release_speed_m_s=0.25,
        )
        np.testing.assert_allclose(points[0]["positions"], contact)
        np.testing.assert_allclose(points[-1]["positions"], self.start)
        self.assertGreater(audit.prepress_joint_speed_rad_s, 0.01)
        self.assertGreater(audit.prepress_cartesian_speed_m_s, 0.01)
        self.assertEqual(audit.press_endpoint_time_sec, 0.0)
        validate_trajectory_payload(self.payload(points))

    def test_press_return_can_continue_through_pointing_to_origin(self):
        pointing = self.start + np.asarray(
            [-0.04, -0.20, 0.16, -0.03, 0.10, 0.04]
        )
        pressed = self.end + np.asarray(
            [0.01, 0.025, -0.03, 0.0, 0.01, 0.0]
        )
        continuous_return, return_audit = (
            generate_continuous_waypoint_trajectory(
                self.end,
                pointing,
                self.start,
                first_cartesian_speed_m_s=0.15,
                second_cartesian_speed_m_s=0.20,
                first_joint_velocity_scale=0.65,
                second_joint_velocity_scale=0.75,
            )
        )
        points, _ = generate_blended_press_inverse_return(
            self.end,
            pressed,
            continuous_return,
            press_speed_m_s=0.15,
            release_speed_m_s=0.20,
        )
        pointing_sample = min(
            points,
            key=lambda point: np.linalg.norm(
                np.asarray(point["positions"]) - pointing
            ),
        )
        np.testing.assert_allclose(
            pointing_sample["positions"], pointing, atol=1e-9
        )
        self.assertGreater(
            np.linalg.norm(pointing_sample["velocities"]), 1e-3
        )
        np.testing.assert_allclose(points[-1]["positions"], self.start)
        self.assertGreater(return_audit.waypoint_joint_speed_rad_s, 1e-3)
        validate_trajectory_payload(self.payload(points))

    def test_panel_distance_target_on_known_contact_path(self):
        panel = np.eye(4)
        panel[:3, 3] = [0.616, 0.077, 0.277]
        panel[:3, 2] = [-0.9998, -0.0172, -0.0071]
        # Restore an orthonormal rotation; only column 2 is used by this helper.
        target, fraction, distance = joint_target_at_panel_distance(
            self.start, self.end, panel, 0.08
        )
        self.assertEqual(target.shape, (6,))
        self.assertGreater(fraction, 0.0)
        self.assertLess(fraction, 1.0)
        self.assertAlmostEqual(distance, 0.08, places=9)

    def test_splice_preserves_future_handoff_state(self):
        active, _ = generate_timed_joint_trajectory(
            self.start, self.end, cartesian_speed_m_s=0.08,
            point_period_sec=0.1,
        )
        handoff = select_future_handoff(
            active, active_elapsed_sec=0.2, lead_sec=0.5
        )
        replacement_end = self.end + np.asarray([0.01, 0.02, -0.02, 0, 0, 0])
        replacement, audit = generate_splice_trajectory(
            handoff, replacement_end, cartesian_speed_m_s=0.08,
            point_period_sec=0.1,
        )
        self.assertEqual(replacement[0]["positions"], handoff.positions.tolist())
        self.assertLess(audit.start_position_error_rad, 1e-12)
        self.assertLess(audit.start_velocity_error_rad_s, 1e-12)
        self.assertLess(audit.start_acceleration_error_rad_s2, 1e-12)
        self.assertLessEqual(audit.cartesian_peak_speed_m_s, 0.08 + 1e-9)
        self.assertLess(audit.encoded_payload_bytes, 60000)

    def test_splice_applies_slower_speed_only_in_final_zone(self):
        active, _ = generate_timed_joint_trajectory(
            self.start, self.end, cartesian_speed_m_s=0.10,
            point_period_sec=0.1,
        )
        handoff = select_future_handoff(
            active, active_elapsed_sec=0.2, lead_sec=0.5
        )
        replacement_end = self.end + np.asarray([0.04, 0.08, -0.08, 0, 0, 0])
        _, audit = generate_splice_trajectory(
            handoff,
            replacement_end,
            cartesian_speed_m_s=0.10,
            end_zone_distance_m=0.05,
            end_zone_speed_m_s=0.05,
            point_period_sec=0.1,
        )
        self.assertLessEqual(audit.cartesian_peak_speed_m_s, 0.10 + 1e-9)
        self.assertLessEqual(audit.end_zone_peak_speed_m_s, 0.05 + 1e-9)
        self.assertEqual(audit.end_zone_distance_m, 0.05)
        self.assertLessEqual(audit.cartesian_max_overshoot_m, 0.002)
        self.assertLessEqual(audit.cartesian_max_backtrack_step_m, 0.0005)
        self.assertLessEqual(audit.cartesian_max_cross_track_m, 0.025)
        self.assertLessEqual(audit.joint_max_overshoot_rad, 0.020)

    def test_feasible_splice_search_skips_incompatible_accelerated_handoff(self):
        end = self.end + np.asarray([0.01, 0.02, -0.02, 0, 0, 0])
        active = [
            {
                "time_from_start_sec": 1.0,
                "positions": self.end.tolist(),
                "velocities": [0.0] * 6,
                # An extreme inherited acceleration makes the bounded C2
                # replacement impossible within the bridge duration limit.
                "accelerations": [100.0] * 6,
            },
            {
                "time_from_start_sec": 1.1,
                "positions": self.end.tolist(),
                "velocities": [0.0] * 6,
                "accelerations": [0.0] * 6,
            },
        ]
        handoff, points, audit, failures = generate_first_feasible_splice(
            active,
            end,
            earliest_handoff_time_sec=1.0,
            cartesian_speed_m_s=0.08,
            point_period_sec=0.1,
        )
        self.assertEqual(handoff.active_index, 1)
        self.assertEqual(len(failures), 1)
        self.assertGreater(len(points), 1)
        self.assertLessEqual(audit.duration_sec, 15.0)


if __name__ == "__main__":
    unittest.main()
    encode_trajectory_udp_payload,
