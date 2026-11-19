import unittest

from piper_handeye.rolling_speed import (
    RollingSpeedConfig,
    RollingSpeedInput,
    RollingSpeedScheduler,
    plan_receding_approach,
)


class RollingSpeedTests(unittest.TestCase):
    def test_initial_plan_runs_fast_when_compute_fits_before_guard(self):
        plan = plan_receding_approach(0.45, 0.50)
        self.assertAlmostEqual(plan.speed_cap_m_s, 0.15)
        self.assertGreaterEqual(plan.expected_guard_arrival_sec, 0.50)
        self.assertEqual(plan.splice_count_limit, 1)

    def test_initial_plan_slows_for_late_compute(self):
        early = plan_receding_approach(0.45, 0.50)
        late = plan_receding_approach(0.45, 3.00)
        self.assertLess(late.speed_cap_m_s, early.speed_cap_m_s)
        self.assertGreaterEqual(late.expected_guard_arrival_sec, 3.00)

    def test_dynamic_guard_contains_braking_latency_and_uncertainty(self):
        cfg = RollingSpeedConfig(
            coarse_target_uncertainty_m=0.02,
            endpoint_margin_m=0.003,
            control_latency_sec=0.12,
        )
        plan = plan_receding_approach(0.45, 0.50, cfg)
        expected = (
            plan.braking_distance_m
            + plan.latency_distance_m
            + cfg.coarse_target_uncertainty_m
            + cfg.endpoint_margin_m
        )
        self.assertAlmostEqual(plan.dynamic_guard_distance_m, expected)
        self.assertGreater(
            plan.brake_entry_distance_m,
            plan.dynamic_guard_distance_m,
        )

    def test_obstacle_cap_is_applied_to_initial_plan(self):
        plan = plan_receding_approach(
            0.45, 0.10, obstacle_speed_cap_m_s=0.04
        )
        self.assertLessEqual(plan.speed_cap_m_s, 0.04)

    def test_slow_valid_frame_rate_increases_predicted_ready_time(self):
        fast = RollingSpeedScheduler().update(
            RollingSpeedInput(
                0.45, 0.1, False, accepted_samples=0,
                required_samples=6, valid_frame_rate_hz=10.0,
            )
        )
        slow = RollingSpeedScheduler().update(
            RollingSpeedInput(
                0.45, 0.1, False, accepted_samples=0,
                required_samples=6, valid_frame_rate_hz=2.0,
            )
        )
        self.assertGreater(slow.predicted_ready_sec, fast.predicted_ready_sec)

    def test_low_predicted_margin_reduces_speed(self):
        fast = RollingSpeedScheduler()
        slow = RollingSpeedScheduler()
        fast_output = fast.update(
            RollingSpeedInput(
                distance_to_target_m=0.30, dt_sec=0.1, target_ready=False,
                predicted_ready_sec=0.3,
            )
        )
        slow_output = slow.update(
            RollingSpeedInput(
                distance_to_target_m=0.10, dt_sec=0.1, target_ready=False,
                predicted_ready_sec=1.5,
            )
        )
        self.assertGreater(fast_output.raw_speed_cap_m_s, slow_output.raw_speed_cap_m_s)

    def test_ready_target_uses_continuous_contact_envelope(self):
        far_scheduler = RollingSpeedScheduler()
        near_scheduler = RollingSpeedScheduler()
        far_scheduler.reset(0.10)
        near_scheduler.reset(0.10)
        far = far_scheduler.update(
            RollingSpeedInput(0.20, 0.1, True)
        )
        near = near_scheduler.update(
            RollingSpeedInput(0.01, 0.1, True)
        )
        self.assertGreater(far.contact_speed_cap_m_s, near.contact_speed_cap_m_s)
        self.assertLess(near.commanded_speed_m_s, far.commanded_speed_m_s)

    def test_stale_target_causes_controlled_deceleration(self):
        scheduler = RollingSpeedScheduler()
        scheduler.reset(0.10)
        output = scheduler.update(
            RollingSpeedInput(
                0.20, 0.1, False, latest_target_age_sec=0.5,
                predicted_ready_sec=1.0,
            )
        )
        self.assertEqual(output.mode, "decelerate_unknown")
        self.assertLess(output.commanded_speed_m_s, 0.10)
        self.assertGreater(output.commanded_speed_m_s, 0.0)

    def test_no_discontinuity_at_ten_centimeters(self):
        scheduler = RollingSpeedScheduler(
            RollingSpeedConfig(max_jerk_m_s3=5.0)
        )
        speeds = []
        for distance in (0.11, 0.105, 0.10, 0.095, 0.09):
            output = scheduler.update(
                RollingSpeedInput(
                    distance, 0.1, False, predicted_ready_sec=0.5,
                )
            )
            speeds.append(output.commanded_speed_m_s)
        self.assertTrue(
            all(abs(current - following) < 0.04 for current, following in zip(speeds, speeds[1:]))
        )

    def test_jerk_is_bounded(self):
        cfg = RollingSpeedConfig(max_jerk_m_s3=1.0)
        scheduler = RollingSpeedScheduler(cfg)
        previous_acceleration = 0.0
        for _ in range(10):
            output = scheduler.update(
                RollingSpeedInput(0.50, 0.1, True)
            )
            self.assertLessEqual(
                abs(output.commanded_acceleration_m_s2 - previous_acceleration),
                cfg.max_jerk_m_s3 * 0.1 + 1e-12,
            )
            previous_acceleration = output.commanded_acceleration_m_s2

    def test_jerk_limited_scheduler_stops_at_terminal_guard(self):
        scheduler = RollingSpeedScheduler()
        distance = 0.30
        stopped = False
        moved_fast = False
        for _ in range(200):
            output = scheduler.update(
                RollingSpeedInput(distance, 0.05, True)
            )
            distance = max(
                0.0, distance - output.commanded_speed_m_s * 0.05
            )
            moved_fast = moved_fast or output.commanded_speed_m_s >= 0.05
            if moved_fast and output.commanded_speed_m_s <= 0.005:
                stopped = True
                break
        self.assertTrue(stopped)
        self.assertGreaterEqual(distance, 0.003)
        self.assertLessEqual(distance, 0.02)


if __name__ == "__main__":
    unittest.main()
