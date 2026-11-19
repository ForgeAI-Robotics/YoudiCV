import copy
import unittest

import numpy as np

from piper_handeye.rolling_horizon import (
    RollingHorizonBuffer,
    RollingHorizonConfig,
    split_rolling_chunks,
)


class RollingHorizonTests(unittest.TestCase):
    def setUp(self):
        self.start = np.asarray([0.23, 1.63, -0.98, 0.39, -0.55, -0.03])

    def point(self, timestamp, offset=0.0, velocity=0.0):
        positions = self.start.copy()
        positions[0] += offset
        velocities = np.zeros(6)
        velocities[0] = velocity
        return {
            "time_from_start_sec": float(timestamp),
            "positions": positions.tolist(),
            "velocities": velocities.tolist(),
            "accelerations": np.zeros(6).tolist(),
        }

    def first_chunk(self):
        return [self.point(0.0), self.point(0.5, 0.05, 0.1), self.point(1.0, 0.1)]

    def test_contiguous_chunks_and_overlap_are_accepted(self):
        buffer = RollingHorizonBuffer("test", opened_monotonic=10.0)
        first = buffer.append(0, self.first_chunk(), now_monotonic=10.1)
        second = buffer.append(
            1,
            [self.point(1.0, 0.1), self.point(1.5, 0.15, 0.1)],
            now_monotonic=10.2,
        )
        self.assertEqual(first.accepted_new_points, 3)
        self.assertEqual(second.accepted_new_points, 1)
        self.assertEqual(len(buffer.points), 4)
        self.assertEqual(buffer.expected_sequence, 2)

    def test_identical_retry_is_idempotent(self):
        buffer = RollingHorizonBuffer("test")
        chunk = self.first_chunk()
        buffer.append(0, chunk)
        retry = buffer.append(0, copy.deepcopy(chunk))
        self.assertTrue(retry.duplicate)
        self.assertEqual(len(buffer.points), 3)

    def test_changed_retry_is_rejected(self):
        buffer = RollingHorizonBuffer("test")
        chunk = self.first_chunk()
        buffer.append(0, chunk)
        changed = copy.deepcopy(chunk)
        changed[-1]["positions"][0] += 0.001
        with self.assertRaisesRegex(ValueError, "different content"):
            buffer.append(0, changed)

    def test_gap_and_discontinuity_are_rejected(self):
        buffer = RollingHorizonBuffer("test")
        buffer.append(0, self.first_chunk())
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            buffer.append(2, [self.point(1.0, 0.1), self.point(1.5, 0.15)])
        broken = [self.point(1.0, 0.101), self.point(1.5, 0.15)]
        with self.assertRaisesRegex(ValueError, "positions discontinuity"):
            buffer.append(1, broken)

    def test_status_reports_ready_low_stale_and_underrun(self):
        config = RollingHorizonConfig(append_timeout_sec=0.5)
        buffer = RollingHorizonBuffer("test", config=config, opened_monotonic=10.0)
        buffer.append(0, self.first_chunk(), now_monotonic=10.1)
        ready = buffer.status(now_monotonic=10.2, playhead_sec=0.2, started=True)
        self.assertTrue(ready["ready"])
        self.assertFalse(ready["low_horizon"])
        low = buffer.status(now_monotonic=10.8, playhead_sec=0.8, started=True)
        self.assertTrue(low["low_horizon"])
        self.assertTrue(low["append_stale"])
        underrun = buffer.status(now_monotonic=11.2, playhead_sec=1.0, started=True)
        self.assertTrue(underrun["underrun"])

    def test_joint_limits_and_velocity_are_rejected(self):
        buffer = RollingHorizonBuffer("test")
        bad_limit = self.first_chunk()
        bad_limit[1]["positions"][0] = 3.0
        with self.assertRaisesRegex(ValueError, "position limits"):
            buffer.append(0, bad_limit)
        bad_speed = self.first_chunk()
        bad_speed[1]["velocities"][0] = 0.5
        with self.assertRaisesRegex(ValueError, "declared velocity"):
            buffer.append(0, bad_speed)

    def test_sealed_and_aborted_sessions_cannot_append(self):
        sealed = RollingHorizonBuffer("sealed")
        sealed.append(0, self.first_chunk())
        sealed.seal()
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            sealed.append(1, [self.point(1.0, 0.1), self.point(1.5, 0.15)])
        aborted = RollingHorizonBuffer("aborted")
        aborted.abort()
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            aborted.append(0, self.first_chunk())

    def test_chunk_split_repeats_exact_tail_and_covers_timeline(self):
        points = [self.point(index * 0.1, index * 0.005) for index in range(16)]
        chunks = split_rolling_chunks(
            points, initial_horizon_sec=0.6, append_extension_sec=0.2
        )
        self.assertGreater(len(chunks), 1)
        self.assertGreaterEqual(chunks[0][-1]["time_from_start_sec"], 0.6)
        for previous, following in zip(chunks, chunks[1:]):
            self.assertEqual(previous[-1], following[0])
        reconstructed = chunks[0] + [point for chunk in chunks[1:] for point in chunk[1:]]
        self.assertEqual(reconstructed, points)

    def test_future_tail_can_be_revised_with_continuous_safe_stop(self):
        buffer = RollingHorizonBuffer("revision")
        initial = [self.point(index * 0.2, index * 0.01) for index in range(6)]
        buffer.append(0, initial)
        replacement = [
            copy.deepcopy(initial[3]),
            self.point(0.8, 0.07),
            self.point(1.0, 0.08),
        ]
        result = buffer.revise_tail(1, replacement, playhead_sec=0.2)
        self.assertAlmostEqual(result.anchor_time_sec, 0.6)
        self.assertEqual(result.removed_points, 2)
        self.assertEqual(buffer.points[-1], replacement[-1])

    def test_revision_inside_frozen_horizon_is_rejected(self):
        buffer = RollingHorizonBuffer("frozen")
        initial = [self.point(index * 0.2, index * 0.01) for index in range(6)]
        buffer.append(0, initial)
        replacement = [copy.deepcopy(initial[2]), self.point(0.8, 0.05)]
        with self.assertRaisesRegex(ValueError, "frozen execution horizon"):
            buffer.revise_tail(1, replacement, playhead_sec=0.2)

    def test_revision_requires_zero_velocity_terminal_fallback(self):
        buffer = RollingHorizonBuffer("terminal")
        initial = [self.point(index * 0.2, index * 0.01) for index in range(6)]
        buffer.append(0, initial)
        replacement = [copy.deepcopy(initial[3]), self.point(0.9, 0.07, velocity=0.1)]
        with self.assertRaisesRegex(ValueError, "zero velocity"):
            buffer.revise_tail(1, replacement, playhead_sec=0.2)


if __name__ == "__main__":
    unittest.main()
