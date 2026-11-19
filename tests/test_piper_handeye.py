import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from piper_handeye.geometry import (
    average_transforms,
    matrix_to_vector,
    pose_to_transform,
    transform,
    vector_to_matrix,
)
from piper_handeye.kinematics import (
    _rotation_matrix_to_rotvec,
    elbow_position,
    elbow_positions_batch,
    forward_kinematics,
    forward_kinematics_batch,
    inverse_kinematics,
    relative_target,
)
from piper_handeye.aruco_pose import marker_to_facing_tcp, pose_error
from piper_handeye.robot import PiperUdpRobot, extract_json


class GeometryTests(unittest.TestCase):
    def test_vector_round_trip(self):
        source = transform(Rotation.from_euler("xyz", [0.2, -0.4, 0.1]).as_matrix(), [1, 2, 3])
        np.testing.assert_allclose(vector_to_matrix(matrix_to_vector(source)), source, atol=1e-12)

    def test_pose_quaternion_order(self):
        source = {"position": {"x": 1, "y": 2, "z": 3},
                  "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}
        np.testing.assert_allclose(pose_to_transform(source), transform(np.eye(3), [1, 2, 3]))

    def test_average_transforms_handles_quaternion_sign(self):
        values = [transform(Rotation.from_euler("z", 0.1 + d).as_matrix(), [d, 0, 0])
                  for d in (-0.01, 0, 0.01)]
        result = average_transforms(values)
        self.assertAlmostEqual(result[0, 3], 0.0, places=12)
        self.assertAlmostEqual(Rotation.from_matrix(result[:3, :3]).as_euler("xyz")[2], 0.1, places=6)


class ProtocolTests(unittest.TestCase):
    def make_robot(self, mode="ascii"):
        return PiperUdpRobot({"controller_ip": "127.0.0.1", "local_ip": "127.0.0.1",
                              "header_mode": mode})

    def test_extract_prefixed_json(self):
        self.assertEqual(extract_json(b"system_state" + json.dumps({"a": 1}).encode()), {"a": 1})

    def test_ascii_packet_layout(self):
        robot = self.make_robot()
        packet = robot._packet({"task_id": "1"})
        self.assertEqual(packet[:4], b"1601")
        self.assertIn(b"bt_control", packet)

    def test_binary_packet_layout(self):
        packet = self.make_robot("binary")._packet({"task_id": "1"})
        self.assertEqual(packet[:2], b"\x16\x01")

    def test_joint_limit_rejection_before_network(self):
        robot = self.make_robot()
        with self.assertRaises(ValueError):
            robot.move_joints([3.0, 1.0, -1.0, 0, 0, 0])


class KinematicsTests(unittest.TestCase):
    def test_fast_rotation_log_matches_scipy_including_near_pi(self):
        rotations = Rotation.from_euler(
            "xyz",
            [
                [0.0, 0.0, 0.0],
                [0.3, -0.4, 0.7],
                [-1.2, 0.5, 2.0],
                [np.deg2rad(179.9999), 0.0, 0.0],
                [np.pi, 0.0, 0.0],
            ],
        )
        for matrix in rotations.as_matrix():
            expected = Rotation.from_matrix(matrix).as_rotvec()
            actual = _rotation_matrix_to_rotvec(matrix)
            np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_batch_elbow_fk_matches_scalar_fk(self):
        values = np.asarray(
            [
                [0.04, 0.105, 0.0, 0.051, 0.204, 0.0],
                [0.12, 0.30, -0.20, 0.10, 0.40, -0.10],
                [-0.20, 0.55, -0.45, -0.15, 0.50, 0.20],
            ]
        )
        expected = np.asarray([elbow_position(value) for value in values])
        np.testing.assert_allclose(
            elbow_positions_batch(values), expected, atol=1e-12
        )

    def test_batch_fk_matches_scalar_fk(self):
        values = np.asarray(
            [
                [0.0, 1.265, -1.134, 0.0, 0.736, 0.0],
                [0.23, 1.63, -0.98, 0.39, -0.55, -0.03],
                [-0.4, 2.0, -1.8, -0.5, 0.8, 1.0],
            ]
        )
        expected = np.asarray([forward_kinematics(value) for value in values])
        np.testing.assert_allclose(
            forward_kinematics_batch(values), expected, atol=1e-12
        )

    def test_ik_recovers_nearby_cartesian_target(self):
        seed = np.array([0.0, 1.265, -1.134, 0.0, 0.736, 0.0])
        target = relative_target(
            forward_kinematics(seed),
            translation_base=(0.01, -0.01, 0.005),
            rotation_tool=(0.04, -0.03, 0.02),
        )
        solved, position_error, angle_error = inverse_kinematics(target, seed)
        self.assertTrue(np.isfinite(solved).all())
        self.assertLess(position_error, 1e-4)
        self.assertLess(angle_error, 1e-4)

    def test_ik_hard_locks_base_z_joint_during_rise(self):
        seed = np.array([0.04, 0.55, -0.45, 0.10, 0.40, -0.10])
        goal = seed + np.array([0.0, 0.04, -0.03, 0.02, -0.01, 0.02])
        solved, position_error, angle_error = inverse_kinematics(
            forward_kinematics(goal),
            seed,
            fixed_joint_indices=(0,),
        )
        self.assertEqual(solved[0], seed[0])
        self.assertLess(position_error, 1e-4)
        self.assertLess(angle_error, 1e-4)

    def test_ik_rejects_invalid_fixed_joint_index(self):
        with self.assertRaises(ValueError):
            inverse_kinematics(
                forward_kinematics(np.zeros(6)),
                np.zeros(6),
                fixed_joint_indices=(6,),
            )

    def test_marker_facing_tcp_has_safe_offset_and_opposite_z(self):
        value = marker_to_facing_tcp(0.12)
        np.testing.assert_allclose(value[:3, 3], [0, 0, 0.12], atol=1e-12)
        np.testing.assert_allclose(value[:3, :3] @ [0, 0, 1], [0, 0, -1], atol=1e-12)
        position, angle = pose_error(value, value)
        self.assertAlmostEqual(position, 0.0)
        self.assertAlmostEqual(angle, 0.0)

    def test_marker_facing_tcp_rejects_contact_distance(self):
        with self.assertRaises(ValueError):
            marker_to_facing_tcp(0.0)


if __name__ == "__main__":
    unittest.main()
