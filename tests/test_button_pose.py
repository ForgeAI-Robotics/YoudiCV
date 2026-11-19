import unittest

import numpy as np

from piper_handeye.button_pose import (
    ButtonDetection,
    estimate_button_camera_pose,
    select_button,
)


class ButtonPoseTests(unittest.TestCase):
    def setUp(self):
        self.camera = np.array(
            [[400.0, 0.0, 320.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]]
        )
        self.detection = ButtonDetection(7, "3", 0.91, (280.0, 140.0, 360.0, 220.0))

    def test_center_pixel(self):
        self.assertEqual(self.detection.center_pixel, (320.0, 180.0))

    def test_select_button_filters_label_and_confidence(self):
        chosen = select_button(
            [
                ButtonDetection(1, "2", 0.99, (0, 0, 10, 10)),
                self.detection,
                ButtonDetection(7, "3", 0.65, (10, 10, 20, 20)),
            ],
            "3",
            0.7,
        )
        self.assertEqual(chosen, self.detection)

    def test_frontoparallel_panel_outputs_camera_point_and_outward_normal(self):
        depth = np.full((360, 640), 0.50, dtype=np.float32)
        pose = estimate_button_camera_pose(
            self.detection, depth, self.camera, min_plane_pixels=100
        )
        np.testing.assert_allclose(pose.p_camera, [0.0, 0.0, 0.50], atol=1e-6)
        np.testing.assert_allclose(pose.normal_camera, [0.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(pose.camera_to_button[:3, 3], pose.p_camera)
        np.testing.assert_allclose(pose.camera_to_button[:3, 2], pose.normal_camera)
        self.assertLess(pose.plane_rms_m, 1e-7)

    def test_sloped_panel_ray_plane_intersection(self):
        height, width = 360, 640
        ys, xs = np.indices((height, width), dtype=float)
        rays_x = (xs - 320.0) / 400.0
        # Plane z = 0.6 + 0.08*x. With x = ray_x*z, solve for z per pixel.
        depth = 0.6 / (1.0 - 0.08 * rays_x)
        shifted = ButtonDetection(7, "3", 0.95, (360.0, 140.0, 440.0, 220.0))
        pose = estimate_button_camera_pose(
            shifted, depth, self.camera, min_plane_pixels=100
        )
        expected_z = 0.6 / (1.0 - 0.08 * 0.2)
        np.testing.assert_allclose(pose.p_camera, [0.2 * expected_z, 0.0, expected_z], atol=2e-5)
        self.assertLess(pose.plane_rms_m, 1e-7)
        self.assertLess(float(pose.normal_camera @ pose.p_camera), 0.0)

    def test_rejects_missing_panel_depth(self):
        depth = np.zeros((360, 640), dtype=np.float32)
        with self.assertRaises(RuntimeError):
            estimate_button_camera_pose(self.detection, depth, self.camera)

    def test_independent_button_rings_override_biased_target_depth(self):
        depth = np.full((360, 640), 0.50, dtype=np.float32)
        # Corrupt the complete target neighbourhood by 40 mm.  A target-local
        # fit alone would confidently return the wrong contact depth.
        depth[120:240, 260:380] = 0.54
        supports = [
            ButtonDetection(1, "1", 0.95, (60.0, 60.0, 120.0, 120.0)),
            ButtonDetection(2, "2", 0.95, (500.0, 60.0, 560.0, 120.0)),
            ButtonDetection(3, "4", 0.95, (60.0, 260.0, 120.0, 320.0)),
        ]
        pose = estimate_button_camera_pose(
            self.detection,
            depth,
            self.camera,
            min_plane_pixels=100,
            panel_support_detections=[self.detection, *supports],
            min_panel_support_detections=3,
        )
        self.assertAlmostEqual(pose.p_camera[2], 0.50, places=5)
        self.assertAlmostEqual(pose.local_global_depth_delta_m, 0.04, places=5)
        self.assertEqual(pose.panel_support_detections, 3)

    def test_rejects_contact_depth_without_enough_independent_supports(self):
        depth = np.full((360, 640), 0.50, dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "independent panel button supports"):
            estimate_button_camera_pose(
                self.detection,
                depth,
                self.camera,
                min_plane_pixels=100,
                panel_support_detections=[self.detection],
                min_panel_support_detections=3,
            )


if __name__ == "__main__":
    unittest.main()
