import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from piper_handeye.visual_search import (
    button_grid_position,
    centered_camera_translation_base,
    closer_camera_tcp_target,
    is_button_centered,
    pixel_centered_camera_tcp_target,
    plan_upward_search,
    predict_button_center_from_grid,
    predicted_pixel_aim_tcp_target,
    raised_and_tilted_camera_tcp_target,
    raised_tcp_target,
)


class VisualSearchTests(unittest.TestCase):
    def test_two_column_number_pattern(self):
        self.assertEqual(button_grid_position("1").column, 0)
        self.assertEqual(button_grid_position("8").column, 0)
        self.assertEqual(button_grid_position("9").column, 1)
        self.assertEqual(button_grid_position("5").column, 0)
        self.assertEqual(button_grid_position("12").column, 1)
        self.assertEqual(button_grid_position("16").row, 7)
        self.assertEqual(button_grid_position("16").column, 1)

    def test_invisible_higher_button_searches_only_up(self):
        plan = plan_upward_search("5", ["1", "2", "3", "8", "9", "10"])
        self.assertEqual(plan.direction, "up_with_wrist")
        self.assertGreater(plan.predicted_lift_m, 0.0)
        self.assertTrue(plan.lift_offsets_m)
        self.assertEqual(tuple(sorted(plan.lift_offsets_m)), plan.lift_offsets_m)
        self.assertTrue(all(value > 0.0 for value in plan.lift_offsets_m))
        self.assertEqual(len(plan.camera_tilt_offsets_deg), len(plan.lift_offsets_m))

    def test_missing_slot_in_visible_span_looks_up_from_lowest_origin(self):
        plan = plan_upward_search("8", ["1", "2", "3", "9", "10"])
        self.assertEqual(plan.direction, "up_with_wrist")
        self.assertEqual(plan.predicted_lift_m, 0.24)
        self.assertTrue(all(value > 0.0 for value in plan.lift_offsets_m))

    def test_button_16_searches_up_in_second_column(self):
        plan = plan_upward_search("16", ["1", "2", "3", "9", "10"])
        self.assertEqual(plan.direction, "up_with_wrist")
        self.assertTrue(plan.lift_offsets_m)
        self.assertGreater(plan.predicted_lift_m, 0.0)
        self.assertLessEqual(max(plan.camera_tilt_offsets_deg), 30.0)

    def test_visible_button_needs_no_search_but_still_needs_pointing(self):
        plan = plan_upward_search("2", ["1", "2", "3"])
        self.assertEqual(plan.direction, "already_visible")
        self.assertEqual(plan.lift_offsets_m, ())

    def test_centering_translation_preserves_optical_depth(self):
        camera = np.eye(4)
        point = np.asarray([0.04, -0.03, 0.45])
        translation = centered_camera_translation_base(camera, point)
        self.assertTrue(np.allclose(translation, [0.04, -0.03, 0.0]))
        moved = camera.copy()
        moved[:3, 3] += translation
        centered = (np.linalg.inv(moved) @ np.r_[point, 1.0])[:3]
        self.assertTrue(np.allclose(centered, [0.0, 0.0, 0.45]))

    def test_center_tolerance_and_origin_is_lowest(self):
        self.assertTrue(is_button_centered([300, 160, 340, 200], [640, 360]))
        self.assertFalse(is_button_centered([500, 160, 540, 200], [640, 360]))
        origin = np.eye(4)
        raised = raised_tcp_target(origin, 0.08)
        self.assertEqual(raised[2, 3], 0.08)
        with self.assertRaises(ValueError):
            raised_tcp_target(origin, -0.01)

    def test_wrist_search_raises_camera_and_tilts_optical_axis(self):
        tcp = np.eye(4)
        handeye = np.eye(4)
        handeye[:3, 3] = [0.02, -0.01, 0.10]
        target_tcp = raised_and_tilted_camera_tcp_target(
            tcp, handeye, 0.04, 12.0
        )
        target_camera = target_tcp @ handeye
        self.assertAlmostEqual(target_camera[2, 3], 0.14)
        expected = Rotation.from_euler("x", 12.0, degrees=True).as_matrix()
        np.testing.assert_allclose(target_camera[:3, :3], expected, atol=1e-12)

    def test_transient_pixel_detection_uses_wrist_to_center(self):
        tcp = np.eye(4)
        handeye = np.eye(4)
        matrix = np.asarray([[1800.0, 0.0, 960.0], [0.0, 1800.0, 540.0], [0.0, 0.0, 1.0]])
        target_tcp = pixel_centered_camera_tcp_target(
            tcp, handeye, [500, 40, 540, 80], [640, 360], matrix, [1920, 1080]
        )
        self.assertFalse(np.allclose(target_tcp[:3, :3], np.eye(3)))
        self.assertTrue(np.allclose(target_tcp[:3, 3], [0.0, 0.0, 0.0]))

    def test_live_columns_predict_8_upward_from_1_2_3_not_down_from_9(self):
        detections = []
        observed = {
            "1": (380.0, 184.0),
            "2": (380.0, 142.0),
            "3": (380.0, 100.0),
            "9": (440.0, 142.0),
            "10": (440.0, 100.0),
        }
        for label, (cx, cy) in observed.items():
            detections.append({
                "label": label,
                "bbox_xyxy": [cx - 10, cy - 10, cx + 10, cy + 10],
            })
        detections.append({
            "label": "B1", "bbox_xyxy": [370, 216, 390, 236]
        })
        prediction = predict_button_center_from_grid("8", detections)
        self.assertEqual(prediction.layout, "camera_inferred_columns")
        self.assertEqual(prediction.target_column, 0)
        self.assertEqual(prediction.anchor_label, "3")
        self.assertAlmostEqual(prediction.rows_above_visible, 5.0)
        np.testing.assert_allclose(prediction.center_xy, [380.0, -110.0])
        self.assertAlmostEqual(prediction.row_gap_px, -42.0)
        self.assertAlmostEqual(prediction.column_gap_px, 60.0)
        self.assertLess(prediction.fit_rms_px, 1e-9)

    def test_live_columns_predict_16_from_nearest_upward_sequence(self):
        detections = []
        for label, cx, cy in (
            ("1", 380, 184), ("2", 380, 142), ("3", 380, 100),
            ("9", 440, 142), ("10", 440, 100),
        ):
            detections.append({
                "label": label,
                "bbox_xyxy": [cx - 10, cy - 10, cx + 10, cy + 10],
            })
        prediction = predict_button_center_from_grid("16", detections)
        self.assertEqual(prediction.target_column, 1)
        self.assertEqual(prediction.anchor_label, "10")
        self.assertAlmostEqual(prediction.rows_above_visible, 6.0)
        np.testing.assert_allclose(prediction.center_xy, [440.0, -152.0])

    def test_sideways_incrementing_pattern_selects_row_major_layout(self):
        detections = []
        for label in ("1", "2", "3", "4", "5", "6"):
            grid = button_grid_position(label, "row_major")
            cx = 380.0 + 60.0 * grid.column
            cy = 100.0 - 42.0 * grid.row
            detections.append({
                "label": label,
                "bbox_xyxy": [cx - 10, cy - 10, cx + 10, cy + 10],
            })
        prediction = predict_button_center_from_grid("8", detections)
        self.assertEqual(prediction.layout, "camera_inferred_columns")
        self.assertEqual(prediction.target_column, 1)
        np.testing.assert_allclose(prediction.center_xy, [440.0, -26.0])
        self.assertLess(prediction.fit_rms_px, 1e-9)

    def test_one_shot_aim_points_at_predicted_row(self):
        tcp = np.eye(4)
        handeye = np.eye(4)
        matrix = np.asarray([
            [1800.0, 0.0, 960.0],
            [0.0, 1800.0, 540.0],
            [0.0, 0.0, 1.0],
        ])
        desired_tcp = predicted_pixel_aim_tcp_target(
            tcp, handeye, [320.0, 0.0], matrix, [1920, 1080],
            [640, 360], 0.50, 0.0,
        )
        scaled = matrix.copy()
        scaled[0, :] /= 3.0
        scaled[1, :] /= 3.0
        ray = np.asarray([
            (320.0 - scaled[0, 2]) / scaled[0, 0],
            (0.0 - scaled[1, 2]) / scaled[1, 1],
            1.0,
        ])
        target = np.r_[ray * 0.50, 1.0]
        target_new = np.linalg.inv(desired_tcp) @ target
        self.assertAlmostEqual(target_new[1], 0.0, places=10)
        self.assertGreater(target_new[2], 0.0)

    def test_one_shot_aim_does_not_yaw_for_other_column(self):
        tcp = np.eye(4)
        handeye = np.eye(4)
        matrix = np.asarray([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 180.0],
            [0.0, 0.0, 1.0],
        ])
        desired_tcp = predicted_pixel_aim_tcp_target(
            tcp, handeye, [420.0, 180.0], matrix, [640, 360],
            [640, 360], 0.40, 0.04,
        )
        # A horizontal prediction does not add yaw.  Only the lift-induced
        # row pitch is allowed, exactly as in the proven 12/15/16 search.
        np.testing.assert_allclose(desired_tcp[:3, :3][:, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(desired_tcp[:3, 3], [0.0, 0.0, 0.04])

    def test_close_up_preserves_camera_orientation_and_moves_forward(self):
        tcp = np.eye(4)
        handeye = np.eye(4)
        target_tcp = closer_camera_tcp_target(tcp, handeye, 0.08)
        target_camera = target_tcp @ handeye
        np.testing.assert_allclose(target_camera[:3, :3], np.eye(3))
        np.testing.assert_allclose(target_camera[:3, 3], [0.0, 0.0, 0.08])

if __name__ == "__main__":
    unittest.main()
