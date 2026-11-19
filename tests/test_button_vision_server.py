import tempfile
import unittest
import json
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import yaml

from run_button_vision_server import VisionService, load_handeye, resolve_path, scaled_camera_matrix


def check_reference_timing_logged_without_hardware():
    service = VisionService.__new__(VisionService)
    service.args = SimpleNamespace(token="test")
    service.camera = MagicMock()
    service.camera.recent_frames.return_value = [
        SimpleNamespace(depth_m=np.ones((4, 4)), sequence=n) for n in range(25)
    ]
    service.safety = MagicMock()
    service.safety.capture_matrix_reference.return_value = {"reference_timing_sec": {"total": .1}}
    output = io.StringIO()
    with redirect_stdout(output):
        result = service.dispatch({"token": "test", "command": "safety_capture_matrix_reference"})
    assert result["ok"]
    assert "total" in result["request_timing_sec"]
    assert json.loads(output.getvalue())["status"] == "complete"
    service.safety.capture_matrix_reference.side_effect = RuntimeError("test failure")
    try:
        output = io.StringIO()
        with redirect_stdout(output):
            service.dispatch({"token": "test", "command": "safety_capture_matrix_reference"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("reference failure must propagate")
    assert json.loads(output.getvalue())["status"] == "failed"


class ButtonVisionServerUtilityTests(unittest.TestCase):
    def test_recognition_snapshot_draws_inference_frame_without_redetecting(self):
        from piper_handeye.button_pose import ButtonDetection
        service = VisionService.__new__(VisionService)
        service.args = SimpleNamespace(token="test")
        service.camera = MagicMock()
        service.detector = MagicMock()
        frame = SimpleNamespace(sequence=10, received_unix=123.,
                                color=np.zeros((36, 64, 3), dtype=np.uint8))
        detection = ButtonDetection(0, "8", .9, (4., 5., 15., 20.))
        sample = SimpleNamespace(sequence=10, frame=frame, target_label="8",
                                 detections=(detection,), detection_enhancement={})
        service.perception = MagicMock()
        service.perception.latest_sample.return_value = sample
        with tempfile.TemporaryDirectory() as directory, patch("run_button_vision_server.ROOT", Path(directory)):
            result = service.dispatch({"token": "test", "command": "recognition_snapshot"})
            self.assertTrue(Path(result["raw_yolo_path"]).is_file())
            self.assertTrue(Path(result["input_path"]).is_file())
            self.assertEqual(result["frame_sequence"], 10)
            self.assertNotIn("enhanced_yolo_path", result)
        service.camera.latest_frame.assert_not_called()
        service.detector.detect.assert_not_called()

    def test_recognition_snapshot_keeps_failed_enhancement_candidates(self):
        service = VisionService.__new__(VisionService)
        service.args = SimpleNamespace(token="test")
        frame = SimpleNamespace(sequence=10, received_unix=123.,
                                color=np.zeros((36, 64, 3), dtype=np.uint8))
        sample = SimpleNamespace(sequence=10, frame=frame, target_label="8",
                                 detections=(), detection_enhancement={
                                     "raw_detections": [], "enhanced_detections": [], "accepted": False})
        service.perception = MagicMock()
        service.perception.latest_sample.return_value = sample
        with tempfile.TemporaryDirectory() as directory, patch("run_button_vision_server.ROOT", Path(directory)):
            result = service.dispatch({"token": "test", "command": "recognition_snapshot"})
            self.assertTrue(Path(result["enhanced_yolo_path"]).is_file())

    def test_failed_down_retry_keeps_same_frame_candidates_without_redetecting(self):
        service = VisionService.__new__(VisionService)
        service.args = SimpleNamespace(token="test")
        service.camera = MagicMock()
        service.detector = MagicMock()
        frame = SimpleNamespace(sequence=10, received_unix=123.,
                                color=np.zeros((360, 640, 3), dtype=np.uint8))
        candidates = [dict(label='M', confidence=.90, bbox_xyxy=[190, 40, 225, 75])]
        audit = dict(source='raw', algorithm='call_button_local_scale', accepted=False,
                     local_candidates=candidates, down_recovery=dict(target_accepted=False))
        service.perception = MagicMock()
        service.perception.latest_sample.return_value = SimpleNamespace(
            sequence=10, frame=frame, target_label='down', detections=(), detection_enhancement=audit)
        with tempfile.TemporaryDirectory() as directory, patch('run_button_vision_server.ROOT', Path(directory)):
            result = service.dispatch({'token': 'test', 'command': 'recognition_snapshot'})
            self.assertTrue(Path(result['tile_yolo_path']).is_file())
            self.assertEqual(result['frame_sequence'], 10)
            self.assertFalse(result['detection_enhancement']['down_recovery']['target_accepted'])
        service.camera.latest_frame.assert_not_called()
        service.detector.detect.assert_not_called()

    def test_light_snapshot_is_rgb_only_and_honors_sequence_barrier(self):
        service = VisionService.__new__(VisionService)
        service.args = SimpleNamespace(token="test")
        frame = SimpleNamespace(
            sequence=12,
            received_unix=100.5,
            color=np.zeros((36, 64, 3), dtype=np.uint8),
        )
        service.camera = MagicMock()
        service.camera.latest_frame.return_value = SimpleNamespace(sequence=99)
        service.perception = MagicMock()
        service.perception.target_label = "8"
        from piper_handeye.button_pose import ButtonDetection
        detection = ButtonDetection(0, "8", .95, (10., 10., 30., 30.))
        service.perception.latest_light_sample.return_value = SimpleNamespace(
            sequence=12, frame=frame, pose=None, detections=(detection,))
        base = {"token": "test", "command": "light_snapshot", "target_label": "8"}
        stale = service.dispatch(dict(base, after_sequence=12))
        self.assertFalse(stale["fresh"])
        with patch("run_button_vision_server.cv2.imwrite", return_value=True) as save, patch(
            "run_button_vision_server.prune_capture_groups"
        ):
            result = service.dispatch(dict(base, after_sequence=11))
        self.assertTrue(result["fresh"])
        self.assertEqual(result["frame_sequence"], 12)
        self.assertEqual(result["image_size"], [64, 36])
        self.assertTrue(result["path"].endswith(".jpg"))
        save.assert_called_once()
        self.assertEqual(result["target_detection"]["frame_sequence"], 12)
        service.camera.latest_frame.assert_not_called()
        service.perception.latest_valid.assert_not_called()
        # Latest inference has no target: must not reuse an older successful box.
        service.perception.latest_light_sample.return_value.detections = ()
        with patch("run_button_vision_server.cv2.imwrite", return_value=True), patch("run_button_vision_server.prune_capture_groups"):
            missing = service.dispatch(dict(base, after_sequence=11))
        self.assertNotIn("target_detection", missing)
        service.perception.latest_light_sample.return_value = None
        self.assertFalse(service.dispatch(dict(base, after_sequence=-1))["fresh"])

    def test_latest_batch_preserves_barrier_and_legacy_response(self):
        from collections import deque
        import threading
        from piper_handeye.live_button import LiveButtonPerception
        service = VisionService.__new__(VisionService)
        service.args = SimpleNamespace(token="test")
        perception = LiveButtonPerception.__new__(LiveButtonPerception)
        perception.target_label = "15"
        perception._condition = threading.Condition()
        perception._samples = deque([
            SimpleNamespace(sequence=n, target_label=label, pose=pose, frame=SimpleNamespace(received_unix=100.0))
            for n, label, pose in [(1,"15",True), (2,"15",True),
                                  (3,"14",True), (4,"15",None),
                                  (5,"15",True), (6,"15",True)]
        ])
        service.perception = perception
        with patch('run_button_vision_server.pose_payload', side_effect=lambda x: {"sequence": x.sequence}), patch('run_button_vision_server.realtime_payload', return_value={}), patch('run_button_vision_server.time.time', return_value=100.2):
            base = {"command": "latest", "token": "test", "target_label": "15"}
            result = service.dispatch(dict(base, after_sequence=2))
            self.assertEqual(result["samples"], [{"sequence":5}, {"sequence":6}])
            self.assertEqual(service.dispatch(dict(base, after_sequence=6))["samples"], [])
            self.assertNotIn("samples", service.dispatch(base))
            self.assertEqual(result["sample"], {"sequence":6})
            self.assertNotIn("realtime", service.dispatch(dict(base, include_realtime=False)))
            self.assertEqual(service.dispatch(dict(base, after_sequence=2, max_batch_age_sec=.1))["samples"], [])
            self.assertEqual(len(service.dispatch(dict(base, after_sequence=2, max_batch_age_sec=.35))["samples"]), 2)

    def test_reference_timing(self):
        check_reference_timing_logged_without_hardware()

    def test_resolve_path_is_relative_to_config(self):
        config = Path("/tmp/example/config.yaml")
        self.assertEqual(resolve_path("models/model.pt", config),
                         Path("/tmp/example/models/model.pt"))

    def test_scaled_camera_matrix_preserves_aspect_ratio(self):
        matrix = np.asarray([[1000.0, 0.0, 960.0],
                             [0.0, 1000.0, 540.0],
                             [0.0, 0.0, 1.0]])
        scaled = scaled_camera_matrix(matrix, [1920, 1080], [640, 360])
        np.testing.assert_allclose(
            scaled,
            [[1000.0 / 3.0, 0.0, 320.0],
             [0.0, 1000.0 / 3.0, 180.0],
             [0.0, 0.0, 1.0]],
        )

    def test_load_handeye_reads_runtime_calibration(self):
        payload = {
            "gripper_to_camera": np.eye(4).tolist(),
            "camera_matrix": np.eye(3).tolist(),
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handeye.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            handeye, matrix, distortion = load_handeye(path)
        np.testing.assert_allclose(handeye, np.eye(4))
        np.testing.assert_allclose(matrix, np.eye(3))
        np.testing.assert_allclose(distortion, np.zeros(5))


if __name__ == "__main__":
    unittest.main()
