import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from piper_handeye.missing_button_capture import save_missing_button_view

import numpy as np


def test_capture_keeps_image_if_health_request_fails():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.jpg"
        source.write_bytes(b"test image")
        vision = MagicMock()
        vision.request.side_effect = [
            {"path": str(source), "frame_sequence": 42},
            {"input_path": str(source), "raw_yolo_path": str(source), "frame_sequence": 40},
            {"frame_sequence": 40, "detections": []},
            RuntimeError("health unavailable"),
        ]
        result = save_missing_button_view(vision, "8", "timeout", output_root=root / "out")
        assert Path(result["image"]).read_bytes() == b"test image"
        stored = json.loads((Path(result["directory"]) / "diagnostic.json").read_text())
        assert stored["snapshot"]["frame_sequence"] == 42
        assert stored["visible_buttons"]["frame_sequence"] == 40
        assert stored["errors"]["health"] == "health unavailable"
        assert Path(result["raw_yolo_path"]).read_bytes() == b"test image"
        assert stored["recognition"]["frame_sequence"] == 40


def test_existing_snapshot_only_requests_cached_recognition_image():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.jpg"
        source.write_bytes(b"observation")
        vision = MagicMock()
        vision.request.return_value = {"input_path": str(source), "raw_yolo_path": str(source)}
        result = save_missing_button_view(
            vision, "8", "missing_box", snapshot={"path": str(source)}, output_root=root / "out")
        assert Path(result["image"]).exists()
        vision.request.assert_called_once_with("recognition_snapshot")


def test_capture_failure_does_not_raise_or_hide_reason():
    with tempfile.TemporaryDirectory() as directory:
        vision = MagicMock()
        vision.request.side_effect = RuntimeError("camera disconnected")
        result = save_missing_button_view(vision, "8", "visibility_timeout", output_root=directory)
        assert result["reason"] == "visibility_timeout"
        assert result["errors"]["capture"] == "camera disconnected"


def test_numpy_diagnostics_are_saved_as_numbers_without_masking_timeout(tmp_path):
    source = tmp_path / 'source.jpg';source.write_bytes(b'frame')
    vision = MagicMock()
    vision.request.return_value = {'matrix': np.eye(4), 'value': np.float32(.25)}
    result = save_missing_button_view(vision, 'down', 'stationary_target_acquisition_failed',
        snapshot={'path': str(source), 'sequence': np.int64(4)}, output_root=tmp_path / 'out')
    stored = json.loads((Path(result['directory']) / 'diagnostic.json').read_text())
    assert stored['recognition']['matrix'] == np.eye(4).tolist()
    assert stored['recognition']['value'] == .25
    assert stored['snapshot']['sequence'] == 4
    assert 'save' not in result['errors'] and 'emit' not in result['errors']


def test_secondary_output_failure_cannot_replace_primary_diagnostic_reason(tmp_path, monkeypatch):
    source = tmp_path / 'source.jpg';source.write_bytes(b'frame')
    vision = MagicMock();vision.request.return_value = {}
    def broken_output(*_args, **_kwargs):raise OSError('output unavailable')
    monkeypatch.setattr('builtins.print', broken_output)
    result = save_missing_button_view(vision, 'down', 'stationary_target_acquisition_failed',
        snapshot={'path': str(source)}, output_root=tmp_path / 'out')
    assert result['reason'] == 'stationary_target_acquisition_failed'
    assert 'OSError' in result['errors']['emit'] and Path(result['image']).exists()
