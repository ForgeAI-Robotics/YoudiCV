from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import math
import queue
import time

import pytest

from bag_handle_v2.infer_bag import DEFAULT_WEIGHTS, predict_frame
from run_bag_detection_gui import BagGUI


def model_for(cls=0, point=(50, 30, 0.9)):
    box = SimpleNamespace(cls=MagicMock(), xyxy=[MagicMock()], conf=MagicMock())
    box.cls.item.return_value = cls
    box.xyxy[0].tolist.return_value = [10, 20, 90, 80]
    box.conf.item.return_value = 0.9
    keypoint = MagicMock()
    keypoint.tolist.return_value = list(point)
    result = SimpleNamespace(boxes=[box], keypoints=SimpleNamespace(data={(0, 0): keypoint}))
    model = MagicMock(names={0: 'graspable_bag', 1: 'invalid_bag', 2: 'uncertain_bag'})
    model.predict.return_value = [result]
    return model


@pytest.mark.parametrize('cls,point,usable', [
    (0, (50, 30, 0.9), True),
    (0, (10, 20, 0.5), True),
    (0, (50, 30, 0.49), False),
    (0, (9, 30, 0.9), False),
    (0, (100, 30, 0.9), False),
    (0, (math.nan, 30, 0.9), False),
    (0, (50, 30, math.inf), False),
    (1, (50, 30, 0.9), False),
    (2, (50, 30, 0.9), False),
])
def test_v2_point_checks(cls, point, usable):
    image = SimpleNamespace(shape=(100, 100, 3))
    data, _ = predict_frame(image, model_for(cls, point))
    obj = data['detections'][0]
    assert obj['grasp_ready_2d'] is usable
    assert (obj['handle_center_xy'] is not None) is usable
    assert (obj['status'] == 'valid') is usable
    assert obj['class_id'] == cls
    assert data['robot_motion_authorized'] is False


def test_frame_inference_reuses_model_and_preserves_source_and_thresholds():
    model = model_for()
    image = SimpleNamespace(shape=(100, 100, 3))
    for _ in range(2):
        data, _ = predict_frame(image, model, device='0', source='frame42')
        assert data['source'] == 'frame42'
        assert data['image_size'] == {'width': 100, 'height': 100}
    assert model.predict.call_count == 2
    model.predict.assert_called_with(image, imgsz=640, conf=0.65, device='0', verbose=False)


def test_wrong_checkpoint_is_rejected_before_prediction():
    model = model_for()
    model.names = {0: 'button'}
    with pytest.raises(ValueError, match='Wrong checkpoint'):
        predict_frame(SimpleNamespace(shape=(100, 100, 3)), model)
    model.predict.assert_not_called()


def test_gui_passes_v2_weights_and_changed_threshold_to_backend(tmp_path):
    gui = BagGUI.__new__(BagGUI)
    gui.arm_process = None
    gui.arm_events = queue.Queue()
    gui.close_pending = False
    gui.abort_return_pending = False
    gui.abort_return_button = MagicMock()
    gui.args = SimpleNamespace(backend_python=Path('/test/env/bin/python'), weights=DEFAULT_WEIGHTS,
                               point_conf=0.6, device='auto', source=Path('paper.jpg'))
    gui.process = None
    gui.session = 0
    gui.conf = MagicMock()
    gui.conf.get.return_value = '0.75'
    gui.status = MagicMock()
    gui.start_button = MagicMock()
    gui.stop_button = MagicMock()
    with patch('run_bag_detection_gui.ROOT', tmp_path), \
         patch('run_bag_detection_gui.subprocess.Popen') as popen, \
         patch('run_bag_detection_gui.threading.Thread'):
        gui.start()
    command = popen.call_args.args[0]
    assert command[command.index('--weights') + 1] == str(DEFAULT_WEIGHTS)
    assert command[command.index('--conf') + 1] == '0.75'
    assert command[command.index('--point-conf') + 1] == '0.6'
    assert command[command.index('--source') + 1] == 'paper.jpg'
    assert gui.process is popen.return_value


def test_gui_discards_frames_from_previous_session_and_stopped_backend():
    gui = BagGUI.__new__(BagGUI)
    gui.arm_process = None
    gui.arm_events = queue.Queue()
    gui.close_pending = False
    gui.abort_return_pending = False
    gui.abort_return_button = MagicMock()
    gui.events = queue.Queue()
    gui.events.put((1, {'type': 'frame'}))
    gui.events.put((2, {'type': 'frame'}))
    gui.session = 2
    gui.closing = False
    gui.stopping = True
    gui.frame = None
    gui.root = MagicMock()
    gui.poll()
    assert gui.frame is None


def test_gui_clears_expired_live_results():
    gui = BagGUI.__new__(BagGUI)
    gui.arm_process = None
    gui.arm_events = queue.Queue()
    gui.close_pending = False
    gui.abort_return_pending = False
    gui.abort_return_button = MagicMock()
    gui.events = queue.Queue()
    gui.session = 1
    gui.closing = False
    gui.frame = {'capture_time': time.time() - 3}
    gui.args = SimpleNamespace(source=None)
    gui.status = MagicMock()
    gui.root = MagicMock()
    gui.render = MagicMock()
    gui.poll()
    assert gui.frame is None
    gui.render.assert_called_once()
