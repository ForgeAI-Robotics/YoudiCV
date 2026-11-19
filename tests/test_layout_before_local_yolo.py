"""Same-frame geometry before a second model call; no camera/robot/model."""
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock
import threading
import time

import numpy as np
import pytest
import yaml

from piper_handeye.button_pose import ButtonDetection
from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.detection_enhancement import detect_with_enhancement_fallback
from piper_handeye.live_button import LiveButtonPerception
from piper_handeye.numeric_layout_handoff import LayoutTaskPolicy, LayoutTaskVision
from piper_handeye.orbbec_v1_stream import LiveRgbdFrame
from tools.opencv_light_worker import RelativeLuminance
from run_button_vision_server import VisionService
from tests.test_numeric_layout_localization import scene, ROOT


def current_frame(frame):
    return replace(frame, received_monotonic=time.monotonic(), received_unix=time.time())


def test_2d_geometry_reads_no_depth_and_produces_no_motion_pose(scene, monkeypatch):
    localizer, f, detections = scene
    class NoDepth:
        def __getattr__(self, name):
            if name == 'depth_m':
                raise AssertionError('2-D geometry read depth')
            return getattr(f, name)
    monkeypatch.setattr('piper_handeye.numeric_layout_localization.estimate_button_camera_pose',
                        Mock(side_effect=AssertionError('2-D geometry estimated a pose')))
    result = localizer.localize_2d(NoDepth(), '2', detections, now_monotonic=10.05)
    assert result['accepted_2d'] and not result['valid_3d']
    assert result['localization_stage'] == '2d_only'
    assert not result['motion_eligible'] and not result['target_detected']
    assert result['yolo_confidence'] is None and 'pose' not in result


def test_missing_target_uses_geometry_before_second_detector_call(scene):
    localizer, f, ds = scene
    detector = NS(detect=Mock(return_value=ds))
    result, audit = detect_with_enhancement_fallback(
        detector, f.color, '2', .7,
        before_numeric_retry=lambda raw: localizer.localize_2d(
            f, '2', raw, now_monotonic=10.05))
    assert result == ds and not any(d.label == '2' for d in result)
    assert detector.detect.call_count == 1
    assert audit['numeric_retry_skipped_by_layout'] is True
    assert not audit['attempted'] and audit['retry_inference_sec'] == 0
    assert audit['pre_retry_layout_2d']['accepted_2d']
    assert audit['raw_detections']


def test_existing_raw_success_never_calls_layout_or_second_model(scene):
    _, f, ds = scene
    detector = NS(detect=Mock(return_value=ds))
    callback = Mock(side_effect=AssertionError('existing target was reinferred'))
    result, audit = detect_with_enhancement_fallback(
        detector, f.color, '1', .7, before_numeric_retry=callback)
    assert result == ds
    callback.assert_not_called()
    assert detector.detect.call_count == 1 and not audit['attempted']


@pytest.mark.parametrize('mode', ['insufficient', 'stale', 'scope', 'error'])
def test_rejected_layout_keeps_original_second_inference(scene, mode):
    l, f, ds = scene
    raw = ds[:1] if mode == 'insufficient' else ds
    target = '8' if mode == 'scope' else '2'
    recovered = (*raw, ButtonDetection(0, target, .95, (169, 214, 191, 236)))
    detector = NS(detect=Mock(side_effect=[raw, recovered]))
    def callback(detections):
        if mode == 'error':
            raise RuntimeError('geometry unavailable')
        return l.localize_2d(f, target, detections,
                             now_monotonic=10.36 if mode == 'stale' else 10.05)
    _, audit = detect_with_enhancement_fallback(
        detector, f.color, target, .7, before_numeric_retry=callback)
    assert detector.detect.call_count == 2
    assert not audit.get('numeric_retry_skipped_by_layout', False)
    assert audit['attempted']


def test_stale_at_end_of_geometry_is_not_allowed_to_skip_retry(scene, monkeypatch):
    l, f, ds = scene
    import piper_handeye.numeric_layout_localization as module
    original = module.infer_layout
    elapsed = [0.]
    def slow_geometry(*a, **kw):
        result = original(*a, **kw)
        elapsed[0] = .31
        return result
    monkeypatch.setattr(module, 'time', NS(perf_counter=lambda: elapsed[0], monotonic=lambda: 10.05))
    monkeypatch.setattr(module, 'infer_layout', slow_geometry)
    result = l.localize_2d(f, '2', ds, now_monotonic=10.05)
    assert not result['accepted_2d'] and not result['valid_3d']
    assert result['reason'] == 'rgbd_frame_stale_after_2d_localization'


def test_geometry_rejects_unusable_crop_even_when_center_was_inferred(scene, monkeypatch):
    l, f, ds = scene
    monkeypatch.setattr('piper_handeye.numeric_layout_localization.infer_layout',
        lambda *a: dict(accepted=True, anchors=['1','3','9'], center=[10,10],
                       estimated_diameter_px=30, reason='controlled geometry'))
    r = l.localize_2d(f, '2', ds, now_monotonic=10.05)
    assert not r['accepted_2d'] and r['reason'] == 'inferred_light_region_incomplete'


def test_early_inferred_roi_reaches_real_opencv_while_depth_is_blocked(scene, tmp_path, monkeypatch):
    localizer, f, ds = scene
    f = current_frame(f)
    entered, release = threading.Event(), threading.Event()
    detector = NS(detect=Mock(return_value=ds))
    worker = LiveButtonPerception(
        NS(wait_for_sequence=lambda *a, **kw: f), detector, localizer.camera_matrix,
        target_label='2', target_confidence=.7, geometry_options={},
        enhancement_fallback_enabled=True, numeric_layout_localizer=localizer,
        layout_before_local_yolo_enabled=True)
    import piper_handeye.numeric_layout_localization as module
    original = module.estimate_button_camera_pose
    def depth(*a, **kw):
        entered.set()
        assert release.wait(3)
        worker._running = False
        return original(*a, **kw)
    monkeypatch.setattr(module, 'estimate_button_camera_pose', depth)
    monkeypatch.setattr('piper_handeye.live_button.estimate_button_camera_pose',
                        Mock(side_effect=AssertionError('inferred ROI entered ordinary YOLO depth')))
    worker.start()
    try:
        assert entered.wait(2)
        light = worker.latest_light_sample()
        assert light.publication_stage == 'layout_2d_before_depth'
        assert light.pose is None and not light.numeric_layout_localization['valid_3d']
        assert worker.latest_sample() is None and worker.latest_valid() is None
        assert not worker.valid_window() and detector.detect.call_count == 1
        service = VisionService.__new__(VisionService)
        service.args = NS(token='test');service.perception = worker
        monkeypatch.setattr('run_button_vision_server.ROOT', tmp_path)
        settings = yaml.safe_load((ROOT/'piper_button_pose.yaml').read_text())['numeric_layout_localization']
        class Client:
            def request(self, command, **options):
                return service.dispatch(dict(command=command, token='test', **options))
        vision = LayoutTaskVision(Client(), LayoutTaskPolicy(settings, '2'))
        recognizer = NS(observe=RelativeLuminance().predict)
        observer = FreshLightObserver(vision, recognizer, '2')
        assert observer.observe('near') == 'OFF'
        row = observer.audit[-1]
        assert row['target_detection'] is None
        assert row['target_region']['source'] == 'panel_layout_inferred'
        assert row['target_region']['confidence'] is None
        assert row['frame_sequence'] == row['target_region']['frame_sequence'] == f.sequence
        assert row['recognition']['region_source'] == 'panel_layout_inferred'
        assert not any(s['name']=='vision.depth_pose' for s in light.timing['spans'])
    finally:
        release.set();worker._thread.join(3)
    assert worker._fatal_error is None
    assert worker.latest_sample().pose is None  # layout depth never becomes ordinary pose
    assert worker.latest_sample().publication_stage == 'completed_rgbd'
    assert worker.latest_light_sample() is light  # no mutation by late depth completion


@pytest.mark.parametrize('target', ['up', 'down'])
def test_arrow_flow_never_calls_numeric_layout_callback(target):
    callback = Mock(side_effect=AssertionError('arrow entered numeric geometry'))
    detector = NS(detect=Mock(return_value=()))
    detect_with_enhancement_fallback(detector, np.zeros((32,32,3), np.uint8),
                                    target, .7, before_numeric_retry=callback)
    callback.assert_not_called()
