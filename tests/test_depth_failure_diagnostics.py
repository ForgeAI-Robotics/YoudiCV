"""Real producer -> existing latest RPC -> persisted task trace; no hardware."""
import json
from unittest.mock import Mock
import pytest
from piper_handeye.button_pose import ButtonDetection
from piper_handeye import pipeline_timing as timing
from tests.test_early_light_publication import worker, service


@pytest.mark.parametrize('message', [
    'only 17 valid local panel pixels around button; require 120',
    'only 52 planar depth inliers; require 120',
    'only 0 valid panel pixels around button; require 120',
    'panel plane RMS 9.4 mm exceeds 6.0 mm',
    'button center ray is parallel to the fitted panel plane',
    'fitted button point is behind the camera',
])
def test_exact_depth_subreason_survives_producer_rpc_and_task_trace(tmp_path, monkeypatch, message):
    monkeypatch.delenv('YOUXIAOMEI_TIMING_FILE', raising=False)
    monkeypatch.setattr(timing, '_recorder', None)
    w = worker(); w.target_label = '6'
    detection = ButtonDetection(0, '6', .95, (50, 40, 90, 80))
    monkeypatch.setattr('piper_handeye.live_button.detect_with_enhancement_fallback',
                        Mock(return_value=((detection,), {})))
    def depth(*a, **k):
        w._running = False
        raise RuntimeError(message)
    call = Mock(side_effect=depth)
    monkeypatch.setattr('piper_handeye.live_button.estimate_button_camera_pose', call)
    w.start(); w._thread.join(2)
    assert not w._thread.is_alive() and w._fatal_error is None
    assert call.call_count == 1
    sample = w.latest_sample()
    assert sample.error == message and sample.pose is None
    assert w.latest_valid() is None and not w.valid_window()
    assert w.latest_light_sample().publication_stage == 'yolo_2d_before_depth'
    srv = service(w, tmp_path, monkeypatch)
    reply = srv.dispatch(dict(command='latest', token='test', target_label='6',
                              include_realtime=False))
    assert reply['sample'] is None
    assert reply['latest_processed_timing']['sequence'] == 100
    assert reply['latest_processed_timing']['pose_valid'] is False
    recorder = timing.configure_timing(tmp_path/'trace.jsonl', 'diagnostic-test')
    try:
        timing.record_vision_response(reply)
        assert recorder.flush()['written']
        events = [json.loads(line) for line in (tmp_path/'trace.jsonl').read_text().splitlines()]
        error = next(e for e in events if e.get('name') == 'vision.depth_pose')
        assert error['status'] == 'error'
        assert error['error_type'] == 'RuntimeError'
        assert error['error_message'] == message
        assert error['target_label'] == '6' and error['frame_sequence'] == 100
    finally:
        timing._recorder = None
        monkeypatch.delenv('YOUXIAOMEI_TIMING_FILE', raising=False)
        monkeypatch.delenv('YOUXIAOMEI_TIMING_TASK', raising=False)


def test_successful_depth_keeps_original_publication_and_no_failure_fields(monkeypatch):
    w = worker(); pose = object()
    monkeypatch.setattr('piper_handeye.live_button.detect_with_enhancement_fallback',
                        Mock(return_value=((ButtonDetection(0,'5',.95,(50,40,90,80)),),{})))
    def depth(*a, **k):
        w._running = False
        return pose
    monkeypatch.setattr('piper_handeye.live_button.estimate_button_camera_pose', depth)
    w.start(); w._thread.join(2)
    assert w.latest_valid().pose is pose
    span = next(s for s in w.latest_sample().timing['spans'] if s['name']=='vision.depth_pose')
    assert span['status']=='ok'
    assert 'error_message' not in span and 'error_type' not in span
