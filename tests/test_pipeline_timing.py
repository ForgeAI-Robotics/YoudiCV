"""Timing must be faithful, bounded, and unable to change control outcomes."""
import json
import os
import subprocess
import sys
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from piper_handeye import pipeline_timing as timing
from piper_handeye.button_pose import ButtonDetection
from piper_handeye.live_button import LiveButtonPerception
from piper_handeye.orbbec_v1_stream import LiveRgbdFrame
from piper_handeye.press_preparation import BackgroundPressPreparation
from piper_handeye.vision_rpc import VisionRpcClient
from tools.analyze_pipeline_timing import analyze, write_analysis


@pytest.fixture(autouse=True)
def isolated_recorder(monkeypatch):
    monkeypatch.delenv('YOUXIAOMEI_TIMING_FILE', raising=False)
    monkeypatch.delenv('YOUXIAOMEI_TIMING_TASK', raising=False)
    monkeypatch.setattr(timing, '_recorder', None)
    yield
    timing._recorder = None
    os.environ.pop('YOUXIAOMEI_TIMING_FILE', None)
    os.environ.pop('YOUXIAOMEI_TIMING_TASK', None)


def test_disabled_timing_preserves_return_and_does_not_enable_itself():
    marker = object()
    @timing.traced('test', 'test')
    def work():
        return marker
    assert work() is marker
    assert timing.get_recorder() is None


def test_nested_spans_have_identity_and_do_not_write_until_finalization(tmp_path):
    recorder = timing.configure_timing(tmp_path/'trace.jsonl', 'test-task')
    with patch.object(type(tmp_path), 'open', side_effect=AssertionError('unexpected timing I/O')):
        with timing.stage('outer', 'planning'):
            with timing.stage('inner', 'ik'):
                pass
    assert len(recorder.spans) == 2
    inner, outer = recorder.spans
    assert inner['parent_span_id'] == outer['span_id']
    assert inner['start_monotonic_sec'] >= outer['start_monotonic_sec']
    assert inner['end_monotonic_sec'] <= outer['end_monotonic_sec']
    assert inner['task_id'] == 'test-task'
    assert recorder.flush()['written']
    count = len((tmp_path/'trace.jsonl').read_text().splitlines())
    recorder.flush()
    assert len((tmp_path/'trace.jsonl').read_text().splitlines()) == count == 3


def test_recorder_failure_does_not_mask_original_control_exception(tmp_path):
    recorder = timing.configure_timing(tmp_path/'trace.jsonl')
    error = RuntimeError('original IK failure')
    @timing.traced('test', 'ik')
    def work():
        raise error
    with patch.object(recorder, 'add', side_effect=OSError('diagnostic failure')):
        with pytest.raises(RuntimeError) as caught:
            work()
    assert caught.value is error and recorder.errors == 1


def test_failed_operation_records_failure_without_changing_exception(tmp_path):
    recorder = timing.configure_timing(tmp_path/'trace.jsonl')
    with pytest.raises(ValueError, match='bad pose'):
        with timing.stage('IK', 'ik'):
            raise ValueError('bad pose')
    assert recorder.spans[0]['status'] == 'error'


def test_storage_is_bounded_and_overflow_is_explicit(tmp_path):
    recorder = timing.TimingRecorder(tmp_path/'trace.jsonl', max_spans=3)
    for index in range(10):
        recorder.add({'name': str(index)})
    assert len(recorder.spans) == 3 and recorder.dropped == 7
    assert recorder.spans[0]['name'] == '0'  # preserve pre-press evidence


def test_save_failure_remains_optional(tmp_path):
    blocked = tmp_path/'file'
    blocked.write_text('not a directory')
    recorder = timing.TimingRecorder(blocked/'trace.jsonl')
    assert not recorder.flush()['written']
    assert recorder.errors == 1


def test_task_finalization_creates_analysis_without_changing_control(tmp_path):
    timing.configure_timing(tmp_path/'trace.jsonl')
    with timing.stage('IK','ik'):
        pass
    result=timing.finish_timing_with_analysis()
    assert Path(result['analysis_directory']).joinpath('耗时分析.txt').is_file()


def test_optional_analysis_failure_is_reported_without_raising(tmp_path):
    timing.configure_timing(tmp_path/'trace.jsonl')
    with patch('tools.analyze_pipeline_timing.write_analysis',side_effect=OSError('analysis unavailable')):
        result=timing.finish_timing_with_analysis()
    assert result['written'] and result['analysis_error']=='OSError'


def test_child_planning_process_inherits_timing_and_flushes_on_exit(tmp_path):
    path=tmp_path/'trace.jsonl'
    timing.configure_timing(path,'shared-task')
    child=subprocess.run([sys.executable,'-c',
        "from piper_handeye.pipeline_timing import stage\nwith stage('child.IK','ik'):\n pass\n"],
        capture_output=True,text=True,timeout=10.)
    assert child.returncode==0,child.stderr
    with timing.stage('parent','planning'):
        pass
    timing.finish_timing()
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    spans=[r for r in rows if r.get('kind')=='span']
    assert {r['name'] for r in spans}=={'child.IK','parent'}
    assert len({r['pid'] for r in spans})==2
    assert {r['task_id'] for r in spans}=={'shared-task'}
    result,_=analyze(rows)
    assert result['foreign_clock_spans_excluded']==0


def test_background_build_interval_distinguishes_covered_work_from_join_wait(tmp_path):
    recorder = timing.configure_timing(tmp_path/'trace.jsonl')
    ready = threading.Event()
    marker = object()
    def build():
        ready.set()
        return marker
    with BackgroundPressPreparation() as preparation:
        preparation.start(build)
        assert ready.wait(1.)
        future_plan,build_seconds=preparation.future.result(timeout=1.)
        assert future_plan is marker and build_seconds>=0
        plan, audit = preparation.finish()
    assert plan is marker and audit['build_completed_before_finish']
    assert audit['build_interval']['end_monotonic_sec'] <= audit['wait_interval']['start_monotonic_sec']
    assert {r['name'] for r in recorder.spans} >= {'press.background_plan_build', 'press.wait_for_background_plan'}
    assert len({r['thread_id'] for r in recorder.spans}) == 2


def test_rpc_timings_preserve_response_and_never_capture_secret_payload(tmp_path):
    recorder = timing.configure_timing(tmp_path/'trace.jsonl')
    source = timing.interval('vision.detection', 'detection', timing.mark())
    response = dict(ok=True, sample=dict(sequence=5, timing=dict(spans=[source])))
    wire = MagicMock()
    wire.makefile.return_value.__enter__.return_value.readline.return_value = json.dumps(response).encode()+b'\n'
    with patch('socket.create_connection') as connect:
        connect.return_value.__enter__.return_value = wire
        client = VisionRpcClient('test', 1, 'TOP-SECRET')
        assert client.request('latest') == response
        assert client.request('latest') == response
    assert sum(r['name']=='vision.detection' for r in recorder.spans) == 1
    assert {r['name'] for r in recorder.spans} >= {'vision.connect', 'vision.send', 'vision.response_wait', 'vision.decode', 'vision.rpc'}
    recorder.flush()
    data = (tmp_path/'trace.jsonl').read_text()
    assert 'TOP-SECRET' not in data and '"token"' not in data


@pytest.mark.parametrize('command',['health','safety','safety_camera_pose','safety_motion_state'])
def test_high_frequency_safety_requests_do_not_accumulate_trace_events(command,tmp_path):
    recorder = timing.configure_timing(tmp_path/'trace.jsonl')
    wire = MagicMock()
    wire.makefile.return_value.__enter__.return_value.readline.return_value = b'{"ok":true}\n'
    with patch('socket.create_connection') as connect:
        connect.return_value.__enter__.return_value = wire
        assert VisionRpcClient('test',1,'secret').request(command) == {'ok':True}
    assert len(recorder.spans) == 0


@pytest.mark.parametrize('payload', [dict(sample=None), dict(sample={'sequence': 1}),
    dict(sample={'sequence': 1,'timing': None}), dict(samples=None)])
def test_old_or_malformed_optional_payload_does_not_change_control(payload, tmp_path):
    timing.configure_timing(tmp_path/'trace.jsonl')
    timing.record_vision_response(payload)


def test_failed_processed_frame_timings_are_diagnostic_not_a_motion_target(tmp_path):
    recorder = timing.configure_timing(tmp_path/'trace.jsonl')
    source = timing.interval('vision.depth_pose', 'depth', timing.mark(), status='error')
    payload = dict(sample=None, latest_processed_timing=dict(sequence=3,
                    pose_valid=False, timing=dict(spans=[source])))
    timing.record_vision_response(payload)
    assert payload['sample'] is None
    assert len(recorder.spans) == 1 and recorder.spans[0]['status'] == 'error'


@pytest.mark.parametrize('failure', [None, 'missing', 'depth', 'detection'])
def test_live_camera_timing_separates_stages_and_preserves_failed_samples(failure):
    clock = SimpleNamespace(value=100.)
    image = np.zeros((36,64,3),np.uint8)
    detection = ButtonDetection(0,'4',.95,(10.,10.,30.,30.))
    camera = MagicMock()
    detector = MagicMock()
    perception = LiveButtonPerception(camera, detector, np.eye(3),
        target_label='4', target_confidence=.7, geometry_options={})
    def wait(*args, **kwargs):
        clock.value += .01
        perception._running = False
        return LiveRgbdFrame(7, clock.value, clock.value+1000.,0,0,image,np.ones((36,64)))
    def detect(*args):
        clock.value += .02
        if failure == 'detection':
            raise RuntimeError('inference failed')
        return [] if failure == 'missing' else [detection]
    pose = object()
    def depth(*args, **kwargs):
        clock.value += .03
        if failure == 'depth':
            raise RuntimeError('bad depth')
        return pose
    camera.wait_for_sequence.side_effect = wait
    detector.detect.side_effect = detect
    perception._running = True
    with patch('time.monotonic', side_effect=lambda: clock.value), patch('time.time', side_effect=lambda: clock.value+1000.), patch('time.perf_counter', side_effect=lambda: clock.value), patch('piper_handeye.live_button.estimate_button_camera_pose', side_effect=depth):
        perception._run()
    sample = perception.latest_sample()
    assert sample.frame.color is image and sample.sequence == 7
    assert sample.pose is (pose if failure is None else None)
    assert bool(sample.error) is (failure is not None)
    assert detector.detect.call_count == 1
    spans = {r['name']:r for r in sample.timing['spans']}
    assert spans['vision.frame_wait']['elapsed_sec'] == pytest.approx(.01)
    assert spans['vision.detection']['elapsed_sec'] == pytest.approx(.02)
    if failure in ('missing', 'detection'):
        assert 'vision.depth_pose' not in spans
    else:
        assert spans['vision.depth_pose']['elapsed_sec'] == pytest.approx(.03)
        assert sample.processing_sec == pytest.approx(.05)
    failed_name = {'depth':'vision.depth_pose','missing':'vision.selection','detection':'vision.detection'}.get(failure)
    if failed_name:
        assert spans[failed_name]['status'] == 'error'
    json.dumps(sample.timing, allow_nan=False)


def row(name, category, start, end, domain='test'):
    return dict(kind='span', name=name, category=category,
                start_monotonic_sec=start, end_monotonic_sec=end, clock_domain=domain)


def session():
    return dict(kind='session', clock_domain='test', start_monotonic_sec=0., end_monotonic_sec=10.)


def test_summary_uses_interval_union_and_does_not_sum_nested_ik_or_parallel_work():
    result, _ = analyze([session(), row('IK','ik',1.,4.), row('IK child','ik',2.,3.),
        row('approach','motion_window',0.,6.), row('build','planning',2.,7.),
        row('join','planning_wait',6.,7.)])
    assert result['task_wall_sec'] == 10.
    assert result['categories']['ik']['union_sec'] == 3.
    assert result['overlap_sec']['ik / motion_window'] == 3.
    assert result['overlap_sec']['motion_window / planning'] == 4.
    assert result['categories']['planning_wait']['union_sec'] == 1.


def test_summary_refuses_to_align_other_device_clocks_and_clips_old_source_frames():
    result, _ = analyze([session(), row('remote','depth',100.,200.,domain='other-device'),
        row('old','capture',-10.,1.), row('bad','ik',5.,4.)])
    assert result['foreign_clock_spans_excluded'] == 1
    assert result['malformed_spans_excluded'] == 1
    assert result['categories']['capture']['union_sec'] == 1.
    assert 'depth' not in result['categories']


def test_analysis_requires_a_known_session_clock():
    with pytest.raises(ValueError,match='missing task session'):
        analyze([row('IK','ik',1.,2.)])


def test_offline_analysis_writes_readable_report_and_timeline(tmp_path):
    source=tmp_path/'input.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in [session(),row('IK','ik',1.,2.)]))
    result=write_analysis(source,tmp_path/'result')
    assert result['categories']['ik']['union_sec']==1.
    assert '不能相加' in (tmp_path/'result/耗时分析.txt').read_text()
    assert json.loads((tmp_path/'result/timing_trace.json').read_text())['traceEvents'][0]['dur']==1e6
