"""Exercise real producer/RPC while depth is blocked; no hardware or model."""
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import threading
import time

import numpy as np
import pytest

from piper_handeye.button_pose import ButtonDetection
from piper_handeye.live_button import LiveButtonPerception, LiveButtonSample
from piper_handeye.orbbec_v1_stream import LiveRgbdFrame
from piper_handeye.button_light_hardware import FreshLightObserver
from run_button_vision_server import VisionService


def frame(sequence=100):
    return LiveRgbdFrame(sequence,time.monotonic(),time.time(),sequence,sequence,
        np.full((120,160,3),100,np.uint8),np.full((120,160),.5))


def worker():
    camera=Mock();camera.latest_sequence=100
    w=LiveButtonPerception(camera,Mock(),np.eye(3),target_label='5',
                           target_confidence=.7,geometry_options={})
    camera.wait_for_sequence.side_effect=lambda *a,**k:frame()
    return w


def sample(f,*,pose=None,record=None):
    return LiveButtonSample(f.sequence,'5',f,pose,None,.01,
        (ButtonDetection(0,'5',.95,(50,40,90,80)),),numeric_yolo_consistency=record or {})


def service(w,tmp_path,monkeypatch):
    s=VisionService.__new__(VisionService);s.args=NS(token='test');s.perception=w
    s.camera=Mock();s.detector=Mock()
    monkeypatch.setattr('run_button_vision_server.ROOT',tmp_path)
    return s


@pytest.mark.parametrize('fail_depth',[False,True])
def test_exact_light_frame_is_available_before_depth_without_a_motion_pose(tmp_path,monkeypatch,fail_depth):
    w=worker();entered=threading.Event();release=threading.Event();pose=object()
    detection=ButtonDetection(0,'5',.95,(50,40,90,80))
    def depth(*a,**k):
        entered.set();assert release.wait(3)
        w._running=False
        if fail_depth:raise RuntimeError('recorded missing depth')
        return pose
    monkeypatch.setattr('piper_handeye.live_button.detect_with_enhancement_fallback',Mock(return_value=((detection,),{})))
    monkeypatch.setattr('piper_handeye.live_button.estimate_button_camera_pose',depth)
    w.start()
    try:
        assert entered.wait(2)
        early=w.latest_light_sample();assert early is not None
        assert early.pose is None and early.error=='depth_pending'
        assert early.publication_stage=='yolo_2d_before_depth'
        assert w.latest_sample() is None and w.latest_valid() is None and not w.valid_window()
        with pytest.raises(TimeoutError):w.collect_after(0,valid_results=1,max_attempts=2,timeout=.02)
        s=service(w,tmp_path,monkeypatch)
        r=s.dispatch(dict(command='light_snapshot',token='test',target_label='5',after_sequence=99))
        assert r['fresh'] and not r['depth_complete'] and r['frame_sequence']==100
        assert r['target_detection']['frame_sequence']==100
        assert r['pairing']=='same_inference_frame'
        assert r['light_publication_stage']=='yolo_2d_before_depth'
        assert not any(x['name']=='vision.depth_pose' for x in r['perception_timing']['spans'])
        assert not s.dispatch(dict(command='light_snapshot',token='test',after_sequence=100))['fresh']
        s.camera.latest_frame.assert_not_called();s.detector.detect.assert_not_called()
    finally:
        release.set();w._thread.join(3)
    assert w._fatal_error is None
    assert w.latest_sample().publication_stage=='completed_rgbd'
    assert (w.latest_valid() is None) is fail_depth
    assert w.latest_light_sample() is early  # completed timing must not mutate it
    assert not any(x['name']=='vision.depth_pose' for x in early.timing['spans'])


def test_switch_away_and_back_does_not_publish_old_inflight_target(monkeypatch):
    w=worker();entered=threading.Event();release=threading.Event()
    monkeypatch.setattr('piper_handeye.live_button.detect_with_enhancement_fallback',lambda *a,**k:((ButtonDetection(0,'5',.95,(50,40,90,80)),),{}))
    def depth(*a,**k):
        entered.set();assert release.wait(3);w._running=False;return object()
    monkeypatch.setattr('piper_handeye.live_button.estimate_button_camera_pose',depth)
    w.start()
    try:
        assert entered.wait(2) and w.latest_light_sample() is not None
        assert w.set_target('4')==100;assert w.latest_light_sample('5') is None
        assert w.set_target('5')==100;assert w.latest_light_sample() is None
    finally:release.set();w._thread.join(3)
    assert w.latest_light_sample() is None
    assert w.latest_sample().sequence==100


def test_inferred_roi_keeps_completed_same_frame_path(monkeypatch):
    w=worker();entered=threading.Event();release=threading.Event()
    def layout(f,label,ds):
        entered.set();assert release.wait(3);w._running=False
        return dict(frame_sequence=f.sequence,target_label=label,valid_3d=True)
    w.numeric_layout_localizer=NS(yolo_consistency_enabled=False,localize=layout)
    monkeypatch.setattr('piper_handeye.live_button.detect_with_enhancement_fallback',lambda *a,**k:((),{}))
    w.start()
    try:
        assert entered.wait(2)
        assert w.latest_light_sample() is None and w.latest_sample() is None
    finally:release.set();w._thread.join(3)
    s=w.latest_light_sample()
    assert s is w.latest_sample() and s.publication_stage=='completed_rgbd'
    assert s.numeric_layout_localization['valid_3d'] and s.pose is None


def test_known_conflict_invalidates_both_caches_while_fallback_is_pending(monkeypatch):
    w=worker();old=sample(frame(90),pose=object());w._samples.append(old);w._light_samples.append(old)
    record=dict(version=1,source='known_panel_neighbour_consistency',enabled=True,
                decision='conflict',target_label='5',frame_sequence=100)
    entered=threading.Event();release=threading.Event()
    def layout(*a):
        entered.set();assert release.wait(3);w._running=False;return {}
    w.numeric_layout_localizer=NS(yolo_consistency_enabled=True,inspect_yolo=lambda *a:record,localize=layout)
    monkeypatch.setattr('piper_handeye.live_button.detect_with_enhancement_fallback',lambda *a,**k:((ButtonDetection(0,'5',.95,(50,40,90,80)),),{}))
    w.start()
    try:
        assert entered.wait(2)
        assert w.latest_valid() is None and not w.valid_window() and w.latest_light_sample() is None
        with pytest.raises(TimeoutError):w.collect_after(0,valid_results=1,max_attempts=3,timeout=.02)
    finally:release.set();w._thread.join(3)
    assert w.latest_light_sample().sequence==100
    assert w.latest_light_sample().publication_stage=='completed_rgbd'
    assert w.latest_valid() is None


def test_collect_cannot_join_pose_samples_across_an_inflight_conflict():
    w=worker();w._running=True
    w._samples.append(sample(frame(90),pose=object()))
    seen=threading.Event();done=threading.Event();result={}
    original=w._conflict_barrier
    def barrier(label):
        value=original(label);seen.set();return value
    w._conflict_barrier=barrier
    def collect():
        try:result['batch']=w.collect_after(0,valid_results=2,max_attempts=5,timeout=2)
        finally:done.set()
    thread=threading.Thread(target=collect);thread.start()
    try:
        assert seen.wait(1)
        with w._condition:
            w._conflict_barriers['5']=100
            w._samples.append(sample(frame(101),pose=object()));w._condition.notify_all()
        assert not done.wait(.03)
        with w._condition:
            w._samples.append(sample(frame(102),pose=object()));w._condition.notify_all()
        assert done.wait(1)
        assert [s.sequence for s in result['batch'].accepted]==[101,102]
    finally:
        w._running=False
        with w._condition:w._condition.notify_all()
        thread.join(2)


def lamp(sequence,age,*,box=True):
    return dict(fresh=True,frame_sequence=sequence,capture_time_unix_sec=time.time()-age,
        image_size=[160,120],path=f'/tmp/controlled_frame_{sequence}.jpg',
        target_detection=(dict(label='5',frame_sequence=sequence,confidence=.95,
                               bbox_xyxy=[50,40,90,80]) if box else None),
        light_publication_stage='yolo_2d_before_depth')


def observer(replies):
    it=iter(replies);last=[None]
    def request(*a,**k):
        try:last[0]=next(it)
        except StopIteration:pass
        return last[0]
    recognizer=Mock();recognizer.observe.return_value=dict(state='ON')
    o=FreshLightObserver(NS(request=request),recognizer,'5',poll_sec=.001,fresh_frame_timeout_sec=.025)
    # Evidence capture itself is tested elsewhere; never open a camera here.
    o.unsafe_waiting_evidence_saved=True
    return o,recognizer


def test_stale_frames_wait_for_fresh_only_after_continuity_is_broken():
    o,rec=observer([lamp(1,.6),lamp(2,.5),lamp(3,.01)])
    assert o.observe('near')=='ON'
    assert rec.observe.call_count==1
    entry=o.audit[-1]
    assert entry['frame_sequence']==3
    assert [r['sequence'] for r in entry['frame_acquisition']['skipped_stale_frames']]==[1,2]


def test_invalid_frame_between_on_results_is_not_hidden():
    o,rec=observer([lamp(1,.01),lamp(2,.5),lamp(3,.01)])
    assert [o.observe('near') for _ in range(3)]==['ON','UNCERTAIN','ON']
    assert rec.observe.call_count==2


def test_only_stale_frames_end_in_uncertain_within_existing_budget():
    o,rec=observer([lamp(1,.6)])
    began=time.monotonic()
    assert o.observe('near')=='UNCERTAIN'
    assert time.monotonic()-began<.3
    assert not o.audit[-1]['recognition']['visibility']['checks']['recent_rgb_time']
    rec.observe.assert_not_called()


def test_current_freshness_threshold_is_not_raised():
    o,rec=observer([lamp(1,.36)])
    assert o.observe('near')=='UNCERTAIN';rec.observe.assert_not_called()


def test_report_identifies_early_2d_and_complete_motion_paths():
    from run_button_light_closed_loop import LIGHT_EVIDENCE_POLICY
    p=LIGHT_EVIDENCE_POLICY['light_rgb_publication']
    assert p['max_rgb_age_sec']==.35
    assert p['ordinary_yolo']=='yolo_2d_before_depth'
    assert p['motion_inputs']=='completed_rgbd_only'
    assert p['inferred_roi']=='completed_same_frame_fallback'

