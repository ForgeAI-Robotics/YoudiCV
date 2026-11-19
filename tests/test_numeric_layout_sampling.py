"""Production shadow RPC/collector plumbing; no hardware/command connections."""
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from piper_handeye.numeric_layout_sampling import (
    LayoutStationarySampler, collect_stationary_layout_shadow,
)
from piper_handeye.realtime_planner import SlidingButtonWindow, TimedButtonTransform


CONFIG = dict(numeric_layout_localization=dict(mode='shadow', target_labels=list('12345'),
                                              profile_sha256='a'*64), button_geometry={})


def state(stamp, *, q=0., x=0.):
    matrix = np.eye(4); matrix[0,3] = x
    return SimpleNamespace(timestamp=stamp, joints=np.full(6,q), base_to_gripper=matrix)


def packet(sequence, capture, *, mono=None, z=.6):
    pose = np.eye(4); pose[2,3] = z
    record = dict(frame_sequence=sequence, rgbd_bundle_sequence=sequence,
                  pairing='same_rgbd_bundle', target_label='2', source='panel_layout_inferred',
                  mode='shadow', motion_eligible=False, target_detected=False,
                  yolo_confidence=None, light_state='not_evaluated',
                  profile_sha256='a'*64, profile_id='test_panel',
                  capture_time_unix_sec=capture, received_monotonic=capture-90 if mono is None else mono,
                  accepted_2d=True, valid_3d=True, bbox_xyxy=[100,100,130,130],
                  geometry=dict(accepted=True, anchors=['1','3','9','10']),
                  pose=dict(camera_to_button=pose.tolist(), plane_rms_m=.0013,
                            panel_support_detections=4))
    return dict(ok=True,fresh=True,valid_3d=True,frame_sequence=sequence,
                capture_time_unix_sec=capture,localization=record)


def sampler(**kwargs):
    return LayoutStationarySampler(CONFIG,np.eye(4),'2',now_unix=100.,now_monotonic=10.,**kwargs)


def feed(s, seq, capture, **kwargs):
    mono=capture-90
    s.observe_robot(state(mono-.02),now_unix=capture-.02,now_monotonic=mono-.02)
    s.observe_robot(state(mono+.02),now_unix=capture+.02,now_monotonic=mono+.02)
    return s.consume(packet(seq,capture,**kwargs),now_unix=capture+.05,now_monotonic=mono+.05)


@pytest.mark.parametrize('count',[2,4])
def test_stable_fresh_distinct_frames_have_base_coordinates_no_fake_confidence(count):
    s=sampler(frame_count=count)
    for i in range(count):
        row=feed(s,i+1,100.1+i*.1)
        assert row['accepted'],row
        assert row['reason']==('stable' if i==count-1 else 'collecting')
    r=s.snapshot(now_unix=100.1+(count-1)*.1+.05)
    assert r['stable'] and r['sample_count']==count
    assert np.asarray(r['base_to_button'])[:3,3]==pytest.approx([0,0,.6])
    assert r['motion_eligible'] is False and r['target_detected'] is False
    assert r['yolo_confidence'] is None and r['light_state']=='not_evaluated'
    assert all(not hasattr(v,'confidence') for v in s.window._samples)


def test_capture_time_interpolation_and_handeye_used_instead_of_latest_state():
    he=np.eye(4);he[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];he[:3,3]=[0,.02,.01]
    s=LayoutStationarySampler(CONFIG,he,'2',now_unix=100,now_monotonic=10)
    s.observe_robot(state(10.08,x=.1),now_unix=100.08,now_monotonic=10.08)
    s.observe_robot(state(10.12,x=.3),now_unix=100.12,now_monotonic=10.12)
    p=packet(1,100.1);p['localization']['pose']['camera_to_button'][0][3]=.03
    row=s.consume(p,now_unix=100.15,now_monotonic=10.15)
    assert row['accepted'],row
    assert np.asarray(row['base_to_button'])[:3,3]==pytest.approx([.2,.05,.61])
    assert row['transform_source']=='capture_time_encoder_interpolation'


@pytest.mark.parametrize('change,reason',[
    (lambda p:p.update(frame_sequence=999),'frame_or_target_mismatch'),
    (lambda p:p['localization'].update(rgbd_bundle_sequence=999),'frame_or_target_mismatch'),
    (lambda p:p['localization'].update(target_label='3'),'frame_or_target_mismatch'),
    (lambda p:p['localization'].update(profile_sha256='b'*64),'profile_changed_or_mismatched'),
    (lambda p:p['localization'].update(yolo_confidence=.9),'invalid_layout_provenance'),
    (lambda p:p['localization'].update(target_detected=True),'invalid_layout_provenance'),
    (lambda p:p['localization'].update(motion_eligible=True),'invalid_layout_provenance'),
    (lambda p:p['localization'].update(received_monotonic=9.9),'capture_clock_mismatch'),
    (lambda p:p['localization'].update(valid_3d=False),'layout_depth_not_valid'),
    (lambda p:p['localization']['pose'].update(plane_rms_m=float('nan')),'plane_rms_above_threshold'),
    (lambda p:p['localization']['pose'].update(plane_rms_m=.007),'plane_rms_above_threshold'),
    (lambda p:p['localization']['pose'].update(panel_support_detections=2),'insufficient_depth_supports'),
    (lambda p:p['localization']['geometry'].update(anchors=['1','1','9']),'invalid_independent_layout_references'),
    (lambda p:p['localization'].pop('bbox_xyxy'),'bbox_xyxy'),
    (lambda p:p['localization'].update(bbox_xyxy=[1,2,0,3]),'invalid_layout_bbox'),
])
def test_bad_evidence_never_enters_window(change,reason):
    s=sampler();s.observe_robot(state(10.08),now_unix=100.08,now_monotonic=10.08)
    s.observe_robot(state(10.12),now_unix=100.12,now_monotonic=10.12)
    p=packet(1,100.1);change(p)
    row=s.consume(p,now_unix=100.15,now_monotonic=10.15)
    assert not row['accepted'] and reason in row['reason'],row
    assert s.snapshot(now_unix=100.15)['sample_count']==0


@pytest.mark.parametrize('count',[2,4])
def test_repeated_frames_do_not_fill_window(count):
    s=sampler(frame_count=count)
    assert feed(s,1,100.1)['accepted']
    for _ in range(8):
        row=s.consume(packet(1,100.1),now_unix=100.15,now_monotonic=10.15)
        assert not row['accepted'] and row['reason']=='not_a_new_frame'
    assert s.snapshot(now_unix=100.15)['sample_count']==1
    assert not s.snapshot(now_unix=100.15)['stable']


def test_joint_motion_discards_entire_old_epoch():
    s=sampler(frame_count=2);assert feed(s,1,100.1)['accepted']
    s.observe_robot(state(10.2,q=.005),now_unix=100.2,now_monotonic=10.2)
    r=s.snapshot(now_unix=100.21)
    assert r['encoder_resets']==1 and r['sample_count']==0
    row=s.consume(packet(2,100.19),now_unix=100.21,now_monotonic=10.21)
    assert not row['accepted'] and row['reason']=='captured_before_stationary_pose'


@pytest.mark.parametrize('robot_times,match',[
    ([10.0,10.2],'interpolation gap'),([10.2,10.22],'outside robot-state history')])
def test_missing_or_gapped_capture_time_history_rejected(robot_times,match):
    s=sampler()
    for t in robot_times:s.observe_robot(state(t),now_unix=t+90,now_monotonic=t)
    r=s.consume(packet(1,100.1),now_unix=100.25,now_monotonic=10.25)
    assert not r['accepted'] and match in r['reason']


@pytest.mark.parametrize('age',[.351,-.051])
def test_age_gate_unchanged(age):
    s=sampler();p=packet(1,100.1)
    assert s.consume(p,now_unix=100.1+age,now_monotonic=10.1+age)['reason']=='sample_age_out_of_range'


def test_spatial_and_normal_quality_match_original_window():
    s=sampler();old=SlidingButtonWindow(size=4,max_input_age_sec=.35,max_latest_age_sec=.35,
                                       max_span_sec=2,max_position_rms_m=.0015,
                                       max_position_jump_m=.003,max_normal_rms_deg=1.5)
    for i,z in enumerate([.6,.601,.61,.602]):
        row=feed(s,i+1,100.1+i*.1,z=z)
        assert row['accepted']
        old.add(TimedButtonTransform(i+1,100.1+i*.1,np.array(row['base_to_button']),.95,.0013),now=100.15+i*.1)
    a=s.window.estimate(now=100.45);b=old.estimate(now=100.45)
    assert not a.stable and a.reasons==b.reasons
    assert a.position_rms_m==b.position_rms_m


def test_freshness_rechecked_when_reading_stable_result():
    s=sampler(frame_count=2);feed(s,1,100.1);feed(s,2,100.2)
    assert s.snapshot(now_unix=100.25)['stable']
    stale=s.snapshot(now_unix=100.56)
    assert not stale['stable'] and stale['base_to_button'] is None


def test_clock_step_rejected():
    s=sampler();r=s.consume(packet(1,100.1),now_unix=100.15,now_monotonic=10.05)
    assert r['reason']=='capture_clock_mismatch'


@pytest.mark.parametrize('target',['up','down','6','16'])
def test_only_current_configured_scope(target):
    with pytest.raises(ValueError,match='1..5'):
        LayoutStationarySampler(CONFIG,np.eye(4),target,now_unix=100,now_monotonic=10)


@pytest.mark.parametrize('count',[True,1,3,5])
def test_no_unsupported_sample_count(count):
    with pytest.raises(ValueError,match='2 or 4'):sampler(frame_count=count)


def test_readonly_collector_runs_existing_rpc_and_encoder_interfaces_only():
    offset=time.time()-time.monotonic();sequence=[0];requests=[];reads=[]
    class Robot:
        def state(self,*,max_age):
            assert max_age==.35;reads.append(threading.current_thread().name)
            return state(time.monotonic())
        def __getattr__(self,name):raise AssertionError('unexpected robot action '+name)
    class Vision:
        def request(self,command,**kwargs):
            assert command=='layout_localization_snapshot'
            assert kwargs['include_rgb'] is False and kwargs['target_label']=='2'
            requests.append(kwargs);time.sleep(.04);sequence[0]+=1
            mono=time.monotonic()-.025
            return packet(sequence[0],mono+offset,mono=mono)
    r=collect_stationary_layout_shadow(Vision(),Robot(),np.eye(4),'2',CONFIG)
    assert r['stable'] and r['sample_count']==4 and r['motion_eligible'] is False
    assert len(requests)>=4 and all(n=='layout-shadow-encoders' for n in reads)
    assert not any(t.name=='layout-shadow-encoders' for t in threading.enumerate())


def test_cancellation_never_calls_vision_or_actuator():
    class Robot:
        def state(self,**kwargs):return state(time.monotonic())
    class Vision:
        def request(self,*a,**k):raise AssertionError('cancelled should not poll')
    with pytest.raises(RuntimeError,match='cancelled'):
        collect_stationary_layout_shadow(Vision(),Robot(),np.eye(4),'2',CONFIG,cancelled=lambda:True)


def test_encoder_reader_failure_propagates_and_stops():
    class Robot:
        def state(self,**kwargs):raise TimeoutError('test missing feedback')
    class Vision:
        def request(self,*a,**k):time.sleep(.02);return {'ok':True}
    with pytest.raises(RuntimeError,match='encoder failure.*missing feedback'):
        collect_stationary_layout_shadow(Vision(),Robot(),np.eye(4),'2',CONFIG)
    assert not any(t.name=='layout-shadow-encoders' for t in threading.enumerate())
