"""Neighbour veto and real task handoff, with no camera, robot or model IO."""
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import pytest

from piper_handeye.button_pose import ButtonDetection
from piper_handeye.numeric_yolo_consistency import inspect_numeric_target, target_conflicted, admitted_detections
from piper_handeye.numeric_layout_localization import NumericLayoutLocalizer
from piper_handeye.numeric_layout_handoff import LayoutTaskVision, LayoutTaskPolicy, require_layout_server_support
from piper_handeye.live_button import LiveButtonPerception, LiveButtonSample
from piper_handeye.orbbec_v1_stream import LiveRgbdFrame

PROFILE=Path(__file__).resolve().parents[1]/'config/numeric_panel_layout_current.json'
COORDINATES=json.loads(PROFILE.read_text())['coordinates']
DIGEST=hashlib.sha256(PROFILE.read_bytes()).hexdigest()


def button(label, actual=None, confidence=.95):
    x,y=COORDINATES[actual or label];cx,cy=220+65*x,300+35*y
    return ButtonDetection(0,label,confidence,(cx-12,cy-12,cx+12,cy+12))


def scene(target='5',wrong=False):
    labels=['1','2','3','4','9','10','11']
    ds=[button(l) for l in labels if l!=target and (not wrong or l!='3')]
    ds.append(button(target,'3' if wrong else None,.99))
    return tuple(ds)


def inspect(ds,target='5'):
    return inspect_numeric_target(COORDINATES,ds,target,[640,360])


def test_high_confidence_three_as_five_rejected_by_independent_neighbours():
    ds=scene(wrong=True);before=copy.deepcopy(ds);r=inspect(ds)
    assert r['decision']=='conflict' and r['alternative_label']=='3'
    assert r['target_confidence']==.99 and r['claimed_cell_distance']>1.9
    assert '5' not in r['anchors'] and len(r['anchors'])>=4
    assert ds==before and ds[-1].label=='5'
    assert all(d.label!='5' for d in admitted_detections(ds,r))


@pytest.mark.parametrize('target',list('12345'))
def test_correct_buttons_not_vetoed(target):
    ds=scene(target=target);r=inspect(ds,target)
    assert r['decision']=='consistent',r
    assert admitted_detections(ds,r) is ds


@pytest.mark.parametrize('count',[0,1,2,3])
def test_insufficient_references_leave_original_admission_unchanged(count):
    ds=tuple([button('5','3'),*[button(l) for l in ['1','2','9'][:count]]])
    r=inspect(ds)
    assert r['decision']=='unverified' and admitted_detections(ds,r) is ds


def test_same_column_only_cannot_veto():
    ds=tuple([button('5','3'),*[button(l) for l in ['1','2','4','6']]])
    assert inspect(ds)['decision']=='unverified'


def test_reference_overlapping_target_does_not_count_as_independent():
    ds=tuple([button('5','3'),button('3'),button('1'),button('2'),button('9')])
    assert inspect(ds)['decision']=='unverified'


def test_duplicate_target_not_overridden():
    ds=scene(wrong=True)+(button('5','6'),)
    assert inspect(ds)['reason']=='no_unique_complete_target'


@pytest.mark.parametrize('field,value',[('frame_sequence',99),('profile_sha256','b'*64),('target_label','2')])
def test_mismatched_audit_cannot_veto_new_frame_or_other_target(field,value):
    ds=scene(wrong=True);r=dict(inspect(ds),frame_sequence=100,profile_sha256=DIGEST)
    r[field]=value
    assert admitted_detections(ds,r,target='5',sequence=100,digest=DIGEST) is ds


def make_worker(wrong=True):
    frame=LiveRgbdFrame(100,10.,1000.,100,101,np.full((360,640,3),140,np.uint8),np.full((360,640),.6))
    localizer=NumericLayoutLocalizer(json.loads(PROFILE.read_text()),np.array([[400,0,320],[0,400,180],[0,0,1]]),
        dict(min_plane_pixels=50),target_labels=list('12345'),profile_sha256=DIGEST,yolo_consistency_enabled=True)
    worker=LiveButtonPerception(Mock(),Mock(),localizer.camera_matrix,target_label='5',target_confidence=.7,
                               geometry_options=localizer.geometry_options,numeric_layout_localizer=localizer)
    def take(*args,**kwargs):
        worker._running=False
        return frame
    worker.camera.wait_for_sequence.side_effect=take
    worker._running=True
    with patch('piper_handeye.live_button.detect_with_enhancement_fallback',return_value=(scene(wrong=wrong),{})),patch('piper_handeye.numeric_layout_localization.time.monotonic',return_value=10.05):
        worker._run()
    assert worker._fatal_error is None
    return worker,frame


def test_actual_perception_keeps_raw_wrong_yolo_but_never_creates_wrong_pose():
    worker,frame=make_worker()
    s=worker.latest_sample()
    assert target_conflicted(s.numeric_yolo_consistency)
    assert s.pose is None and 'numeric_yolo_layout_conflict' in s.error
    assert next(d for d in s.detections if d.label=='5').center_pixel==button('3').center_pixel
    assert worker.latest_valid() is None and worker.valid_window()==()
    # Real depth estimator can locate the requested, geometrically separate 5.
    assert s.numeric_layout_localization['valid_3d']
    assert s.numeric_layout_localization['yolo_confidence'] is None
    assert s.numeric_layout_localization['center_xy'][1]<button('3').center_pixel[1]-50


def test_cached_valid_pose_before_conflict_cannot_be_reused():
    w,f=make_worker();blocked=w._samples[-1]
    old=LiveButtonSample(99,'5',f,object(),None,0.,scene(wrong=True))
    w._samples.appendleft(old)
    assert w.latest_valid('5') is None and not w.valid_window('5')
    later=LiveButtonSample(101,'5',f,object(),None,0.,scene())
    w._samples.append(later)
    assert w.latest_valid('5') is later and w.valid_window('5')==(later,)


def test_task_visible_view_filters_conflict_without_changing_server_raw_evidence():
    w,f=make_worker();s=w.latest_sample()
    raw=dict(frame_sequence=100,numeric_yolo_consistency=s.numeric_yolo_consistency,
             detections=[dict(label=d.label,confidence=d.confidence,bbox_xyxy=list(d.bbox_xyxy)) for d in s.detections])
    policy=LayoutTaskPolicy(dict(mode='shadow',task_fallback_enabled=True,yolo_consistency_enabled=True,
                                target_labels=list('12345'),profile_sha256=DIGEST),'5')
    v=LayoutTaskVision(NS(request=lambda *_a,**_k:raw),policy)
    result=v.request('visible_buttons')
    assert any(d['label']=='5' for d in raw['detections'])
    assert not any(d['label']=='5' for d in result['detections'])
    assert result['yolo_detections'] is raw['detections']


def test_new_client_rejects_old_service_before_motion():
    config=dict(numeric_layout_localization=dict(mode='shadow',task_fallback_enabled=True,
        yolo_consistency_enabled=True,target_labels=list('12345'),profile_sha256=DIGEST))
    health=dict(numeric_layout_task_handoff=dict(version=1,enabled=True,profile_sha256=DIGEST))
    with pytest.raises(RuntimeError,match='consistency'):require_layout_server_support(health,config,'5')
    health['numeric_yolo_consistency']=dict(version=1,enabled=True,profile_sha256=DIGEST)
    require_layout_server_support(health,config,'5')


CASES=json.loads((Path(__file__).parent/'fixtures/numeric_yolo_consistency/recorded_conflicts.json').read_text())


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['id'])
def test_recorded_three_as_five_conflicts_are_rejected_without_relabel(case):
    ds=tuple(ButtonDetection(0,d['label'],d['confidence'],tuple(d['bbox_xyxy'])) for d in case['detections'])
    audit=inspect_numeric_target(COORDINATES,ds,'5',case['image_size'])
    assert audit['decision']=='conflict' and audit['alternative_label']==case['expected_alternative']
    assert next(d for d in ds if d.label=='5').confidence>.85


def test_recorded_conflict_transitions_to_existing_bounded_look_up_not_wrong_press():
    from piper_handeye.numeric_visibility import wait_numeric_visibility
    from test_numeric_visibility_recovery import Clock
    case=CASES[0];ds=tuple(ButtonDetection(0,d['label'],d['confidence'],tuple(d['bbox_xyxy'])) for d in case['detections'])
    audit=inspect(ds);clock=Clock()
    class Client:
        seq=0
        def request(self,*args,**kwargs):
            self.seq+=1
            return dict(frame_sequence=self.seq,capture_time_unix_sec=time.time(),image_size=[640,360],
                detections=[d for d in case['detections'] if d['confidence']>=.5],
                detection_enhancement=dict(raw_detections=case['detections']),
                numeric_yolo_consistency=dict(audit,frame_sequence=self.seq,profile_sha256=DIGEST))
    settings=dict(mode='shadow',task_fallback_enabled=True,yolo_consistency_enabled=True,
                  target_labels=list('12345'),profile_sha256=DIGEST)
    response,visible=wait_numeric_visibility(LayoutTaskVision(Client(),LayoutTaskPolicy(settings,'5')),
        target_label='5',min_confidence=.7,sequence_barrier=0,timeout_sec=4.,
        clock=clock.time,sleep=clock.sleep)
    assert visible is False and not any(d['label']=='5' for d in response['detections'])
    assert response['numeric_visibility_audit'][-1]['mode']=='grid_prediction'
    assert response['numeric_visibility_audit'][-1]['rejected_by_layout']
    assert clock.now<.5  # rejected raw scores do not add the weak-hint wait.


def test_actual_lamp_rpc_does_not_offer_wrong_raw_five_as_light_region(tmp_path,monkeypatch):
    import run_button_vision_server as server
    worker,frame=make_worker();sample=worker.latest_sample()
    service=server.VisionService.__new__(server.VisionService)
    service.args=NS(token='test')
    service.perception=Mock();service.perception.target_label='5'
    service.perception.latest_sample.return_value=sample
    service.perception.latest_light_sample.return_value=sample
    monkeypatch.setattr(server,'ROOT',tmp_path)
    monkeypatch.setattr(server.cv2,'imwrite',lambda *a,**kw:True)
    monkeypatch.setattr(server,'prune_capture_groups',lambda *a,**kw:None)
    monkeypatch.setattr(server.time,'time',lambda:1000.05)
    response=service.dispatch(dict(command='light_snapshot',token='test',target_label='5'))
    assert 'target_detection' not in response
    assert response['numeric_yolo_consistency']['alternative_label']=='3'
    assert sample.detections[-1].label=='5'
