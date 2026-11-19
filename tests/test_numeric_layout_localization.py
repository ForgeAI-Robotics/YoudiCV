"""Same-frame shadow integration, no camera/robot/service calls."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch

import numpy as np
import pytest
import yaml

from piper_handeye.button_pose import ButtonDetection
from piper_handeye.live_button import LiveButtonPerception
from piper_handeye.numeric_layout_localization import NumericLayoutLocalizer
from piper_handeye.numeric_layout_geometry import infer_layout
from piper_handeye.orbbec_v1_stream import LiveRgbdFrame
from tools.offline_numeric_panel_layout_v3 import infer_image as offline_infer
from run_button_vision_server import VisionService,layout_localization_payload

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def scene():
    p=ROOT/'config/numeric_panel_layout_current.json';profile=json.loads(p.read_text())
    k=np.array([[400.,0,320],[0,400.,180],[0,0,1]])
    localizer=NumericLayoutLocalizer(profile,k,dict(min_plane_pixels=50),target_labels=['1','2','3','4','5'],profile_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    frame=LiveRgbdFrame(100,10.,1000.,800,805,np.full((360,640,3),150,np.uint8),np.full((360,640),.6))
    labels=['1','3','9','10','4','11']
    ds=[]
    for label in labels:
        x,y=profile['coordinates'][label];cx,cy=180+45*x,260+35*y
        ds.append(ButtonDetection(0,label,.95,(cx-11,cy-11,cx+11,cy+11)))
    return localizer,frame,tuple(ds)


def test_same_frame_inference_goes_through_real_depth_estimator(scene):
    localizer,frame,ds=scene;r=localizer.localize(frame,'2',ds,now_monotonic=10.05)
    assert r['accepted_2d'] and r['valid_3d'],r
    assert r['frame_sequence']==r['rgbd_bundle_sequence']==100
    assert r['color_timestamp']==800 and r['depth_timestamp']==805
    assert r['pose']['p_camera_m'][2]==pytest.approx(.6)
    assert r['pose']['panel_support_detections']>=3
    assert r['motion_eligible'] is False and r['target_detected'] is False
    assert r['yolo_confidence'] is None and r['light_state']=='not_evaluated'
    assert not any(d.label=='2' for d in ds)
    json.dumps(r,allow_nan=False)


def test_no_depth_does_not_produce_3d_target(scene):
    l,f,d=scene;f=copy.copy(f);object.__setattr__(f,'depth_m',np.full((360,640),np.nan))
    r=l.localize(f,'2',d,now_monotonic=10.05)
    assert r['accepted_2d'] and not r['valid_3d']
    assert 'pose' not in r


def test_depth_is_current_bundle_never_previous_call(scene):
    l,f,d=scene;r1=l.localize(f,'2',d,now_monotonic=10.05)
    f2=LiveRgbdFrame(101,10.1,1000.1,900,905,f.color,np.full((360,640),.8))
    r2=l.localize(f2,'2',d,now_monotonic=10.15)
    assert r1['pose']['p_camera_m'][2]==pytest.approx(.6)
    assert r2['pose']['p_camera_m'][2]==pytest.approx(.8)
    assert r2['frame_sequence']==101


def test_latest_missing_anchors_not_old_candidate(scene):
    l,f,d=scene;assert l.localize(f,'2',d,now_monotonic=10.05)['valid_3d']
    r=l.localize(f,'2',(),now_monotonic=10.05)
    assert not r['accepted_2d'] and not r['valid_3d'] and 'pose' not in r


def test_ambiguous_neighbour_dropped_not_all_reference_evidence(scene):
    l,f,d=scene;original=tuple(d);d=d+(ButtonDetection(0,'11',.9,(490,40,512,62)),)
    r=l.localize(f,'2',d,now_monotonic=10.05)
    assert r['valid_3d'],r
    assert r['reference_filter']['ambiguous_labels']==['11']
    assert r['reference_filter']['discarded_count']==2
    assert '11' not in r['geometry']['anchors']
    assert d[:-1]==original


@pytest.mark.parametrize('target',['up','down','8','16'])
def test_only_configured_numeric_targets(scene,target):
    l,f,d=scene
    assert l.localize(f,target,d,now_monotonic=10.05)['reason']=='target_outside_configured_layout_scope'


def test_existing_yolo_success_not_overridden_or_reinferred(scene):
    l,f,d=scene
    with patch('piper_handeye.numeric_layout_localization.infer_layout',side_effect=AssertionError):
        r=l.localize(f,'1',d,now_monotonic=10.05)
    assert r['reason']=='existing_yolo_target_available'


def test_ambiguous_real_target_not_overridden(scene):
    l,f,d=scene;dd=d+(ButtonDetection(0,'1',.95,(100,100,120,120)),)
    assert l.localize(f,'1',dd,now_monotonic=10.05)['reason']=='ambiguous_strong_target_not_overridden'


@pytest.mark.parametrize('now',[10.36,9.9])
def test_stale_or_future_frame_rejected(scene,now):
    l,f,d=scene
    assert l.localize(f,'2',d,now_monotonic=now)['reason']=='rgbd_frame_stale'


def test_rgb_and_depth_size_mismatch_rejected(scene):
    l,f,d=scene;f=copy.copy(f);object.__setattr__(f,'depth_m',np.ones((180,320)))
    assert l.localize(f,'2',d,now_monotonic=10.05)['reason']=='rgb_depth_shape_mismatch'


def test_off_default_and_profile_hash_checked_before_start(scene,tmp_path):
    l,_,_=scene
    assert NumericLayoutLocalizer.from_config({},tmp_path/'a.yaml',l.camera_matrix,{},{}) is None
    settings=dict(mode='shadow',profile_path=str(ROOT/'config/numeric_panel_layout_current.json'),profile_sha256='bad',target_labels=['2'])
    with pytest.raises(ValueError,match='SHA256'):
        NumericLayoutLocalizer.from_config(settings,tmp_path/'a.yaml',l.camera_matrix,{},.7)
    with pytest.raises(ValueError,match='off/shadow'):
        NumericLayoutLocalizer.from_config({'mode':'enabled'},tmp_path/'a.yaml',l.camera_matrix,{},.7)


def test_runtime_config_matches_profile(scene):
    l,_,_=scene;c=yaml.safe_load((ROOT/'piper_button_pose.yaml').read_text())
    configured=NumericLayoutLocalizer.from_config(c['numeric_layout_localization'],ROOT/'piper_button_pose.yaml',l.camera_matrix,c['button_geometry'],.7)
    assert configured.targets==frozenset('12345')


def make_service(scene):
    l,f,d=scene;record=l.localize(f,'2',d,now_monotonic=10.05)
    sample=SimpleNamespace(sequence=100,target_label='2',frame=f,pose=None,detections=d,numeric_layout_localization=record,timing={})
    service=VisionService.__new__(VisionService);service.args=SimpleNamespace(token='test')
    service.perception=MagicMock();service.perception.target_label='2'
    service.perception.latest_sample.return_value=sample;service.perception.latest_valid.return_value=None
    service.perception.latest_light_sample.return_value=sample
    service.camera=MagicMock();service.safety=MagicMock();service.detector=MagicMock()
    return service,sample


def test_readonly_rpc_uses_exact_sample_no_camera_or_inference(scene):
    service,sample=make_service(scene)
    with patch('run_button_vision_server.time.time',return_value=1000.05),patch('run_button_vision_server.cv2.imwrite',side_effect=AssertionError):
        r=service.dispatch(dict(command='layout_localization_snapshot',token='test',target_label='2',include_rgb=True))
    assert r['fresh'] and r['valid_3d'] and r['frame_sequence']==100 and r['jpeg_base64']
    service.camera.latest_frame.assert_not_called();service.detector.detect.assert_not_called();service.safety.reset.assert_not_called()


def test_rpc_barrier_and_freshness(scene):
    service,_=make_service(scene)
    with patch('run_button_vision_server.time.time',return_value=1001.):
        r=service.dispatch(dict(command='layout_localization_snapshot',token='test'))
    assert not r['fresh'] and not r['valid_3d']
    assert not service.dispatch(dict(command='layout_localization_snapshot',token='test',after_sequence=100))['fresh']


def test_wrong_frame_bundle_and_label_rejected(scene):
    _,sample=make_service(scene)
    sample.numeric_layout_localization['frame_sequence']=99
    with pytest.raises(RuntimeError,match='inconsistent'):layout_localization_payload(sample)
    sample.numeric_layout_localization['frame_sequence']=100;sample.numeric_layout_localization['target_label']='3'
    with pytest.raises(RuntimeError,match='inconsistent'):layout_localization_payload(sample)


def test_shadow_never_appears_as_ordinary_3d_pose_or_lamp_box(scene):
    service,sample=make_service(scene)
    with patch('run_button_vision_server.time.time',return_value=1000.05),patch('run_button_vision_server.realtime_payload',return_value={}):
        latest=service.dispatch(dict(command='latest',token='test'))
        visible=service.dispatch(dict(command='visible_buttons',token='test'))
        with patch('run_button_vision_server.cv2.imwrite',return_value=True),patch('run_button_vision_server.prune_capture_groups'):
            lamp=service.dispatch(dict(command='light_snapshot',token='test',target_label='2'))
    assert latest['sample'] is None and latest['numeric_layout_localization']['valid_3d']
    assert not any(d['label']=='2' for d in visible['detections'])
    assert 'target_detection' not in lamp and lamp['numeric_layout_localization']['valid_3d']
    assert sample.pose is None


def test_live_worker_attaches_shadow_to_failed_same_frame_without_promoting(scene):
    localizer,frame,ds=scene
    class Camera:
        latest_sequence=100
        def wait_for_sequence(self,sequence,timeout):
            worker._running=False
            return frame
    worker=LiveButtonPerception(Camera(),MagicMock(),localizer.camera_matrix,target_label='2',target_confidence=.7,geometry_options={},numeric_layout_localizer=localizer)
    worker._running=True
    with patch('piper_handeye.live_button.detect_with_enhancement_fallback',return_value=(ds,{})),patch('piper_handeye.numeric_layout_localization.time.monotonic',return_value=10.05):worker._run()
    sample=worker._samples[-1]
    assert sample.frame is frame and sample.sequence==100
    assert sample.pose is None and sample.error
    assert sample.numeric_layout_localization['valid_3d']
    assert not worker.valid_window('2')


def test_production_geometry_matches_frozen_offline_on_recorded_cases():
    base=ROOT/'tests/fixtures/numeric_panel_layout_offline/20261027_numeric_panel_layout';m=json.loads((base/'manifest.json').read_text())
    import cv2
    for f in m['frames']:
        if f['split']!='validation':continue
        image=cv2.imread(str(base/f['image']))
        for target in f['targets']:
            aa=[dict(label=l,**v) for l,v in f['buttons'].items() if l!=target]
            a=infer_layout(m['template']['coordinates'],aa,target,image);b=offline_infer(m['template']['coordinates'],aa,target,image)
            assert a['accepted']==b['accepted'] and a['reason']==b['reason']
            if a['accepted']:np.testing.assert_allclose(a['center'],b['center'],atol=1e-6)
