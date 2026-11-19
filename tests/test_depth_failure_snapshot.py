"""Explicit diagnostic snapshots preserve exact failed RGB-D data, without inference."""
from dataclasses import replace
import json
import time
from unittest.mock import Mock
import numpy as np
import pytest
from tests.test_early_light_publication import frame, sample, worker, service


def setup_service(tmp_path, monkeypatch):
    w=worker();f=frame();f.depth_m[0,0]=np.nan
    error='panel plane RMS 42.1 mm exceeds 6.0 mm'
    x=replace(sample(f),error=error,timing={'spans':[dict(name='vision.depth_pose',status='error',
                error_type='RuntimeError',error_message=error,target_label='5')]})
    w._samples.append(x)
    return w,x,service(w,tmp_path,monkeypatch)


def test_explicit_failure_capture_uses_exact_sample_rgb_depth_and_calibration(tmp_path,monkeypatch):
    w,x,s=setup_service(tmp_path,monkeypatch)
    r=s.dispatch(dict(command='depth_failure_snapshot',token='test',target_label='5',after_sequence=99))
    assert r['frame_sequence']==100
    with np.load(r['bundle_path'],allow_pickle=False) as b:
        np.testing.assert_array_equal(b['color'],x.frame.color)
        np.testing.assert_array_equal(b['depth_m'],x.frame.depth_m)
        np.testing.assert_array_equal(b['camera_matrix'],w.camera_matrix)
    m=json.loads(open(r['metadata_path']).read())
    assert m['error_message']==x.error and m['error_type']=='RuntimeError'
    assert m['frame_sequence']==100 and m['pairing']=='same_completed_rgbd_bundle'
    assert m['geometry_options']==w.geometry_options
    assert m['detections'][0]['label']=='5'
    s.camera.latest_frame.assert_not_called();s.detector.detect.assert_not_called()
    assert w.latest_sample() is x and w.latest_valid() is None


@pytest.mark.parametrize('fault',['stale','pose_valid','no_depth_error','no_message','wrong_target','duplicate'])
def test_invalid_or_unrelated_sample_is_never_saved(tmp_path,monkeypatch,fault):
    w,x,s=setup_service(tmp_path,monkeypatch);options={}
    if fault=='stale':x=replace(x,frame=replace(x.frame,received_unix=time.time()-.5))
    if fault=='pose_valid':x=replace(x,pose=object())
    if fault=='no_depth_error':x=replace(x,timing={'spans':[]})
    if fault=='no_message':x=replace(x,error=None)
    if fault=='wrong_target':options['target_label']='6'
    if fault=='duplicate':options['after_sequence']=100
    w._samples.clear();w._samples.append(x)
    with pytest.raises(RuntimeError):s.dispatch(dict(command='depth_failure_snapshot',token='test',**options))
    assert not list(tmp_path.rglob('*.npz'))


def test_failure_snapshot_still_requires_existing_authentication(tmp_path,monkeypatch):
    w,x,s=setup_service(tmp_path,monkeypatch)
    with pytest.raises(PermissionError):s.dispatch(dict(command='depth_failure_snapshot',token='wrong'))
    assert not list(tmp_path.rglob('*.npz'))
