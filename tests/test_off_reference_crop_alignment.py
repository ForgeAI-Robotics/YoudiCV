"""Actual crop-jitter failures plus lamp-change/pose/age controls; no robot."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from piper_handeye.off_reference_alignment import recompare_stationary_off
from piper_handeye.paired_button_light import PairedButtonLight, extract_patch, compare_patches
from tests.test_numeric_postpress_registration import synthetic_images

FIXTURE=Path(__file__).parent/'fixtures/off_reference_crop_jitter'
MANIFEST=json.loads((FIXTURE/'manifest.json').read_text())
FRAMES={r['sequence']:r for r in MANIFEST['frames']}


def snapshot(row):
    image=FIXTURE/row['file'];assert hashlib.sha256(image.read_bytes()).hexdigest()==row['sha256']
    return dict(path=str(image),frame_sequence=row['sequence'],capture_time_unix_sec=row['captured'],
                fresh=True,image_size=[640,360],target_detection=dict(row['target_detection']))


def pair(monkeypatch,a=125289,b=125293,*,pose=True):
    aa,bb=snapshot(FRAMES[a]),snapshot(FRAMES[b])
    monkeypatch.setattr('piper_handeye.paired_button_light.time.time',lambda:FRAMES[b]['captured']+.01)
    p=PairedButtonLight(require_stationary_reference=True,stationary_check=Mock(return_value=pose))
    p.begin_attempt();return p,aa,bb


@pytest.mark.parametrize('a,b',MANIFEST['pairs'])
def test_actual_rejected_off_pairs_recover_without_threshold_changes(monkeypatch,a,b):
    p,aa,bb=pair(monkeypatch,a,b)
    old=compare_patches(extract_patch(aa),extract_patch(bb),'OFF',stationary=True)
    assert old['state']=='UNCERTAIN' and old['background_error']>10
    hook=Mock(side_effect=AssertionError('common crop should not need feature registration'))
    monkeypatch.setattr('piper_handeye.off_reference_alignment.register_reference',hook)
    p.collect(aa,'OFF',stationary=True);p.collect(bb,'OFF',stationary=True)
    r=p.last_reference_comparison
    assert r['state']=='OFF' and len(p.references)==2
    assert r['background_error']<3 and r['background_max_mean_error']==10
    assert r['background_max_p90_error']==25
    assert r['comparison_method']=='common_pixel_crop'
    assert r['off_reference_alignment']['accepted']
    assert r['reference_box']==aa['target_detection']['bbox_xyxy']
    assert r['current_box']==bb['target_detection']['bbox_xyxy']
    assert r['comparison_reference_box']==r['comparison_current_box']
    # No mutation of the original evidence's crop, real box or capture time.
    assert p.references[0].box.tolist()==aa['target_detection']['bbox_xyxy']
    assert np.array_equal(p.references[0].gray,extract_patch(aa).gray)
    assert p.references[0].captured==aa['capture_time_unix_sec']
    hook.assert_not_called()


@pytest.mark.parametrize('target',['up','down'])
def test_call_reference_collection_uses_existing_crop_recovery(monkeypatch,target):
    # Relabel an established crop-jitter fixture to test routing, not arrow accuracy.
    p,aa,bb=pair(monkeypatch)
    aa['target_detection']['label']=target;bb['target_detection']['label']=target
    p.collect(aa,'OFF',stationary=True);p.collect(bb,'OFF',stationary=True)
    result=p.last_reference_comparison
    assert result['state']=='OFF' and len(p.references)==2
    assert result['off_reference_alignment']['policy']['scope']=='stationary_call_pre_press_off_only'
    assert all(r.target_label==target for r in p.references)
    assert result['background_max_mean_error']==10 and result['background_max_p90_error']==25


@pytest.mark.parametrize('target',['up','down'])
@pytest.mark.parametrize('fault',['moving','pose','not_fresh','mixed_frame'])
def test_call_crop_recovery_keeps_motion_pose_and_frame_guards(monkeypatch,target,fault):
    p,aa,bb=pair(monkeypatch,pose=fault!='pose')
    aa['target_detection']['label']=target;bb['target_detection']['label']=target
    if fault=='not_fresh':bb['fresh']=False
    if fault=='mixed_frame':bb['target_detection']['frame_sequence']=aa['frame_sequence']
    p.collect(aa,'OFF',stationary=fault!='moving')
    if fault=='mixed_frame':
        with pytest.raises(ValueError,match='image_box_frame_mismatch'):
            p.collect(bb,'OFF',stationary=True)
    else:p.collect(bb,'OFF',stationary=fault!='moving')
    assert not p.references


@pytest.mark.parametrize('fault',['moving','no_stationary_mode','pose','stale','not_fresh',
                                 'other_label','large_shift','large_scale'])
def test_original_rejection_remains_without_alignment_admission(monkeypatch,fault):
    p,aa,bb=pair(monkeypatch,pose=fault!='pose')
    if fault=='no_stationary_mode':p.require_stationary_reference=False
    if fault=='stale':monkeypatch.setattr('piper_handeye.paired_button_light.time.time',lambda:FRAMES[125293]['captured']+.36)
    if fault=='not_fresh':bb['fresh']=False
    if fault=='other_label':bb['target_detection']['label']='down'
    if fault in {'large_shift','large_scale'}:
        bb['target_detection']=dict(bb['target_detection'])
        bb['target_detection']['bbox_xyxy']=np.array(bb['target_detection']['bbox_xyxy'])+(
            np.array([12,0,12,0]) if fault=='large_shift' else np.array([0,0,12,12]))
        bb['target_detection']['bbox_xyxy']=bb['target_detection']['bbox_xyxy'].tolist()
    p.collect(aa,'OFF',stationary=True);p.collect(bb,'OFF',stationary=fault!='moving')
    assert len(p.references)==0 and p.last_reference_comparison['state']=='UNCERTAIN'


def test_passing_old_path_never_runs_alignment(monkeypatch):
    p,aa,bb=pair(monkeypatch);bb={**bb,'path':aa['path'],
        'target_detection':{**aa['target_detection'],'frame_sequence':bb['frame_sequence']}}
    hook=Mock(side_effect=AssertionError('passing pair reinferred'))
    monkeypatch.setattr('piper_handeye.paired_button_light.recompare_stationary_off',hook)
    p.collect(aa,'OFF',stationary=True);p.collect(bb,'OFF',stationary=True)
    assert len(p.references)==2 and p.last_reference_comparison['state']=='OFF';hook.assert_not_called()


def _synthetic_pair(tmp_path,**options):
    a,b=synthetic_images(tmp_path,**options)
    a=replace(a,target_label='3')
    # Real 2,-4px camera translation plus 4px width YOLO jitter. A shared crop
    # cannot explain the translation by resize, so background fitting is needed.
    box=np.array([128,84,196,148],float)
    image=cv2.imread(b.image);x1,y1,x2,y2=box.astype(int)
    gray=cv2.resize(cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY),(64,64),
                    interpolation=cv2.INTER_AREA).astype(np.float32)
    return a,replace(b,box=box,gray=gray,target_label='3')


def test_background_alignment_can_rescue_real_small_image_motion(tmp_path):
    a,b=_synthetic_pair(tmp_path)
    r,audit=recompare_stationary_off(a,b,compare_patches,stationary_check=lambda:True,
                                    freshness_check=lambda:True,cache={})
    assert audit['accepted'] and r['state']=='OFF'
    assert r['comparison_method']=='background_registered_crop'
    assert audit['background_registration']['accepted']
    assert audit['background_registration']['authorization_context']=='pre_press_off_reference'


@pytest.mark.parametrize('target',['3','up','down'])
@pytest.mark.parametrize('kind',['lit','covered','nonuniform_background'])
def test_real_lamp_change_or_occlusion_cannot_become_off_reference(tmp_path,kind,target):
    a,b=_synthetic_pair(tmp_path,lit=kind=='lit',covered=kind=='covered')
    a=replace(a,target_label=target);b=replace(b,target_label=target)
    if kind=='nonuniform_background':
        image=cv2.imread(b.image);image[:120]=np.minimum(image[:120].astype(int)+70,255).astype(np.uint8)
        cv2.imwrite(b.image,image)
        x1,y1,x2,y2=b.box.astype(int)
        b=replace(b,gray=cv2.resize(cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY),
                                   (64,64),interpolation=cv2.INTER_AREA).astype(np.float32))
    r,audit=recompare_stationary_off(a,b,compare_patches,stationary_check=lambda:True,
                                    freshness_check=lambda:True,cache={})
    assert r is None and not audit['accepted']


@pytest.mark.parametrize('fault',['time','sequence','wrong_label','missing_image','size_change','border','small'])
def test_invalid_original_data_are_not_recovered(tmp_path,fault):
    a,b=(extract_patch(snapshot(FRAMES[s])) for s in MANIFEST['pairs'][0])
    if fault=='time':b=replace(b,captured=a.captured+6.01)
    if fault=='sequence':b=replace(b,sequence=a.sequence)
    if fault=='wrong_label':b=replace(b,target_label='5')
    if fault=='missing_image':a=replace(a,image=str(tmp_path/'missing.jpg'))
    if fault=='size_change':
        path=tmp_path/'other.jpg';cv2.imwrite(str(path),np.zeros((480,640,3),np.uint8));b=replace(b,image=str(path))
    if fault=='border':a=replace(a,box=np.array([1,1,76,76.]));b=replace(b,box=a.box.copy())
    if fault=='small':a=replace(a,box=np.array([30,30,50,50.]));b=replace(b,box=a.box.copy())
    r,audit=recompare_stationary_off(a,b,compare_patches,stationary_check=lambda:True,
                                    freshness_check=lambda:True,cache={})
    assert r is None and not audit['accepted']


def test_new_frame_still_uses_its_own_matching_box(monkeypatch):
    p,aa,bb=pair(monkeypatch);p.collect(aa,'OFF',stationary=True)
    bb['target_detection']['frame_sequence']=aa['frame_sequence']
    with pytest.raises(ValueError,match='image_box_frame_mismatch'):p.collect(bb,'OFF',stationary=True)
    assert not p.references


def test_pose_lost_during_comparison_never_adds_off_references(monkeypatch):
    p,aa,bb=pair(monkeypatch)
    # Admission starts healthy, then recheck after the image comparison fails.
    p.stationary_check=Mock(side_effect=[True,False,False,False])
    p.collect(aa,'OFF',stationary=True);p.collect(bb,'OFF',stationary=True)
    assert not p.references and not p.last_reference_comparison['off_reference_alignment']['accepted']


def test_expired_alignment_budget_never_admits_a_good_common_crop(monkeypatch):
    a,b=(extract_patch(snapshot(FRAMES[s])) for s in MANIFEST['pairs'][0])
    import piper_handeye.off_reference_alignment as module
    clock=[0.];monkeypatch.setattr(module,'time',NS(perf_counter=lambda:clock[0]))
    def delayed(*args,**kw):
        r=compare_patches(*args,**kw);clock[0]=.201;return r
    r,audit=recompare_stationary_off(a,b,delayed,stationary_check=lambda:True,
                                    freshness_check=lambda:True,cache={})
    assert r is None and not audit['accepted']


def test_during_motion_off_collection_never_runs_new_fallback(monkeypatch):
    p,aa,bb=pair(monkeypatch)
    hook=Mock(side_effect=AssertionError('moving collector called alignment'))
    monkeypatch.setattr('piper_handeye.paired_button_light.recompare_stationary_off',hook)
    p.collect(aa,'OFF');p.collect(bb,'OFF')
    assert not p.references;hook.assert_not_called()
