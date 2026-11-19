"""Actual rejected 4-frame, provenance/phase gates and adverse images; no robot."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from piper_handeye.button_background_registration import register_reference, POLICY
from piper_handeye.paired_button_light import (
    ButtonPatch, PairedButtonLight, compare_patches, extract_patch,
)
from tests.test_panel_background_reference import GLARE, snapshot as glare_snapshot


FIXTURE=Path(__file__).parent/'fixtures/numeric_postpress_registration'
FRAMES=json.loads((FIXTURE/'manifest.json').read_text())['frames']


def snapshot(frame,label='4'):
    p=FIXTURE/frame['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==frame['sha256']
    return dict(path=str(p),frame_sequence=frame['sequence'],
                capture_time_unix_sec=frame['captured'],fresh=True,image_size=[640,360],
                target_detection=dict(label=label,confidence=.95,
                    frame_sequence=frame['sequence'],bbox_xyxy=frame['box']))


def pair(monkeypatch,*,pose=True):
    ref=extract_patch(snapshot(FRAMES[0]))
    p=PairedButtonLight(require_stationary_reference=True,stationary_check=Mock(return_value=pose))
    p.begin_attempt();p.references.append(ref);p.stationary_reference_ready=True
    p.frozen=True;p.press_started=ref.captured+.01;p.last_reference_sequence=ref.sequence
    monkeypatch.setattr('piper_handeye.paired_button_light.time.time',lambda:FRAMES[1]['captured']+.01)
    return p


def test_real_four_keeps_original_rejection_and_registration_rescues_same_frame(monkeypatch):
    before,after=(extract_patch(snapshot(f)) for f in FRAMES[:2])
    raw=compare_patches(before,after,'ON',stationary=True,strong_on=True)
    assert raw['state']=='UNCERTAIN'
    assert raw['background_error']==pytest.approx(10.066128730773926)
    p=pair(monkeypatch);r=p.classify(snapshot(FRAMES[1]),'ON',stationary=True,strong_on=True)
    assert r['state']=='ON'
    assert r['background_error']<=10
    assert r['background_registration']['accepted'] is True
    assert r['unaligned_comparison']['state']=='UNCERTAIN'
    assert r['reference_box']==FRAMES[0]['box']
    assert r['comparison_reference_box']==FRAMES[1]['box']
    assert p.references[0].box.tolist()==FRAMES[0]['box']
    assert r['background_registration']['raw_center_shift_fraction'][1]>.06
    assert r['background_registration']['policy']['max_center_shift_fraction']==.10
    assert r['background_registration']['policy']['max_log_size_change']==.08


@pytest.mark.parametrize('fault',['moving','pose','arrow','no_strong_on','stale','not_fresh','unconfirmed_reference'])
def test_actual_frame_requires_all_stationary_numeric_authorizations(monkeypatch,fault):
    p=pair(monkeypatch,pose=fault!='pose');s=snapshot(FRAMES[1])
    if fault=='arrow':s['target_detection']['label']='down'
    if fault=='stale':monkeypatch.setattr('piper_handeye.paired_button_light.time.time',lambda:FRAMES[1]['captured']+.36)
    if fault=='not_fresh':s['fresh']=False
    if fault=='unconfirmed_reference':p.stationary_reference_ready=False
    r=p.classify(s,'ON',stationary=fault!='moving',strong_on=fault!='no_strong_on')
    assert r['state']=='UNCERTAIN'
    assert 'background_registration' not in r


def test_no_extra_registration_on_passing_off(monkeypatch):
    p=pair(monkeypatch);hook=Mock(side_effect=AssertionError('OFF must not invoke registration'))
    monkeypatch.setattr('piper_handeye.paired_button_light.register_reference',hook)
    r=p.classify(snapshot(FRAMES[2]),'OFF',stationary=True,strong_on=True)
    assert r['state']=='OFF';hook.assert_not_called()


def test_pose_check_wait_cannot_admit_an_expired_picture(monkeypatch):
    p=pair(monkeypatch);clock=[FRAMES[1]['captured']+.01]
    monkeypatch.setattr('piper_handeye.paired_button_light.time.time',lambda:clock[0])
    def delayed_pose():
        clock[0]+=.4
        return True
    p.stationary_check=delayed_pose
    hook=Mock(side_effect=AssertionError('do not register stale picture'))
    monkeypatch.setattr('piper_handeye.paired_button_light.register_reference',hook)
    assert p.classify(snapshot(FRAMES[1]),'ON',stationary=True,strong_on=True)['state']=='UNCERTAIN'
    hook.assert_not_called()


@pytest.mark.parametrize('fault',['time','sequence','shift','scale','small','missing_image','frame_size','border'])
def test_registration_is_not_a_general_view_override(tmp_path,fault):
    a,b=(extract_patch(snapshot(f)) for f in FRAMES[:2])
    if fault=='time':b.captured=a.captured+31
    if fault=='sequence':b.sequence=a.sequence
    if fault=='shift':b.box=b.box+np.array([20,0,20,0])
    if fault=='scale':b.box=b.box+np.array([0,0,12,12])
    if fault=='small':a.box=np.array([0,0,20,20])
    if fault=='missing_image':a.image=str(tmp_path/'missing.jpg')
    if fault=='frame_size':
        dst=tmp_path/'smaller.png';cv2.imwrite(str(dst),np.zeros((120,160,3),np.uint8));b.image=str(dst)
    if fault=='border':a.box=np.array([0,0,70,70]);b.box=a.box.copy()
    gray,r=register_reference(a,b)
    assert gray is None and not r['accepted']


def synthetic_images(tmp_path,*,lit=False,exposure=0,covered=False):
    rng=np.random.default_rng(77)
    scene=rng.integers(30,120,(240,320,3),dtype=np.uint8)
    scene=cv2.GaussianBlur(scene,(3,3),0)
    cv2.circle(scene,(160,120),32,(24,24,24),-1)
    cv2.putText(scene,'4',(143,133),cv2.FONT_HERSHEY_SIMPLEX,.9,(100,100,100),2,cv2.LINE_AA)
    first=tmp_path/'before.png';cv2.imwrite(str(first),scene)
    if lit:cv2.putText(scene,'4',(143,133),cv2.FONT_HERSHEY_SIMPLEX,.9,(235,235,235),2,cv2.LINE_AA)
    scene=np.clip(scene.astype(float)+exposure,0,255).astype(np.uint8)
    after=cv2.warpAffine(scene,np.array([[1.,0.,2.],[0.,1.,-4.]]),(320,240))
    if covered:after[60:165,100:220]=200
    second=tmp_path/'after.png';cv2.imwrite(str(second),after)
    a=ButtonPatch(np.zeros((64,64)),np.array([128,88,192,152]),1,100.,str(first))
    b=ButtonPatch(np.zeros((64,64)),np.array([130,84,194,148]),2,101.,str(second))
    b.gray=cv2.resize(cv2.cvtColor(after[84:148,130:194],cv2.COLOR_BGR2GRAY),(64,64)).astype(np.float32)
    return a,b


@pytest.mark.parametrize('exposure',[0,25,-15])
def test_registration_does_not_turn_global_lighting_into_on(tmp_path,exposure):
    a,b=synthetic_images(tmp_path,exposure=exposure)
    registered,r=register_reference(a,b)
    assert r['accepted'],r
    assert compare_patches(replace(a,gray=registered,box=b.box.copy()),b,'ON',
                           stationary=True,strong_on=True)['state']!='ON'


def test_local_button_change_is_preserved_by_background_fit(tmp_path):
    a,b=synthetic_images(tmp_path,lit=True)
    registered,r=register_reference(a,b)
    assert r['accepted'],r
    assert compare_patches(replace(a,gray=registered,box=b.box.copy()),b,'ON',
                           stationary=True,strong_on=True)['state']=='ON'


def test_occlusion_is_not_registration_authority(tmp_path):
    a,b=synthetic_images(tmp_path,covered=True)
    registered,r=register_reference(a,b)
    if registered is not None:
        assert compare_patches(replace(a,gray=registered,box=b.box.copy()),b,'ON',
                               stationary=True,strong_on=True)['state']!='ON'


@pytest.mark.parametrize('record',GLARE)
@pytest.mark.parametrize('scale',[1,2])
def test_real_unlit_glare_cannot_be_promoted_through_numeric_classify(monkeypatch,tmp_path,record,scale):
    a=glare_snapshot(record['before']);b=glare_snapshot(record['after'])
    for s in (a,b):s['target_detection']['label']='4';s['fresh']=True
    if scale==2:
        for index,s in enumerate((a,b)):
            raw=cv2.imread(s['path']);path=tmp_path/f'glare_{index}.png'
            cv2.imwrite(str(path),cv2.resize(raw,None,fx=2,fy=2))
            s['path']=str(path)
            s['target_detection']['bbox_xyxy']=(np.array(s['target_detection']['bbox_xyxy'])*2).tolist()
    ref=extract_patch(a);p=PairedButtonLight(require_stationary_reference=True,stationary_check=lambda:True)
    p.references.append(ref);p.stationary_reference_ready=True;p.frozen=True
    p.press_started=ref.captured+.001;p.last_reference_sequence=ref.sequence
    monkeypatch.setattr('piper_handeye.paired_button_light.time.time',lambda:b['capture_time_unix_sec']+.01)
    assert p.classify(b,'ON',stationary=True,strong_on=True)['state']!='ON'


def test_registration_selects_original_best_reference_once(monkeypatch):
    p=pair(monkeypatch)
    p.references.append(replace(p.references[0],sequence=17257))
    original=__import__('piper_handeye.paired_button_light',fromlist=['register_reference']).register_reference
    hook=Mock(wraps=original);monkeypatch.setattr('piper_handeye.paired_button_light.register_reference',hook)
    r=p.classify(snapshot(FRAMES[1]),'ON',stationary=True,strong_on=True)
    assert r['state']=='ON' and hook.call_count==1


def test_reported_limits_match_executed_policy_and_two_on_requirement():
    from run_button_light_closed_loop import LIGHT_EVIDENCE_POLICY
    from piper_handeye.button_light_hardware import POST_PRESS_ON_CONFIRMATION_FRAMES
    policy=LIGHT_EVIDENCE_POLICY['numeric_post_press_registration']
    for key in ('max_center_shift_fraction','max_log_size_change'):
        assert policy[key]==POLICY[key]
    assert POST_PRESS_ON_CONFIRMATION_FRAMES==2

