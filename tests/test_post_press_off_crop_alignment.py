"""Round 77: occluded motion UNKNOWNs stay separate from stationary OFFs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import hashlib
import json
import time

import cv2
import numpy as np
import pytest

from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.off_reference_alignment import recompare_stationary_post_press_off
from piper_handeye.paired_button_light import PairedButtonLight, extract_patch, compare_patches
from tests.test_panel_background_reference import SCENES, snapshot as lamp_snapshot

FIXTURE = Path(__file__).parent/'fixtures/post_press_off_crop_jitter'
FRAMES = json.loads((FIXTURE/'manifest.json').read_text())['frames']
BEFORE = [f for f in FRAMES if f['position']=='before_press']
AFTER = [f for f in FRAMES if f['position'] in {'near','near_recheck'}]
COVERED = [f for f in FRAMES if f['position']=='press_monitor']


def snapshot(frame):
    path=FIXTURE/frame['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==frame['sha256']
    return dict(path=str(path),frame_sequence=frame['sequence'],
        capture_time_unix_sec=frame['captured'],fresh=True,image_size=[640,360],
        target_detection=frame['target_detection'])


def frozen_pair(monkeypatch, frame=None, *, pose=True):
    clock=[(frame or AFTER[0])['captured']+.01]
    monkeypatch.setattr('piper_handeye.paired_button_light.time.time',lambda:clock[0])
    pair=PairedButtonLight(require_stationary_reference=True,
                          stationary_check=Mock(return_value=pose))
    pair.begin_attempt()
    pair.references.extend(extract_patch(snapshot(f)) for f in BEFORE)
    pair.stationary_reference_ready=True;pair.frozen=True
    pair.press_started=BEFORE[-1]['captured']+.01
    pair.last_reference_sequence=BEFORE[-1]['sequence']
    return pair,clock


@pytest.mark.parametrize('frame',AFTER,ids=lambda f:str(f['sequence']))
def test_real_nine_stationary_frames_confirm_off_without_changing_limits(monkeypatch,frame):
    pair,_=frozen_pair(monkeypatch,frame)
    references=[(p.sequence,p.box.copy(),p.gray.copy(),p.image,p.captured) for p in pair.references]
    raw=pair.classify(snapshot(frame),'OFF',stationary=True)
    assert raw['state']==frame['original_state']
    result=pair.classify(snapshot(frame),'OFF',stationary=True,
                         allow_post_press_off_alignment=True)
    assert result['state']=='OFF'
    assert result['background_max_mean_error']==10
    assert result['background_max_p90_error']==25
    assert result['center_error']<=8 and result['changed_fraction']<.02
    if raw['state']=='UNCERTAIN':
        assert result['post_press_off_alignment']['accepted']
        assert result['comparison_method']=='post_press_common_pixel_crop'
        assert result['comparison_reference_box']==result['comparison_current_box']
        assert result['reference_sequence']==raw['reference_sequence']
        assert result['reference_selection_background_error']==raw['background_error']
    else:assert 'post_press_off_alignment' not in result
    assert len(pair.references)==2 and pair.frozen
    for old,new in zip(references,pair.references):
        assert old[0]==new.sequence and old[3]==new.image and old[4]==new.captured
        np.testing.assert_array_equal(old[1],new.box)
        np.testing.assert_array_equal(old[2],new.gray)


def test_occluded_first_two_frames_do_not_poison_stationary_off_recheck(monkeypatch):
    pair,clock=frozen_pair(monkeypatch)
    phase=[0]; served=set(); pending=iter(AFTER[:3])
    def request(*args,**kwargs):
        if phase[0]<2:
            if phase[0] in served:return {'fresh':False}
            frame=COVERED[phase[0]];served.add(phase[0])
        else:frame=next(pending)
        clock[0]=frame['captured']+.01
        return snapshot(frame)
    recognizer=NS(observe=Mock(return_value={'state':'OFF','recognizer':'opencv_self_contrast'}))
    observer=FreshLightObserver(NS(request=request),recognizer,'down',paired_light=pair)
    assert observer.observe('press_monitor')=='UNCERTAIN'
    phase[0]=1
    assert observer.observe('press_monitor')=='UNCERTAIN'
    recognizer.observe.assert_not_called()
    phase[0]=2
    # Existing recheck starts a fresh streak, independent of motion UNKNOWNs.
    result=observer.reobserve_post_press('near')
    assert result['confirmed_state']=='OFF' and result['samples']==3
    assert [f['frame_sequence'] for f in result['frames']]==[f['sequence'] for f in AFTER[:3]]
    assert all(f['frame_sequence'] not in {x['sequence'] for x in COVERED}
               for f in result['frames'])
    assert pair.frozen and len(pair.references)==2


def test_recovered_stationary_off_reaches_existing_fresh_target_retry(monkeypatch):
    from piper_handeye.button_light_flow import ButtonLightFlow
    from tests.test_button_light_flow import FakeHardware
    pair,clock=frozen_pair(monkeypatch)
    clock[0]=AFTER[0]['captured']-.05
    pending=iter(AFTER[:3])
    def request(*args,**kwargs):
        frame=next(pending);clock[0]=frame['captured']+.01;return snapshot(frame)
    observer=FreshLightObserver(NS(request=request),
        NS(observe=lambda *a,**kw:{'state':'OFF'}),'down',paired_light=pair)
    class Hardware(FakeHardware):
        def reobserve_post_press(self,position):
            self.actions.append(('recheck',position));return observer.reobserve_post_press(position)
    hardware=Hardware(['OFF','OFF'],['UNCERTAIN']*3+['ON'])
    result=ButtonLightFlow(hardware).run()
    assert result.status=='arrived_after_verified_press' and result.press_attempts==2
    assert hardware.actions.count(('refresh',))==2
    assert hardware.actions.count(('press',1))==1 and hardware.actions.count(('press',2))==1
    # This verifies authorization/dispatch only; no physical retry is simulated.


@pytest.mark.parametrize('label',['up','down','1','3','5','16'])
def test_supported_buttons_share_geometry_without_changing_classification(monkeypatch,label):
    pair,_=frozen_pair(monkeypatch)
    pair.references=type(pair.references)((replace(p,target_label=label) for p in pair.references),maxlen=64)
    s=snapshot(AFTER[0]);s['target_detection']={**s['target_detection'],'label':label}
    result=pair.classify(s,'OFF',stationary=True,allow_post_press_off_alignment=True)
    assert result['state']=='OFF' and result['post_press_off_alignment']['accepted']


def test_passing_off_and_actual_on_never_call_new_fallback(monkeypatch):
    hook=Mock(side_effect=AssertionError('passing OFF/absolute ON must keep original path'))
    monkeypatch.setattr('piper_handeye.paired_button_light.recompare_stationary_post_press_off',hook)
    pair,_=frozen_pair(monkeypatch,AFTER[-1])
    assert pair.classify(snapshot(AFTER[-1]),'OFF',stationary=True,
                         allow_post_press_off_alignment=True)['state']=='OFF'
    pair.classify(snapshot(AFTER[-1]),'ON',stationary=True,allow_post_press_off_alignment=True)
    hook.assert_not_called()


def test_reported_post_press_policy_matches_runtime():
    from piper_handeye.off_reference_alignment import POST_PRESS_POLICY
    from run_button_light_closed_loop import LIGHT_EVIDENCE_POLICY
    policy=LIGHT_EVIDENCE_POLICY['post_press_off_crop_alignment']
    for key in ('min_button_size_px','max_center_shift_fraction','max_log_size_change',
                'max_pair_span_sec','processing_budget_sec'):
        assert policy[key]==POST_PRESS_POLICY[key]


@pytest.mark.parametrize('fault',['moving','opt_out','pose','stale','not_fresh','unconfirmed',
                                'no_stationary_mode','wrong_label','mixed_frame'])
def test_rejected_inputs_cannot_use_new_off_comparison(monkeypatch,fault):
    pair,clock=frozen_pair(monkeypatch,pose=fault!='pose');s=snapshot(AFTER[0])
    if fault=='stale':clock[0]=s['capture_time_unix_sec']+.36
    if fault=='not_fresh':s['fresh']=False
    if fault=='unconfirmed':pair.stationary_reference_ready=False
    if fault=='no_stationary_mode':pair.require_stationary_reference=False
    if fault=='wrong_label':s['target_detection']={**s['target_detection'],'label':'up'}
    if fault=='mixed_frame':s['target_detection']={**s['target_detection'],'frame_sequence':1}
    if fault=='mixed_frame':
        with pytest.raises(ValueError,match='image_box_frame_mismatch'):
            pair.classify(s,'OFF',stationary=True,allow_post_press_off_alignment=True)
        return
    result=pair.classify(s,'OFF',stationary=fault!='moving',
        allow_post_press_off_alignment=fault!='opt_out')
    assert result['state']!='OFF'


@pytest.mark.parametrize('fault',['pose_during','age_during','budget','budget_during_pose','large_scale','large_shift',
                                'small','border','wrong_label','sequence','time','missing_image'])
def test_alignment_keeps_pose_age_geometry_and_time_bounds(monkeypatch,fault):
    pair,_=frozen_pair(monkeypatch);a=pair.references[0];b=extract_patch(snapshot(AFTER[0]))
    pose=Mock(return_value=True);fresh=Mock(return_value=True)
    if fault=='pose_during':pose=Mock(side_effect=[True,False,False])
    if fault=='age_during':fresh=Mock(side_effect=[True,True,False,False])
    if fault=='large_scale':b=replace(b,box=b.box+np.array([0,0,10,10]))
    if fault=='large_shift':b=replace(b,box=b.box+np.array([10,0,10,0]))
    if fault=='small':a=replace(a,box=np.array([20,20,50,50.]));b=replace(b,box=a.box.copy())
    if fault=='border':a=replace(a,box=np.array([1,1,76,76.]));b=replace(b,box=a.box.copy())
    if fault=='wrong_label':b=replace(b,target_label='up')
    if fault=='sequence':b=replace(b,sequence=a.sequence)
    if fault=='time':b=replace(b,captured=a.captured+30.01)
    if fault=='missing_image':a=replace(a,image='/nonexistent/round77.jpg')
    compare=compare_patches
    if fault in {'budget','budget_during_pose'}:
        import piper_handeye.off_reference_alignment as module
        clock=[0.];monkeypatch.setattr(module,'time',NS(perf_counter=lambda:clock[0]))
        if fault=='budget':
            def compare(*args,**kwargs):
                r=compare_patches(*args,**kwargs);clock[0]=.201;return r
        else:
            def pose():clock[0]=.201;return True
    result,audit=recompare_stationary_post_press_off(a,b,compare,
        stationary_check=pose,freshness_check=fresh,cache={})
    assert result is None and not audit['accepted']


@pytest.mark.parametrize('scene',SCENES,ids=lambda s:s['label'])
def test_real_lit_images_cannot_become_off_even_with_wrong_absolute_off(scene):
    for frame in scene['after']:
        sa=lamp_snapshot(scene['before'][0]);sb=lamp_snapshot(frame)
        sa['target_detection']['label']=scene['label'];sb['target_detection']['label']=scene['label']
        a,b=extract_patch(sa),extract_patch(sb)
        result,audit=recompare_stationary_post_press_off(a,b,compare_patches,
            stationary_check=lambda:True,freshness_check=lambda:True,cache={})
        assert result is None and not audit['accepted']


@pytest.mark.parametrize('kind',['lit','covered','saturated','nonuniform_background'])
def test_modified_lamp_or_occlusion_cannot_authorize_retry(monkeypatch,tmp_path,kind):
    pair,_=frozen_pair(monkeypatch);a=pair.references[0];b=extract_patch(snapshot(AFTER[0]))
    image=cv2.imread(b.image)
    if kind=='lit':image[158:190,318:341]=230
    if kind=='covered':image[145:205,298:359]=0
    if kind=='saturated':image[145:205,298:359]=255
    if kind=='nonuniform_background':image[130:220,280:328]=230
    path=tmp_path/(kind+'.png');cv2.imwrite(str(path),image);b=replace(b,image=str(path))
    result,audit=recompare_stationary_post_press_off(a,b,compare_patches,
        stationary_check=lambda:True,freshness_check=lambda:True,cache={})
    assert result is None and not audit['accepted']


@pytest.mark.parametrize('position',['press_monitor','prepress','prepress_confirmation'])
def test_motion_and_other_poses_never_opt_in(monkeypatch,position):
    pair,clock=frozen_pair(monkeypatch)
    observer=FreshLightObserver(NS(request=lambda *a,**k:snapshot(AFTER[0])),
        NS(observe=lambda *a,**k:{'state':'OFF'}),'down',paired_light=pair)
    result=observer.observe(position)
    assert result=='UNCERTAIN'
    assert 'post_press_off_alignment' not in observer.audit[-1]['recognition']['paired_comparison']
