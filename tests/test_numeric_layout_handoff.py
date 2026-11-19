"""Opt-in numeric task boundaries, no robot/camera/network calls."""
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock
import time

import numpy as np
import pytest

from piper_handeye.numeric_layout_handoff import (
    LayoutTaskPolicy, LayoutTaskVision, selected_button_region, task_vision,
    require_layout_server_support, SourceAwareButtonWindow, timed_target, target_batch,
)
from piper_handeye.button_light_hardware import FreshLightObserver, target_is_safely_visible
from piper_handeye.numeric_layout_sampling import LayoutTransform
from piper_handeye.realtime_planner import SlidingButtonWindow


SETTINGS=dict(mode='shadow',task_fallback_enabled=True,target_labels=list('12345'),profile_sha256='a'*64)


def evidence(seq=1, *, capture=None, target='2', matrix=None, box=None, image_size=None):
    capture=time.time() if capture is None else capture
    matrix=np.eye(4) if matrix is None else np.asarray(matrix)
    return dict(schema=1,source='panel_layout_inferred',mode='shadow',target_label=target,
                target_detected=False,yolo_confidence=None,motion_eligible=False,
                frame_sequence=seq,rgbd_bundle_sequence=seq,capture_time_unix_sec=capture,
                received_monotonic=capture-(time.time()-time.monotonic()),
                pairing='same_rgbd_bundle',profile_sha256='a'*64,profile_id='test_panel',
                accepted_2d=True,valid_3d=True,light_state='not_evaluated',
                geometry=dict(accepted=True,anchors=[s for s in ['1','3','9','10'] if s!=target]),
                bbox_xyxy=box or [100,100,130,130],image_size=image_size or [640,360],
                pose=dict(camera_to_button=matrix.tolist(),plane_rms_m=.001,
                          panel_support_detections=4,p_camera_m=matrix[:3,3].tolist(),
                          normal_camera=matrix[:3,2].tolist()))


def response(e):
    return dict(ok=True,fresh=True,frame_sequence=e['frame_sequence'],
                capture_time_unix_sec=e['capture_time_unix_sec'],image_size=e['image_size'],
                numeric_layout_localization=deepcopy(e),detections=[],sample=None)


def decorate(e,command='light_snapshot',settings=None):
    raw=response(e);client=NS(request=lambda *_a,**_k:deepcopy(raw))
    vision=LayoutTaskVision(client,LayoutTaskPolicy(settings or SETTINGS,e['target_label']))
    return vision.request(command,target_label=e['target_label'])


def test_default_and_arrows_have_no_task_fallback():
    plain=object()
    assert task_vision(plain,{},'2') is plain
    for target in ['up','down','6','16']:
        assert task_vision(plain,dict(numeric_layout_localization=SETTINGS),target) is plain
    assert task_vision(plain,dict(numeric_layout_localization={**SETTINGS,'task_fallback_enabled':False}),'2') is plain


def test_layout_roi_is_separate_and_does_not_invent_target_confidence():
    snap=decorate(evidence())
    assert snap['detections']==[] and 'target_detection' not in snap
    region=selected_button_region(snap)
    assert region['source']=='panel_layout_inferred' and region['confidence'] is None
    assert region['target_detected'] is False
    visible,audit=target_is_safely_visible(snap)
    assert visible and audit['confidence'] is None
    assert audit['checks']['layout_region_qualified'] is True
    assert 'confidence_ok' not in audit['checks']


def test_genuine_yolo_box_wins_and_detections_are_not_changed():
    snap=decorate(evidence());raw=dict(label='2',confidence=.95,frame_sequence=1,bbox_xyxy=[10,20,30,40])
    snap['target_detection']=raw
    assert selected_button_region(snap) is raw


@pytest.mark.parametrize('mutate',[
    lambda e:e.update(profile_sha256='b'*64),lambda e:e.update(target_detected=True),
    lambda e:e.update(yolo_confidence=.99),lambda e:e.update(rgbd_bundle_sequence=999),
    lambda e:e.update(capture_time_unix_sec=time.time()-1),lambda e:e.update(accepted_2d=False),
    lambda e:e.update(bbox_xyxy=[0,0,30,30]),lambda e:e['geometry'].update(anchors=['1','1','3']),
])
def test_invalid_layout_never_supplies_a_lamp_crop(mutate):
    e=evidence();mutate(e)
    snap=decorate(e)
    assert selected_button_region(snap) is None
    assert not target_is_safely_visible(snap)[0]


def test_inferred_box_cannot_be_replaced_or_used_on_another_image():
    snap=decorate(evidence());snap['layout_target_region']['bbox_xyxy'][0]+=2
    assert selected_button_region(snap) is None
    snap=decorate(evidence());snap['frame_sequence']=999
    assert selected_button_region(snap) is None
    snap=decorate(evidence());snap['image_size']=[1280,720]
    assert selected_button_region(snap) is None
    assert selected_button_region(decorate(evidence()),target_label='3') is None


def test_lamp_2d_region_does_not_need_depth_but_movement_does():
    e=evidence();e['valid_3d']=False;e.pop('pose')
    assert selected_button_region(decorate(e)) is not None
    assert 'layout_sample' not in decorate(e,'latest')


def test_layout_motion_sample_is_separate_and_quality_window_unchanged():
    policy=LayoutTaskPolicy(SETTINGS,'2');sample=policy.sample(evidence())
    base=np.eye(4)
    admitted=timed_target(sample,base,target_label='2')
    assert isinstance(admitted,LayoutTransform) and not hasattr(admitted,'confidence')
    original=SlidingButtonWindow(size=2,max_input_age_sec=.35,max_latest_age_sec=.35)
    window=SourceAwareButtonWindow.from_window(original)
    for name in ('size','min_confidence','max_input_age_sec','max_span_sec','max_position_jump_m'):
        assert getattr(window,name)==getattr(original,name)
    assert window.add(admitted)
    assert not window.estimate().stable
    e=evidence(2);sample2=policy.sample(e)
    assert window.add(timed_target(sample2,base,target_label='2'))
    assert window.estimate().stable
    r=decorate(e,'latest')
    assert r['sample'] is None and r['layout_sample']['confidence'] is None
    assert len(target_batch(r))==1 and len(target_batch(r,after_sequence=2))==0


def test_modified_or_invalid_depth_never_becomes_a_motion_sample():
    e=evidence();e['pose']['p_camera_m']=[float('nan'),0,0]
    assert LayoutTaskPolicy(SETTINGS,'2').sample(e) is None
    e=evidence();sample=LayoutTaskPolicy(SETTINGS,'2').sample(e);sample['confidence']=.9
    with pytest.raises(ValueError,match='invalid inferred'):timed_target(sample,np.eye(4),target_label='2')


def test_old_or_different_server_is_rejected_before_task_execution():
    config=dict(numeric_layout_localization=SETTINGS)
    for health in [{},{'numeric_layout_task_handoff':dict(version=1,enabled=False,profile_sha256='a'*64)},
                   {'numeric_layout_task_handoff':dict(version=1,enabled=True,profile_sha256='b'*64)}]:
        with pytest.raises(RuntimeError,match='matching vision/safety'):
            require_layout_server_support(health,config,'2')
    require_layout_server_support(dict(numeric_layout_task_handoff=dict(version=1,enabled=True,
                                  profile_sha256='a'*64)),config,'2')


def opencv_scene(tmp_path):
    from tests.test_paired_button_light import patch_image,snapshot
    from tools.opencv_light_worker import RelativeLuminance
    from piper_handeye.paired_button_light import PairedButtonLight
    pair=PairedButtonLight(require_stationary_reference=True,stationary_check=lambda:True)
    pair.begin_attempt();scene=NS(increase=0,sequence=0)
    class Vision:
        def request(self,*_a,**_k):
            scene.sequence+=1
            s=snapshot(tmp_path,patch_image(scene.sequence,increase=scene.increase));s.pop('target_detection')
            e=evidence(scene.sequence,capture=s['capture_time_unix_sec'],box=[30,30,94,94],image_size=[128,128])
            s['layout_target_region']=LayoutTaskPolicy(SETTINGS,'2').region(e)
            return s
    model=RelativeLuminance()
    observer=FreshLightObserver(Vision(),NS(observe=model.predict),'2',paired_light=pair)
    return observer,pair,scene


def test_real_opencv_observer_and_before_after_pair_accept_inferred_roi(tmp_path):
    observer,pair,scene=opencv_scene(tmp_path)
    observer.capture_stationary_off_reference(timeout_sec=.5)
    assert pair.stationary_reference_ready
    pair.freeze()
    scene.increase=150;assert observer.observe('near')=='ON'
    assert observer.observe('near_confirmation')=='ON'
    assert observer.audit[-1]['target_detection'] is None
    assert observer.audit[-1]['target_region']['source']=='panel_layout_inferred'
    assert observer.audit[-1]['recognition']['paired_comparison']['current_region_source']=='panel_layout_inferred'


@pytest.mark.parametrize('failed_attempts',[0,1,3])
def test_production_flow_with_real_opencv_inferred_regions_and_retry(tmp_path,failed_attempts):
    from piper_handeye.button_light_flow import ButtonLightFlow,ButtonFlowConfig
    from run_button_light_closed_loop import press_attempt_mode
    observer,pair,scene=opencv_scene(tmp_path);calls=[]
    class Hardware:
        waiting_position_is_origin=True
        def observe_light(self,position):return observer.observe(position)
        def wait_for_next_observation(self):pass
        def refresh_button_target(self):calls.append('refresh')
        def press_button(self,attempt):
            calls.append(press_attempt_mode(attempt))
            scene.increase=0;pair.begin_attempt()
            observer.capture_stationary_off_reference(timeout_sec=.5);pair.freeze()
            scene.increase=0 if attempt<=failed_attempts else 150
            return {'status':'simulated_contact','feedback_contact_confirmed':True}
        def move_to_near_observation(self,distance):
            assert distance==.05;calls.append('near');return 'near'
        def move_to_waiting_observation(self):
            calls.append('watch');scene.increase=0;return {'verified':True}
        def move_to_origin(self):calls.append('origin');return {'verified':True}
    result=ButtonLightFlow(Hardware(),ButtonFlowConfig(target_label='2')).run()
    if failed_attempts==3:
        assert result.status=='button_not_lit_after_max_attempts' and result.press_attempts==3
        assert 'watch' not in calls
    else:
        assert result.status=='arrived_after_verified_press'
        assert result.press_attempts==failed_attempts+1
        assert calls[-2:]==['watch','origin']
    assert calls.count('moving_rgbd_receding_approach')==1
    assert calls.count('current_observation_to_prepress')==result.press_attempts-1
    assert calls.count('origin')==1


def test_existing_stationary_entry_collects_layout_and_retains_actual_provenance():
    from run_online_replan_no_contact import collect_stationary_target
    config=dict(numeric_layout_localization=SETTINGS,detector={},button_geometry={})
    class Robot:
        def state(self,**kw):return NS(timestamp=time.monotonic(),joints=np.zeros(6),base_to_gripper=np.eye(4))
        def __getattr__(self,name):raise AssertionError('unexpected robot action '+name)
    sequence=[0]
    class Client:
        def request(self,command,**kw):
            assert command=='latest'
            time.sleep(.035);sequence[0]+=1
            return response(evidence(sequence[0],capture=time.time()-.02))
    vision=LayoutTaskVision(Client(),LayoutTaskPolicy(SETTINGS,'2'))
    estimate,audit=collect_stationary_target(vision,Robot(),np.eye(4),'2',config)
    assert estimate.stable and estimate.sample_count==4
    accepted=[r for r in audit if r['accepted']]
    assert all(r['source']=='panel_layout_inferred' and r['confidence'] is None
               and r['target_detected'] is False for r in accepted)
    assert audit[-1]['acquisition_summary']['position_sources']==['panel_layout_inferred']
    from piper_handeye.numeric_visibility import recover_numeric_target
    robot=Robot()
    robot.state=lambda **kw:NS(timestamp=time.monotonic(),joints=np.zeros(6),base_to_gripper=np.eye(4),raw={'hardware_ok':1})
    acquire=Mock(side_effect=AssertionError('valid same-target reference should not reacquire'))
    _,_,_,reused=recover_numeric_target('2','2',estimate,audit,np.zeros(6),robot,acquire,image_size=[640,360])
    assert reused


def test_two_fresh_layout_visibility_frames_use_existing_numeric_entry():
    from piper_handeye.numeric_visibility import wait_numeric_visibility
    records=[response(evidence(1)),response(evidence(2))]
    client=NS(request=lambda *_a,**_k:records.pop(0))
    vision=LayoutTaskVision(client,LayoutTaskPolicy(SETTINGS,'2'))
    r,ready=wait_numeric_visibility(vision,target_label='2',min_confidence=.7,sequence_barrier=0)
    assert ready and r['frame_sequence']==2
    assert r['visibility_resolution']=='qualified_same_frame_layout'
    assert r['detections']==[]


@pytest.mark.parametrize('enabled',[False,True])
def test_depth_safety_reference_uses_layout_only_when_explicitly_enabled(enabled):
    from piper_handeye.depth_safety import DepthPanelSafetyMonitor,DepthSafetyConfig
    e=evidence();e['pose']['p_camera_m']=[0,0,.6];e['pose']['camera_to_button'][2][3]=.6
    sample=NS(sequence=1,target_label='2',pose=None,numeric_layout_localization=e,
              frame=NS(received_unix=e['capture_time_unix_sec'],received_monotonic=e['received_monotonic']))
    monitor=DepthPanelSafetyMonitor(Mock(),lambda:sample,DepthSafetyConfig(),
                 numeric_layout_task_settings={**SETTINGS,'task_fallback_enabled':enabled})
    monitor.reset('2');monitor._update_reference(sample)
    assert (monitor._reference is not None) is enabled
    assert sample.pose is None
    if enabled:assert monitor._reference_position_source=='panel_layout_inferred'


@pytest.mark.parametrize('direct',[False,True])
def test_actual_moving_replan_orchestrator_uses_two_inferred_samples(monkeypatch,direct):
    import run_button_light_closed_loop as m
    from tests.test_direct_reference_approach import test_real_receding_orchestrator_uses_last_solved_endpoint
    original=m.execute_receding_prepress
    def with_layout(**kw):
        original_vision=kw['vision'];cursor=[0]
        class Client:
            def request(self,*a,**k):
                raw=original_vision.request(*a,**k)
                samples=raw['samples'];sample=samples[min(cursor[0],len(samples)-1)];cursor[0]+=1
                e=evidence(sample['sequence'],capture=sample['capture_time_unix_sec'],
                           matrix=sample['camera_to_button'])
                return {**response(e),'samples':[]}
        kw['vision']=LayoutTaskVision(Client(),LayoutTaskPolicy(SETTINGS,'2'))
        kw['target_label']='2';kw['config']['numeric_layout_localization']=SETTINGS
        return original(**kw)
    monkeypatch.setattr(m,'execute_receding_prepress',with_layout)
    test_real_receding_orchestrator_uses_last_solved_endpoint(monkeypatch,direct,False,None)
