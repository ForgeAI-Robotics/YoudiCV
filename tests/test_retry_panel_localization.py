"""No-motion tests of fresh-pixel localization despite unusable near depth."""
import ast
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import threading
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from piper_handeye.retry_panel_localization import (
    RetryPanelReference, RetryPanelSampler, capture_camera_matrix,
    collect_retry_panel_target, intersect_rgb_ray, POLICY,
)
from piper_handeye.numeric_layout_handoff import LayoutTaskPolicy, LayoutTaskVision
from piper_handeye.realtime_planner import WindowEstimate


K = np.array([[300., 0, 320], [0, 300., 180], [0, 0, 1.]])
CONFIG = dict(depth=dict(capture_size=[640, 360]), detector=dict(target_confidence=.70),
              calibration=dict(rgb_calibration_size=[1920, 1080]))
SETTINGS = dict(mode='shadow', task_fallback_enabled=True, target_labels=['4'],
                profile_sha256='a'*64)


def reference(target='4', task='task', z=.19):
    t = np.eye(4); t[2, 3] = z
    e = WindowEstimate(True, (), t, (10, 11), 2, .05, .1, 0., 0., 0.)
    rows = [dict(accepted=True, plane_rms_m=.001), dict(accepted=True, plane_rms_m=.002)]
    return RetryPanelReference.freeze(e, rows, target_label=target, task_id=task)


def state(mono, *, x=0., q=0., healthy=1):
    t = np.eye(4);t[0, 3] = x
    return NS(timestamp=mono, joints=np.ones(6)*q, base_to_gripper=t, raw=dict(hardware_ok=healthy))


def packet(seq, capture, *, u=320., confidence=.94, target='4'):
    return dict(ok=True, frame_sequence=seq, capture_time_unix_sec=capture,
                image_size=[640, 360], detections=[dict(label=target, confidence=confidence,
                bbox_xyxy=[u-35, 145, u+35, 215])])


def sampler(**overrides):
    args=dict(reference=reference(), handeye=np.eye(4), camera_matrix=K,
              image_size=[640, 360], expected_joints=np.zeros(6), target_label='4',
              task_id='task', min_confidence=.7, now_unix=100., now_monotonic=10.)
    args.update(overrides)
    return RetryPanelSampler(**args)


def feed(s, seq=1, capture=100.1, *, u=320., x=0., q=0.):
    mono=capture-90
    s.observe_robot(state(mono-.02, x=x, q=q), now_unix=capture-.02, now_monotonic=mono-.02)
    s.observe_robot(state(mono+.02, x=x, q=q), now_unix=capture+.02, now_monotonic=mono+.02)
    return s.consume(packet(seq, capture, u=u, target=s.reference.target_label),
                     now_unix=capture+.03, now_monotonic=mono+.03)


def test_camera_intrinsics_use_capture_resolution_not_calibration_resolution():
    full=K.copy();full[:2]*=3
    assert np.array_equal(capture_camera_matrix(full, CONFIG), K)
    bad=deepcopy(CONFIG);bad['depth']['capture_size']=[640,480]
    with pytest.raises(ValueError, match='aspect ratio'):capture_camera_matrix(full,bad)


def test_reference_is_immutable_and_only_freezes_verified_fit():
    r=reference();m=r.matrix;m[2,3]=9
    assert r.matrix[2,3]==.19
    e=WindowEstimate(False, ('bad',), np.eye(4), (1,2), 2, 0., .1, 0., 0., 0.)
    with pytest.raises(ValueError, match='stable'):RetryPanelReference.freeze(e,[],target_label='4',task_id='x')
    e=WindowEstimate(True, (), np.eye(4), (1,2), 2, 0., .1, 0., 0., 0.)
    for rows in [[], [dict(accepted=True,plane_rms_m=.01)]*2,
                 [dict(accepted=True,plane_rms_m=float('nan'))]*2]:
        with pytest.raises(ValueError, match='fit evidence'):
            RetryPanelReference.freeze(e,rows,target_label='4',task_id='x')


def test_fresh_ray_moves_target_in_plane_without_replacing_normal_or_depth():
    r=reference();out,correction=intersect_rgb_ray(np.eye(4),K,[286,145,356,215],r)
    assert out[:3,3]==pytest.approx([.19/300,0,.19])
    assert np.array_equal(out[:3,:3],r.matrix[:3,:3]) and correction==pytest.approx(.19/300)


def test_camera_rotation_and_translation_are_applied():
    camera=np.eye(4);camera[:3,:3]=Rotation.from_euler('y',1,degrees=True).as_matrix()
    camera[0,3]=.002
    out,_=intersect_rgb_ray(camera,K,[285,145,355,215],reference())
    assert out[:3,3]==pytest.approx([.002+.19*np.tan(np.deg2rad(1)),0,.19])


@pytest.mark.parametrize('q,reason',[(.1,'left its verified'),(float('nan'),'invalid')])
def test_lost_or_invalid_encoder_hold_stops(q,reason):
    s=sampler()
    with pytest.raises(RuntimeError,match=reason):
        s.observe_robot(state(10.1,q=q),now_unix=100.1,now_monotonic=10.1)


@pytest.mark.parametrize('healthy',[0,None,2])
def test_unhealthy_hardware_stops(healthy):
    with pytest.raises(RuntimeError,match='unhealthy'):
        sampler().observe_robot(state(10.1,healthy=healthy),now_unix=100.1,now_monotonic=10.1)


@pytest.mark.parametrize('changes',[dict(task_id='other'),dict(target_label='5')])
def test_reference_cannot_cross_tasks_or_targets(changes):
    with pytest.raises(ValueError,match='different task or target'):sampler(**changes)


@pytest.mark.parametrize('target',['4','up','down'])
def test_two_new_frames_required_and_no_current_depth_is_read(target):
    s=sampler(reference=reference(target=target),target_label=target);assert feed(s)['accepted']
    assert not s.snapshot(now_unix=100.14)[0].stable
    row=feed(s,2,100.2,u=320.25);assert row['accepted'],row
    e,rows,d=s.snapshot(now_unix=100.24)
    assert e.stable and e.sample_count==2 and e.position_sources==('yolo',)
    assert d['reference']['depth_source']=='frozen_verified_task_panel_not_current_near_depth'
    assert all('plane_rms_m' not in row and 'reference_plane_rms_m' in row for row in rows)
    assert e.max_position_jump_m==pytest.approx(.19*.25/300)
    assert not hasattr(s.window._samples[0],'confidence')


@pytest.mark.parametrize('mutate,reason',[
    (lambda p:p.update(capture_time_unix_sec=99.9),'captured_before_stationary_pose'),
    (lambda p:p.update(capture_time_unix_sec=99.7),'sample_age_out_of_range'),
    (lambda p:p.update(image_size=[1280,720]),'capture_resolution_changed'),
    (lambda p:p.update(detections=[]),'no_qualified_fresh_button_region'),
    (lambda p:p['detections'][0].update(label='5'),'no_qualified_fresh_button_region'),
    (lambda p:p['detections'][0].update(confidence=.69),'no_qualified_fresh_button_region'),
    (lambda p:p['detections'][0].update(confidence=float('nan')),'no_qualified_fresh_button_region'),
    (lambda p:p['detections'][0].update(bbox_xyxy=[1,145,71,215]),'button_not_fully_visible'),
    (lambda p:p['detections'].append(p['detections'][0].copy()),'ambiguous_target_detections'),
    (lambda p:p['detections'][0].update(bbox_xyxy=[350,145,420,215]),'moved'),
])
def test_bad_images_never_authorize_retry(mutate,reason):
    s=sampler();s.observe_robot(state(10.08),now_unix=100.08,now_monotonic=10.08)
    s.observe_robot(state(10.12),now_unix=100.12,now_monotonic=10.12)
    p=packet(1,100.1);mutate(p)
    row=s.consume(p,now_unix=100.15,now_monotonic=10.15)
    assert not row['accepted'] and reason in row['reason'],row
    assert s.snapshot(now_unix=100.15)[0].sample_count==0


def test_duplicate_clock_step_and_missing_history_rejected():
    s=sampler();assert feed(s)['accepted']
    assert s.consume(packet(1,100.1),now_unix=100.15,now_monotonic=10.15)['reason']=='not_a_new_inference_frame'
    s=sampler();p=packet(1,100.1)
    assert s.consume(p,now_unix=100.15,now_monotonic=10.05)['reason']=='capture_clock_mismatch'
    s=sampler();s.observe_robot(state(10.2),now_unix=100.2,now_monotonic=10.2)
    row=s.consume(p,now_unix=100.25,now_monotonic=10.25)
    assert 'outside robot-state history' in row['reason']


def test_interpolation_uses_capture_pose_and_handeye_not_latest_pose():
    h=np.eye(4);h[0,3]=.001;s=sampler(handeye=h)
    s.observe_robot(state(10.08,x=.001),now_unix=100.08,now_monotonic=10.08)
    s.observe_robot(state(10.12,x=.003),now_unix=100.12,now_monotonic=10.12)
    row=s.consume(packet(1,100.1),now_unix=100.15,now_monotonic=10.15)
    assert row['accepted'] and row['base_to_button'][0][3]==pytest.approx(.003)


def test_temporal_jump_and_image_freshness_limits_remain_strict():
    s=sampler();feed(s);assert feed(s,2,100.2,u=327)['accepted']
    e,_,_=s.snapshot(now_unix=100.23)
    assert not e.stable and any('max jump' in r for r in e.reasons)
    s=sampler();feed(s);feed(s,2,100.2)
    assert not s.snapshot(now_unix=100.56)[0].stable


def test_encoder_motion_discards_old_samples():
    s=sampler();feed(s)
    s.observe_robot(state(10.2,q=.005),now_unix=100.2,now_monotonic=10.2)
    e,_,d=s.snapshot(now_unix=100.2)
    assert e.sample_count==0 and d['encoder_resets']==1
    row=s.consume(packet(2,100.19),now_unix=100.21,now_monotonic=10.21)
    assert row['reason']=='captured_before_stationary_pose'


def test_parallel_and_behind_camera_intersections_rejected():
    c=np.eye(4);c[:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix()
    with pytest.raises(ValueError,match='oblique'):intersect_rgb_ray(c,K,[285,145,355,215],reference())
    c=np.eye(4);c[2,3]=.3
    with pytest.raises(ValueError,match='behind'):intersect_rgb_ray(c,K,[285,145,355,215],reference())


def test_layout_veto_is_preserved():
    s=sampler();s.observe_robot(state(10.08),now_unix=100.08,now_monotonic=10.08)
    s.observe_robot(state(10.12),now_unix=100.12,now_monotonic=10.12)
    p=packet(1,100.1);p['numeric_yolo_consistency']=dict(version=1,
        source='known_panel_neighbour_consistency',enabled=True,decision='conflict',
        target_label='4',frame_sequence=1)
    row=s.consume(p,now_unix=100.15,now_monotonic=10.15)
    assert row['reason']=='numeric_yolo_layout_conflict'


def test_explicit_layout_2d_recovery_still_works_without_near_depth():
    s=sampler();s.observe_robot(state(10.08),now_unix=100.08,now_monotonic=10.08)
    s.observe_robot(state(10.12),now_unix=100.12,now_monotonic=10.12)
    p=packet(1,100.1);p['detections']=[]
    evidence=dict(source='panel_layout_inferred',mode='shadow',target_label='4',target_detected=False,
        yolo_confidence=None,motion_eligible=False,frame_sequence=1,rgbd_bundle_sequence=1,
        capture_time_unix_sec=100.1,pairing='same_rgbd_bundle',profile_sha256='a'*64,
        profile_id='panel',accepted_2d=True,valid_3d=False,geometry=dict(accepted=True,anchors=['2','3','11']),
        bbox_xyxy=[285,145,355,215],image_size=[640,360])
    region=LayoutTaskPolicy(SETTINGS,'4').region(evidence,sequence=1,capture=100.1,now=100.15)
    p['layout_target_region']=region;p['layout_requested_target_label']='4'
    row=s.consume(p,now_unix=100.15,now_monotonic=10.15)
    assert row['accepted'] and row['source']=='panel_layout_inferred' and row['confidence'] is None
    p['frame_sequence']=2;p['capture_time_unix_sec']=100.11
    assert not s.consume(p,now_unix=100.15,now_monotonic=10.15)['accepted']


def test_collector_only_reads_existing_2d_results_and_encoders():
    requests=[];checks=[];seq=[0]
    class Robot:
        def state(self,*,max_age):
            assert max_age==.35
            return state(time.monotonic())
        def __getattr__(self,name):raise AssertionError('actuator access '+name)
    class Vision:
        def request(self,command,**options):
            assert command=='visible_buttons';requests.append(command);time.sleep(.04);seq[0]+=1
            # NO depth / camera_to_button / JPEG / new inference entry point.
            return packet(seq[0],time.time()-.025)
    e,rows=collect_retry_panel_target(Vision(),Robot(),np.eye(4),reference(),CONFIG,K,
        np.zeros(6),target_label='4',task_id='task',check_hold=lambda:checks.append(1))
    assert e.stable and len(rows)>=2 and len(checks)>=4
    assert not any(t.name=='retry-panel-encoders' for t in threading.enumerate())


def test_stop_callback_cancels_without_commands():
    class Robot:
        def state(self,**kwargs):return state(time.monotonic())
    class Vision:
        def request(self,*a,**k):raise AssertionError('must stop before polling')
    def stop():raise RuntimeError('independent stop active')
    with pytest.raises(RuntimeError,match='independent stop'):
        collect_retry_panel_target(Vision(),Robot(),np.eye(4),reference(),CONFIG,K,np.zeros(6),
            target_label='4',task_id='task',check_hold=stop)
    assert not any(t.name=='retry-panel-encoders' for t in threading.enumerate())


def test_no_region_times_out_with_explicit_retry_diagnostics():
    class Robot:
        def state(self,**kwargs):return state(time.monotonic())
    class Vision:
        def request(self,command,**kwargs):
            time.sleep(.04)
            p=packet(time.monotonic_ns(),time.time()-.025);p['detections']=[]
            return p
    started=time.monotonic()
    with pytest.raises(RuntimeError,match='no stable fresh button target') as failure:
        collect_retry_panel_target(Vision(),Robot(),np.eye(4),reference(),CONFIG,K,np.zeros(6),
            target_label='4',task_id='task',check_hold=lambda:None)
    d=failure.value.stationary_target_failure
    assert 3<=time.monotonic()-started<3.5 and d['sample_count']==0
    assert d['localization_method']==POLICY and not d['stable']
    assert any(r.get('reason')=='no_qualified_fresh_button_region' for r in d['audit'])
    assert not any(t.name=='retry-panel-encoders' for t in threading.enumerate())


def test_encoder_failure_and_rpc_failure_stop_collector_and_join_reader():
    class Robot:
        def state(self,**kwargs):return state(time.monotonic(),healthy=0)
    class Vision:
        def request(self,*args,**kwargs):time.sleep(.04);return packet(1,time.time()-.025)
    with pytest.raises(RuntimeError,match='encoder failure'):
        collect_retry_panel_target(Vision(),Robot(),np.eye(4),reference(),CONFIG,K,np.zeros(6),
            target_label='4',task_id='task',check_hold=lambda:None)
    class GoodRobot:
        def state(self,**kwargs):return state(time.monotonic())
    class BrokenVision:
        def request(self,*args,**kwargs):raise ConnectionError('vision disconnected')
    with pytest.raises(ConnectionError,match='vision disconnected'):
        collect_retry_panel_target(BrokenVision(),GoodRobot(),np.eye(4),reference(),CONFIG,K,np.zeros(6),
            target_label='4',task_id='task',check_hold=lambda:None)
    assert not any(t.name=='retry-panel-encoders' for t in threading.enumerate())


def test_production_refresh_uses_original_first_approach_and_new_retry_collector():
    # Execute the actual nested production callback with fake boundaries; this
    # catches accidental application to the first press or a new reference on retry.
    tree=ast.parse((Path(__file__).resolve().parents[1]/'run_button_light_closed_loop.py').read_text())
    fn=deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='refreshed_plan'))
    calls=[];r=reference();matrix=r.matrix
    est=WindowEstimate(True,(),matrix,(10,11),2,.01,.1,0.,0.,0.)
    audit=[dict(accepted=True,plane_rms_m=.001)]*2
    def first(*a,**k):calls.append('depth_first');return est,deepcopy(audit)
    def retry(*a,**k):
        calls.append('rgb_retry');assert a[3].sequence_range==(10,11);return est,deepcopy(audit)
    def build(*a,**kwargs):
        calls.append('build');assert kwargs['reference_distance_m']==.08
        assert np.array_equal(kwargs['reference_joints'],np.ones(6)*.01)
        return 'plan'
    def prepare(*a):
        calls.append('prepare8');return dict(observation_joints=(np.ones(6)*.01).tolist())
    setup=ast.parse('def outer():\n refresh_count=0\n retry_panel_reference=None\n') .body[0]
    setup.body += [fn,ast.parse('return refreshed_plan').body[0]]
    ns=dict(collect_stationary_target=first,collect_retry_panel_target=retry,build_plan=build,
        RetryPanelReference=RetryPanelReference,capture_camera_matrix=capture_camera_matrix,
        RETRY_PREPARATION_DISTANCE_M=.08,
        config=CONFIG,full_camera_matrix=K*3,task_id='task',report={},checkpoint=lambda *a:None,
        vision=object(),robot=object(),handeye=np.eye(4),args=NS(target_label='4',
            validate_plan_only=False,observation_only=False),latest_target={},
        executor=NS(plan=NS(observation_joints=np.zeros(6)),prepare_retry_observation=prepare),
        safety=NS(_assert_clear=lambda:None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[setup],type_ignores=[])),'production_refresh','exec'),ns)
    callback=ns['outer']();assert callback() is None
    assert callback()=='plan';assert callback()=='plan'
    assert calls==['depth_first','prepare8','rgb_retry','build','prepare8','rgb_retry','build']


def test_historical4_boxes_stay_stable_when_bad_near_depth_is_not_reused():
    fixture=json.loads((Path(__file__).parent/'fixtures/retry_panel_localization/historical4.json').read_text())
    for case in fixture['cases']:
        e=WindowEstimate(True,(),np.asarray(case['initial_plane']),(1,4),4,.01,.1,0.,0.,0.)
        r=RetryPanelReference.freeze(e,[dict(accepted=True,plane_rms_m=case['initial_rms_m'])]*4,
                                    target_label='4',task_id='replay')
        s=RetryPanelSampler(r,case['handeye'],case['capture_camera_matrix'],[640,360],
            case['observation_joints'],target_label='4',task_id='replay',min_confidence=.7,
            now_unix=100,now_monotonic=10)
        old=s._window()
        from piper_handeye.numeric_layout_sampling import LayoutTransform
        first=case['samples'][0]['capture_time_unix_sec']
        for i,x in enumerate(case['samples']):
            stamp=100.1+x['capture_time_unix_sec']-first;mono=stamp-90
            for t in [mono-.02,mono+.02]:
                state=NS(timestamp=t,joints=np.asarray(case['observation_joints']),
                    base_to_gripper=np.asarray(case['static_base_to_gripper']),raw=dict(hardware_ok=1))
                s.observe_robot(state,now_unix=t+90,now_monotonic=t)
            p=packet(x['sequence'],stamp);p['detections'][0].update(
                bbox_xyxy=x['bbox_xyxy'],confidence=x['confidence'])
            row=s.consume(p,now_unix=stamp+.03,now_monotonic=mono+.03)
            assert row['accepted'],(case['source_report'],row)
            old.add(LayoutTransform(x['sequence'],stamp,np.asarray(x['bad_near_base_to_button']),
                                    x['plane_rms_m']),now=stamp+.03)
        new,_,_=s.snapshot(now_unix=stamp+.03)
        prior=old.estimate(now=stamp+.03)
        assert not prior.stable and prior.max_position_jump_m>.06
        assert new.stable and new.max_position_jump_m<.001


def test_cycle_fingerprint_includes_retry_implementation(tmp_path):
    from piper_handeye.button_cycle_gui import session_metadata
    root=Path(__file__).resolve().parents[1]
    hold=tmp_path/'hold.json';hold.write_text('{}')
    data=session_metadata(NS(config=root/'piper_button_pose.yaml',hold_report=hold))
    assert str(root/'piper_handeye/retry_panel_localization.py') in data['sha256']
