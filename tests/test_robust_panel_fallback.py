"""Exact field failures and independent wrong-plane controls; no robot/model."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
import hashlib
import json
import numpy as np
import pytest
from piper_handeye import button_pose as m

FIXTURES = Path(__file__).parent/'fixtures/robust_panel_fallback'
FRAMES = json.loads((FIXTURES/'manifest.json').read_text())['frames']


def recorded(row):
    path = FIXTURES/row['bundle']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    with np.load(path, allow_pickle=False) as b:
        depth, matrix = b['depth_m'], b['camera_matrix']
    ds = [m.ButtonDetection(j,x['label'],x['confidence'],tuple(x['bbox_xyxy']))
          for j,x in enumerate(row['detections'])]
    return m.select_button(ds,'6',.7),depth,matrix,[d for d in ds if d.confidence>=.7]


@pytest.mark.parametrize('row',FRAMES,ids=lambda r:str(r['frame_sequence']))
def test_exact_gripper_contamination_recovers_with_original_quality_limits(row,monkeypatch):
    target,depth,matrix,supports = recorded(row)
    with monkeypatch.context() as old:
        old.setattr(m,'_fit_panel_plane_robust',Mock(side_effect=RuntimeError('disabled')))
        with pytest.raises(RuntimeError,match='panel plane RMS'):
            m.estimate_button_camera_pose(target,depth,matrix,panel_support_detections=supports,**row['geometry_options'])
    pose = m.estimate_button_camera_pose(target,depth,matrix,panel_support_detections=supports,**row['geometry_options'])
    audit = pose.panel_plane_fit
    assert audit['method']=='bounded_ransac_fallback_v1'
    assert audit['original_failure']==row['recorded_error']
    assert pose.plane_rms_m<=.006 and audit['consensus_fraction']>=.70
    assert audit['qualified_support_regions']>=3 and audit['local_depth_delta_m']<=.012
    assert audit['probe_points']<=2048 and audit['trials']<=32 and audit['elapsed_sec']<=.20
    # Independent investigation control, never a production label exclusion.
    clean=m.estimate_button_camera_pose(target,depth,matrix,
        panel_support_detections=[d for d in supports if d.label!=row['contaminated_reference_label']],**row['geometry_options'])
    assert clean.panel_plane_fit is None
    assert np.linalg.norm(pose.p_camera-clean.p_camera)<.0001


@pytest.mark.parametrize('row',FRAMES,ids=lambda r:str(r['frame_sequence']))
def test_successful_control_is_bit_identical_and_does_not_use_fallback(row,monkeypatch):
    target,depth,matrix,supports=recorded(row)
    supports=[d for d in supports if d.label!=row['contaminated_reference_label']]
    pose=m.estimate_button_camera_pose(target,depth,matrix,panel_support_detections=supports,**row['geometry_options'])
    forbidden=Mock(side_effect=AssertionError('fallback on successful frame'))
    monkeypatch.setattr(m,'_fit_panel_plane_robust',forbidden)
    again=m.estimate_button_camera_pose(target,depth,matrix,panel_support_detections=supports,**row['geometry_options'])
    for name in ['p_camera','normal_camera','camera_to_button']:
        np.testing.assert_array_equal(getattr(pose,name),getattr(again,name))
    for name in ['plane_rms_m','plane_inliers','depth_median_m','local_global_depth_delta_m','center_depth_m']:
        assert getattr(pose,name)==getattr(again,name)
    forbidden.assert_not_called()


def scene(centers=((50,50),(150,50),(50,150),(150,150)),contamination=.03):
    rng=np.random.default_rng(38)
    uv=np.concatenate([rng.uniform(-8,8,(500,2))+c for c in centers])
    z=np.full(len(uv),.5);z[:int(len(z)*contamination)]=.146
    points=m._backproject(uv,z,np.array([[400,0,100],[0,400,100],[0,0,1.]]))
    ds=[m.ButtonDetection(j,str(j),.9,(x-10,y-10,x+10,y+10)) for j,(x,y) in enumerate(centers)]
    target=m.ButtonDetection(8,'6',.9,(90,90,110,110))
    return points,uv,ds,target,(np.array([0,0,.5]),np.array([0.,0.,-1.]),.001,200),np.array([0.,0.,1.])


def rescue(data):
    return m._fit_panel_plane_robust(*data,min_inliers=120,distance_threshold_m=.004,max_rms_m=.006,outer_scale=1.8)


def test_finds_panel_without_label_whitelist():
    fit,audit=rescue(scene())
    assert fit[2]<1e-7 and audit['consensus_fraction']==pytest.approx(.97)
    assert fit[0][2]==pytest.approx(.5) and audit['qualified_support_regions']==4


@pytest.mark.parametrize('centers',[
    ((50,50),(50,50)),((50,50),(50,50),(50,50),(50,50)),
    ((50,50),(100,100),(150,150)),((50,50),(51,50),(50,51),(51,51)),
])
def test_single_corner_duplicates_or_line_cannot_supply_spatial_coverage(centers):
    with pytest.raises(RuntimeError,match='support regions|spatially spread'):
        rescue(scene(centers))


@pytest.mark.parametrize('change',['local_depth','local_normal','local_rms','local_parallel','behind'])
def test_wrong_foreground_or_incompatible_local_plane_rejected(change):
    data=list(scene());center,normal,rms,count=data[4]
    if change=='local_depth':center=np.array([0,0,.146])
    if change=='local_normal':normal=np.array([.5,0,-np.sqrt(.75)])
    if change=='local_rms':rms=.020
    if change=='local_parallel':normal=np.array([1.,0,0])
    if change=='behind':center=np.array([0,0,-.5])
    data[4]=(center,normal,rms,count)
    with pytest.raises(RuntimeError,match='target-local|parallel'):rescue(data)


def test_competing_planes_without_majority_rejected():
    data=list(scene(contamination=0));data[0][:800,2]=.146;data[0][800:1500,2]=.30
    with pytest.raises(RuntimeError,match='70%'):rescue(data)


def test_tiny_valid_reference_patches_do_not_supply_region_support():
    data=list(scene(contamination=0))
    data[2][1:]=[replace(d,bbox_xyxy=(300,300,320,320)) for d in data[2][1:]]
    with pytest.raises(RuntimeError,match='3 supported reference regions'):rescue(data)


def test_budget_exhaustion_rejects_without_relaxing_image_freshness(monkeypatch):
    ticks=iter([0.,.201]);monkeypatch.setattr(m.time,'monotonic',lambda:next(ticks))
    with pytest.raises(RuntimeError,match='0.20 s budget'):rescue(scene())


def test_missing_local_depth_is_not_rescued(monkeypatch):
    forbidden=Mock(side_effect=AssertionError('must not run'));monkeypatch.setattr(m,'_fit_panel_plane_robust',forbidden)
    data=scene()
    with pytest.raises(RuntimeError,match='valid local panel pixels'):
        m.estimate_button_camera_pose(data[3],np.zeros((200,200)),np.eye(3),panel_support_detections=data[2])
    forbidden.assert_not_called()


def test_audit_survives_real_producer_rpc_and_persistent_timing(tmp_path,monkeypatch):
    from tests.test_early_light_publication import worker,service
    from piper_handeye import pipeline_timing as timing
    from run_button_vision_server import pose_payload
    w=worker()
    pose=m.estimate_button_camera_pose(m.ButtonDetection(0,'5',.9,(50,40,90,80)),
        np.full((120,160),.5),np.array([[400,0,80],[0,400,60],[0,0,1.]]))
    _,audit=rescue(scene());pose=replace(pose,panel_plane_fit=audit)
    monkeypatch.setattr('piper_handeye.live_button.detect_with_enhancement_fallback',Mock(return_value=((pose.detection,),{})))
    def depth(*a,**k):w._running=False;return pose
    monkeypatch.setattr('piper_handeye.live_button.estimate_button_camera_pose',depth)
    w.start();w._thread.join(2)
    assert w._fatal_error is None and not w._thread.is_alive()
    sample=w.latest_valid();assert pose_payload(sample)['panel_plane_fit']==audit
    srv=service(w,tmp_path,monkeypatch)
    reply=srv.dispatch(dict(command='latest',token='test',target_label='5',include_realtime=False))
    span=next(s for s in reply['latest_processed_timing']['timing']['spans'] if s['name']=='vision.depth_pose')
    assert span['status']=='ok' and span['panel_plane_fit']==audit
    monkeypatch.delenv('YOUXIAOMEI_TIMING_FILE',raising=False);monkeypatch.setattr(timing,'_recorder',None)
    recorder=timing.configure_timing(tmp_path/'trace.jsonl','robust-panel-test')
    try:
        timing.record_vision_response(reply);assert recorder.flush()['written']
        rows=[json.loads(line) for line in (tmp_path/'trace.jsonl').read_text().splitlines()]
        assert next(x for x in rows if x.get('name')=='vision.depth_pose')['panel_plane_fit']==audit
    finally:
        timing._recorder=None
        monkeypatch.delenv('YOUXIAOMEI_TIMING_FILE',raising=False)
        monkeypatch.delenv('YOUXIAOMEI_TIMING_TASK',raising=False)
