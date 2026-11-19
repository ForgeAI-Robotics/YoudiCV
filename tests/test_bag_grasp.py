from types import SimpleNamespace
from unittest.mock import MagicMock
import json
import time
import numpy as np
import pytest

from piper_handeye import bag_grasp as bg
from piper_handeye.kinematics import forward_kinematics, forward_kinematics_batch
from tools.bag_grasp_worker import Task, chassis_clear


Q=np.array([-.03286455,.30660199,-.33629004,0,.11883947,.09683087])
K=np.array([[357.66,0,322.52],[0,358.33,177.28],[0,0,1.]])
T=np.eye(4)

def obj():
    return {'grasp_ready_2d':True,'status':'valid','class_id':0,'confidence':.85,
            'handle_center_xy':[315.,110.], 'bbox_xyxy':[249.,99.,400.,340.],
            'keypoint_confidence':.98}


def test_registered_depth_center_and_coordinates():
    depth=np.full((360,640),.247)
    r=bg.localize([obj()],depth,K,np.zeros(5),Q,T)
    expected=(forward_kinematics(Q)@np.r_[(315-K[0,2])*.247/K[0,0],(110-K[1,2])*.247/K[1,1],.247,1])[:3]
    np.testing.assert_allclose(r['base_xyz_m'],expected)
    assert r['depth_m']==.247 and r['depth_valid_fraction']==1


@pytest.mark.parametrize('case',['missing','ambiguous','wrongclass','weak','weakpoint','nanpoint',
                                 'outside','clipped','hole','sparse','jump','nan','far','near','wrongsize'])
def test_bad_target_or_depth_never_produces_motion_target(case):
    o=obj();ds=[o];depth=np.full((360,640),.247)
    if case=='missing':ds=[]
    if case=='ambiguous':ds=[o,obj()]
    if case=='wrongclass':o['class_id']=1
    if case=='weak':o['confidence']=.64
    if case=='weakpoint':o['keypoint_confidence']=.49
    if case=='nanpoint':o['handle_center_xy']=[float('nan'),110]
    if case=='outside':o['handle_center_xy']=[100,110]
    if case=='clipped':o['bbox_xyxy'][2]=640
    if case=='hole':depth[110,315]=0
    if case=='sparse':depth[108:110,313:318]=0
    if case=='jump':depth[108:110,313:318]=.300
    if case=='nan':depth[110,315]=np.nan
    if case=='far':depth[:]=1.1
    if case=='near':depth[:]=.1
    if case=='wrongsize':depth=depth[:180]
    with pytest.raises(ValueError):bg.localize(ds,depth,K,np.zeros(5),Q,T)


def samples():
    return [{'sequence':i,'captured_unix':100+i*.1,
             'target':{'base_xyz_m':[.4,0,.4],'pixel':[315,110]}} for i in range(4)]


@pytest.mark.parametrize('case',['short','duplicate','stale','future','jump','pixel','timeorder','span','nan'])
def test_unstable_or_stale_series_rejected(case):
    ss=samples()
    if case=='short':ss=ss[:3]
    if case=='duplicate':ss[1]['sequence']=0
    if case=='jump':ss[-1]['target']['base_xyz_m'][0]+=.007
    if case=='pixel':ss[-1]['target']['pixel'][0]+=9
    if case=='timeorder':ss[2]['captured_unix']=99
    if case=='span':ss[-1]['captured_unix']=102
    if case=='nan':ss[-1]['target']['base_xyz_m'][1]=float('nan')
    now=101 if case=='stale' else 99 if case=='future' else 100.31
    with pytest.raises(ValueError):bg.stable_target(ss,now=now)


def test_four_fresh_consistent_frames_return_original_base_point():
    np.testing.assert_allclose(bg.stable_target(samples(),now=100.31),[.4,0,.4])


def test_observation_requires_matching_calibration_and_valid_encoders(tmp_path):
    p=tmp_path/'observation.json'
    bg.save_observation(Q,'sha',path=p)
    np.testing.assert_allclose(bg.load_observation('sha',path=p),Q)
    with pytest.raises(ValueError,match='标定'):bg.load_observation('other',path=p)
    q=Q.copy();q[2]=.04
    with pytest.raises(ValueError):bg.save_observation(q,'sha',path=p)
    data=json.loads(p.read_text());data['joints_rad'][0]=float('nan');p.write_text(json.dumps(data))
    with pytest.raises(ValueError):bg.load_observation('sha',path=p)


def test_contact_plan_has_bounded_endpoint_and_preserves_tool_rotation():
    config=__import__('yaml').safe_load((bg.ROOT/'piper_button_pose.yaml').read_text())
    k,d,tc,_=bg.calibration(config)
    point=bg.localize([obj()],np.full((360,640),.247),k,d,Q,tc)['base_xyz_m']
    plan=bg.contact_plan(Q,point,tc)
    pre=forward_kinematics(plan['precontact_joints']);end=forward_kinematics(plan['touch_joints'])
    assert np.linalg.norm(end[:3,3]-point)<.001
    assert abs(np.linalg.norm(end[:3,3]-pre[:3,3])-.020)<.001
    assert np.max(np.abs(end[:3,:3]-forward_kinematics(Q)[:3,:3]))<.01
    poses=forward_kinematics_batch(np.array([p['positions'] for p in plan['forward']]))
    assert np.linalg.norm(poses[-1,:3,3]-poses[0,:3,3])<.022


@pytest.mark.parametrize('target',[[float('nan'),0,0],[.9,0,.3],[.1,0,.3]])
def test_target_out_of_bounded_forward_workspace_rejected(target):
    with pytest.raises(ValueError):bg.contact_plan(Q,target,np.eye(4))


@pytest.mark.parametrize('sample',[
    {'available':False},
    {'available':True,'age_sec':.6,'velocity':[0]*6},
    {'available':True,'age_sec':0,'velocity':[.02,0,0,0,0,0]},
    {'available':True,'age_sec':0,'velocity':[0,0,0,0,.03,0]},
    {'available':True,'age_sec':0,'velocity':[float('nan')]*6},
])
def test_moving_stale_or_nonfinite_chassis_blocks_task(sample):
    with pytest.raises(RuntimeError):chassis_clear(sample)


def task():
    t=Task(SimpleNamespace(gui_heartbeat=False,execute=True,action='touch',conf=.65))
    t.stopper=MagicMock();t.stopper.health.return_value={'latched':False}
    t.emit=MagicMock();t.timed=MagicMock();t.stopped_at=MagicMock()
    t.odom=MagicMock();t.odom.snapshot.return_value={'available':True,'age_sec':0,'velocity':[0]*6}
    t.motion_owned=True
    t.independent_joints=MagicMock(return_value=Q.copy())
    return t


def test_stop_without_task_ownership_cannot_stop_another_gui():
    t=task();t.motion_owned=False;t.moving=True;t.abort('stop')
    t.stopper.stop.assert_not_called()


def test_stop_during_motion_uses_independent_bridge_and_never_clears_latch():
    t=task();t.moving=True;t.abort('stop')
    t.stopper.stop.assert_called_once()
    t.stopper.reset_latch.assert_not_called()
    assert t.cancel.is_set()


@pytest.mark.parametrize('reason',['cancelled','staleguard','latched','chassis'])
def test_motion_not_sent_when_final_gate_fails(reason):
    t=task()
    if reason=='cancelled':t.cancel.set()
    if reason=='staleguard':t.last_guard=time.monotonic()-1
    if reason=='latched':t.stopper.health.return_value={'latched':True}
    if reason=='chassis':t.odom.snapshot.return_value={'available':False}
    with pytest.raises(RuntimeError):t.execute_path([{'positions':Q.tolist()}],'test')
    t.timed.execute.assert_not_called()


def test_unknown_execution_result_stops_without_resending():
    t=task();t.timed.execute.side_effect=TimeoutError('state unknown')
    with pytest.raises(TimeoutError):t.execute_path([{'positions':Q.tolist()}],'test')
    assert t.timed.execute.call_count==1
    t.stopper.stop.assert_called_once()
    assert t.moving and t.report['actual_motion_requested']


def test_gui_heartbeat_timeout_blocks_sending_motion():
    t=task();t.args.gui_heartbeat=True;t.require_heartbeat=True;t.heartbeat=time.monotonic()-2
    with pytest.raises(RuntimeError):t.check()
    t.timed.execute.assert_not_called()


def test_initial_full_box_and_near_contact_partial_box_are_distinct():
    o=obj();o['bbox_xyxy'][3]=360
    depth=np.full((360,640),.247)
    with pytest.raises(ValueError):bg.localize([o],depth,K,np.zeros(5),Q,T)
    r=bg.localize([o],depth,K,np.zeros(5),Q,T,require_full_box=False)
    assert r['depth_valid_fraction']==1
    o['handle_center_xy']=[315,359]
    with pytest.raises(ValueError):bg.localize([o],depth,K,np.zeros(5),Q,T,require_full_box=False)


def fixed_plan():
    return {'approach':[{'positions':Q.tolist()}],
            'forward':[{'positions':(Q+.001).tolist()}],
            'precontact_joints':Q.tolist(),
            'normal_base':[-1.,0.,0.], 'target_base_m':[.44,-.01,.46]}


def test_initial_target_runs_contact_even_when_near_field_recognition_is_unavailable():
    t=task();p=fixed_plan();frozen=json.loads(json.dumps(p))
    t.stable_frames=MagicMock(side_effect=AssertionError('不能再次识别'))
    t.predict_frame=MagicMock(side_effect=AssertionError('不能再次推理'))
    t.execute_path=MagicMock(side_effect=[{}, {'feedback_contact_confirmed':True}])
    t.guarded=MagicMock()
    result=t.execute_fixed_target(p)
    assert result['feedback_contact_confirmed']
    t.stable_frames.assert_not_called();t.predict_frame.assert_not_called()
    assert p==frozen
    assert t.execute_path.call_count==2
    assert t.guarded.execute.call_args.kwargs['contact_panel_origin_base']==p['target_base_m']
    assert t.guarded.execute.call_args.kwargs['dry_run'] is True
    assert t.execute_path.call_args.kwargs['guarded'] is p


def test_stop_after_approach_prevents_fixed_target_contact():
    t=task();t.guarded=MagicMock()
    t.execute_path=MagicMock(side_effect=lambda *a,**k:t.cancel.set())
    with pytest.raises(RuntimeError):t.execute_fixed_target(fixed_plan())
    assert t.execute_path.call_count==1
    t.guarded.execute.assert_not_called()


def test_stop_during_contact_revalidation_prevents_contact():
    t=task();t.guarded=MagicMock();t.execute_path=MagicMock()
    t.guarded.execute.side_effect=lambda *a,**k:t.cancel.set()
    with pytest.raises(RuntimeError):t.execute_fixed_target(fixed_plan())
    assert t.execute_path.call_count==1


def test_unconfirmed_contact_result_does_not_trigger_an_extra_motion_attempt():
    t=task();t.guarded=MagicMock()
    t.execute_path=MagicMock(side_effect=[{}, {'feedback_contact_confirmed':False}])
    assert t.execute_fixed_target(fixed_plan())['feedback_contact_confirmed'] is False
    assert t.execute_path.call_count==2


def cycle_plan():
    return bg.grasp_cycle_plan(Q,[.43726,-.01056,.45913])


def test_cycle_overtravel_is_seven_cm_in_horizontal_plane_and_reverses_exact_paths():
    plan=cycle_plan()
    delta=np.array(plan['grasp_base_m'])-plan['target_base_m']
    assert np.linalg.norm(delta)==pytest.approx(.07)
    assert delta[2]==0
    expected=forward_kinematics(Q)[:3,2].copy();expected[2]=0;expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(delta,expected*.07)
    for name in ('approach','insertion'):
        path=plan[name];rev=plan['withdraw_'+name]
        np.testing.assert_allclose([p['positions'] for p in path],[p['positions'] for p in rev[::-1]])
        np.testing.assert_allclose([p['velocities'] for p in path],-np.array([p['velocities'] for p in rev[::-1]]))
    actual=forward_kinematics(np.array(plan['grasp_joints']))
    assert np.linalg.norm(actual[:3,3]-plan['grasp_base_m'])<.001
    assert np.linalg.norm(actual[:3,:3]-forward_kinematics(Q)[:3,:3])<.015


@pytest.mark.parametrize('point',[[float('nan'),0,0],[.9,0,.4],[.1,0,.3]])
def test_cycle_bad_or_unbounded_initial_point_rejected(point):
    with pytest.raises(ValueError):bg.grasp_cycle_plan(Q,point)


def test_cycle_exact_order_and_same_localization_for_pick_and_place():
    t=task();plan=cycle_plan();events=[]
    t.validate_path=MagicMock()
    t.execute_path=lambda p,n:events.append(('move',p))
    t.set_gripper=lambda opened,q:events.append(('gripper',opened,q))
    t.stable_frames=MagicMock(side_effect=AssertionError('must not relocalize'))
    t.execute_grasp_cycle(plan)
    assert [e[0] for e in events]==['move','move','gripper','move','move','move','move','move','move','gripper','move','move']
    assert events[2][1] is False and events[9][1] is True
    assert [e[1] for e in events if e[0]=='move']==[plan[name] for name in bg.CYCLE_PATH_NAMES]
    assert t.report['cycle_completed'] and not t.report['grasp_confirmed']
    assert t.report['return_to_origin_on_success'] and not t.report['possibly_holding_bag']
    t.stable_frames.assert_not_called()
    t.validate_path.assert_not_called()  # Global preflight + execute-side checks remain.


def test_cycle_lifts_vertical_five_cm_and_lowers_before_opening():
    plan=cycle_plan()
    grasp_fk=forward_kinematics(np.array(plan['grasp_joints']))
    np.testing.assert_allclose(np.array(plan['raised_base_m'])-grasp_fk[:3,3],[0,0,.05],atol=1e-12)
    poses=forward_kinematics_batch(np.array([p['positions'] for p in plan['lift']]))
    delta=poses[:,:3,3]-poses[0,:3,3]
    assert delta[-1,2]==pytest.approx(.05,abs=.001)
    assert np.max(np.linalg.norm(delta[:,:2],axis=1))<=.003
    assert np.min(np.diff(delta[:,2]))>=-.0001
    for forward,reverse in [('lift','lower'),('carry_out_near','carry_back_near'),('carry_out_far','carry_back_far')]:
        np.testing.assert_allclose([p['positions'] for p in plan[forward]],[p['positions'] for p in plan[reverse][::-1]])
        np.testing.assert_allclose([p['velocities'] for p in plan[forward]],-np.array([p['velocities'] for p in plan[reverse][::-1]]))
    np.testing.assert_allclose(plan['lower'][-1]['positions'],plan['grasp_joints'])
    np.testing.assert_allclose(plan['carry_out_far'][-1]['positions'],Q)


@pytest.mark.parametrize('stop_after',range(1,11))
def test_cycle_stop_at_each_movement_boundary_never_continues_or_releases(stop_after):
    t=task();plan=cycle_plan();moves=[]
    t.validate_path=MagicMock()
    def move(p,n):
        moves.append(n)
        if len(moves)==stop_after:t.cancel.set()
    t.execute_path=move
    def gripper(*_):t.check()
    t.set_gripper=MagicMock(side_effect=gripper)
    with pytest.raises(RuntimeError):t.execute_grasp_cycle(plan)
    assert len(moves)==stop_after
    assert not t.report.get('return_to_origin_on_success')


def clamp_state(position,**values):
    return {'gripper_position':position,'clamp_open':position>=.35,'joint_feedback_age_sec':.01,**values}


def test_cycle_gripper_uses_teleop_toggle_to_open_and_known_close_command():
    t=task();t.live=MagicMock()
    t.live.status.return_value={'status':clamp_state(0)}
    t.live.toggle_clamp.return_value={'status':clamp_state(1.396)}
    t.set_gripper(True,Q)
    t.live.toggle_clamp.assert_called_once()
    t.live.close_clamp.assert_not_called()
    t.live.status.return_value={'status':clamp_state(1.396)}
    t.live.close_clamp.return_value={'status':clamp_state(.01,last_gripper_target=0)}
    t.set_gripper(False,Q)
    t.live.close_clamp.assert_called_once()
    assert len(t.report['gripper_actions'])==2


@pytest.mark.parametrize('opening,position',[(True,1.396),(False,.01)])
def test_gripper_already_at_requested_state_does_not_toggle(opening,position):
    t=task();t.live=MagicMock();t.live.status.return_value={'status':clamp_state(position)}
    t.set_gripper(opening,Q)
    t.live.toggle_clamp.assert_not_called();t.live.close_clamp.assert_not_called()


@pytest.mark.parametrize('case',['stale','nan','inconsistent','no_motion','timeout','cancel'])
def test_bad_or_unknown_gripper_response_stops_without_retry(case):
    t=task();t.live=MagicMock();before=clamp_state(1.396)
    after=clamp_state(.01,last_gripper_target=0)
    if case=='stale':before['joint_feedback_age_sec']=.3
    if case=='nan':before['joint_feedback_age_sec']=float('nan')
    if case=='inconsistent':before['clamp_open']=False
    if case=='no_motion':after=clamp_state(1.396)
    if case=='timeout':t.live.close_clamp.side_effect=TimeoutError('unknown result')
    if case=='cancel':t.live.close_clamp.side_effect=lambda **_:(t.cancel.set() or {'status':after})
    t.live.status.return_value={'status':before}
    t.live.close_clamp.return_value={'status':after}
    with pytest.raises((RuntimeError,TimeoutError)):t.set_gripper(False,Q)
    assert t.live.close_clamp.call_count<=1
    t.live.toggle_clamp.assert_not_called()


def test_monitor_uses_independent_feedback_while_clamp_blocks_5015_longer_than_status_timeout():
    import threading
    t=task();t.live=MagicMock();t.camera_watch=True
    t.live.status.return_value={'status':clamp_state(0)}
    started=threading.Event();release=threading.Event();errors=[]
    def slow_open(**kwargs):
        assert kwargs['timeout_sec']==6.5
        started.set()
        assert release.wait(3)
        return {'status':clamp_state(1.396)}
    t.live.toggle_clamp.side_effect=slow_open
    t.capture=MagicMock(return_value={'sequence':1,'depth':np.ones((360,640))})
    t.healthy=MagicMock(side_effect=AssertionError('5015 is synchronously blocked'))
    def run():
        try:t.set_gripper(True,Q)
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=run);thread.start()
    try:
        assert started.wait(2)
        start=time.monotonic();count=0
        while time.monotonic()-start<.55:
            assert t.monitor_frame(t.live,MagicMock())['sequence']==1
            assert t.capture.call_args.kwargs['q_reader'] is t.independent_joints
            count+=1;time.sleep(.02)
        assert count>=10 and not t.cancel.is_set()
        t.healthy.assert_not_called()
        assert t.live.status.call_count==1
    finally:release.set();thread.join(3)
    assert not errors,errors
    assert not thread.is_alive() and not t.gripper_active.is_set()
    t.stopper.stop.assert_not_called()


@pytest.mark.parametrize('case',['hardware','motion','missing'])
def test_independent_fault_or_arm_drift_during_clamp_still_stops_supervision(case):
    t=task();t.arm_query_gate=MagicMock();t.arm_query_gate.acquire.return_value=False
    t.camera_watch=True;t.gripper_active.set();t.gripper_expected=Q.copy()
    if case=='motion':t.independent_joints.return_value=Q+np.array([.01,0,0,0,0,0])
    else:t.independent_joints.side_effect=RuntimeError('独立反馈不可用')
    t.capture=MagicMock()
    with pytest.raises(RuntimeError):t.monitor_frame(MagicMock(),MagicMock())
    t.capture.assert_not_called()


def test_monitor_returns_to_original_5015_health_checks_after_clamp_finishes():
    t=task();t.camera_watch=True;t.healthy=MagicMock();t.capture=MagicMock(return_value={'sequence':8})
    live=MagicMock();vision=MagicMock()
    assert t.monitor_frame(live,vision)=={'sequence':8}
    t.healthy.assert_called_once_with(live)
    t.capture.assert_called_once_with(vision,live)


def test_independent_feedback_rejects_hardware_fault_and_nonfinite_joints():
    for raw,q in [({'hardware_ok':0},Q),({'hardware_ok':1},[float('nan')]*6)]:
        t=task();t.robot=MagicMock();t.robot.state.return_value=SimpleNamespace(raw=raw,joints=q)
        with pytest.raises((RuntimeError,ValueError)):Task.independent_joints(t)
        t.robot.state.assert_called_once_with(max_age=.25)


def test_depth_monitor_capture_can_read_encoder_stream_without_5015_rpc(tmp_path):
    import cv2
    t=task();t.live=MagicMock()
    t.live.live_joints.side_effect=AssertionError('5015 blocked')
    rgb=tmp_path/'rgb.jpg';depth=tmp_path/'depth.png'
    cv2.imwrite(str(rgb),np.zeros((360,640,3),dtype=np.uint8))
    cv2.imwrite(str(depth),np.full((360,640),300,dtype=np.uint16))
    vision=MagicMock();vision.request.return_value={'path':str(rgb),'depth_mm_path':str(depth),
        'capture_time_unix_sec':time.time(),'frame_sequence':123}
    reader=MagicMock(return_value=Q.copy())
    frame=t.capture(vision,t.live,q_reader=reader)
    assert frame['sequence']==123 and np.median(frame['depth'])==pytest.approx(.3)
    assert reader.call_count==2
    t.live.live_joints.assert_not_called()


def test_reachable_vertical_lift_avoids_old_joint_interpolation_arc():
    # Synthetic reachable configuration, not a reconstruction of the missing failure frame.
    q=np.array([-.02,1.7209704724874504,-1.9517024158970828,-.1,.03506805351965811,.2])
    goal=forward_kinematics(q);goal[2,3]+=.05
    end,pe,ae=bg.inverse_kinematics(goal,q,regularization=.001,max_nfev=180)
    assert pe<.001 and ae<np.deg2rad(.5)
    old,_=bg.trajectory(q,end)
    old_fk=forward_kinematics_batch(np.array([p['positions'] for p in old]))
    assert np.max(np.linalg.norm(old_fk[:,:2,3]-old_fk[0,:2,3],axis=1))>.003
    path,end,audit=bg.vertical_lift_trajectory(q)
    fk=forward_kinematics_batch(np.array([p['positions'] for p in path]))
    assert audit['max_xy_drift_m']<.0001
    assert np.max(np.linalg.norm(fk[:,:2,3]-fk[0,:2,3],axis=1))<.0001
    assert fk[-1,2,3]-fk[0,2,3]==pytest.approx(.05,abs=.001)
    assert np.max(np.abs(np.array([p['velocities'] for p in path]))/(bg.PIPER_MAX_VELOCITIES*.25))<=1
    np.testing.assert_allclose(path[0]['velocities'],0,atol=1e-12)
    np.testing.assert_allclose(path[-1]['velocities'],0,atol=1e-12)
    np.testing.assert_allclose(path[0]['accelerations'],0,atol=1e-12)
    np.testing.assert_allclose(path[-1]['accelerations'],0,atol=1e-12)


def test_vertical_path_still_rejects_lift_above_workspace_ceiling():
    with pytest.raises(ValueError,match='IK不可达|关节不连续'):
        bg.vertical_lift_trajectory(np.array([0,1.4348966077905356,-2.8310430648105696,
                                             0,-.08781919281139379,0]))
