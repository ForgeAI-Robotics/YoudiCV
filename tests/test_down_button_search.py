import time
from unittest.mock import patch
import numpy as np
import pytest
from piper_handeye.call_button_search import down_search_anchor, choose_down_search_j5_step
from run_online_replan_no_contact import wait_for_stable_visibility_after_target_switch, wait_for_target_after_motion


def response(sequence,down=.2,up=.92,age=0):
    return dict(frame_sequence=sequence,capture_time_unix_sec=time.time()-age,image_size=[640,360],
                detections=[dict(label='up',confidence=up,bbox_xyxy=[190,0,225,32]),
                            dict(label='down',confidence=down,bbox_xyxy=[194,40,227,75])])


class Vision:
    def __init__(self,frames):self.frames=iter(frames)
    def request(self,*a,**k):return next(self.frames)


def test_two_low_down_frames_resolve_to_search_only_with_clipped_strong_up():
    r,visible=wait_for_stable_visibility_after_target_switch(Vision([response(1),response(2)]),
            target_label='down',min_confidence=.7,sequence_barrier=0)
    assert not visible and r['visibility_resolution']=='up_anchor_search_only'


def test_stable_down_keeps_normal_no_search_path():
    r,visible=wait_for_stable_visibility_after_target_switch(Vision([response(1,down=.9),response(2,down=.9)]),
            target_label='down',min_confidence=.7,sequence_barrier=0)
    assert visible and r['visibility_resolution']=='stable_down_target'


def test_one_good_down_must_not_bypass_stability_due_to_anchor():
    r,visible=wait_for_stable_visibility_after_target_switch(Vision([response(1),response(2,down=.9)]),
            target_label='down',min_confidence=.7,sequence_barrier=0)
    assert not visible


def test_duplicates_stale_frames_and_jumps_do_not_count_twice():
    frames=[response(1,age=1),response(2),response(2),response(3)]
    frames[3]['detections'][0]['bbox_xyxy']=[250,0,285,32]
    frames.append(response(4));frames.append(response(5))
    r,visible=wait_for_stable_visibility_after_target_switch(Vision(frames),target_label='down',min_confidence=.7,sequence_barrier=0)
    assert r['frame_sequence']==5 and not visible


@pytest.mark.parametrize('kind',['weak','ambiguous','tiny','outside','nan'])
def test_bad_anchor_never_authorizes_search(kind):
    r=response(1)
    if kind=='weak':r['detections'][0]['confidence']=.69
    if kind=='ambiguous':r['detections'].append(r['detections'][0].copy())
    if kind=='tiny':r['detections'][0]['bbox_xyxy']=[190,0,200,10]
    if kind=='outside':r['detections'][0]['bbox_xyxy']=[190,-20,225,10]
    if kind=='nan':r['detections'][0]['bbox_xyxy']=[190,0,float('nan'),32]
    assert down_search_anchor(r) is None


def test_no_target_or_anchor_times_out_without_motion():
    class Empty:
        n=0
        def request(self,*a,**k):
            self.n+=1
            return response(self.n,down=.3,up=.4)
    with patch('run_online_replan_no_contact.save_missing_button_view'):
        with pytest.raises(RuntimeError,match='guarded search unavailable'):
            wait_for_stable_visibility_after_target_switch(Empty(),target_label='down',min_confidence=.7,sequence_barrier=0,timeout_sec=.01)


def test_reacquisition_requires_fresh_full_high_confidence_down():
    class Scene:
        n=0
        def request(self,*a,**k):
            self.n+=1
            return response(self.n,down=.3 if self.n==1 else .95)
    r,target=wait_for_target_after_motion(Scene(),target_label='down',min_confidence=.7,require_complete=True)
    assert target is not None and r['frame_sequence']==3


@pytest.mark.parametrize('sign',[1,-1])
def test_wrist_direction_follows_camera_geometry_instead_of_label(sign):
    def fk(q):
        transform=np.eye(4);transform[1,3]=sign*q[4]
        return transform
    k=np.array([[500,0,320],[0,500,180],[0,0,1.]])
    with patch('piper_handeye.call_button_search.forward_kinematics',side_effect=fk):
        q,audit=choose_down_search_j5_step(np.zeros(6),np.eye(4),[0,-.16,.5],k,[640,360],[640,360])
    assert q[4]==-.05*sign
    assert np.count_nonzero(q)==1
    assert audit['view_error_after']<audit['view_error_before']


def test_search_refuses_step_that_worsens_view():
    def fk(q):
        transform=np.eye(4);transform[1,3]=q[4];return transform
    k=np.array([[500,0,320],[0,500,180],[0,0,1.]])
    with patch('piper_handeye.call_button_search.forward_kinematics',side_effect=fk):
        with pytest.raises(RuntimeError,match='cannot further improve'):
            choose_down_search_j5_step(np.zeros(6),np.eye(4),[0,-.054,.5],k,[640,360],[640,360])


def test_call_button_policy_is_separate_from_numeric_reference_policy():
    from run_button_light_closed_loop import requires_fixed_call_reference
    assert requires_fixed_call_reference('up')
    assert requires_fixed_call_reference('down')
    assert not requires_fixed_call_reference('8')
    assert not requires_fixed_call_reference('16')


def test_anchor_behind_camera_never_produces_a_search_step():
    k=np.array([[500,0,320],[0,500,180],[0,0,1.]])
    with patch('piper_handeye.call_button_search.forward_kinematics',return_value=np.eye(4)):
        with pytest.raises(RuntimeError,match='in front'):
            choose_down_search_j5_step(np.zeros(6),np.eye(4),[0,0,-.5],k,[640,360],[640,360])
