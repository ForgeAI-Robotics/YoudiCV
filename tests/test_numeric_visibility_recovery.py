import copy
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from piper_handeye.numeric_visibility import (
    wait_numeric_visibility, recover_numeric_target, emit_numeric_progress,
)
from piper_handeye.visual_search import plan_upward_search


class Clock:
    def __init__(self):self.now=0
    def time(self):return self.now
    def sleep(self,t):self.now+=max(.03,t)


def frame(seq, score=.95, *, age=0, box=(298,10,323,37), empty=False):
    return dict(frame_sequence=seq,capture_time_unix_sec=time.time()-age,
                image_size=[640,360],detections=[] if empty else [dict(
                    label='4',confidence=score,bbox_xyxy=list(box))])


class Vision:
    def __init__(self,frames):self.frames=iter(frames);self.last=None
    def request(self,*a,**k):
        self.last=next(self.frames,self.last)
        return copy.deepcopy(self.last)


def wait(frames,**kwargs):
    clock=Clock();vision=Vision(frames)
    return wait_numeric_visibility(vision,target_label='4',min_confidence=.7,
        sequence_barrier=0,timeout_sec=1.,clock=clock.time,sleep=clock.sleep,**kwargs)


def test_weak_then_two_good_frames_recovers_without_motion_or_reducing_threshold():
    check=Mock();response,ready=wait([frame(1,.55),frame(2),frame(3)],check_hold=check)
    assert ready and response['frame_sequence']==3 and check.call_count==3


@pytest.mark.parametrize('gap',[frame(2,empty=True),frame(2,.69),frame(2,age=1)])
def test_nonconsecutive_evidence_does_not_count_as_two_confirmations(gap):
    r,ready=wait([frame(1),gap,frame(3),frame(4)])
    assert ready and r['frame_sequence']==4


def test_duplicates_and_jumping_boxes_cannot_confirm_a_target():
    r,ready=wait([frame(1),frame(1),frame(2,box=(330,10,355,37)),frame(3),frame(4)])
    assert ready and r['frame_sequence']==4


@pytest.mark.parametrize('frames',[[frame(1)], [frame(1,age=1),frame(2,age=1)],
                                   [frame(1,.69),frame(2,.69)]])
def test_missing_fresh_confirmations_fail_within_budget(frames):
    with pytest.raises(RuntimeError,match='no stable fresh visibility'):
        wait(frames)


def test_hardware_or_stop_error_is_not_converted_into_perception_retry():
    with pytest.raises(RuntimeError,match='stop latch'):
        wait([frame(1),frame(2)],check_hold=Mock(side_effect=RuntimeError('stop latch')))


def test_conflicting_duplicate_targets_and_invalid_boxes_do_not_authorize_motion():
    a=frame(1);a['detections'].append(dict(label='4',confidence=.94,bbox_xyxy=[400,10,425,37]))
    b=copy.deepcopy(a);b['frame_sequence']=2
    for frames in ([a,b],[frame(1,box=[-1,10,24,37]),frame(2,box=[-1,10,24,37])]):
        with pytest.raises(RuntimeError,match='no stable fresh visibility'):
            wait(frames)


def test_missing_timestamps_never_count_as_fresh_images():
    a=frame(1);b=frame(2)
    del a['capture_time_unix_sec'];del b['capture_time_unix_sec']
    with pytest.raises(RuntimeError,match='no stable fresh visibility'):
        wait([a,b])


def reference():
    samples=[dict(sequence=i,capture_time_unix_sec=time.time()-.01,
                  accepted=True,confidence=.95,bbox_xyxy=[298,10,323,37]) for i in range(1,5)]
    samples[-1]['acquisition_summary']=dict(target_label='4',window_sample_count=4)
    estimate=SimpleNamespace(stable=True,base_to_button=np.eye(4))
    state=SimpleNamespace(joints=np.zeros(6),raw={'hardware_ok':1})
    robot=Mock();robot.state.return_value=state
    return estimate,samples,robot


def test_real_failure_branch_already_visible_promotes_same_target_reference():
    plan=plan_upward_search('4',['1','2','4','9','10'],camera_rows_above_visible=0,
                            camera_target_column=0)
    assert plan.direction=='already_visible'
    estimate,samples,robot=reference();acquire=Mock()
    e,s,q,reused=recover_numeric_target('4','4',estimate,samples,np.zeros(6),
                                      robot,acquire,image_size=[640,360])
    assert reused and e is estimate and s is samples
    acquire.assert_not_called()


@pytest.mark.parametrize('fault',['wrong_label','stale','drift','incomplete','weak','unstable','duplicate'])
def test_invalid_reference_is_never_promoted_without_fresh_acquisition(fault):
    e,s,r=reference();anchor='4'
    if fault=='wrong_label':anchor='2'
    if fault=='stale':s[-1]['capture_time_unix_sec']-=1
    if fault=='drift':r.state.return_value.joints[2]=.004
    if fault=='incomplete':s[-1]['bbox_xyxy']=[298,0,323,27]
    if fault=='weak':s[-1]['confidence']=.69
    if fault=='unstable':e.stable=False
    if fault=='duplicate':s[-1]['sequence']=s[-2]['sequence']
    new_e,new_s,_=reference();acquire=Mock(return_value=(new_e,new_s))
    result=recover_numeric_target('4',anchor,e,s,np.zeros(6),r,acquire,image_size=[640,360])
    assert not result[-1] and result[0] is new_e
    acquire.assert_called_once()


def test_hardware_failure_and_failed_reacquisition_cannot_return_ready():
    e,s,r=reference();r.state.return_value.raw['hardware_ok']=0
    with pytest.raises(RuntimeError,match='hardware'):
        recover_numeric_target('4','4',e,s,np.zeros(6),r,Mock(),image_size=[640,360])
    e,s,r=reference()
    with pytest.raises(RuntimeError,match='target still missing'):
        recover_numeric_target('4','2',e,s,np.zeros(6),r,
                               Mock(side_effect=RuntimeError('target still missing')),image_size=[640,360])


def test_live_progress_is_bound_to_current_task_and_never_claims_press_success(tmp_path,monkeypatch):
    path=tmp_path/'status.jsonl'
    monkeypatch.setenv('YOUXIAOMEI_TASK_ID','current-task')
    monkeypatch.setenv('YOUXIAOMEI_MISSION_STATUS_OUTBOX',str(path))
    emit_numeric_progress('numeric_visibility_waiting',target_label='4')
    emit_numeric_progress('numeric_target_recovered',target_label='4')
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert all(r['task_id']=='current-task' for r in rows)
    assert [r['sequence'] for r in rows]==[1,2]
    assert rows[-1]['outcome']=='active' and not any(r['phase']=='button_on_confirmed' for r in rows)
