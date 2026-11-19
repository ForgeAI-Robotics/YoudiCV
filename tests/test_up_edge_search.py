"""An edge UP cue permits guarded search, never lamp/press authorization."""
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from piper_handeye.call_button_search import up_search_anchor
from piper_handeye.observation_visibility import complete_button_box
from piper_handeye.kinematics import forward_kinematics
from run_online_replan_no_contact import (
    collect_stationary_target,
    complete_visible_target,
    wait_for_stable_visibility_after_target_switch,
    wait_for_target_after_motion,
)


FIXTURE = Path(__file__).parent / 'fixtures' / 'up_edge_search'
RECORDED = json.loads((FIXTURE / 'manifest.json').read_text())


def frame(sequence, *, age=0, box=None, down=False):
    detections = copy.deepcopy(RECORDED['detections'])
    if box is not None:
        detections[0]['bbox_xyxy'] = box
    if down:
        detections.append(dict(label='down', confidence=.94,
                               bbox_xyxy=[174,45,207,80]))
    return dict(frame_sequence=sequence, capture_time_unix_sec=time.time()-age,
                image_size=RECORDED['image_size'], detections=detections)


class Vision:
    def __init__(self, frames):
        self.frames = iter(frames)

    def request(self, *args, **kwargs):
        return next(self.frames)


def wait(frames, *, barrier=0):
    return wait_for_stable_visibility_after_target_switch(
        Vision(frames), target_label='up', min_confidence=.7,
        sequence_barrier=barrier)


def test_original_failure_box_is_search_only_and_original_rgb_is_preserved():
    assert hashlib.sha256((FIXTURE/'20261021_093256_camera.jpg').read_bytes()).hexdigest() == RECORDED['camera_sha256']
    response = frame(1)
    assert not complete_button_box(response['detections'][0]['bbox_xyxy'],
                                   response['image_size'])
    assert complete_visible_target(response, 'up', .7) is None
    assert up_search_anchor(response)['label'] == 'up'


def test_parent_and_pointing_acquisition_both_resolve_real_edge_geometry_to_search():
    # Both production callers use this resolver and reacquire after set_target.
    for barrier in [10, 100]:
        response, observation_ready = wait(
            [frame(barrier+1), frame(barrier+2)], barrier=barrier)
        assert not observation_ready
        assert response['visibility_resolution'] == 'up_edge_search_only'
        assert up_search_anchor(response)['label'] == 'up'


def test_complete_down_keeps_priority_as_existing_safety_reference():
    response, ready = wait([frame(1, down=True), frame(2, down=True)])
    assert not ready
    assert response['visibility_resolution'] == 'down_anchor_search_only'
    assert up_search_anchor(response)['label'] == 'down'


def test_complete_up_keeps_direct_path_without_search():
    response, ready = wait([frame(1, box=[170,10,205,47]),
                            frame(2, box=[170,10,205,47])])
    assert ready and complete_visible_target(response,'up',.7) is not None
    assert up_search_anchor(response) is None


def test_missing_and_duplicate_frames_cannot_join_two_edge_cues():
    empty = frame(3); empty['detections'] = []
    response, ready = wait([frame(0), frame(1), frame(1), empty,
                            frame(4), frame(4), frame(5)])
    assert not ready and response['frame_sequence'] == 5


def test_mode_changes_and_stale_or_jumping_frames_reset_streak():
    responses = [frame(1), frame(2, down=True), frame(3, age=1),
                 frame(4), frame(5, box=[230,4,264,40]), frame(6), frame(7)]
    response, ready = wait(responses)
    assert not ready and response['frame_sequence'] == 7


@pytest.mark.parametrize('kind', ['weak','nan_confidence','ambiguous','outside',
                                 'tiny','nan_box','bad_size'])
def test_bad_edge_cue_never_authorizes_search(kind):
    response = frame(1)
    cue = response['detections'][0]
    if kind == 'weak': cue['confidence'] = .69
    if kind == 'nan_confidence': cue['confidence'] = float('nan')
    if kind == 'ambiguous': response['detections'].append(copy.deepcopy(cue))
    if kind == 'outside': cue['bbox_xyxy'] = [170,-1,205,40]
    if kind == 'tiny': cue['bbox_xyxy'] = [170,1,180,11]
    if kind == 'nan_box': cue['bbox_xyxy'][1] = float('nan')
    if kind == 'bad_size': response['image_size'] = [float('nan'),360]
    assert up_search_anchor(response) is None


def test_no_qualified_cue_fails_without_authorizing_adjustment():
    class Empty:
        n = 0
        def request(self,*args,**kwargs):
            self.n += 1
            response = frame(self.n)
            response['detections'] = []
            return response
    with patch('run_online_replan_no_contact.save_missing_button_view'):
        with pytest.raises(RuntimeError,match='no stable fresh visibility'):
            wait_for_stable_visibility_after_target_switch(
                Empty(),target_label='up',min_confidence=.7,
                sequence_barrier=0,timeout_sec=.01)


def test_reacquisition_after_search_requires_two_new_complete_frames():
    class Scene:
        n = 0
        def request(self,*args,**kwargs):
            self.n += 1
            return frame(self.n, box=([170,4,205,40] if self.n in [1,3]
                                      else [170,10,205,46]))
    response, detection = wait_for_target_after_motion(
        Scene(), target_label='up', min_confidence=.7, require_complete=True)
    assert response['frame_sequence'] == 5
    assert complete_button_box(detection['bbox_xyxy'], response['image_size'])


def test_edge_up_alone_never_becomes_a_qualified_post_motion_observation():
    class EdgeOnly:
        n = 0
        def request(self,*args,**kwargs):
            self.n += 1
            return frame(self.n)
    with patch('run_online_replan_no_contact.save_missing_button_view'):
        _, detection = wait_for_target_after_motion(
            EdgeOnly(),target_label='up',min_confidence=.7,
            require_complete=True,timeout_sec=.01)
    assert detection is None


@pytest.mark.parametrize('depth', [.3,.4,.6,.8])
def test_existing_negative_j5_moves_failure_cue_away_from_top_edge(depth):
    # Assumed depth sweep with actual calibration and the saved origin joints;
    # this is a projection check, not a collision or physical movement test.
    root = Path(__file__).parent.parent
    calibration = yaml.safe_load((root/'ik_handeye_refine_output/handeye.yaml').read_text())
    handeye = np.array(calibration['gripper_to_camera'])
    matrix = np.array(calibration['camera_matrix']); matrix[:2] /= 3.
    joints = np.array([-.032,.107,-.001,0,.115,.098])
    box = RECORDED['detections'][0]['bbox_xyxy']
    pixel = np.array([(box[0]+box[2])*.5,(box[1]+box[3])*.5,1.])
    xyz = np.linalg.solve(matrix,pixel)*depth
    anchor = forward_kinematics(joints)@handeye@np.r_[xyz,1.]
    after = joints.copy(); after[4] -= .05
    camera = (np.linalg.inv(forward_kinematics(after)@handeye)@anchor)[:3]
    projected = matrix@camera
    assert projected[1]/projected[2] > pixel[1]+20
    assert np.count_nonzero(after-joints) == 1


@pytest.mark.parametrize('bad_depth', [False,True])
def test_edge_reference_still_needs_four_stable_fresh_depth_samples(bad_depth):
    clock = [100.]
    class Scene:
        n = 0
        def request(self, command, **kwargs):
            assert command == 'latest' and kwargs['target_label'] == 'up'
            clock[0] += .1; self.n += 1
            pose = np.eye(4); pose[2,3] = .5
            return {'sample':dict(sequence=self.n,
                capture_time_unix_sec=clock[0]-.05,
                camera_to_button=pose.tolist(),confidence=.93,
                plane_rms_m=.02 if bad_depth else .001,
                bbox_xyxy=RECORDED['detections'][0]['bbox_xyxy'])}
    robot = SimpleNamespace(state=lambda **kwargs:SimpleNamespace(
        joints=np.zeros(6),base_to_gripper=np.eye(4)))
    with patch('run_online_replan_no_contact.time.time',lambda:clock[0]), \
         patch('run_online_replan_no_contact.time.monotonic',lambda:clock[0]), \
         patch('run_online_replan_no_contact.save_missing_button_view'):
        if bad_depth:
            with pytest.raises(RuntimeError,match='no stable fresh button target'):
                collect_stationary_target(Scene(),robot,np.eye(4),'up',
                                          {'detector':{},'button_geometry':{}})
        else:
            estimate, audit = collect_stationary_target(Scene(),robot,np.eye(4),'up',
                                          {'detector':{},'button_geometry':{}})
            assert estimate.stable and estimate.sample_count == 4
            assert all(row['accepted'] for row in audit[-4:])
            assert all(not complete_button_box(row['bbox_xyxy'],[640,360])
                       for row in audit[-4:])  # Safety cue only, not lamp-ready.
