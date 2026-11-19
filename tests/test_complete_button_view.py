import time
from unittest.mock import patch
import pytest
from piper_handeye.observation_visibility import complete_button_box
from piper_handeye.button_light_hardware import FreshLightObserver
from run_online_replan_no_contact import complete_visible_target, wait_for_target_after_motion, should_center_visible_target


@pytest.mark.parametrize('box',[[148,0,184,28],[148,4,184,35],[630,30,650,60],[10,20,18,28]])
def test_clipped_or_too_small_button_is_not_observation_ready(box):
    assert not complete_button_box(box,[640,360])


def test_visible_target_needs_full_crop_and_complete_targets_keep_original_behavior():
    response={'image_size':[640,360],'detections':[
        {'label':'up','confidence':.93,'bbox_xyxy':[148,0,184,28]},
        {'label':'down','confidence':.85,'bbox_xyxy':[153,38,187,73]}]}
    assert complete_visible_target(response,'up',.7) is None
    assert complete_visible_target(response,'down',.7) is not None
    assert should_center_visible_target(auto_pointing_enabled=True,legacy_center_requested=False,
        target_visible_at_origin=True,target_was_pointed=True,incomplete_view=True)
    assert not should_center_visible_target(auto_pointing_enabled=True,legacy_center_requested=False,
        target_visible_at_origin=True,target_was_pointed=False,incomplete_view=False)


def test_j5_search_requires_two_full_post_motion_images():
    class Vision:
        def __init__(self):self.calls=0
        def request(self,*args,**kwargs):
            self.calls+=1
            return dict(frame_sequence=self.calls,capture_time_unix_sec=time.time(),image_size=[640,360],
                detections=[dict(label='up',confidence=.95,bbox_xyxy=[148,0 if self.calls==1 else 10,184,40])])
    vision=Vision();response,found=wait_for_target_after_motion(vision,target_label='up',min_confidence=.7,require_complete=True)
    assert found is not None and response['frame_sequence']==3


def test_duplicate_yolo_frame_does_not_confirm_complete_view_twice():
    class Vision:
        def __init__(self):self.calls=0
        def request(self,*args,**kwargs):
            self.calls+=1
            return dict(frame_sequence=1 if self.calls<3 else 2,capture_time_unix_sec=time.time(),image_size=[640,360],
                detections=[dict(label='up',confidence=.95,bbox_xyxy=[148,10,184,40])])
    vision=Vision();response,found=wait_for_target_after_motion(vision,target_label='up',min_confidence=.7,require_complete=True)
    assert found is not None and vision.calls==3


@pytest.mark.parametrize('bad', ['crop','old_rgb','old_box'])
def test_invalid_lamp_view_never_calls_recognizer(bad):
    class Vision:
        def request(self,command,**kwargs):
            return dict(fresh=True,frame_sequence=30,path='/tmp/no-image-needed.jpg',image_size=[640,360],
                capture_time_unix_sec=time.time()-(1 if bad=='old_rgb' else 0),
                target_detection=dict(frame_sequence=20 if bad=='old_box' else 30,confidence=.95,
                                      bbox_xyxy=[148,0 if bad=='crop' else 10,184,40]))
    class Recognizer:
        def observe(self,*args,**kwargs):raise AssertionError('unsafe crop reached classifier')
    observer=FreshLightObserver(Vision(),Recognizer(),'up')
    with patch('piper_handeye.button_light_hardware.save_missing_button_view',return_value={}):
        assert observer.observe('waiting')=='UNCERTAIN'
    assert observer.audit[-1]['recognition']['recognizer']=='skipped_unsafe_target_crop'
