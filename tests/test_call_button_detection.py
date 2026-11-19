import time
from unittest.mock import patch
import numpy as np
import pytest
from piper_handeye.button_pose import ButtonDetection
from piper_handeye.detection_enhancement import detect_with_enhancement_fallback
from run_online_replan_no_contact import wait_for_stable_visibility_after_target_switch


def det(label, confidence, box):
    return ButtonDetection(0,label,confidence,tuple(box))


class Detector:
    def __init__(self, results): self.results=iter(results);self.shapes=[]
    def detect(self,image): self.shapes.append(image.shape);return next(self.results)


IMAGE=np.zeros((360,640,3),np.uint8)


def test_clipped_up_triggers_tile_even_if_raw_confident_and_retains_down_anchor():
    d=Detector([[det('up',.92,[190,0,225,32])],
                [det('up',.95,[190,0,225,32]),det('down',.94,[194,40,227,75])]])
    output,audit=detect_with_enhancement_fallback(d,IMAGE,'up',.7)
    assert audit['source']=='call_tile' and d.shapes==[(360,640,3),(320,320,3)]
    assert {x.label for x in output}=={'up','down'}
    assert next(x for x in output if x.label=='down').bbox_xyxy==(194,40,227,75)


def test_crop_offset_restored_and_source_image_unchanged():
    image=IMAGE.copy();d=Detector([[det('down',.3,[480,240,514,275])],[det('down',.93,[160,200,194,235])]])
    output,audit=detect_with_enhancement_fallback(d,image,'down',.7)
    assert audit['tile_xyxy']==[320,40,640,360]
    assert max(output,key=lambda x:x.confidence).bbox_xyxy==(480,240,514,275)
    assert np.array_equal(image,IMAGE)


def test_good_full_target_and_ambiguous_raw_do_not_retry():
    for raw in [[det('down',.9,[190,40,225,75])],
                [det('up',.9,[190,0,225,32]),det('up',.91,[390,0,425,32])]]:
        d=Detector([raw]);out,audit=detect_with_enhancement_fallback(d,IMAGE,raw[0].label,.7)
        assert len(d.shapes)==1 and not audit['attempted'] and out==tuple(raw)


@pytest.mark.parametrize('bad',[
    det('down',.69,[194,40,227,75]),det('down',.9,[310,40,320,75]),
    det('up',.9,[20,310,55,320]),det('down',float('nan'),[194,40,227,75]),
    det('down',.9,[float('nan'),40,227,75]),det('8',.99,[194,40,227,75])])
def test_weak_crop_clipped_nonfinite_or_other_label_cannot_authorize(bad):
    d=Detector([[],[bad]]);out,audit=detect_with_enhancement_fallback(d,IMAGE,'up',.7)
    assert not out and not audit['accepted']


def test_conflicting_strong_labels_in_same_region_not_merged():
    raw=[det('up',.9,[190,0,225,32])]
    d=Detector([raw,[det('down',.95,[190,0,225,32])]])
    out,audit=detect_with_enhancement_fallback(d,IMAGE,'up',.7)
    assert out==tuple(raw) and audit['reason']=='conflicting_call_labels'


def test_successful_tile_hint_requires_new_inference_on_next_frame():
    d=Detector([[],[det('down',.94,[194,40,227,75])],[],[]])
    first,a=detect_with_enhancement_fallback(d,IMAGE,'down',.7)
    second,b=detect_with_enhancement_fallback(d,IMAGE,'down',.7)
    assert first and not second and len(d.shapes)==4
    assert a['tile_xyxy']==b['tile_xyxy'] and not d._call_tile_hints


def test_unknown_location_scans_one_tile_per_frame_never_six_in_one_frame():
    d=Detector([[]]*12);tiles=[]
    for i in range(6):
        _,a=detect_with_enhancement_fallback(d,IMAGE,'up',.7);tiles.append(tuple(a['tile_xyxy']))
    assert len(set(tiles))==6 and len(d.shapes)==12


def test_clipped_up_routes_to_full_down_anchor_not_observation_success():
    class Vision:
        n=0
        def request(self,*args,**kw):
            self.n+=1
            return dict(frame_sequence=self.n,capture_time_unix_sec=time.time(),image_size=[640,360],
                        detections=[dict(label='up',confidence=.95,bbox_xyxy=[190,0,225,32]),
                                    dict(label='down',confidence=.94,bbox_xyxy=[194,40,227,75])])
    response,visible=wait_for_stable_visibility_after_target_switch(
        Vision(),target_label='up',min_confidence=.7,sequence_barrier=0)
    assert not visible and response['frame_sequence']==2


def test_stale_up_frames_never_authorize_motion():
    class Vision:
        n=0
        def request(self,*a,**k):
            self.n+=1
            return dict(frame_sequence=self.n,capture_time_unix_sec=time.time()-1,image_size=[640,360],
                        detections=[dict(label='up',confidence=.99,bbox_xyxy=[190,40,225,75])])
    with patch('run_online_replan_no_contact.save_missing_button_view'):
        with pytest.raises(RuntimeError,match='no stable fresh visibility'):
            wait_for_stable_visibility_after_target_switch(Vision(),target_label='up',
                min_confidence=.7,sequence_barrier=0,timeout_sec=.01)
