import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from piper_handeye.button_pose import ButtonDetection
from piper_handeye.detection_enhancement import detect_with_enhancement_fallback
from piper_handeye.numeric_button_detection import numeric_panel_crop, retry_numeric_button_tile


FIXTURE = Path(__file__).parent/'fixtures/numeric_visibility'
RECORD = json.loads((FIXTURE/'manifest.json').read_text())


def d(label, box, score=.95):
    return ButtonDetection(0,label,score,tuple(box))


def scene():
    image = cv2.imread(str(FIXTURE/'failure.jpg'))
    rows = RECORD['frames'][0]['live_recognition']['detection_enhancement']['raw_detections']
    raw = tuple(d(r['label'],r['bbox_xyxy'],r['confidence']) for r in rows)
    roi = numeric_panel_crop(image,raw)
    x,y,_,_ = roi
    mapped = [d('4',[298,10,323,37]), d('1',[300,106,324,130]),
              d('2',[299,75,324,100]),d('9',[337,75,361,100])]
    local = [d(a.label,[a.bbox_xyxy[0]-x,a.bbox_xyxy[1]-y,
                      a.bbox_xyxy[2]-x,a.bbox_xyxy[3]-y],a.confidence) for a in mapped]
    return image,raw,roi,local


def test_recorded_failure_crop_is_real_and_panel_roi_improves_input_scale():
    for f in RECORD['frames']:
        assert hashlib.sha256((FIXTURE/f['image']).read_bytes()).hexdigest() == f['sha256']
    image,raw,roi,_ = scene()
    assert roi[0]<298 and roi[2]>361 and roi[1]==0 and roi[3]>150
    assert max(roi[2]-roi[0],roi[3]-roi[1]) < image.shape[1]*.6


def test_same_frame_target_can_replace_raw_14_with_independent_neighbour_support():
    image,raw,roi,local = scene(); detector=Mock();detector.detect.return_value=local
    original=image.copy();result,audit=retry_numeric_button_tile(detector,image,raw,'4',.7)
    assert audit['source']=='numeric_tile' and audit['accepted']
    assert set(audit['independent_supports']) >= {'1','9'}
    assert '14' in audit['displaced_raw_labels']
    target=next(a for a in result if a.label=='4')
    assert target.bbox_xyxy == (298,10,323,37)
    assert np.array_equal(image,original)
    assert detector.detect.call_count==1


@pytest.mark.parametrize('fault',['weak','conflict','ambiguous','missing_support','bad_box','clipped','nan'])
def test_bad_local_evidence_never_promotes_target(fault):
    image,raw,roi,local=scene();x,y,right,bottom=roi
    if fault=='weak':local[0]=d('4',local[0].bbox_xyxy,.69)
    if fault=='conflict':local.append(d('14',local[0].bbox_xyxy,.94))
    if fault=='ambiguous':local.append(d('4',[65,45,89,72],.94))
    if fault=='missing_support':local=local[:1]
    if fault=='bad_box':local[0]=d('4',[-2,10,20,37])
    if fault=='clipped':local[0]=d('4',[1,10,25,37])
    if fault=='nan':local[0]=d('4',local[0].bbox_xyxy,float('nan'))
    detector=Mock();detector.detect.return_value=local
    result,audit=retry_numeric_button_tile(detector,image,raw,'4',.7)
    assert not audit['accepted'] and result is raw


def test_unrelated_strong_target_is_not_hidden_and_disabled_path_is_unchanged():
    image,raw,_,_=scene();strong=raw+(d('4',[200,100,225,126]),)
    detector=Mock();detector.detect.return_value=strong
    result,audit=detect_with_enhancement_fallback(detector,image,'4',.7)
    assert result==strong and not audit['attempted'] and detector.detect.call_count==1
    detector=Mock();detector.detect.return_value=raw
    result,audit=detect_with_enhancement_fallback(detector,image,'4',.7,enabled=False)
    assert result==raw and detector.detect.call_count==1


def test_numeric_fallback_uses_one_extra_inference_and_never_stacks_clahe():
    image,raw,_,local=scene();detector=Mock();detector.detect.side_effect=[raw,local]
    with patch('piper_handeye.detection_enhancement.enhance_detection_image') as enhance:
        result,audit=detect_with_enhancement_fallback(detector,image,'4',.7)
    assert audit['source']=='numeric_tile' and detector.detect.call_count==2
    enhance.assert_not_called()


def test_insufficient_numeric_context_keeps_original_enhancement_policy():
    detector=Mock();detector.detect.side_effect=[[],[d('4',[50,50,75,75])]]
    _,audit=detect_with_enhancement_fallback(detector,np.zeros((100,100,3),np.uint8),'4',.7)
    assert audit['source']=='enhanced' and detector.detect.call_count==2


def test_conflict_with_strong_raw_numeric_label_requires_point_nine_local_score():
    image,raw,_,local=scene()
    raw=tuple(d(a.label,a.bbox_xyxy,.98 if a.label=='14' else a.confidence) for a in raw)
    local[0]=d('4',local[0].bbox_xyxy,.89)
    detector=Mock();detector.detect.return_value=local
    result,audit=retry_numeric_button_tile(detector,image,raw,'4',.7)
    assert result is raw and audit['reason']=='numeric_scale_conflict_requires_strong_local_evidence'


def test_raw_non_numeric_category_cannot_be_overwritten_by_a_numeric_retry():
    image,raw,_,local=scene()
    raw=raw+(d('open',[298,10,323,37],.99),)
    detector=Mock();detector.detect.return_value=local
    result,audit=retry_numeric_button_tile(detector,image,raw,'4',.7)
    assert result is raw and audit['reason']=='raw_non_numeric_button_conflict'


def test_call_buttons_still_use_the_existing_call_only_policy():
    image,raw,_,_=scene();detector=Mock();detector.detect.return_value=raw
    with patch('piper_handeye.detection_enhancement.retry_call_button_tile',return_value=(raw,dict(attempted=False))) as call, \
         patch('piper_handeye.detection_enhancement.retry_numeric_button_tile') as numeric:
        detect_with_enhancement_fallback(detector,image,'up',.7)
    call.assert_called_once();numeric.assert_not_called()


def test_recognition_evidence_draws_numeric_retry_on_its_original_frame(tmp_path):
    from types import SimpleNamespace
    from run_button_vision_server import VisionService
    service=VisionService.__new__(VisionService);service.args=SimpleNamespace(token='test')
    service.camera=Mock();service.detector=Mock();service.perception=Mock()
    frame=SimpleNamespace(sequence=99,received_unix=123.,color=np.zeros((100,160,3),np.uint8))
    mapped=[dict(label='4',confidence=.94,bbox_xyxy=[40,20,64,47])]
    service.perception.latest_sample.return_value=SimpleNamespace(
        sequence=99,frame=frame,target_label='4',detections=(),
        detection_enhancement=dict(source='raw',algorithm='numeric_panel_local_scale',
                                   accepted=False,raw_detections=[],tile_detections=mapped))
    with patch('run_button_vision_server.ROOT',tmp_path):
        result=service.dispatch({'command':'recognition_snapshot','token':'test'})
    assert result['frame_sequence']==99 and Path(result['tile_yolo_path']).is_file()
    service.camera.latest_frame.assert_not_called();service.detector.detect.assert_not_called()
