"""Replay actual missing-7 evidence without a camera, ROS, model or motion."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from piper_handeye.numeric_search_evidence import numeric_search_evidence
from piper_handeye.numeric_visibility import wait_numeric_visibility
from piper_handeye.visual_search import predict_button_center_from_grid


RECORD = json.loads((Path(__file__).parent/'fixtures/numeric_search_context/recorded_7_missing.json').read_text())


@pytest.fixture
def frames(monkeypatch):
    rows = copy.deepcopy(RECORD['frames'])
    monkeypatch.setattr('time.time', lambda: rows[-1]['capture_time_unix_sec']+.01)
    return rows


def test_real_failed_frames_recover_a_complete_anchor_without_creating_target(frames):
    for frame in frames[:1]:
        before = copy.deepcopy(frame)
        with pytest.raises(RuntimeError, match='no upward-only column'):
            predict_button_center_from_grid('7', frame['detections'])
        detections, prediction, audit = numeric_search_evidence(frame, '7', .7)
        assert frame == before
        assert prediction.anchor_label == '3'
        assert prediction.center_xy[1] < 0 and prediction.fit_rms_px < 1
        assert prediction.rows_above_visible == pytest.approx(4, abs=.05)
        assert '7' not in [d['label'] for d in detections]
        assert {'1','2','9','10'} <= set(audit['independent_supports'])
        assert audit['purpose'] == 'upward_search_only' and audit['target_detected'] is False
        assert '4' not in audit['labels']  # clipped top row is not a depth anchor


def test_next_recorded_frame_with_3_misread_as_8_is_rejected(frames):
    # The actual next frame is inconsistent, not a second independent success.
    with pytest.raises(RuntimeError, match='no upward-only column'):
        numeric_search_evidence(frames[1], '7', .7)


def test_working_raw_grid_is_preserved_without_replacing_detections(frames):
    frame = frames[0]
    frame['detections'] = copy.deepcopy(frame['detection_enhancement']['tile_detections'])
    detections, prediction, audit = numeric_search_evidence(frame, '7', .7)
    assert detections is frame['detections'] and audit == {'source':'raw'}


@pytest.mark.parametrize('fault', ['stale','future','missing_time','frame_mismatch',
    'wrong_algorithm','no_retry','crop_outside','weak_correction','few_supports',
    'local_conflict','non_numeric_conflict','duplicate_label','direct_target'])
def test_invalid_local_context_does_not_authorize_search(frames, fault):
    frame = frames[0]
    audit = frame['detection_enhancement']
    local = audit['tile_detections']
    correction = next(d for d in local if d['label']=='3')
    if fault == 'stale':frame['capture_time_unix_sec'] -= 1
    if fault == 'future':frame['capture_time_unix_sec'] += 1
    if fault == 'missing_time':del frame['capture_time_unix_sec']
    if fault == 'frame_mismatch':audit['frame_sequence'] = frame['frame_sequence']-1
    if fault == 'wrong_algorithm':audit['algorithm'] = 'lab_clahe2_gamma075'
    if fault == 'no_retry':audit['attempted'] = False
    if fault == 'crop_outside':audit['tile_xyxy'][0] = -1
    if fault == 'weak_correction':correction['confidence'] = .89
    if fault == 'few_supports':
        audit['tile_detections'] = [d for d in local if d['label'] in ('3','1')]
    if fault == 'local_conflict':
        local.append(dict(label='6',confidence=.90,bbox_xyxy=correction['bbox_xyxy']))
    if fault == 'non_numeric_conflict':
        frame['detections'].append(dict(label='open',confidence=.98,bbox_xyxy=correction['bbox_xyxy']))
    if fault == 'duplicate_label':
        local.append(dict(label='3',confidence=.95,bbox_xyxy=[365,150,392,175]))
    if fault == 'direct_target':
        frame['detections'].append(dict(label='7',confidence=.95,bbox_xyxy=[500,100,525,125]))
    with pytest.raises(RuntimeError):
        numeric_search_evidence(frame,'7',.7)


class Clock:
    def __init__(self):self.now=0
    def time(self):return self.now
    def sleep(self,t):self.now+=max(.03,t)


def wait(frames):
    clock = Clock()
    iterator = iter(frames)
    last = frames[-1]
    vision = Mock()
    vision.request.side_effect = lambda *a,**k: copy.deepcopy(next(iterator,last))
    hold = Mock()
    result = wait_numeric_visibility(vision,target_label='7',min_confidence=.7,
        sequence_barrier=0,timeout_sec=.5,clock=clock.time,sleep=clock.sleep,check_hold=hold)
    return result, hold


def stable_test_frames(frames):
    # Synthetic continuation tests scheduling only, never claims live success.
    a = copy.deepcopy(frames[0])
    b = copy.deepcopy(a)
    b['frame_sequence'] += 2
    b['capture_time_unix_sec'] = frames[1]['capture_time_unix_sec']
    return [a,b]


def test_two_consistent_new_test_frames_return_search_only_not_target_visible(frames):
    sequence = stable_test_frames(frames)
    (response, ready), hold = wait(sequence)
    assert not ready and hold.call_count==2
    assert response['frame_sequence']==sequence[-1]['frame_sequence']
    assert response['numeric_search_evidence']['source']=='numeric_tile_context'
    assert response['detections']==sequence[-1]['detections']
    assert [r['consecutive'] for r in response['numeric_visibility_audit']]==[1,2]


@pytest.mark.parametrize('gap',['duplicate','same_capture','stale','bad_context','source_change'])
def test_single_frame_or_interrupted_context_cannot_confirm_search(frames,gap):
    a,b = stable_test_frames(frames)
    if gap=='duplicate':sequence=[a,a]
    else:
        bad=copy.deepcopy(b)
        if gap=='same_capture':bad['capture_time_unix_sec']=a['capture_time_unix_sec']
        if gap=='stale':bad['capture_time_unix_sec']-=1
        if gap=='bad_context':bad['detection_enhancement']['algorithm']='wrong'
        if gap=='source_change':bad['detections']=bad['detection_enhancement']['tile_detections']
        sequence=[a,bad]
    with pytest.raises(RuntimeError,match='no stable fresh visibility'):
        wait(sequence)


def test_actual_recorded_misclassification_interrupts_search_confirmation(frames):
    with pytest.raises(RuntimeError,match='no stable fresh visibility'):
        wait(frames)
