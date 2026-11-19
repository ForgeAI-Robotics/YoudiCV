"""No motion: retain real failed frames and gate startup on emergency feedback."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import time

import cv2
import numpy as np
import pytest

from piper_handeye.failure_image_archive import FailureImageArchive,remove_success_images,MAX_PER_BUCKET
from piper_handeye.cycle_fault_recovery import EmergencyFeedbackError,require_emergency_released,emergency_stop_reason


def observation(tmp_path,seq=1,position='before_press',*,mismatch=False):
    image=np.zeros((80,100,3),dtype=np.uint8);image[20:50,30:60]=(20,100,220)
    p=tmp_path/('frame_%d.jpg'%seq)
    assert cv2.imwrite(str(p),image)
    return dict(image=str(p),position=position,frame_sequence=seq,state='ON',
        capture_time_unix_sec=time.time(),target_detection=dict(frame_sequence=seq+(1 if mismatch else 0),bbox_xyxy=[30,20,60,50]),
        recognition=dict(legacy_state='OFF',state='ON',local_regions=[dict(contrast=86.5)]))


def test_failed_reference_survives_pruned_temporary_images_with_exact_pixels(tmp_path):
    archive=FailureImageArchive(tmp_path/'persistent_failure');archive.begin_attempt(1)
    expected={}
    for i in range(14):
        entry=observation(tmp_path,i);raw=Path(entry['image']).read_bytes();expected[i]=raw
        archive.capture(entry);Path(entry['image']).unlink()
    saved=archive.finalize(success=False,context=dict(reason='cannot_confirm_off'))
    report=json.loads(Path(saved['record']).read_text())
    assert saved['saved_frames']==14 and len(report['records'])==14
    for record in report['records']:
        seq=record['observation']['frame_sequence'];raw=expected[seq]
        assert Path(record['original_image']).read_bytes()==raw
        assert record['sha256']==hashlib.sha256(raw).hexdigest()
        expected_crop=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)[20:50,30:60]
        assert np.array_equal(cv2.imread(record['crop_image']),expected_crop)
        assert record['observation']['recognition']['local_regions'][0]['contrast']==86.5


def test_success_does_not_create_persistent_image_archive(tmp_path):
    archive=FailureImageArchive(tmp_path/'persistent_failure');archive.begin_attempt(1)
    archive.capture(observation(tmp_path))
    result=archive.finalize(success=True,context={})
    assert result['retained'] is False and not archive.directory.exists()


def test_failed_first_attempt_retained_even_when_retry_succeeds(tmp_path):
    archive=FailureImageArchive(tmp_path/'failure');archive.capture(observation(tmp_path,1,'waiting'))
    archive.begin_attempt(1);archive.capture(observation(tmp_path,2))
    archive.begin_attempt(2);archive.capture(observation(tmp_path,3))
    saved=archive.finalize(success=True,context={'final_result':'retry_success'})
    report=json.loads(Path(saved['record']).read_text())
    assert report['failed_attempts']==[1]
    assert {r['observation']['frame_sequence'] for r in report['records']}=={1,2}


def test_mismatched_region_saves_original_without_inventing_crop(tmp_path):
    archive=FailureImageArchive(tmp_path/'failure');archive.capture(observation(tmp_path,mismatch=True))
    result=archive.finalize(success=False,context={})
    record=json.loads(Path(result['record']).read_text())['records'][0]
    assert Path(record['original_image']).exists() and record['crop_image'] is None
    assert 'same-frame' in record['crop_error']


def test_failed_task_without_capture_records_why_no_picture_exists(tmp_path):
    archive=FailureImageArchive(tmp_path/'failure')
    archive.capture(dict(image='/missing/frame.jpg',frame_sequence=1))
    result=archive.finalize(success=False,context={'failure_stage':'startup'})
    report=json.loads(Path(result['record']).read_text())
    assert report['saved_frames']==0 and report['capture_errors']


def test_buffer_has_per_stage_bound(tmp_path):
    archive=FailureImageArchive(tmp_path/'failure')
    for i in range(40):archive.capture(observation(tmp_path,i))
    result=archive.finalize(success=False,context={})
    assert result['saved_frames']==MAX_PER_BUCKET
    frames={r['observation']['frame_sequence'] for r in json.loads(Path(result['record']).read_text())['records']}
    assert {0,1,39}.issubset(frames) and 2 not in frames


def test_success_cleanup_only_owns_current_run_images(tmp_path):
    old=tmp_path/'old';old.mkdir();(old/'old.jpg').write_bytes(b'old')
    run=tmp_path/'new';run.mkdir();(run/'sample.jpg').write_bytes(b'current')
    (run/'report.json').write_text('{}');(run/'linked.jpg').symlink_to(old/'old.jpg')
    assert remove_success_images(run)==1
    assert (old/'old.jpg').read_bytes()==b'old'
    assert (run/'report.json').exists() and (run/'linked.jpg').is_symlink()


@pytest.mark.parametrize('sample,code',[
    ({'available':False,'error':None},'feedback_unavailable'),
    ({'available':True,'engaged':False,'age_sec':.1,'error':'disconnected'},'feedback_error'),
    ({'available':True,'engaged':False,'age_sec':1.501},'feedback_stale'),
    ({'available':True,'engaged':True,'age_sec':.1},'emergency_engaged'),
    ({'available':True,'engaged':False,'age_sec':float('nan')},'feedback_invalid'),
    ({'available':True,'engaged':None,'age_sec':.1},'feedback_invalid')])
def test_emergency_reason_and_original_feedback_are_distinct(sample,code):
    with pytest.raises(EmergencyFeedbackError) as failure:require_emergency_released(sample)
    assert failure.value.code==code and failure.value.audit['emergency_feedback']==sample
    assert emergency_stop_reason(failure.value)==('cycle_physical_estop' if code=='emergency_engaged' else 'cycle_feedback_unavailable')


def test_emergency_age_limit_is_not_relaxed():
    assert require_emergency_released(dict(available=True,engaged=False,age_sec=1.5))
    with pytest.raises(EmergencyFeedbackError):require_emergency_released(dict(available=True,engaged=False,age_sec=1.50001))


def test_actual_readiness_waits_for_first_emergency_packet_before_other_checks():
    from test_button_cycle_worker import ready_probe
    ready=ready_probe();ready.requires_emergency_feedback=True;ready.emergency=Mock()
    ready.emergency.snapshot.return_value=dict(available=False,error=None)
    with pytest.raises(EmergencyFeedbackError,match='尚未收到'):ready()
    ready.arm.status.assert_not_called()
    ready.emergency.snapshot.return_value=dict(available=True,engaged=False,age_sec=.1,error=None)
    assert ready()['at_origin'] is True
    ready.emergency.snapshot.return_value=dict(available=True,engaged=True,age_sec=.1,error=None)
    with pytest.raises(EmergencyFeedbackError,match='已按下'):ready()
    ready.arm.enable.assert_not_called()


def test_startup_packet_wait_does_not_consume_first_button(tmp_path):
    from test_button_cycle import setup
    from piper_handeye.button_cycle import CycleSettings
    from test_button_cycle_worker import ready_probe
    ready=ready_probe();ready.requires_emergency_feedback=True;ready.emergency=Mock()
    ready.emergency.snapshot.return_value=dict(available=False,error=None)
    runner,dispatch,store,clock,events=setup(tmp_path,CycleSettings('内呼',('1','2'),1,failure_policy='recover_wait'),ready)
    runner.resume();runner.tick();clock.advance(.2);runner.tick()
    assert not dispatch.commands and store.cursor==0 and not store.rows
    ready.emergency.snapshot.return_value=dict(available=True,engaged=False,age_sec=.1,error=None)
    clock.advance(1.);runner.tick();clock.advance(.6);runner.tick()
    assert dispatch.commands==[1] and store.cursor==0 and not store.rows


def test_emergency_transition_log_keeps_sample_without_repeating_every_poll(tmp_path):
    from test_button_cycle_worker import ready_probe
    ready=ready_probe();ready.store.directory=tmp_path;ready.emergency=Mock()
    ready.emergency.snapshot.return_value=dict(available=False,error=None)
    for _ in range(2):
        with pytest.raises(EmergencyFeedbackError):ready.emergency_guard()
    ready.emergency.snapshot.return_value=dict(available=True,engaged=False,age_sec=.07,error=None)
    ready.emergency_guard();ready.emergency_guard()
    rows=[json.loads(s) for s in (tmp_path/'emergency_events.jsonl').read_text().splitlines()]
    assert [r['reason_code'] for r in rows]==['feedback_unavailable','released']
    assert rows[1]['sample']['age_sec']==.07


def test_real_observer_buffers_before_press_on_even_when_reference_rejects(tmp_path):
    from piper_handeye.button_light_hardware import FreshLightObserver
    from piper_handeye.paired_button_light import PairedButtonLight
    class Vision:
        n=0
        def request(self,*args,**kwargs):
            self.n+=1;entry=observation(tmp_path,self.n)
            return dict(fresh=True,path=entry['image'],frame_sequence=self.n,
                capture_time_unix_sec=time.time(),image_size=[100,80],target_detection=dict(label='1',confidence=.95,frame_sequence=self.n,bbox_xyxy=[30,20,60,50]))
    archive=FailureImageArchive(tmp_path/'failure');archive.begin_attempt(1)
    observer=FreshLightObserver(Vision(),NS(observe=lambda *a,**kw:dict(state='ON',legacy_state='OFF')),'1',
        paired_light=PairedButtonLight(require_stationary_reference=True),failure_image_archive=archive)
    with pytest.raises(RuntimeError,match='cannot confirm OFF'):observer.capture_stationary_off_reference(timeout_sec=.05)
    for image in tmp_path.glob('frame_*.jpg'):image.unlink()
    saved=archive.finalize(success=False,context={'failure':'cannot_confirm_off'})
    assert saved['saved_frames']>0
    record=json.loads(Path(saved['record']).read_text())['records'][0]
    assert record['observation']['recognition']['state']=='ON'
    assert Path(record['crop_image']).exists()
