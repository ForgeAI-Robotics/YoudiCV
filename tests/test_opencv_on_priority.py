"""Actual observer -> flow ON priority, comparison exceptions and robot health."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import time
import numpy as np
import pytest

from piper_handeye.button_light_hardware import FreshLightObserver, DryValidatedSegmentExecutor
from piper_handeye.button_light_flow import ButtonLightFlow
from tests.test_paired_button_light import patch_image, snapshot
from tests.test_button_light_flow import FakeHardware


def observer(tmp_path, *, state='ON', comparison=None, error=None):
    data=snapshot(tmp_path,patch_image(3,increase=100))
    data['target_detection']['label']='2'
    comparison=comparison or dict(state='UNCERTAIN',reason='incomparable_background')
    pair=NS(classify=Mock(side_effect=error,return_value=comparison),reset_pending=Mock())
    raw=dict(state=state,legacy_state=state,recognizer='opencv_self_contrast')
    recognizer=NS(observe=Mock(return_value=raw))
    health=Mock(return_value=dict(healthy=True,hardware_ok=1,feedback_age_sec=.01))
    o=FreshLightObserver(NS(request=lambda *a,**kw:data),recognizer,'2',paired_light=pair,
                        post_press_health_check=health)
    return o,data,health


@pytest.mark.parametrize('reason',['incomparable_background','view_geometry_changed',
    'saturated_button','no_comparable_off_reference','image_predates_press',None])
def test_comparison_unknown_cannot_veto_real_single_image_on(tmp_path,reason):
    o,data,health=observer(tmp_path,comparison=dict(state='UNCERTAIN',reason=reason))
    assert o.observe('near')=='ON'
    r=o.audit[-1]['recognition']
    assert r['absolute_state']=='ON' and r['paired_comparison']['state']=='UNCERTAIN'
    assert r['opencv_on_priority']['comparison_reason']==reason
    assert health.called
    assert o.post_press_opencv_result('near')['source']=='opencv_immediate_on_without_paired_off'


@pytest.mark.parametrize('error',[ValueError('invalid crop pairing'),OSError('comparison image unavailable'),
                                RuntimeError('image comparison failed'),TypeError('invalid comparison result'),
                                KeyError('missing comparison metadata')])
def test_image_comparison_exception_keeps_raw_on_and_records_exception(tmp_path,error):
    o,_,_=observer(tmp_path,error=error)
    assert o.observe('near')=='ON'
    r=o.audit[-1]['recognition']
    assert r['absolute_state']=='ON'
    assert r['paired_comparison']['exception_type']==type(error).__name__
    assert r['opencv_on_priority']['comparison_exception']==type(error).__name__


@pytest.mark.parametrize('position',['near','near_confirmation','near_recheck','prepress',
                                    'prepress_confirmation','press_monitor'])
def test_postpress_positions_use_same_per_frame_rule(tmp_path,position):
    o,_,_=observer(tmp_path)
    assert o.observe(position)=='ON'
    assert o.audit[-1]['recognition']['opencv_on_priority']['state']=='ON'


@pytest.mark.parametrize('paired',['ON','OFF'])
def test_explicit_pair_on_unchanged_and_explicit_off_never_overridden(tmp_path,paired):
    o,_,health=observer(tmp_path,comparison=dict(state=paired))
    assert o.observe('near')==paired
    assert 'opencv_on_priority' not in o.audit[-1]['recognition']
    health.assert_not_called()


@pytest.mark.parametrize('state',['OFF','UNCERTAIN'])
def test_no_actual_single_image_on_cannot_become_on(tmp_path,state):
    o,_,health=observer(tmp_path,state=state)
    assert o.observe('near')!='ON'
    health.assert_not_called()


@pytest.mark.parametrize('fault',['stop','communication','unhealthy'])
def test_independent_motion_failures_propagate_outside_comparison_handler(tmp_path,fault):
    o,_,health=observer(tmp_path)
    if fault=='unhealthy':health.return_value={'healthy':False}
    else:health.side_effect=RuntimeError(fault)
    with pytest.raises(RuntimeError):o.observe('near')
    assert not o.audit


def test_no_health_callback_cannot_enable_priority(tmp_path):
    o,_,health=observer(tmp_path);o.post_press_health_check=None
    assert o.observe('near')=='UNCERTAIN'


def test_invalid_single_input_cannot_fabricate_on(tmp_path):
    o,s,health=observer(tmp_path)
    s['target_detection']['frame_sequence']=1
    assert o.observe('near',timeout_sec=.01)=='UNCERTAIN'
    o.recognizer.observe.assert_not_called();health.assert_not_called()


def test_comparison_delay_does_not_erase_previously_valid_single_on(tmp_path,monkeypatch):
    import piper_handeye.button_light_hardware as h
    o,s,health=observer(tmp_path);wall=s['capture_time_unix_sec'];clock=[wall+.01]
    monkeypatch.setattr(h.time,'time',lambda:clock[0])
    def compare(*a,**kw):clock[0]+=.4;return dict(state='UNCERTAIN',reason='late_comparison')
    o.paired_light.classify=compare
    assert o.observe('near')=='ON'
    e=o.audit[-1]['recognition']['opencv_on_priority']
    assert e['absolute_input_age_sec']<.35 and e['image_age_at_adoption_sec']>.35


@pytest.mark.parametrize('fault',['stale','hardware','nonfinite','stop'])
def test_real_independent_health_callback_uses_live_feedback_and_stop(fault):
    e=DryValidatedSegmentExecutor.__new__(DryValidatedSegmentExecutor)
    safety=Mock();e.motion_safety=safety
    if fault=='stop':safety._assert_clear.side_effect=RuntimeError('independent stop')
    q=np.zeros(6)
    if fault=='nonfinite':q[3]=float('nan')
    e.robot=NS(latest_state=lambda **kw:NS(joints=q,timestamp=time.monotonic()-(.3 if fault=='stale' else 0),
                                         raw={'hardware_ok':0 if fault=='hardware' else 1}))
    with pytest.raises(RuntimeError):e.post_press_motion_health()


def test_motion_monitor_adoption_source_is_reported_and_off_wait_remains():
    hw=FakeHardware(['OFF','OFF'],[])
    hw.move_to_near_observation=lambda _:dict(position='press_monitor',confirmed_state='ON',
        evidence=dict(opencv_adopted=True,frames=[dict(frame_sequence=10),dict(frame_sequence=11)]))
    result=ButtonLightFlow(hw).run()
    assert result.status=='arrived_after_verified_press' and result.press_attempts==1
    assert next(e for e in result.events if e['event']=='press_verified_on')['confirmation_source']=='opencv_on_priority_two_frame_motion_monitor'
    assert any(e['event']=='post_press_opencv_adopted' for e in result.events)
    assert next(e for e in result.events if e['event']=='button_became_off')['frames']==3
