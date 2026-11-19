"""Requested post-press adoption, real observer gates, reporting and retry limits."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
from copy import deepcopy
import time

import pytest

from piper_handeye.button_light_flow import ButtonLightFlow
from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.button_light_runtime import ButtonLightRuntime
from piper_handeye.button_cycle import TaskEvidence, summarize, CycleStore, CycleSettings
from piper_handeye.arm_report import execution_evidence
from piper_handeye.mission_status import normalize_flow_event, describe_phase_zh
from tests.test_button_light_flow import FakeHardware


class AdoptingHardware(FakeHardware):
    def __init__(self, waiting, near, absolute=None, recovery=None):
        super().__init__(waiting, near)
        self.absolute = absolute
        self.recovery = recovery
    def post_press_opencv_result(self, position):
        self.actions.append(('opencv', position))
        return self.absolute
    def reobserve_post_press(self, position):
        self.actions.append(('recheck', position))
        return self.recovery


@pytest.mark.parametrize('near', [['UNCERTAIN']*3, ['OFF']+['UNCERTAIN']*3])
def test_first_unknown_then_opencv_on_moves_to_off_wait_without_another_press(near):
    hw = AdoptingHardware(['OFF','OFF'], near, absolute={'state':'ON','frame_sequence':21})
    flow = ButtonLightFlow(hw);result=flow.run()
    assert result.status == 'arrived_after_verified_press' and result.press_attempts == 1
    assert hw.actions.count(('press',1)) == 1
    assert not any(a[0]=='recheck' for a in hw.actions)
    adopted=next(e for e in result.events if e['event']=='post_press_opencv_adopted')
    assert adopted['state']=='ON' and adopted['evidence']['uncertain_frames']==1
    on=next(e for e in result.events if e['event']=='press_verified_on')
    assert on['confirmation_source']=='opencv_immediate_on_without_paired_off'
    off=next(e for e in result.events if e['event']=='button_became_off')
    assert off['frames']==3
    assert hw.actions[-1]==('origin',)


@pytest.mark.parametrize('absolute', [None, {'state':'UNCERTAIN'}, {'state':'OFF'}])
def test_no_valid_absolute_on_keeps_bounded_recheck_then_unknown_failure(absolute):
    hw=AdoptingHardware(['OFF'],['UNCERTAIN']*3,absolute=absolute)
    result=ButtonLightFlow(hw).run()
    assert result.status=='post_press_confirmation_uncertain'
    assert result.press_attempts==1 and hw.actions[-1]==('origin',)
    assert ('recheck','near') in hw.actions
    assert not any(e['event']=='post_press_opencv_adopted' for e in result.events)


@pytest.mark.parametrize('recovery', [
    {'confirmed_state':'ON','source':'single_paired_on_stationary_recheck_v2','frames':[{'frame_sequence':22}]},
    {'confirmed_state':'ON','source':'opencv_after_bounded_stationary_recheck_v1','opencv_adopted':True},
])
def test_recheck_single_paired_or_opencv_on_proceeds_to_normal_off_and_origin(recovery):
    hw=AdoptingHardware(['OFF','OFF'],['UNCERTAIN']*3,recovery=recovery)
    result=ButtonLightFlow(hw).run()
    assert result.status=='arrived_after_verified_press' and result.press_attempts==1
    assert any(e['event']=='post_press_opencv_adopted' for e in result.events)==bool(recovery.get('opencv_adopted'))


def test_recheck_opencv_off_follows_original_refresh_and_retry_path():
    recovery={'confirmed_state':'OFF','source':'opencv_after_bounded_stationary_recheck_v1','opencv_adopted':True}
    hw=AdoptingHardware(['OFF','OFF'],['UNCERTAIN']*3+['ON','ON'],recovery=recovery)
    result=ButtonLightFlow(hw).run()
    assert result.status=='arrived_after_verified_press' and result.press_attempts==2
    assert hw.actions.count(('refresh',))==2
    assert any(e['event']=='post_press_opencv_adopted' and e['state']=='OFF' for e in result.events)
    assert hw.actions.count(('waiting',))==1


def test_opencv_off_adoption_never_exceeds_three_press_attempts():
    hw=AdoptingHardware(['OFF'],['UNCERTAIN']*9,recovery={
        'confirmed_state':'OFF','opencv_adopted':True,'source':'opencv_after_bounded_stationary_recheck_v1'})
    r=ButtonLightFlow(hw).run()
    assert r.status=='button_not_lit_after_max_attempts' and r.press_attempts==3
    assert hw.actions[-1]==('origin',)


def test_initial_unknown_cannot_use_post_press_opencv():
    hw=AdoptingHardware(['UNCERTAIN']*3,[],absolute={'state':'ON'})
    r=ButtonLightFlow(hw).run()
    assert r.status=='initial_observation_uncertain' and r.press_attempts==0
    assert not any(a[0]=='opencv' for a in hw.actions)


def test_failed_physical_press_does_not_use_visual_adoption():
    hw=AdoptingHardware(['OFF'],['UNCERTAIN']*3,absolute={'state':'ON'})
    hw.press_button=Mock(side_effect=RuntimeError('independent stop active'))
    with pytest.raises(RuntimeError,match='independent stop'):
        ButtonLightFlow(hw).run()
    assert not any(a[0]=='opencv' for a in hw.actions)


def test_adopted_on_still_requires_three_fresh_off_and_successful_return():
    hw=AdoptingHardware(['OFF','UNCERTAIN','OFF'],['UNCERTAIN']*3,absolute={'state':'ON'})
    hw.move_to_origin=Mock(side_effect=RuntimeError('origin failed'))
    flow=ButtonLightFlow(hw)
    with pytest.raises(RuntimeError,match='origin failed'):flow.run()
    assert next(e for e in flow.events if e['event']=='button_became_off')['frames']==3


def observer_entry(state='ON', position='near', sequence=21):
    return {'position':position,'frame_sequence':sequence,'capture_time_unix_sec':time.time()-.01,
        'state':'UNCERTAIN','image':'/tmp/existing-same-frame.jpg',
        'target_region':{'label':'2','frame_sequence':sequence,'bbox_xyxy':[1,1,80,80]},
        'visibility':{'visible':True},'recognition':{
            'recognizer':'opencv_self_contrast','absolute_state':state,
            'paired_comparison':{'state':'UNCERTAIN','reason':'nonlocal_rise_for_panel_reference','reference_count':2}}}


def observer():
    pair=NS(frozen=True, require_stationary_reference=True,stationary_reference_ready=True,
            stationary_check=Mock(return_value=True),last_reference_sequence=10,
            press_started=time.time()-.5)
    o=FreshLightObserver(Mock(),Mock(),'2',paired_light=pair)
    o.audit=[observer_entry()];o.last_sequence=21
    return o


@pytest.mark.parametrize('state',['ON','OFF'])
def test_observer_returns_actual_same_frame_absolute_state_and_uncertainty_reason(state):
    o=observer();o.audit[-1]['recognition']['absolute_state']=state
    r=o.post_press_opencv_result('near')
    assert r['state']==state and r['frame_sequence']==21
    assert r['comparison_state']=='UNCERTAIN'
    assert r['stationary_observation_verified'] is True


@pytest.mark.parametrize('fault',['wrong_position','hidden','mismatch','stale','predates_press',
    'unknown_absolute','wrong_recognizer','no_reference','missing_reference','view_changed',
    'saturated','bad_pose','unfrozen','not_stationary','old_audit'])
def test_invalid_images_geometry_or_pose_do_not_become_opencv_success(fault):
    o=observer();e=o.audit[-1];p=e['recognition']['paired_comparison']
    if fault=='wrong_position':e['position']='press_monitor'
    if fault=='hidden':e['visibility']['visible']=False
    if fault=='mismatch':e['target_region']['frame_sequence']=20
    if fault=='stale':e['capture_time_unix_sec']=time.time()-.36
    if fault=='predates_press':e['capture_time_unix_sec']=o.paired_light.press_started-.01
    if fault=='unknown_absolute':e['recognition']['absolute_state']='UNCERTAIN'
    if fault=='wrong_recognizer':e['recognition']['recognizer']='skipped_unsafe_target_crop'
    if fault=='no_reference':p['reference_count']=0
    if fault=='missing_reference':p['reason']='no_comparable_off_reference'
    if fault=='view_changed':p['reason']='view_geometry_changed'
    if fault=='saturated':p['reason']='saturated_button'
    if fault=='bad_pose':o.paired_light.stationary_check.return_value=False
    if fault=='unfrozen':o.paired_light.frozen=False
    if fault=='not_stationary':o.paired_light.require_stationary_reference=False
    if fault=='old_audit':o.last_sequence=22
    assert o.post_press_opencv_result('near') is None


def test_new_invalid_frame_cannot_reuse_previous_opencv_on():
    o=observer();o.audit.append({**observer_entry(sequence=22),'visibility':{'visible':False}});o.last_sequence=22
    assert o.post_press_opencv_result('near') is None


def test_encoder_exception_propagates_instead_of_visual_fallback():
    o=observer();o.paired_light.stationary_check.side_effect=RuntimeError('feedback unavailable')
    with pytest.raises(RuntimeError,match='feedback unavailable'):o.post_press_opencv_result('near')


def test_encoder_check_delay_cannot_adopt_an_image_that_became_stale(monkeypatch):
    o=observer();now=o.audit[-1]['capture_time_unix_sec']+.01;clock=[now]
    monkeypatch.setattr('piper_handeye.button_light_hardware.time.time',lambda:clock[0])
    def check():clock[0]+=.4;return True
    o.paired_light.stationary_check=check
    assert o.post_press_opencv_result('near') is None


@pytest.mark.parametrize('state',['ON','OFF'])
def test_recheck_budget_exhaustion_uses_latest_valid_opencv_state(state):
    o=observer();calls=[]
    def take(position,**kwargs):
        calls.append(position);seq=21+len(calls);o.audit.append(observer_entry(state,position,seq));o.last_sequence=seq
        return 'UNCERTAIN'
    o.observe=take
    r=o.reobserve_post_press('near')
    assert r['confirmed_state']==state and r['opencv_adopted'] is True
    assert r['samples']==(1 if state=='ON' else 6)
    assert len(calls)==r['samples']
    assert r['evidence']['frame_sequence']==(22 if state=='ON' else 27)


def test_recheck_one_qualified_on_is_enough_and_does_not_request_second_frame():
    o=observer();o.observe=Mock(return_value='ON')
    r=o.reobserve_post_press('near')
    assert r['confirmed_state']=='ON' and r['samples']==1
    assert len(r['frames'])==1 and o.observe.call_count==1
    assert not r.get('opencv_adopted')


@pytest.mark.parametrize('fault', ['stale_after_processing', 'bad_pose', 'missing_pose_check'])
def test_single_paired_on_still_requires_fresh_frame_and_current_fixed_pose(fault):
    o=observer();o.observe=Mock(return_value='ON')
    if fault=='stale_after_processing':o.audit[-1]['capture_time_unix_sec']=time.time()-.36
    if fault=='bad_pose':o.paired_light.stationary_check.return_value=False
    if fault=='missing_pose_check':o.paired_light.stationary_check=None
    assert o.reobserve_post_press('near')['confirmed_state'] is None


def test_recheck_missing_latest_valid_result_does_not_fabricate_on():
    o=observer();o.observe=Mock(return_value='UNCERTAIN');o.audit[-1]['position']='near_recheck'
    o.audit[-1]['visibility']['visible']=False
    assert o.reobserve_post_press('near')['confirmed_state'] is None


def test_runtime_routes_optional_absolute_evidence_without_changing_observation():
    route=Mock(return_value={'state':'ON'})
    rt=ButtonLightRuntime(observe=lambda p:'UNCERTAIN',next_frame=lambda:None,
        refresh_target=lambda:None,press=lambda a:None,move_near=lambda d:None,
        move_waiting=lambda:None,move_origin=lambda:None,post_press_opencv=route)
    assert rt.observe_light('near').value=='UNCERTAIN'
    assert rt.post_press_opencv_result('near')=={'state':'ON'}
    route.assert_called_once_with('near')


def test_adoption_is_audited_separately_in_terminal_cycle_row_and_csv(tmp_path):
    flow_events=[{'event':'post_press_opencv_adopted','state':'ON','attempt':1},
                 {'event':'press_verified_on','attempt':1,'confirmation_source':'opencv_after_three_uncertain'},
                 {'event':'button_became_off','frames':3}]
    report={'events':flow_events,'press_execution_started':True,'press_attempts':1,
        'flow_status':'arrived_after_verified_press','returned_to_origin':True,
        'origin_verification':{'verified':True,'hardware_ok':1},'process_exit_code':0}
    terminal=execution_evidence(report)
    assert terminal['opencv_fallback_used'] is True
    assert terminal['on_confirmation_source']=='opencv_after_three_uncertain'
    ev=TaskEvidence('task')
    for f in flow_events:
        ev.add({'mission_status_events':[{'task_id':'task','producer':'arm_pressing',
             'emitted_monotonic_sec':2.,**normalize_flow_event(f)}]})
    ev.terminal=terminal;row=ev.result(0,started=1.,finished=4.)
    assert row['full_cycle_success'] and row['opencv_fallback_used']
    assert '按OpenCV采用ON' in row['reason']
    store=CycleStore.create(tmp_path/'group',CycleSettings('内呼',('2',),1),{})
    store.append({'execution_id':'e','index':0,'round':1,'button':'2',**row})
    text=(store.directory/'results.csv').read_text(encoding='utf-8-sig')
    assert '采用OpenCV后备' in text and 'opencv_after_three_uncertain' in text
    assert summarize(store.rows)['opencv_fallback_items']==1
    assert summarize([{'counted':True}])['opencv_fallback_items']==0


def test_gui_status_does_not_call_adopted_on_a_paired_confirmation():
    adopted=normalize_flow_event({'event':'post_press_opencv_adopted','state':'ON'})
    on=normalize_flow_event({'event':'press_verified_on','confirmation_source':'opencv_after_three_uncertain'})
    assert adopted['phase']=='post_press_opencv_adopted'
    assert on['phase']=='button_on_adopted'
    assert 'OpenCV' in describe_phase_zh(on)
