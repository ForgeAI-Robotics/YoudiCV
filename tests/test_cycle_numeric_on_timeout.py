"""Virtual time, fake hardware and isolated cycle stores only."""
import ast
from itertools import repeat
from pathlib import Path
from unittest.mock import Mock

import pytest

from piper_handeye.button_light_flow import (
    ButtonLightFlow, ButtonFlowConfig, production_flow_config,
    NUMERIC_CYCLE_ON_TIMEOUT_FLOW, NUMERIC_CYCLE_INITIAL_ON_TIMEOUT_FLOW)
from piper_handeye.button_cycle import TaskEvidence, CycleSettings, summarize
from piper_handeye.arm_report import terminal_event
from piper_handeye.mission_status import normalize_flow_event
from run_button_light_closed_loop import flow_status_exit_code, retain_button_run_images
from test_external_call_off_timeout import Clock, Hardware
from test_button_cycle import setup, begin


def simulate(monkeypatch, *, initial='OFF', after='ON', fail=None, cycle=True, label='4'):
    clock=Clock()
    monkeypatch.setattr('piper_handeye.button_light_flow.time.monotonic', clock.monotonic)
    hardware=Hardware(clock,initial=initial,after=after,fail=fail)
    marks=[]
    result=ButtonLightFlow(hardware, production_flow_config(label,{},cycle_numeric_wait=cycle),
        event_sink=lambda _,e:marks.append((clock.now,dict(e)))).run()
    return result,hardware,marks


@pytest.mark.parametrize('label',[str(i) for i in range(1,17)])
def test_only_explicit_numeric_cycles_enable_ten_second_rule(label):
    assert production_flow_config(label,{},cycle_numeric_wait=True).cycle_numeric_on_timeout_sec==10.
    assert production_flow_config(label,{}).cycle_numeric_on_timeout_sec is None


@pytest.mark.parametrize('label',['up','down','open','17','0'])
def test_call_and_other_targets_never_get_numeric_timeout(label):
    assert production_flow_config(label,{},cycle_numeric_wait=True).cycle_numeric_on_timeout_sec is None


@pytest.mark.parametrize('seconds',[0,9,11,True,float('nan'),float('inf')])
def test_timeout_policy_rejects_invalid_values(seconds):
    with pytest.raises(ValueError):ButtonFlowConfig(target_label='4',cycle_numeric_on_timeout_sec=seconds)


def test_after_press_on_ten_seconds_returns_without_claiming_off(monkeypatch):
    result,h,marks=simulate(monkeypatch)
    assert result.status==NUMERIC_CYCLE_ON_TIMEOUT_FLOW and h.pressed==1 and h.returned
    timeout=next(e for _,e in marks if e['event']=='numeric_cycle_on_wait_timeout')
    assert timeout['elapsed_continuous_on_sec']==10.
    assert timeout['after_verified_press'] is True and timeout['last_observed_state']=='ON'
    assert timeout['off_confirmed'] is False
    assert not any(e['event']=='button_became_off' for _,e in marks)
    assert flow_status_exit_code(result.status)==0


def test_next_round_initially_on_is_consumed_without_extra_press(monkeypatch):
    result,h,marks=simulate(monkeypatch,initial='ON')
    assert result.status==NUMERIC_CYCLE_INITIAL_ON_TIMEOUT_FLOW
    assert h.pressed==0 and h.returned
    assert next(e for _,e in marks if e['event']=='numeric_cycle_on_wait_timeout')['after_verified_press'] is False


@pytest.mark.parametrize('break_state',['UNCERTAIN','OFF'])
def test_uncertain_or_off_breaks_continuous_on_timer(monkeypatch,break_state):
    result,_,marks=simulate(monkeypatch,after=['ON']*8+[break_state]+['ON']*11)
    observed_break=next(t for t,e in marks if e['event']=='light_observed' and e['position']=='waiting' and e['state']==break_state)
    timeout_time,timeout=next((t,e) for t,e in marks if e['event']=='numeric_cycle_on_wait_timeout')
    assert timeout_time-observed_break>=11 and timeout['elapsed_continuous_on_sec']==10.
    assert result.status==NUMERIC_CYCLE_ON_TIMEOUT_FLOW


def test_no_on_timeout_during_uncertain_observation(monkeypatch):
    result,_,marks=simulate(monkeypatch,after=['UNCERTAIN']*35+['OFF']*3)
    assert result.status=='arrived_after_verified_press'
    assert not any(e['event']=='numeric_cycle_on_wait_timeout' for _,e in marks)


def test_normal_measured_off_is_unchanged(monkeypatch):
    result,h,marks=simulate(monkeypatch,after=['ON']*3+['OFF']*3)
    assert result.status=='arrived_after_verified_press' and h.returned
    assert any(e['event']=='button_became_off' for _,e in marks)


def test_normal_numeric_mission_waits_longer_than_ten_seconds(monkeypatch):
    result,_,marks=simulate(monkeypatch,cycle=False,after=['ON']*35+['OFF']*3)
    assert result.status=='arrived_after_verified_press'
    assert not any(e['event']=='numeric_cycle_on_wait_timeout' for _,e in marks)


@pytest.mark.parametrize('failure',['origin','observation'])
def test_timeout_never_swallows_robot_or_return_failure(monkeypatch,failure):
    with pytest.raises(RuntimeError):simulate(monkeypatch,fail=failure)


def completed_timeout(tid,*,initial=False,origin=True,rc=0,hard=False,elapsed=10.):
    audit=dict(target_label='4',timeout_sec=10.,elapsed_continuous_on_sec=elapsed,
               last_observed_state='ON',off_confirmed=False,after_verified_press=not initial,
               release_policy='cycle_numeric_on_timeout_skip')
    flow=NUMERIC_CYCLE_INITIAL_ON_TIMEOUT_FLOW if initial else NUMERIC_CYCLE_ON_TIMEOUT_FLOW
    report=dict(task_id=tid,target_label='4',flow_status=flow,process_exit_code=rc,
        returned_to_origin=origin,origin_verification=dict(verified=origin,hardware_ok=1),
        press_execution_started=not initial,press_attempts=0 if initial else 1,
        numeric_cycle_on_wait_timeout=audit)
    events=[dict(task_id=tid,producer='arm_pressing',raw_event='button_already_on' if initial else 'press_verified_on',emitted_monotonic_sec=101.),
        dict(normalize_flow_event(dict(event='numeric_cycle_on_wait_timeout',**audit)),task_id=tid,producer='arm_pressing'),
        terminal_event(report)]
    if hard:events.insert(0,dict(task_id=tid,producer='arm_pressing',raw_event='guarded_press_failed',phase='press_failed'))
    return dict(event='motion_completed',task_id=tid,return_code=rc,mission_status_events=events,arm_report=events[-1]['evidence'])


@pytest.mark.parametrize('initial',[False,True])
def test_timeout_evidence_counts_truthfully_and_is_degraded(initial):
    e=completed_timeout('t',initial=initial)
    t=TaskEvidence('t');t.add(e);r=t.result(0,started=100.,finished=120.)
    assert r['on_timeout_skipped'] and r['safe_to_continue'] and r['counted']
    assert r['on_confirmed'] is (not initial)
    assert not r['off_confirmed'] and not r['full_cycle_success']
    assert r['press_attempts']==(0 if initial else 1)
    assert e['mission_status_events'][-1]['outcome']=='degraded'
    summary=summarize([r]);assert summary['on_timeout_skips']==1
    assert summary['full_cycle_success']==0 and summary['lit_success']==(0 if initial else 1)


@pytest.mark.parametrize('changes',[dict(origin=False),dict(rc=1),dict(hard=True),dict(elapsed=9.)])
def test_invalid_timeout_evidence_cannot_authorize_next_button(changes):
    e=completed_timeout('t',**changes);t=TaskEvidence('t');t.add(e)
    r=t.result(e['return_code'],started=100.,finished=120.)
    assert not r['on_timeout_skipped'] and not r['safe_to_continue']


@pytest.mark.parametrize('policy',['pause','continue_safe','recover_wait'])
@pytest.mark.parametrize('initial',[False,True])
def test_timeout_continues_next_unexecuted_button_after_verified_return(tmp_path,policy,initial):
    runner,d,store,clock,_=setup(tmp_path,CycleSettings('内呼',('4','5'),1,failure_policy=policy))
    begin(runner,clock);d.events.append(completed_timeout('task1',initial=initial));runner.tick()
    assert d.commands==[4,5] and store.cursor==1
    assert store.rows[0]['on_timeout_skipped'] and not store.rows[0]['full_cycle_success']
    assert runner.auto_recovery is None


def test_return_failure_still_prevents_next_button(tmp_path):
    runner,d,store,clock,_=setup(tmp_path)
    begin(runner,clock);d.events.append(completed_timeout('task1',origin=False));runner.tick()
    assert d.commands==[1] and runner.state=='paused'


def test_timeout_at_last_item_finishes_group(tmp_path):
    runner,d,store,clock,_=setup(tmp_path,CycleSettings('内呼',('4',),1,failure_policy='recover_wait'))
    begin(runner,clock);d.events.append(completed_timeout('task1'));runner.tick()
    assert runner.state=='completed' and store.cursor==1 and not d.session_reserved


@pytest.mark.parametrize('status,success',[(NUMERIC_CYCLE_ON_TIMEOUT_FLOW,False),(NUMERIC_CYCLE_INITIAL_ON_TIMEOUT_FLOW,False),('arrived_after_verified_press',True)])
def test_timeout_images_are_archived_after_cleanup(tmp_path,status,success):
    archive=Mock();archive.finalize.return_value={}
    (tmp_path/'image.png').write_bytes(b'run owned image')
    report=dict(process_exit_code=0,flow_status=status)
    retain_button_run_images(report,archive,tmp_path,'t','4')
    assert archive.finalize.call_args.kwargs['success'] is success
    assert (tmp_path/'image.png').exists() is (not success)


def test_cycle_context_is_set_only_by_explicit_test_worker():
    source=(Path(__file__).resolve().parents[1]/'tools/button_cycle_worker.py').read_text()
    tree=ast.parse(source)
    matches=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
             and any(isinstance(n,ast.Constant) and n.value=='YOUXIAOMEI_CYCLE_NUMERIC_ON_TIMEOUT' for n in ast.walk(node))]
    assert len(matches)==1
    expression=ast.unparse(matches[0].value)
    assert 'store.settings.group' in expression and "'内呼'" in expression


def test_failure_archive_is_wired_into_main_final_cleanup():
    source=(Path(__file__).resolve().parents[1]/'run_button_light_closed_loop.py').read_text()
    main=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    finalizers=[node.finalbody for node in ast.walk(main) if isinstance(node,ast.Try) and node.finalbody]
    matched=False
    for body in finalizers:
        calls=[n.func.id for statement in body for n in ast.walk(statement)
               if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        if 'retain_button_run_images' in calls:
            assert calls.index('collect_cleanup_errors')<calls.index('retain_button_run_images')
            matched=True
    assert matched
