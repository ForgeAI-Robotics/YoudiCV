"""Offline checks: reduce redundant waiting without changing motion or stop gates."""
from pathlib import Path
from unittest.mock import MagicMock,patch
import pytest
from run_direct_return_to_origin import prepare_return_mode
from run_voice_command_gateway import CommandDispatcher
from tools.bag_origin_worker import OriginTask
from piper_handeye.bag_task_timing import timed_stage


def test_bag_return_delegates_mode_handoff_without_running_ssh():
    with patch('run_direct_return_to_origin.set_can_joint_mode_while_holding',side_effect=AssertionError('duplicate SSH')) as old:
        audit=prepare_return_mode(MagicMock(),{'status':'ok','busy':False,'safety_handoff_v2':True},bridge_handoff=True)
    assert audit['delegated_to_bridge'] is True
    assert audit['local_ssh_mode_switch'] is False
    assert audit['control_mode']=='bridge_handoff_pending_execution'
    old.assert_not_called()


@pytest.mark.parametrize('health',[{}, {'status':'failed','safety_handoff_v2':True},
    {'status':'ok','busy':True,'safety_handoff_v2':True}, {'status':'ok','busy':False,'safety_handoff_v2':False}])
def test_bridge_handoff_refuses_unknown_busy_or_unsupported_bridge(health):
    with patch('run_direct_return_to_origin.set_can_joint_mode_while_holding') as old:
        with pytest.raises(RuntimeError):prepare_return_mode(MagicMock(),health,bridge_handoff=True)
    old.assert_not_called()  # No retry through another motion route.


def test_existing_voice_return_retains_original_mode_handoff_by_default():
    live=MagicMock()
    with patch('run_direct_return_to_origin.set_can_joint_mode_while_holding',return_value={'legacy':True}) as old:
        assert prepare_return_mode(live,{})=={'legacy':True}
    old.assert_called_once_with(live)


def test_opt_in_return_flag_preserves_the_rest_of_production_command(tmp_path):
    commands=[]
    for enabled in (False,True):
        d=CommandDispatcher(execute=True,config_path=tmp_path/'cfg.yaml',hold_report=tmp_path/'hold.yaml',
                            origin_return_bridge_handoff=enabled)
        d._acquire_motion_lock=MagicMock()
        with patch('run_voice_command_gateway.subprocess.Popen') as popen:
            d._start_origin_operation('return')
        commands.append(popen.call_args.args[0])
    assert '--bridge-mode-handoff' not in commands[0]
    assert commands[1]==commands[0]+['--bridge-mode-handoff']
    assert '--speed-m-s' not in commands[1] and '--joint-velocity-scale' not in commands[1]


def test_only_bag_adapter_enables_bridge_handoff():
    with patch('tools.bag_origin_worker.CommandDispatcher') as cls:
        OriginTask(MagicMock())
    assert cls.call_args.kwargs['origin_return_bridge_handoff'] is True


def test_stage_timing_records_failure_without_masking_or_retrying_it():
    report={}
    with pytest.raises(TimeoutError):
        with timed_stage(report,'motion',planned_duration_sec=3):
            raise TimeoutError('execution unknown')
    assert len(report['timings'])==1
    timing=report['timings'][0]
    assert timing['outcome']=='failed' and timing['duration_sec']>=0
    assert timing['planned_duration_sec']==3
