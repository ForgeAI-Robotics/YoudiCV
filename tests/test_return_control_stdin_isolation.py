"""No robot/network: child processes cannot steal the GUI control channel."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from run_voice_command_gateway import CommandDispatcher
from run_direct_return_to_origin import _run_orangepi_command
from piper_handeye.voice_command import VoiceCommand, VoiceIntent
from tools.button_cycle_worker import control_shutdown_reason


@pytest.mark.parametrize('operation', ['return', 'check', 'press'])
def test_dispatcher_children_have_no_gui_input_pipe(tmp_path, monkeypatch, operation):
    monkeypatch.setenv('YOUXIAOMEI_ROBOT_MOTION_LOCK', str(tmp_path/'lock'))
    monkeypatch.setenv('YOUXIAOMEI_MISSION_STATUS_OUTBOX', str(tmp_path/'outbox'))
    dispatcher = CommandDispatcher(execute=True, config_path=tmp_path/'config', hold_report=tmp_path/'hold')
    launch = Mock(return_value=SimpleNamespace(pid=123, poll=lambda: None))
    monkeypatch.setattr('run_voice_command_gateway.subprocess.Popen', launch)
    try:
        if operation == 'press':
            dispatcher.dispatch_motion(VoiceCommand(VoiceIntent.PRESS_BUTTON, '4', '4', button=4))
        else:
            dispatcher.dispatch_origin_operation(operation)
        assert launch.call_args.kwargs['stdin'] == subprocess.DEVNULL
        assert launch.call_args.kwargs['start_new_session'] is True
    finally:
        dispatcher._release_motion_lock()


def test_ssh_wrapper_does_not_receive_parent_stdin(monkeypatch):
    run = Mock(return_value=SimpleNamespace(stdout='ok'))
    monkeypatch.setattr('run_direct_return_to_origin.subprocess.run', run)
    assert _run_orangepi_command('true') == 'ok'
    assert run.call_args.kwargs['stdin'] == subprocess.DEVNULL
    assert run.call_args.kwargs['timeout'] == 4.


@pytest.mark.parametrize('interrupted,disconnected,error,age,expected', [
    (False, False, None, 2.99, None),
    (False, False, None, 3., None),
    (False, False, None, 3.01, 'GUI心跳超时'),
    (False, True, None, .01, 'GUI控制通道已关闭'),
    (False, True, 'ValueError', .01, 'GUI控制通道读取异常（ValueError）'),
    (True, False, None, .01, '收到退出信号'),
])
def test_disconnect_causes_are_distinguished_without_relaxing_watchdog(interrupted,disconnected,error,age,expected):
    result=control_shutdown_reason(interrupted=interrupted,disconnected=disconnected,
                                   reader_error=error,heartbeat_age=age)
    assert result is None if expected is None else expected in result


# An SSH-like subprocess toggles its standard-input file flags and consumes it.
# With inherited stdin this reproduces the parent's false EOF; with the real
# production helper it touches only /dev/null. No SSH or hardware is called.
WORKER = r'''
import sys,threading,subprocess,time,json,fcntl,os
from run_direct_return_to_origin import _run_orangepi_command
import run_direct_return_to_origin as module
ended=threading.Event();seen=[]
def receive():
    while True:
        line=sys.stdin.readline()
        if not line:ended.set();return
        seen.append(line.strip())
thread=threading.Thread(target=receive,daemon=True);thread.start()
print('ready',flush=True)
time.sleep(.05)
flags=fcntl.fcntl(0,fcntl.F_GETFL)
ssh_like="import os,time,sys;from pathlib import Path;os.set_blocking(0,False);Path(sys.argv[1]).write_text('ready');time.sleep(.24)"
if sys.argv[1]=='production':
    module.ORANGEPI_SSH=(sys.executable,'-c',ssh_like)
    _run_orangepi_command(sys.argv[2],timeout_sec=3.)
else:
    subprocess.run([sys.executable,'-c',ssh_like,sys.argv[2]],check=True,timeout=3.)
time.sleep(.06)
print(json.dumps(dict(reader_ended=ended.is_set(),received=len(seen),
    input_flags_unchanged=flags==fcntl.fcntl(0,fcntl.F_GETFL))),flush=True)
'''


@pytest.mark.parametrize('mode', ['production', 'baseline'])
def test_real_subprocess_preserves_heartbeat_reader_and_reproduces_old_failure(mode, tmp_path):
    marker=tmp_path/'ssh_input_flags_ready'
    process=subprocess.Popen([sys.executable,'-c',WORKER,mode,str(marker)],stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                             cwd=Path(__file__).resolve().parents[1])
    try:
        assert process.stdout.readline().strip()=='ready'
        # Explicit child acknowledgement avoids a scheduler-dependent sleep:
        # only feed after the child has actually changed its stdin flags.
        import time
        deadline=time.monotonic()+3.
        while not marker.exists() and time.monotonic()<deadline:
            time.sleep(.005)
        assert marker.exists(), 'SSH-like child did not acknowledge its input setup' 
        process.stdin.write('heartbeat\n');process.stdin.flush()
        line=process.stdout.readline()
        result=json.loads(line)
        assert process.wait(timeout=4.)==0
        if mode=='production':
            assert result['reader_ended'] is False
            assert result['input_flags_unchanged'] is True
            assert result['received']>=1
        else:
            assert result['reader_ended'] is True
    finally:
        if process.poll() is None:process.kill();process.wait()
        process.stdin.close();process.stdout.close();process.stderr.close()
