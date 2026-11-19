"""Production dispatcher lock and read-only startup/telemetry boundaries."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import struct

import pytest

from run_voice_command_gateway import CommandDispatcher
from piper_handeye.voice_command import VoiceCommand, VoiceIntent
from tools.button_cycle_worker import (
    Readiness, SessionDispatcher, check_recorded_files, control_timestamp, decode_full_odometry, require_stationary, stop_before_exit,
)


def test_whole_session_retains_original_global_lock_across_child_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("YOUXIAOMEI_ROBOT_MOTION_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("YOUXIAOMEI_MISSION_STATUS_OUTBOX", str(tmp_path / "outbox"))
    dispatcher = SessionDispatcher(execute=True, config_path=tmp_path / "cfg", hold_report=tmp_path / "hold")
    other = CommandDispatcher(execute=True, config_path=tmp_path / "cfg", hold_report=tmp_path / "hold")
    launched = Mock(return_value=SimpleNamespace(pid=123, poll=lambda: 0))
    monkeypatch.setattr("run_voice_command_gateway.subprocess.Popen", launched)
    dispatcher.reserve_session()
    dispatcher.dispatch_motion(VoiceCommand(VoiceIntent.PRESS_BUTTON, "4", "4", button=4))
    dispatcher.refresh()
    assert dispatcher.active is None and dispatcher.session_reserved
    with pytest.raises(RuntimeError, match="占用"):
        other._acquire_motion_lock()
    argv = launched.call_args.args[0]
    assert Path(argv[1]).name == "run_button_batch.py" and argv[2] == "4"
    dispatcher.release_session()
    other._acquire_motion_lock()
    other._release_motion_lock()


@pytest.mark.parametrize("change", [dict(available=False), dict(error="lost"), dict(age_sec=.501),
    dict(age_sec=float('nan')), dict(velocity=[0., .01, 0., 0., 0., 0.]),
    dict(velocity=[0., 0., 0., 0., 0., .021]), dict(velocity=[float('nan')]*6),
    dict(velocity=[0.]*5)])
def test_stale_or_moving_chassis_rejects_cycle_start(change):
    sample = dict(available=True, error=None, age_sec=.01, velocity=[0.]*6)
    sample.update(change)
    with pytest.raises(RuntimeError): require_stationary(sample)


def test_odometry_decoder_preserves_sideways_and_roll_pitch_speeds():
    # header seq/time, empty frame and child strings, pose and covariance.
    prefix = bytes(12) + struct.pack('<I', 0) + struct.pack('<I', 0) + bytes((7+36)*8)
    velocity = [.001, .002, .003, .004, .005, .006]
    assert decode_full_odometry(prefix + struct.pack('<6d', *velocity))["velocity"] == velocity


def test_recorded_config_change_refuses_resume(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("original")
    metadata = dict(sha256={str(config): hashlib.sha256(config.read_bytes()).hexdigest()})
    check_recorded_files(metadata)
    config.write_text("changed")
    with pytest.raises(RuntimeError, match="新建"):
        check_recorded_files(metadata)


def ready_probe():
    probe = Readiness.__new__(Readiness)
    probe.dispatcher = SimpleNamespace(origin_return_bridge_handoff=False)
    probe.requires_emergency_feedback = False
    probe.odom = Mock()
    probe.odom.snapshot.return_value = dict(available=True, error=None, age_sec=.01, velocity=[0.]*6)
    probe.store = SimpleNamespace(plan=dict(metadata={}))
    probe.origin = [0.]*6
    probe.arm = Mock()
    probe.arm.status.return_value = dict(status=dict(active=False, other_motion_active=False,
        gripper_command_active=False, fault_reason=None, actual_joints_rad=[0.]*6, joint_feedback_age_sec=.01))
    probe.vision = Mock()
    probe.vision.request.return_value = dict(status="ready", camera_sequence=1)
    probe.stopper = Mock()
    probe.stopper.health.return_value = dict(status="ok", latched=False)
    probe.stop_cfg = dict(token="test-only")
    return probe


@pytest.mark.parametrize("change", [dict(active=True), dict(other_motion_active=True),
    dict(gripper_command_active=True), dict(fault_reason="failed"), dict(joint_feedback_age_sec=.251),
    dict(joint_feedback_age_sec=float('nan')), dict(actual_joints_rad=[.031]*6),
    dict(actual_joints_rad=[float('nan')]*6), dict(actual_joints_rad=[0.]*5)])
def test_readiness_only_reads_and_rejects_busy_stale_or_wrong_origin(change):
    probe = ready_probe()
    probe.arm.status.return_value['status'].update(change)
    with pytest.raises(RuntimeError): probe()
    probe.arm.enable.assert_not_called()
    probe.arm.disable.assert_not_called()
    probe.stopper.reset_latch.assert_not_called()


def test_ready_probe_never_enables_or_clears_stop():
    probe = ready_probe()
    assert probe()["at_origin"]
    assert [call.args[0] for call in probe.vision.request.call_args_list] == ["health"]
    probe.arm.enable.assert_not_called()
    probe.stopper.reset_latch.assert_not_called()
    probe.stopper.health.return_value['latched'] = True
    with pytest.raises(RuntimeError, match="锁存"): probe()


def test_voice_gui_motion_and_voice_methods_reject_cycle_ownership():
    from run_voice_assistant_gui import VoiceAssistantGui
    gui = VoiceAssistantGui.__new__(VoiceAssistantGui)
    gui.cycle_test_busy = lambda: True
    gui.status = Mock()
    gui.backend = Mock()
    gui.execute_manual_floor()
    gui.execute_direction("up")
    gui.check_origin()
    gui.return_origin()
    assert gui.set_voice_enabled(True) is False
    assert gui.resume_trip_from_controller() is False
    gui.backend.send.assert_not_called()
    gui.cycle_abort_callback = Mock()
    gui.abort_and_return()
    gui.cycle_abort_callback.assert_called_once()


def test_controller_cannot_enable_return_or_voice_during_cycle():
    from tools.dualsense_input_gui import InputViewer
    viewer = InputViewer.__new__(InputViewer)
    viewer.external_task_guard = lambda: True
    viewer.control_status = Mock()
    viewer._toggle_control()
    viewer._start_arm_return("arm", "mode_switch")
    viewer._refresh_mode(None)
    viewer._apply_control(None)
    viewer._set_voice_enabled(True)
    assert viewer.control_status.set.call_count == 5


def test_stop_file_failure_still_stops_producer_and_waits_before_exit(monkeypatch):
    process = Mock(pid=876)
    process.poll.return_value = None
    dispatcher = Mock(active=process)
    dispatcher.stop.side_effect = OSError("config file unavailable")
    def refresh(): dispatcher.active = None
    dispatcher.refresh.side_effect = refresh
    readiness = Mock(stop_cfg=dict(token="test-only"))
    readiness.stopper.stop.side_effect = RuntimeError("bridge unreachable")
    kill = Mock()
    monkeypatch.setattr("tools.button_cycle_worker.os.killpg", kill)
    monkeypatch.setattr("tools.button_cycle_worker.time.sleep", lambda _: None)
    stop_before_exit(dispatcher, readiness)
    readiness.stopper.stop.assert_called_once()
    kill.assert_called_once()
    assert dispatcher.active is None and dispatcher.return_after_abort is False


@pytest.mark.parametrize("timestamp", [None, True, float('nan'), float('inf'), 98., 101.])
def test_stale_queued_resume_cannot_start_after_stop(timestamp):
    with pytest.raises(ValueError):
        control_timestamp(dict(issued_monotonic_sec=timestamp), 100., 1.5)
    assert control_timestamp(dict(issued_monotonic_sec=99.9), 100., 1.5) == 99.9
