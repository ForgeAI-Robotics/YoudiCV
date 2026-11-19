from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from run_voice_command_gateway import CommandDispatcher


@pytest.fixture(autouse=True)
def isolate_mission_status_outbox(tmp_path, monkeypatch):
    # Mocked dispatcher tests must never acquire the live robot's global lock.
    monkeypatch.setenv("YOUXIAOMEI_ROBOT_MOTION_LOCK", str(tmp_path / "motion.lock"))
    monkeypatch.setenv(
        "YOUXIAOMEI_MISSION_STATUS_OUTBOX", str(tmp_path / "mission-status.jsonl")
    )


class FakeProcess:
    def __init__(self, pid, return_code=None):
        self.pid = pid
        self.return_code = return_code

    def poll(self):
        return self.return_code

    def wait(self, timeout=None):
        if self.return_code is None:
            self.return_code = -15
        return self.return_code


def test_abort_waits_for_old_process_exit_before_starting_return(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "safety_stop:\n"
        "  host: 127.0.0.1\n"
        "  port: 5013\n"
        "  token: test-token\n",
        encoding="utf-8",
    )
    dispatcher = CommandDispatcher(
        execute=True, config_path=config, hold_report=tmp_path / "hold.yaml"
    )
    old = FakeProcess(101)
    replacement = FakeProcess(202)
    dispatcher.active = old
    stop_client = SimpleNamespace(stop=Mock(return_value={"status": "stopped"}))

    with patch(
        "run_voice_command_gateway.SafetyStopUdpClient",
        return_value=stop_client,
    ), patch("run_voice_command_gateway.os.killpg") as kill, patch(
        "run_voice_command_gateway.subprocess.Popen", return_value=replacement
    ) as launch:
        started = dispatcher.abort_and_return()
        assert started["event"] == "abort_return_started"
        assert dispatcher.active is old
        assert dispatcher.return_after_abort
        kill.assert_called_once()
        launch.assert_not_called()

        assert dispatcher.refresh() is None
        launch.assert_not_called()

        old.return_code = -15
        returning = dispatcher.refresh()
        assert returning["event"] == "abort_completed_return_started"
        assert returning["aborted_process_return_code"] == -15
        assert dispatcher.active is replacement
        assert dispatcher.active_operation == "return"
        launch.assert_called_once()


def test_abort_without_active_task_starts_return_immediately(tmp_path):
    dispatcher = CommandDispatcher(
        execute=True,
        config_path=tmp_path / "config.yaml",
        hold_report=tmp_path / "hold.yaml",
    )
    replacement = FakeProcess(303)
    with patch(
        "run_voice_command_gateway.subprocess.Popen", return_value=replacement
    ):
        event = dispatcher.abort_and_return()
    assert event["event"] == "origin_return_started"
    assert dispatcher.active_operation == "return"


def test_stop_holds_motion_lock_until_child_exit(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "safety_stop:\n"
        "  host: 127.0.0.1\n"
        "  port: 5013\n"
        "  token: test-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "YOUXIAOMEI_ROBOT_MOTION_LOCK", str(tmp_path / "robot-motion.lock")
    )
    first = CommandDispatcher(
        execute=True, config_path=config, hold_report=tmp_path / "hold.yaml"
    )
    second = CommandDispatcher(
        execute=True, config_path=config, hold_report=tmp_path / "hold.yaml"
    )
    first._acquire_motion_lock()

    class LockCheckingProcess(FakeProcess):
        pid = 404

        def wait(self, timeout=None):
            with pytest.raises(RuntimeError, match="占用机械臂任务执行权"):
                second._acquire_motion_lock()
            self.return_code = -15
            return self.return_code

    first.active = LockCheckingProcess(404)
    stop_client = SimpleNamespace(stop=Mock(return_value={"status": "stopped"}))
    with patch(
        "run_voice_command_gateway.SafetyStopUdpClient",
        return_value=stop_client,
    ), patch("run_voice_command_gateway.os.killpg"):
        result = first.stop(reason="test_stop")

    assert result["event"] == "stop"
    assert first.active is None
    second._acquire_motion_lock()
    second._release_motion_lock()


def test_stop_timeout_keeps_process_and_motion_lock(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "safety_stop:\n"
        "  host: 127.0.0.1\n"
        "  port: 5013\n"
        "  token: test-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "YOUXIAOMEI_ROBOT_MOTION_LOCK", str(tmp_path / "robot-motion.lock")
    )
    first = CommandDispatcher(
        execute=True, config_path=config, hold_report=tmp_path / "hold.yaml"
    )
    second = CommandDispatcher(
        execute=True, config_path=config, hold_report=tmp_path / "hold.yaml"
    )
    first._acquire_motion_lock()

    class StuckProcess(FakeProcess):
        def wait(self, timeout=None):
            raise __import__("subprocess").TimeoutExpired("test", timeout)

    process = StuckProcess(505)
    first.active = process
    stop_client = SimpleNamespace(stop=Mock(return_value={"status": "stopped"}))
    with patch(
        "run_voice_command_gateway.SafetyStopUdpClient",
        return_value=stop_client,
    ), patch("run_voice_command_gateway.os.killpg"):
        result = first.stop(reason="test_timeout")

    assert result["event"] == "stop_pending"
    assert first.active is process
    with pytest.raises(RuntimeError, match="占用机械臂任务执行权"):
        second._acquire_motion_lock()
    first._release_motion_lock()


def test_plain_stop_cancels_pending_abort_return_chain(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "safety_stop:\n"
        "  host: 127.0.0.1\n"
        "  port: 5013\n"
        "  token: test-token\n",
        encoding="utf-8",
    )
    dispatcher = CommandDispatcher(
        execute=True, config_path=config, hold_report=tmp_path / "hold.yaml"
    )
    dispatcher.active = FakeProcess(606)
    dispatcher.return_after_abort = True
    stop_client = SimpleNamespace(stop=Mock(return_value={"status": "stopped"}))
    with patch(
        "run_voice_command_gateway.SafetyStopUdpClient",
        return_value=stop_client,
    ), patch("run_voice_command_gateway.os.killpg"), patch(
        "run_voice_command_gateway.subprocess.Popen"
    ) as launch:
        result = dispatcher.stop(reason="cancel_return_chain")
    assert result["event"] == "stop"
    assert dispatcher.active is None
    assert not dispatcher.return_after_abort
    launch.assert_not_called()
