from run_android_command_gui import (
    AndroidCommandGui,
    MISSION_MODE_FLOOR_ONLY,
    MISSION_MODE_TRIP,
    dispatch_android_target,
)
from piper_handeye.android_agx_protocol import AcceptedPacketCache, decode_packet, encode_packet
import pytest


class FakeDispatcher:
    def __init__(self):
        self.calls = []

    def dispatch_motion(self, command):
        self.calls.append(("floor_only", command))
        return {"event": "motion_started", "button": command.button}

    def start_trip(self, command):
        self.calls.append(("trip", command))
        return {"event": "motion_started", "trip_destination": command.button}


def test_floor_only_mode_dispatches_target_directly_without_trip_setup():
    dispatcher = FakeDispatcher()
    result = dispatch_android_target(dispatcher, 15, MISSION_MODE_FLOOR_ONLY)

    assert result == {"event": "motion_started", "button": 15}
    assert len(dispatcher.calls) == 1
    mode, command = dispatcher.calls[0]
    assert mode == "floor_only"
    assert command.button == 15


def test_trip_mode_keeps_existing_staged_trip_dispatch():
    dispatcher = FakeDispatcher()
    result = dispatch_android_target(dispatcher, 8, MISSION_MODE_TRIP)

    assert result == {"event": "motion_started", "trip_destination": 8}
    assert dispatcher.calls[0][0] == "trip"
    assert dispatcher.calls[0][1].button == 8


def test_unknown_mode_fails_closed_without_dispatching():
    dispatcher = FakeDispatcher()
    with pytest.raises(ValueError, match="未知 Android 任务模式"):
        dispatch_android_target(dispatcher, 8, "unexpected")
    assert dispatcher.calls == []


class FakeVariable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def test_ack_failure_does_not_misreport_a_started_task_as_start_failure():
    gui = AndroidCommandGui.__new__(AndroidCommandGui)
    gui.arm_enabled = True
    gui.pending_target = 5
    gui.pending_packet = decode_packet(
        encode_packet(31, "bt_control", {"command": 10, "value": "5"}),
        ("192.168.168.110", 5005),
    )
    gui.accepted_target = None
    gui.accepted_packets = AcceptedPacketCache()
    gui.last_source = gui.pending_packet.source
    gui.mission_mode = FakeVariable(MISSION_MODE_FLOOR_ONLY)
    gui.task_state = FakeVariable("无任务")
    gui.dispatcher = FakeDispatcher()
    gui.listener = type("Listener", (), {
        "reply": lambda self, source: (_ for _ in ()).throw(OSError("network down"))
    })()
    messages = []
    gui._append = messages.append

    gui._execute_pending(packet=gui.pending_packet)

    assert gui.task_state.get() == "正在按目标楼层 5"
    assert gui.accepted_packets.contains(gui.pending_packet)
    assert any("ACK 发送失败" in message for message in messages)
    assert any("任务已接受" in message for message in messages)
    assert not any("任务启动失败" in message for message in messages)
