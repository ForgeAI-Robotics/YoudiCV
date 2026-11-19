import pytest

from run_voice_assistant_gui import parse_floor_entry
from piper_handeye.voice_runtime import dialogue_relevant_transcript, handle_control_message


class FakeDispatcher:
    def __init__(self, active=None):
        self.active = active
        self.commands = []
        self.current_floor = None
        self.trip_stage = None

    def set_current_floor(self, value):
        self.current_floor = value

    def resume_trip(self):
        self.commands.append("resume_trip")
        return {"event": "motion_started", "trip_stage": "floor"}

    def dispatch_motion(self, command):
        self.commands.append(command)
        return {"event": "motion_started", "button": command.button}

    def dispatch_origin_operation(self, operation):
        self.commands.append(operation)
        return {"event": f"origin_{operation}_started", "operation": operation}

    def abort_and_return(self):
        self.commands.append("abort_and_return")
        return {"event": "abort_return_started"}


def test_floor_entry_accepts_only_configured_buttons():
    assert parse_floor_entry(" 15 ") == 15
    assert parse_floor_entry("十五楼") == 15
    assert parse_floor_entry("十五") == 15
    assert parse_floor_entry("送外卖到15楼") == 15
    assert parse_floor_entry("按第九层") == 9
    for value in ("", "0", "17", "1.5", "返回原点"):
        with pytest.raises(ValueError):
            parse_floor_entry(value)


def test_manual_floor_uses_the_shared_dispatcher():
    dispatcher = FakeDispatcher()
    enabled, events = handle_control_message(
        {"command": "manual_floor", "floor": 8},
        dispatcher,
        voice_enabled=False,
    )
    assert not enabled
    assert dispatcher.commands[0].button == 8
    assert [event["event"] for event in events] == [
        "manual_command", "motion_started"
    ]


def test_controller_can_set_starting_floor_and_resume_staged_trip():
    dispatcher = FakeDispatcher()
    enabled, events = handle_control_message(
        {"command": "set_current_floor", "floor": 16},
        dispatcher,
        voice_enabled=False,
    )
    assert enabled is False
    assert dispatcher.current_floor == 16
    assert events == [{"event": "current_floor_set", "floor": 16}]

    _, events = handle_control_message(
        {"command": "resume_trip"}, dispatcher, voice_enabled=False
    )
    assert events == [{"event": "motion_started", "trip_stage": "floor"}]
    assert dispatcher.commands == ["resume_trip"]


def test_manual_floor_preserves_typed_command_for_chat():
    dispatcher = FakeDispatcher()
    _, events = handle_control_message(
        {"command": "manual_floor", "floor": 15, "text": "送外卖到十五楼"},
        dispatcher,
        voice_enabled=False,
    )
    assert events[0]["text"] == "送外卖到十五楼"


def test_manual_floor_is_rejected_while_shared_dispatcher_is_busy():
    dispatcher = FakeDispatcher(active=object())
    _, events = handle_control_message(
        {"command": "manual_floor", "floor": 8},
        dispatcher,
        voice_enabled=True,
    )
    assert not dispatcher.commands
    assert events[-1]["event"] == "manual_command_rejected"


@pytest.mark.parametrize("target", ["up", "down"])
def test_manual_direction_button_uses_shared_press_pipeline(target):
    dispatcher = FakeDispatcher()
    enabled, events = handle_control_message(
        {"command": "manual_button", "target": target},
        dispatcher,
        voice_enabled=False,
    )
    assert not enabled
    assert dispatcher.commands[0].button == target
    assert [event["event"] for event in events] == [
        "manual_command", "motion_started"
    ]


def test_manual_direction_button_is_rejected_while_busy():
    dispatcher = FakeDispatcher(active=object())
    _, events = handle_control_message(
        {"command": "manual_button", "target": "up"},
        dispatcher,
        voice_enabled=False,
    )
    assert not dispatcher.commands
    assert events[-1]["event"] == "manual_command_rejected"


def test_manual_floor_protocol_rejects_coerced_values():
    dispatcher = FakeDispatcher()
    for value in (True, 8.5, "8", None):
        _, events = handle_control_message(
            {"command": "manual_floor", "floor": value},
            dispatcher,
            voice_enabled=False,
        )
        assert events == [{
            "event": "manual_command_rejected",
            "message": "楼层必须是 1–16 的整数",
        }]
    assert not dispatcher.commands


def test_voice_toggle_changes_only_voice_state():
    dispatcher = FakeDispatcher()
    enabled, events = handle_control_message(
        {"command": "set_voice_enabled", "enabled": True},
        dispatcher,
        voice_enabled=False,
    )
    assert enabled
    assert events == [{"event": "voice_state", "enabled": True}]
    assert not dispatcher.commands


@pytest.mark.parametrize("command,operation", [
    ("origin_check", "check"),
    ("origin_return", "return"),
])
def test_origin_controls_share_the_dispatcher(command, operation):
    dispatcher = FakeDispatcher()
    enabled, events = handle_control_message(
        {"command": command}, dispatcher, voice_enabled=True
    )
    assert enabled
    assert dispatcher.commands == [operation]
    assert events == [{
        "event": f"origin_{operation}_started", "operation": operation
    }]


def test_origin_controls_are_rejected_while_any_task_is_active():
    dispatcher = FakeDispatcher(active=object())
    _, events = handle_control_message(
        {"command": "origin_return"}, dispatcher, voice_enabled=False
    )
    assert not dispatcher.commands
    assert events[0]["event"] == "origin_operation_rejected"


def test_abort_and_return_is_allowed_while_task_is_active():
    dispatcher = FakeDispatcher(active=object())
    _, events = handle_control_message(
        {"command": "abort_and_return"}, dispatcher, voice_enabled=False
    )
    assert dispatcher.commands == ["abort_and_return"]
    assert events == [{"event": "abort_return_started"}]


def test_ui_dpi_environment_is_bounded(monkeypatch):
    # Tk/X11 rendering is covered by the GUI smoke test; keep malformed DPI
    # input from silently creating an unusable oversized window.
    import run_voice_assistant_gui as gui

    class FakeTkCall:
        def call(self, *args):
            return None

    class FakeRoot:
        tk = FakeTkCall()

    monkeypatch.setenv("YOUXIAOMEI_UI_DPI", "500")
    with pytest.raises(ValueError):
        gui.configure_crisp_ui(FakeRoot())


def test_chat_filters_non_speech_but_keeps_wake_and_commands():
    assert not dialogue_relevant_transcript("(音乐)", "（请留意）", "优小妹")
    assert not dialogue_relevant_transcript("一二三", "", "优小妹")
    assert dialogue_relevant_transcript("没事了", "", "优小妹")
    assert not dialogue_relevant_transcript("123", "", "优小妹")
    assert dialogue_relevant_transcript("优小妹", "", "优小妹")
    assert dialogue_relevant_transcript(
        "送外卖到八楼", "送外卖到8楼", "优小妹"
    )
