from piper_handeye.gamepad_teleop import GamepadSample
from tools.dualsense_input_gui import InputViewer


def test_mic_button_from_bluetooth_full_report():
    report = bytearray(78)
    report[0] = 0x31
    assert InputViewer._mic_pressed_from_report(bytes(report)) is False
    report[11] = 0x04
    assert InputViewer._mic_pressed_from_report(bytes(report)) is True


def test_mic_button_from_usb_report():
    report = bytearray(64)
    report[0] = 0x01
    assert InputViewer._mic_pressed_from_report(bytes(report)) is False
    report[10] = 0x04
    assert InputViewer._mic_pressed_from_report(bytes(report)) is True


def test_mic_button_rejects_other_or_short_reports():
    assert InputViewer._mic_pressed_from_report(b"") is None
    assert InputViewer._mic_pressed_from_report(b"\x31" * 4) is None
    assert InputViewer._mic_pressed_from_report(b"\x02" + b"\0" * 63) is None


class _TextState:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


def test_four_button_double_chord_enables_only_after_release_to_neutral():
    viewer = object.__new__(InputViewer)
    viewer.ui_mode = "chassis"
    viewer.last_mode_chord_pressed = False
    viewer.mode_chord_control_seen = False
    viewer.last_control_chord_pressed = False
    viewer.control_chord_seen = False
    viewer.control_chord_first_at = None
    viewer.control_chord_enable_armed = False
    viewer.control_chord_enable_deadline = 0.0
    viewer.last_circle_pressed = False
    viewer.mode_switch_first_at = None
    viewer.circle_pressed_at = None
    viewer.circle_first_click_at = None
    viewer.circle_hold_fired = False
    viewer.control_enabled = False
    viewer.control_confirm_pending = False
    viewer.control_confirm_deadline = 0.0
    viewer.control_status = _TextState()
    events = []
    enabled = []
    viewer._journal = lambda event, **fields: events.append((event, fields))
    viewer._toggle_control = lambda: enabled.append(True)
    viewer._start_arm_return = lambda *_args: (_ for _ in ()).throw(
        AssertionError("control chord must not switch mode")
    )

    chord = GamepadSample(
        left_trigger=1.0, right_trigger=1.0, l1=True, r1=True,
    )
    released = GamepadSample()
    viewer._refresh_mode(chord)
    viewer._refresh_mode(released)
    assert enabled == []
    viewer._refresh_mode(chord)
    assert enabled == []
    viewer._refresh_mode(released)

    assert enabled == [True]
    assert viewer.control_confirm_pending is True
    assert "control_chord_double_click_confirmed" in [e for e, _ in events]
    assert "control_chord_neutral_confirmed" in [e for e, _ in events]
