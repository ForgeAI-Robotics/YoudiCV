import pytest

from piper_handeye.dualsense_evdev import (
    ABS_HAT0X,
    ABS_HAT0Y,
    ABS_RX,
    ABS_RY,
    ABS_RZ,
    ABS_X,
    ABS_Y,
    ABS_Z,
    BTN_EAST,
    BTN_NORTH,
    BTN_SOUTH,
    BTN_TL,
    BTN_TR,
    BTN_WEST,
    BTN_START,
    BTN_THUMBL,
    BTN_THUMBR,
    EV_ABS,
    EV_KEY,
    AxisRange,
    DualSenseEventState,
)


def _state():
    return DualSenseEventState({
        ABS_X: AxisRange(0, 255, 4),
        ABS_Y: AxisRange(0, 255, 4),
        ABS_RX: AxisRange(0, 255, 4),
        ABS_RY: AxisRange(0, 255, 4),
        ABS_Z: AxisRange(0, 255),
        ABS_RZ: AxisRange(0, 255),
        ABS_HAT0X: AxisRange(-1, 1),
        ABS_HAT0Y: AxisRange(-1, 1),
    })


def test_dualsense_mapping_matches_reserved_gamepad_contract():
    state = _state()
    for code, value in {
        ABS_X: 255, ABS_Y: 0, ABS_RX: 191, ABS_RY: 64,
        ABS_Z: 51, ABS_RZ: 204, ABS_HAT0X: -1, ABS_HAT0Y: 1,
    }.items():
        state.update(EV_ABS, code, value)
    for code in (BTN_START, BTN_THUMBL, BTN_THUMBR, BTN_SOUTH, BTN_EAST,
                 BTN_NORTH, BTN_WEST, BTN_TL, BTN_TR):
        state.update(EV_KEY, code, 1)
    sample = state.sample()
    assert sample.left_x == pytest.approx(1.0)
    assert sample.left_y == pytest.approx(-1.0)
    assert sample.right_x == pytest.approx(0.498, abs=0.01)
    assert sample.right_y == pytest.approx(-0.498, abs=0.01)
    assert sample.left_trigger == pytest.approx(0.2)
    assert sample.right_trigger == pytest.approx(0.8)
    assert (sample.dpad_x, sample.dpad_y) == (-1.0, 1.0)
    assert sample.menu and sample.a and sample.b and sample.y
    assert sample.left_stick_button and sample.right_stick_button
    assert sample.square and sample.l1 and sample.r1


def test_disconnect_produces_fail_safe_sample():
    state = _state()
    state.connected = False
    assert state.sample().connected is False


def test_invalid_axis_range_is_rejected():
    with pytest.raises(ValueError):
        AxisRange(10, 10)
