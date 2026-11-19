import pytest

from piper_handeye.gamepad_teleop import (
    ChassisMotionLimiter, DualModeGamepad, GamepadSample, TeleopMode,
)


def released(**values):
    return GamepadSample(**values)


def switch_chord(**values):
    return released(left_stick_button=True, right_stick_button=True, **values)


def double_press(pad, sample):
    first = pad.update(sample)
    pad.update(released())
    second = pad.update(sample)
    return first, second


def enter_arm(pad):
    first, confirmed = double_press(pad, switch_chord())
    assert first.actions == ("mode_switch_confirmation_requested",)
    assert confirmed.mode is TeleopMode.ARM_RETURNING
    assert confirmed.actions == (
        "chassis_stop", "disable_other_functions", "return_arm_origin"
    )
    pad.update(released())
    entered = pad.complete_arm_return(success=True)
    assert entered.mode is TeleopMode.ARM
    return entered


def test_chassis_uses_right_trigger_forward_and_left_trigger_reverse():
    pad = DualModeGamepad()
    intent = pad.update(released(left_x=0.55, right_trigger=0.8))
    assert intent.mode is TeleopMode.CHASSIS
    assert intent.chassis_linear_x_m_s > 0
    assert intent.chassis_angular_z_rad_s < 0
    assert pad.update(released(right_trigger=0.8, left_trigger=0.8)).chassis_linear_x_m_s == 0
    reverse = pad.update(released(left_trigger=1.0))
    assert reverse.chassis_linear_x_m_s == pytest.approx(-0.1)


def test_reverse_mirrors_steering_direction_like_racing_game():
    pad = DualModeGamepad()
    forward = pad.update(released(left_x=0.55, right_trigger=0.8))
    reverse = pad.update(released(left_x=0.55, left_trigger=0.8))
    assert forward.chassis_angular_z_rad_s < 0
    assert reverse.chassis_angular_z_rad_s > 0
    assert abs(reverse.chassis_angular_z_rad_s) == pytest.approx(
        abs(forward.chassis_angular_z_rad_s)
    )


def test_chassis_matches_vendor_limits_and_does_not_attenuate_steering_twice():
    pad = DualModeGamepad()
    slow = pad.update(released(left_x=1.0, right_trigger=0.2))
    fast = pad.update(released(left_x=1.0, right_trigger=1.0))
    assert fast.chassis_linear_x_m_s == pytest.approx(0.1)
    assert slow.chassis_angular_z_rad_s == pytest.approx(-0.4)
    assert fast.chassis_angular_z_rad_s == pytest.approx(-0.4)


def test_motion_limiter_decelerates_and_holds_zero_before_reverse():
    now = [0.0]
    limiter = ChassisMotionLimiter(
        linear_accel_m_s2=1.0,
        linear_decel_m_s2=1.0,
        reversal_hold_sec=0.30,
        clock=lambda: now[0],
    )
    limiter.update(0.4, 0.0)
    for _ in range(8):
        now[0] += 0.05
        forward, _ = limiter.update(0.4, 0.0)
    assert forward == pytest.approx(0.4)
    outputs = []
    for _ in range(8):
        now[0] += 0.05
        outputs.append(limiter.update(-0.4, 0.0)[0])
    assert min(outputs) >= 0.0
    assert outputs[-1] == 0.0
    for _ in range(5):
        now[0] += 0.05
        assert limiter.update(-0.4, 0.0)[0] == 0.0
    now[0] += 0.05
    assert limiter.update(-0.4, 0.0)[0] < 0.0


def test_motion_limiter_release_is_a_controlled_stop():
    now = [0.0]
    limiter = ChassisMotionLimiter(
        linear_accel_m_s2=1.0,
        linear_decel_m_s2=0.5,
        clock=lambda: now[0],
    )
    limiter.update(0.4, 0.0)
    for _ in range(8):
        now[0] += 0.05
        limiter.update(0.4, 0.0)
    now[0] += 0.05
    stopped, _ = limiter.update(0.0, 0.0)
    assert stopped == pytest.approx(0.375)


def test_mode_switch_requires_two_released_chord_presses_and_origin_return():
    pad = DualModeGamepad()
    first = pad.update(switch_chord(right_trigger=1.0))
    assert first.mode is TeleopMode.CHASSIS
    assert first.actions == ("mode_switch_confirmation_requested",)
    assert first.chassis_linear_x_m_s == 0
    held = pad.update(switch_chord(right_trigger=1.0))
    assert held.mode is TeleopMode.CHASSIS
    pad.update(released())
    confirmed = pad.update(switch_chord())
    assert confirmed.mode is TeleopMode.ARM_RETURNING
    assert pad.complete_arm_return(success=True).mode is TeleopMode.ARM


def test_arm_direct_mapping_covers_all_six_joints():
    pad = DualModeGamepad(); enter_arm(pad)
    intent = pad.update(released(
        dpad_x=1.0, dpad_y=-1.0, left_y=-1.0,
        r1=True, right_y=-1.0, right_trigger=1.0,
    ))
    assert intent.arm_joint_velocities_rad_s == pytest.approx(
        (-0.18, 0.18, -0.18, 0.18, -0.18, 0.18)
    )
    assert intent.chassis_linear_x_m_s == 0


def test_arm_opposite_inputs_reverse_each_joint():
    pad = DualModeGamepad(); enter_arm(pad)
    intent = pad.update(released(
        dpad_x=-1.0, dpad_y=1.0, left_y=1.0,
        l1=True, right_y=1.0, left_trigger=1.0,
    ))
    assert intent.arm_joint_velocities_rad_s == pytest.approx(
        (0.18, -0.18, 0.18, -0.18, 0.18, -0.18)
    )


def test_unused_horizontal_sticks_do_not_move_arm():
    pad = DualModeGamepad(); enter_arm(pad)
    intent = pad.update(released(left_x=1.0, right_x=-1.0))
    assert intent.arm_joint_velocities_rad_s == pytest.approx((0, 0, 0, 0, 0, 0))


def test_cross_toggles_claw_and_triangle_toggles_voice():
    pad = DualModeGamepad(); enter_arm(pad)
    claw = pad.update(released(a=True))
    assert claw.actions == ("toggle_clamp",)
    pad.update(released())
    voice_on = pad.update(released(y=True))
    assert voice_on.actions == ("enable_voice_elevator_mode",)
    pad.update(released())
    voice_off = pad.update(released(y=True))
    assert voice_off.actions == ("disable_voice_elevator_mode",)


def test_circle_three_second_hold_closes_control_and_returns():
    now = [10.0]
    pad = DualModeGamepad(clock=lambda: now[0]); enter_arm(pad)
    assert pad.update(released(b=True)).actions == ()
    pad.update(released())
    assert pad.update(released(b=True)).actions == ()
    now[0] += 2.9
    assert pad.update(released(b=True)).actions == ()
    now[0] += 0.1
    confirmed = pad.update(released(b=True))
    assert confirmed.mode is TeleopMode.ARM_RETURNING
    assert confirmed.actions == (
        "arm_stop", "disable_other_functions", "return_arm_origin"
    )
    ready = pad.complete_arm_return(success=True)
    assert ready.mode is TeleopMode.ARM
    assert ready.actions == ("arm_at_origin", "arm_mode_entered")


def test_circle_double_press_returns_then_resumes_control():
    now = [10.0]
    pad = DualModeGamepad(clock=lambda: now[0]); enter_arm(pad)
    assert pad.update(released(b=True)).actions == ()
    assert pad.update(released()).actions == ()
    now[0] += 0.2
    assert pad.update(released(b=True)).actions == ()
    confirmed = pad.update(released())
    assert confirmed.mode is TeleopMode.ARM_RETURNING
    assert confirmed.actions == (
        "arm_stop", "temporarily_disable_other_functions",
        "return_arm_origin", "resume_control_after_return",
    )
    ready = pad.complete_arm_return(success=True)
    assert ready.actions == (
        "arm_at_origin", "arm_mode_entered", "resume_control"
    )


def test_switching_back_disables_voice_and_locks_chassis_until_origin_success():
    pad = DualModeGamepad(); enter_arm(pad)
    pad.update(released(y=True)); pad.update(released())
    _, returning = double_press(pad, switch_chord())
    assert returning.mode is TeleopMode.ARM_RETURNING
    assert returning.actions == (
        "arm_stop", "disable_other_functions",
        "disable_voice_elevator_mode", "return_arm_origin",
    )
    blocked = pad.update(released(right_trigger=1.0))
    assert blocked.actions == ("awaiting_arm_origin",)
    failed = pad.complete_arm_return(success=False)
    assert failed.actions == ("arm_origin_failed_outputs_locked",)
    ready = pad.complete_arm_return(success=True)
    assert ready.mode is TeleopMode.CHASSIS


def test_double_press_confirmation_expires():
    now = [10.0]
    pad = DualModeGamepad(clock=lambda: now[0], confirmation_timeout_sec=1.0)
    pad.update(switch_chord()); pad.update(released())
    now[0] += 1.1
    expired_second = pad.update(switch_chord())
    assert expired_second.mode is TeleopMode.CHASSIS
    assert expired_second.actions == ("mode_switch_confirmation_requested",)


def test_disconnect_stops_all_outputs():
    pad = DualModeGamepad()
    stopped = pad.update(released(connected=False, right_trigger=1.0))
    assert stopped.chassis_linear_x_m_s == 0
    assert stopped.actions == ("controller_disconnected_stop",)


def test_options_no_longer_switches_mode():
    pad = DualModeGamepad()
    assert pad.update(released(menu=True)).mode is TeleopMode.CHASSIS
    assert pad.update(released(menu=False)).mode is TeleopMode.CHASSIS


def test_releasing_both_triggers_stops_linear_and_angular_motion():
    pad = DualModeGamepad()
    assert pad.update(released(right_trigger=1.0)).chassis_linear_x_m_s > 0
    stopped = pad.update(released(left_x=1.0))
    assert stopped.chassis_linear_x_m_s == 0
    assert stopped.chassis_angular_z_rad_s == 0
    assert stopped.chassis_braking


def test_trigger_rest_noise_does_not_move_chassis():
    pad = DualModeGamepad()
    stopped = pad.update(released(right_trigger=0.05, left_trigger=0.04, left_x=1.0))
    assert stopped.chassis_linear_x_m_s == 0
    assert stopped.chassis_angular_z_rad_s == 0


def test_centering_left_stick_restores_straight_drive_immediately():
    pad = DualModeGamepad()
    turning = pad.update(released(right_trigger=0.8, left_x=0.75))
    straight = pad.update(released(right_trigger=0.8, left_x=0.0))
    assert straight.chassis_linear_x_m_s == pytest.approx(turning.chassis_linear_x_m_s)
    assert straight.chassis_angular_z_rad_s == 0.0
