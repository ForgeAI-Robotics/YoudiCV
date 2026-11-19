"""Hardware-independent two-mode gamepad intent and safety state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Mapping


class TeleopMode(str, Enum):
    CHASSIS = "chassis"
    ARM = "arm"
    ARM_RETURNING = "arm_returning"


@dataclass(frozen=True)
class GamepadSample:
    """Normalized Xbox-style input; axes and triggers are in [-1,1] and [0,1]."""

    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    dpad_x: float = 0.0
    dpad_y: float = 0.0
    menu: bool = False
    left_stick_button: bool = False
    right_stick_button: bool = False
    a: bool = False
    b: bool = False
    y: bool = False
    square: bool = False
    l1: bool = False
    r1: bool = False
    connected: bool = True


@dataclass(frozen=True)
class TeleopConfig:
    deadzone: float = 0.10
    chassis_max_linear_m_s: float = 0.10
    chassis_max_angular_rad_s: float = 0.40
    arm_joint_speed_rad_s: float = 0.18

    def __post_init__(self) -> None:
        if not 0.0 <= self.deadzone < 0.5:
            raise ValueError("deadzone must be in 0..0.5")
        if min(
            self.chassis_max_linear_m_s,
            self.chassis_max_angular_rad_s,
            self.arm_joint_speed_rad_s,
        ) <= 0:
            raise ValueError("teleoperation limits must be positive")


@dataclass(frozen=True)
class TeleopIntent:
    mode: TeleopMode
    chassis_linear_x_m_s: float = 0.0
    chassis_angular_z_rad_s: float = 0.0
    arm_joint_velocities_rad_s: tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    arm_joint_speed_rad_s: float = 0.18
    actions: tuple[str, ...] = ()
    clamp_open: bool = True
    chassis_braking: bool = False


class ChassisMotionLimiter:
    """Rate-limit chassis commands and force every reversal through zero."""

    def __init__(
        self,
        *,
        linear_accel_m_s2: float = 0.10,
        linear_decel_m_s2: float = 0.12,
        angular_accel_rad_s2: float = 0.40,
        angular_decel_rad_s2: float = 0.50,
        reversal_hold_sec: float = 0.30,
        clock=time.monotonic,
    ) -> None:
        values = (
            linear_accel_m_s2, linear_decel_m_s2,
            angular_accel_rad_s2, angular_decel_rad_s2,
            reversal_hold_sec,
        )
        if min(values) <= 0:
            raise ValueError("motion limiter values must be positive")
        self.linear_accel_m_s2 = float(linear_accel_m_s2)
        self.linear_decel_m_s2 = float(linear_decel_m_s2)
        self.angular_accel_rad_s2 = float(angular_accel_rad_s2)
        self.angular_decel_rad_s2 = float(angular_decel_rad_s2)
        self.reversal_hold_sec = float(reversal_hold_sec)
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self.linear = 0.0
        self.angular = 0.0
        self._last_at: float | None = None
        self._direction = 0
        self._zero_since: float | None = None

    @staticmethod
    def _approach(current: float, target: float, rate: float, dt: float) -> float:
        step = max(0.0, rate * dt)
        if target > current:
            return min(target, current + step)
        return max(target, current - step)

    def update(self, target_linear: float, target_angular: float) -> tuple[float, float]:
        now = float(self._clock())
        if self._last_at is None:
            dt = 0.0
        else:
            # A delayed GUI callback must not create one large velocity jump.
            dt = min(0.05, max(0.0, now - self._last_at))
        self._last_at = now

        epsilon = 1e-4
        target_direction = (
            1 if target_linear > epsilon else -1 if target_linear < -epsilon else 0
        )
        effective_linear = float(target_linear)
        if target_direction and self._direction and target_direction != self._direction:
            effective_linear = 0.0
            if abs(self.linear) <= epsilon:
                self.linear = 0.0
                if self._zero_since is None:
                    self._zero_since = now
                elif now - self._zero_since >= self.reversal_hold_sec:
                    self._direction = target_direction
                    self._zero_since = None
                    effective_linear = float(target_linear)
            else:
                self._zero_since = None
        elif target_direction:
            if self._direction == 0:
                self._direction = target_direction
            self._zero_since = None

        linear_rate = (
            self.linear_accel_m_s2
            if abs(effective_linear) > abs(self.linear)
            and effective_linear * self.linear >= 0.0
            else self.linear_decel_m_s2
        )
        self.linear = self._approach(
            self.linear, effective_linear, linear_rate, dt
        )
        if abs(self.linear) <= epsilon and effective_linear == 0.0:
            self.linear = 0.0
            if self._zero_since is None:
                self._zero_since = now

        angular_rate = (
            self.angular_accel_rad_s2
            if abs(target_angular) > abs(self.angular)
            and target_angular * self.angular >= 0.0
            else self.angular_decel_rad_s2
        )
        self.angular = self._approach(
            self.angular, float(target_angular), angular_rate, dt
        )
        return self.linear, self.angular


def _bounded(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def _deadzone(value: float, threshold: float) -> float:
    value = _bounded(value, -1.0, 1.0)
    magnitude = abs(value)
    if magnitude <= threshold:
        return 0.0
    scaled = (magnitude - threshold) / (1.0 - threshold)
    return scaled if value > 0 else -scaled


class DualModeGamepad:
    """Convert edge-aware controller samples into mutually exclusive intents."""

    def __init__(
        self,
        config: TeleopConfig | None = None,
        *,
        clock=time.monotonic,
        confirmation_timeout_sec: float = 1.5,
        return_hold_sec: float = 3.0,
        initial_mode: TeleopMode = TeleopMode.CHASSIS,
    ):
        self.config = config or TeleopConfig()
        if confirmation_timeout_sec <= 0:
            raise ValueError("confirmation_timeout_sec must be positive")
        if return_hold_sec <= 0:
            raise ValueError("return_hold_sec must be positive")
        self._clock = clock
        self.confirmation_timeout_sec = float(confirmation_timeout_sec)
        self.return_hold_sec = float(return_hold_sec)
        if initial_mode not in {TeleopMode.CHASSIS, TeleopMode.ARM}:
            raise ValueError("initial_mode must be chassis or arm")
        self.mode = initial_mode
        self.clamp_open = True
        self.voice_enabled = False
        self._return_target_mode = TeleopMode.CHASSIS
        self._confirmation_started: dict[str, float] = {}
        self._previous_buttons: dict[str, bool] = {
            "menu": False, "a": False, "b": False, "y": False,
            "square": False, "l1": False, "r1": False,
        }
        self._previous_mode_chord = False
        self._return_hold_started: float | None = None
        self._return_hold_fired = False
        self._return_first_click_at: float | None = None
        self._resume_control_after_return = False

    def _rising(self, sample: GamepadSample, name: str) -> bool:
        current = bool(getattr(sample, name))
        rising = current and not self._previous_buttons[name]
        self._previous_buttons[name] = current
        return rising

    def _confirmed_double_press(self, name: str, rising: bool) -> tuple[bool, bool]:
        """Return (first_press, confirmed) for two released clicks in a window."""
        if not rising:
            return False, False
        now = float(self._clock())
        started = self._confirmation_started.get(name)
        if started is not None and now - started <= self.confirmation_timeout_sec:
            self._confirmation_started.pop(name, None)
            return False, True
        self._confirmation_started[name] = now
        return True, False

    def update(self, sample: GamepadSample) -> TeleopIntent:
        previous_b = self._previous_buttons["b"]
        edges = {name: self._rising(sample, name) for name in self._previous_buttons}
        b_falling = previous_b and not bool(sample.b)
        mode_chord = bool(sample.left_stick_button and sample.right_stick_button)
        mode_switch_rising = mode_chord and not self._previous_mode_chord
        self._previous_mode_chord = mode_chord
        if not sample.connected:
            return TeleopIntent(
                mode=self.mode,
                actions=("controller_disconnected_stop",),
                clamp_open=self.clamp_open,
                chassis_braking=True,
            )

        if self.mode is TeleopMode.ARM_RETURNING:
            return TeleopIntent(
                mode=self.mode,
                actions=("awaiting_arm_origin",),
                clamp_open=self.clamp_open,
                chassis_braking=True,
            )

        first_switch, confirmed_switch = self._confirmed_double_press(
            "mode_switch", mode_switch_rising
        )
        if first_switch:
            return TeleopIntent(
                mode=self.mode,
                actions=("mode_switch_confirmation_requested",),
                clamp_open=self.clamp_open,
                chassis_braking=True,
            )
        if confirmed_switch:
            self._confirmation_started.clear()
            source = self.mode
            self._return_target_mode = (
                TeleopMode.ARM if source is TeleopMode.CHASSIS else TeleopMode.CHASSIS
            )
            self.mode = TeleopMode.ARM_RETURNING
            actions = [
                "chassis_stop" if source is TeleopMode.CHASSIS else "arm_stop",
                "disable_other_functions",
            ]
            if self.voice_enabled:
                self.voice_enabled = False
                actions.append("disable_voice_elevator_mode")
            actions.append("return_arm_origin")
            return TeleopIntent(
                mode=self.mode,
                actions=tuple(actions),
                clamp_open=self.clamp_open,
                chassis_braking=True,
            )

        if edges["b"]:
            self._return_hold_started = float(self._clock())
            self._return_hold_fired = False
        long_return = (
            bool(sample.b)
            and not self._return_hold_fired
            and self._return_hold_started is not None
            and float(self._clock()) - self._return_hold_started
            >= self.return_hold_sec
        )
        double_return = False
        if b_falling and not self._return_hold_fired:
            now = float(self._clock())
            if (
                self._return_first_click_at is not None
                and now - self._return_first_click_at
                <= self.confirmation_timeout_sec
            ):
                double_return = True
                self._return_first_click_at = None
            else:
                self._return_first_click_at = now
            self._return_hold_started = None
        if long_return:
            self._return_hold_fired = True
            self._return_first_click_at = None
        if long_return or double_return:
            self._confirmation_started.clear()
            source = self.mode
            self._return_target_mode = source
            self.mode = TeleopMode.ARM_RETURNING
            self._resume_control_after_return = double_return
            actions = [
                "chassis_stop" if source is TeleopMode.CHASSIS else "arm_stop",
                "temporarily_disable_other_functions"
                if double_return else "disable_other_functions",
            ]
            if self.voice_enabled:
                self.voice_enabled = False
                actions.append("disable_voice_elevator_mode")
            actions.append("return_arm_origin")
            if double_return:
                actions.append("resume_control_after_return")
            return TeleopIntent(
                mode=self.mode,
                actions=tuple(actions),
                clamp_open=self.clamp_open,
                chassis_braking=True,
            )

        cfg = self.config
        if self.mode is TeleopMode.CHASSIS:
            # Racing-game contract: R2 is forward speed and L2 is reverse
            # speed. Trigger deadzones suppress resting sensor noise. Releasing
            # both triggers produces an explicit full stop, including zero
            # angular velocity even if the steering stick is displaced.
            forward = _deadzone(sample.right_trigger, cfg.deadzone)
            reverse = _deadzone(sample.left_trigger, cfg.deadzone)
            drive = forward - reverse
            steer = _deadzone(sample.left_x, cfg.deadzone)
            stopped = abs(drive) <= 1e-12
            return TeleopIntent(
                mode=self.mode,
                chassis_linear_x_m_s=drive * cfg.chassis_max_linear_m_s,
                # RosAria treats linear and angular velocity as independent
                # command components.  Reverse steering is mirrored like a
                # racing game: the same stick direction turns the chassis in
                # the opposite yaw direction while backing up.
                chassis_angular_z_rad_s=(
                    0.0 if stopped else
                    (-steer if drive > 0.0 else steer)
                    * cfg.chassis_max_angular_rad_s
                ),
                clamp_open=self.clamp_open,
                chassis_braking=stopped,
            )

        actions: list[str] = []
        if edges["a"]:
            # The bridge derives open/closed from live gripper feedback.  A
            # second predicted state here diverges after reconnects.
            actions.append("toggle_clamp")
        if edges["y"]:
            self.voice_enabled = not self.voice_enabled
            actions.append(
                "enable_voice_elevator_mode"
                if self.voice_enabled else "disable_voice_elevator_mode"
            )

        # Fixed direct mapping. Signs match the physical motion at the normal
        # Signs match the operator-confirmed physical directions.
        speed = cfg.arm_joint_speed_rad_s
        trigger = (
            _deadzone(sample.right_trigger, cfg.deadzone)
            - _deadzone(sample.left_trigger, cfg.deadzone)
        )
        joint_velocities = (
            -_deadzone(sample.dpad_x, cfg.deadzone) * speed,
            -_deadzone(sample.dpad_y, cfg.deadzone) * speed,
            _deadzone(sample.left_y, cfg.deadzone) * speed,
            (float(sample.r1) - float(sample.l1)) * speed,
            _deadzone(sample.right_y, cfg.deadzone) * speed,
            trigger * speed,
        )
        return TeleopIntent(
            mode=self.mode,
            arm_joint_velocities_rad_s=joint_velocities,
            arm_joint_speed_rad_s=speed,
            actions=tuple(actions),
            clamp_open=self.clamp_open,
            chassis_braking=True,
        )

    def complete_arm_return(self, *, success: bool) -> TeleopIntent:
        if self.mode is not TeleopMode.ARM_RETURNING:
            raise RuntimeError("arm return completion received outside return state")
        if not success:
            return TeleopIntent(
                mode=self.mode,
                actions=("arm_origin_failed_outputs_locked",),
                clamp_open=self.clamp_open,
                chassis_braking=True,
            )
        self.mode = self._return_target_mode
        target = self.mode
        resume = self._resume_control_after_return
        self._resume_control_after_return = False
        return TeleopIntent(
            mode=self.mode,
            actions=(
                "arm_at_origin",
                "arm_mode_entered" if target is TeleopMode.ARM else "chassis_mode_entered",
            ) + (("resume_control",) if resume else ()),
            clamp_open=self.clamp_open,
            chassis_braking=True,
        )


def sample_from_mapping(values: Mapping[str, object]) -> GamepadSample:
    """Stable boundary for a future evdev/ROS joy/UDP input adapter."""
    names = GamepadSample.__dataclass_fields__
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError(f"unknown gamepad fields: {sorted(unknown)}")
    return GamepadSample(**values)
