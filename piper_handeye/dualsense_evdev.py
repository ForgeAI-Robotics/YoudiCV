"""DualSense evdev input adapter with no robot output dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
import select
from typing import Any, Iterable

from .gamepad_teleop import GamepadSample


# Stable Linux input-event ABI values from linux/input-event-codes.h. Keeping
# them local makes mapping tests independent of whether python-evdev is present.
EV_KEY = 0x01
EV_ABS = 0x03
ABS_X = 0x00
ABS_Y = 0x01
ABS_Z = 0x02
ABS_RX = 0x03
ABS_RY = 0x04
ABS_RZ = 0x05
ABS_HAT0X = 0x10
ABS_HAT0Y = 0x11
BTN_SOUTH = 0x130
BTN_EAST = 0x131
BTN_WEST = 0x134
BTN_NORTH = 0x133
BTN_TL = 0x136
BTN_TR = 0x137
BTN_START = 0x13B
BTN_THUMBL = 0x13D
BTN_THUMBR = 0x13E


@dataclass(frozen=True)
class AxisRange:
    minimum: int
    maximum: int
    flat: int = 0

    def __post_init__(self) -> None:
        if self.maximum <= self.minimum:
            raise ValueError("axis maximum must exceed minimum")


def _centered(value: int, limits: AxisRange) -> float:
    centre = (limits.minimum + limits.maximum) / 2.0
    delta = float(value) - centre
    if abs(delta) <= max(0, limits.flat):
        return 0.0
    span = limits.maximum - centre if delta >= 0 else centre - limits.minimum
    return max(-1.0, min(1.0, delta / span))


def _trigger(value: int, limits: AxisRange) -> float:
    return max(
        0.0,
        min(1.0, (float(value) - limits.minimum) / (limits.maximum - limits.minimum)),
    )


@dataclass
class DualSenseEventState:
    """Translate Linux input events into the project's normalized sample."""

    axis_ranges: dict[int, AxisRange]
    axes: dict[int, int] = field(default_factory=dict)
    buttons: dict[int, bool] = field(default_factory=dict)
    connected: bool = True

    def update(self, event_type: int, code: int, value: int) -> None:
        if int(event_type) == EV_ABS and int(code) in self.axis_ranges:
            self.axes[int(code)] = int(value)
        elif int(event_type) == EV_KEY:
            self.buttons[int(code)] = bool(value)

    def _axis(self, code: int, *, trigger: bool = False) -> float:
        limits = self.axis_ranges.get(code)
        if limits is None:
            return 0.0
        default = limits.minimum if trigger else round(
            (limits.minimum + limits.maximum) / 2
        )
        value = self.axes.get(code, default)
        return _trigger(value, limits) if trigger else _centered(value, limits)

    def sample(self) -> GamepadSample:
        return GamepadSample(
            left_x=self._axis(ABS_X),
            left_y=self._axis(ABS_Y),
            right_x=self._axis(ABS_RX),
            right_y=self._axis(ABS_RY),
            left_trigger=self._axis(ABS_Z, trigger=True),
            right_trigger=self._axis(ABS_RZ, trigger=True),
            dpad_x=self._axis(ABS_HAT0X),
            dpad_y=self._axis(ABS_HAT0Y),
            menu=self.buttons.get(BTN_START, False),       # DualSense Options
            left_stick_button=self.buttons.get(BTN_THUMBL, False),
            right_stick_button=self.buttons.get(BTN_THUMBR, False),
            a=self.buttons.get(BTN_SOUTH, False),         # Cross
            b=self.buttons.get(BTN_EAST, False),          # Circle
            y=self.buttons.get(BTN_NORTH, False),         # Triangle
            square=self.buttons.get(BTN_WEST, False),
            l1=self.buttons.get(BTN_TL, False),
            r1=self.buttons.get(BTN_TR, False),
            connected=bool(self.connected),
        )


class DualSenseEvdevAdapter:
    """Read one DualSense through evdev without producing control commands."""

    MATCH_NAMES = ("dualsense", "wireless controller")

    def __init__(self, device: str = "auto", *, evdev_module: Any | None = None):
        if evdev_module is None:
            try:
                import evdev as evdev_module  # type: ignore[no-redef]
            except ImportError as exc:
                raise RuntimeError(
                    "python-evdev is required for the DualSense input adapter"
                ) from exc
        self.evdev = evdev_module
        self.device = self._open(device)
        self.state = DualSenseEventState(self._axis_ranges())
        self._prime_state()

    def _open(self, requested: str) -> Any:
        if requested != "auto":
            return self.evdev.InputDevice(str(requested))
        matches = []
        for path in self.evdev.list_devices():
            candidate = self.evdev.InputDevice(path)
            name = str(candidate.name or "").lower()
            if any(token in name for token in self.MATCH_NAMES):
                matches.append(candidate)
            else:
                candidate.close()
        if not matches:
            raise FileNotFoundError("no DualSense evdev input device found")
        matches.sort(key=lambda item: str(item.path))
        for extra in matches[1:]:
            extra.close()
        return matches[0]

    def _axis_ranges(self) -> dict[int, AxisRange]:
        ranges = {}
        for code in (
            ABS_X, ABS_Y, ABS_Z, ABS_RX, ABS_RY, ABS_RZ, ABS_HAT0X, ABS_HAT0Y
        ):
            try:
                info = self.device.absinfo(code)
            except OSError:
                continue
            if info is not None and int(info.max) > int(info.min):
                ranges[code] = AxisRange(int(info.min), int(info.max), int(info.flat))
        required = {ABS_X, ABS_Y, ABS_Z, ABS_RX, ABS_RY, ABS_RZ}
        missing = sorted(required - ranges.keys())
        if missing:
            raise RuntimeError(f"DualSense is missing required axes: {missing}")
        return ranges

    def _prime_state(self) -> None:
        for code in self.state.axis_ranges:
            info = self.device.absinfo(code)
            self.state.axes[code] = int(info.value)
        try:
            active: Iterable[int] = self.device.active_keys(verbose=False)
        except OSError:
            active = ()
        self.state.buttons.update({int(code): True for code in active})

    def read_sample(self, timeout_sec: float = 0.02) -> GamepadSample:
        """Drain available events and return the latest normalized state."""
        if not 0.0 <= float(timeout_sec) <= 1.0:
            raise ValueError("timeout_sec must be in 0..1 second")
        try:
            readable, _, _ = select.select([self.device.fd], [], [], timeout_sec)
            if readable:
                for event in self.device.read():
                    self.state.update(event.type, event.code, event.value)
        except (EOFError, OSError):
            self.state.connected = False
        return self.state.sample()

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.device.path),
            "name": str(self.device.name),
            "phys": str(self.device.phys),
            "uniq": str(self.device.uniq),
        }

    def close(self) -> None:
        self.state.connected = False
        self.device.close()

    def __enter__(self) -> "DualSenseEvdevAdapter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
