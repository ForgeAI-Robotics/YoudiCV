"""Production-facing boundaries for the verified button-light workflow.

The motion and vision implementations are deliberately injected. This lets the
same orchestration run against a virtual scene now and real services later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from piper_handeye.button_light_flow import LightState


@dataclass
class ButtonLightRuntime:
    """Adapter joining a light observer and segmented robot motion callbacks."""

    observe: Callable[[str], str]
    next_frame: Callable[[], None]
    refresh_target: Callable[[], None]
    press: Callable[[int], Any]
    move_near: Callable[[float], str | dict | None]
    move_waiting: Callable[[], dict | None]
    move_origin: Callable[[], dict | None]
    waiting_position_is_origin: bool = False
    initial_recheck: Callable[[], bool] | None = None
    initial_finished: Callable[[], None] | None = None
    post_press_recheck: Callable[[str], dict | None] | None = None
    initial_observation: Callable[[Any], LightState] | None = None
    post_press_opencv: Callable[[str], dict | None] | None = None

    def post_press_opencv_result(self, position: str) -> dict | None:
        return self.post_press_opencv(position) if self.post_press_opencv else None

    def reobserve_post_press(self, position: str) -> dict | None:
        return self.post_press_recheck(position) if self.post_press_recheck else None

    def recheck_initial_observation(self) -> bool:
        return bool(self.initial_recheck and self.initial_recheck())

    def finish_initial_observation(self) -> None:
        if self.initial_finished:
            self.initial_finished()

    def observe_light(self, position: str) -> LightState:
        return LightState(self.observe(position))

    def wait_for_next_observation(self) -> None:
        self.next_frame()

    def refresh_button_target(self) -> None:
        self.refresh_target()

    def press_button(self, attempt: int) -> Any:
        return self.press(attempt)

    def move_to_near_observation(self, panel_distance_m: float) -> str | dict | None:
        return self.move_near(panel_distance_m)

    def move_to_waiting_observation(self) -> dict | None:
        return self.move_waiting()

    def move_to_origin(self) -> dict | None:
        return self.move_origin()
