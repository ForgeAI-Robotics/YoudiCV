#!/usr/bin/env python3
"""Run the full button-light workflow against deterministic virtual scenes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque

from piper_handeye.button_light_flow import ButtonFlowConfig, ButtonLightFlow, LightState


SCENARIOS = {
    "initial_on": {"waiting": ["ON", "ON", "OFF"]},
    "press_once": {"waiting": ["OFF", "OFF"], "near": ["ON"]},
    "press_twice": {"waiting": ["OFF", "OFF"], "near": ["OFF", "ON"]},
    "three_failures": {"waiting": ["OFF"], "near": ["OFF", "OFF", "OFF"]},
    "uncertain_then_recovers": {
        "waiting": ["UNCERTAIN", "OFF", "OFF"],
        "near": ["UNCERTAIN", "ON"],
    },
    "visible_at_origin": {"waiting": ["OFF", "OFF"], "near": ["ON"]},
}


class VirtualScene:
    def __init__(self, observations: dict[str, list[str]], *, waiting_is_origin=False):
        self.waiting_position_is_origin = waiting_is_origin
        self.observations = defaultdict(deque)
        for position, states in observations.items():
            frames=[v for value in states for v in ([value] if value=="UNCERTAIN" else [value]* (3 if position=="waiting" else 2))]
            self.observations[position].extend(LightState(value) for value in frames)
        self.actions: list[dict] = []

    def observe_light(self, position):
        if position.endswith("_confirmation"):position=position.removesuffix("_confirmation")
        if not self.observations[position]:
            raise RuntimeError(f"virtual scene has no {position} observation left")
        state = self.observations[position].popleft()
        self.actions.append({"action": "observe", "position": position, "state": state.value})
        return state

    def wait_for_next_observation(self):
        self.actions.append({"action": "wait_frame"})

    def refresh_button_target(self):
        self.actions.append({"action": "refresh_target"})

    def press_button(self, attempt):
        self.actions.append({"action": "press", "attempt": attempt})

    def move_to_near_observation(self, panel_distance_m):
        self.actions.append({"action": "move_near", "distance_m": panel_distance_m})

    def move_to_waiting_observation(self):
        self.actions.append({"action": "move_waiting"})

    def move_to_origin(self):
        self.actions.append({"action": "move_origin"})


def run_scenario(name: str) -> dict:
    scene = VirtualScene(
        SCENARIOS[name], waiting_is_origin=(name == "visible_at_origin")
    )
    result = ButtonLightFlow(scene, ButtonFlowConfig()).run()
    return {
        "scenario": name,
        "status": result.status,
        "press_attempts": result.press_attempts,
        "events": result.events,
        "actions": scene.actions,
        "robot_commands_sent": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=[*SCENARIOS, "all"], default="all", nargs="?")
    parser.add_argument("--output")
    args = parser.parse_args()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    report = {"mode": "virtual_scene", "results": [run_scenario(name) for name in names]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
