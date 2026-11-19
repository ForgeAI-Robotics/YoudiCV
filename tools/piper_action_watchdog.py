"""Bounded actionlib wait used by the OrangePi motion bridges.

This module intentionally has no ROS imports so its safety behaviour can be
tested on the Jetson without a robot or ROS master.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol


class ActionWaitClient(Protocol):
    def wait_for_result(self, timeout: object) -> bool: ...
    def cancel_goal(self) -> None: ...
    def cancel_all_goals(self) -> None: ...


def require_safety_generation(*, safety_generation, safety_latched, initial_generation):
    """A stop during mode handoff/baseline may not become the new baseline."""
    if safety_latched() or int(safety_generation()) != int(initial_generation):
        raise RuntimeError("trajectory cancelled during safety handoff")


def send_goal_with_safety(client, goal, *, safety_generation, safety_latched,
                          initial_generation):
    """Check both sides of send_goal, including stop-before-send cancellation races."""
    values = dict(safety_generation=safety_generation, safety_latched=safety_latched,
                  initial_generation=initial_generation)
    require_safety_generation(**values)
    client.send_goal(goal)
    try:
        require_safety_generation(**values)
    except BaseException:
        client.cancel_all_goals()
        raise


def wait_for_action_result(
    client: ActionWaitClient,
    *,
    safety_generation: Callable[[], int],
    initial_generation: int,
    duration_sec: float,
    duration_factory: Callable[[float], object],
    poll_sec: float = 0.05,
    grace_sec: float = 5.0,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Wait in short slices, with an independent wall-clock deadline.

    ``rospy.Duration`` is appropriate for each short actionlib wait, but must
    not be the only timeout authority: a stalled action callback or ROS clock
    must never leave the client waiting indefinitely after a safety cancel.
    """
    if duration_sec < 0.0 or poll_sec <= 0.0 or grace_sec < 0.0:
        raise ValueError("invalid action wait timing")
    deadline = monotonic() + float(duration_sec) + float(grace_sec)
    while True:
        current_generation = int(safety_generation())
        if current_generation > int(initial_generation):
            client.cancel_all_goals()
            raise RuntimeError(
                "trajectory cancelled by visual safety generation %d"
                % current_generation
            )
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            client.cancel_goal()
            raise RuntimeError("trajectory action timed out")
        if client.wait_for_result(duration_factory(min(float(poll_sec), remaining))):
            return
