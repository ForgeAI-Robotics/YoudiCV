import pytest
import time

from piper_handeye.mission_coordinator import MissionCoordinator, MissionProtocolError


def _event(mission, source_seq, phase, *, action_token=None, evidence=None,
           target=None, floor=None):
    event = {
        "schema_version": 1,
        "sequence": source_seq,
        "event_id": f"{phase}-{source_seq}",
        "task_id": f"task-{phase}-{source_seq}",
        "mission_id": mission,
        "phase": phase,
        "evidence": evidence or {},
    }
    if action_token is not None:
        event["permit_token"] = action_token
    if target is not None:
        event["target"] = target
    if floor is not None:
        event["floor"] = floor
    return event


def test_green_lights_only_follow_fresh_interlocks_and_full_sequence():
    manager = MissionCoordinator()
    mission = "mission-test-1"
    manager.start(mission, 16, 1)

    with pytest.raises(MissionProtocolError, match="不满足"):
        manager.request_green_light("arm", "press_call_button")
    nav_call = manager.request_green_light("navigation", "navigate_to_call_panel")
    manager.ingest("navigation", _event(
        mission, 1, "nav_call_panel_arrived", action_token=nav_call["permit_token"]
    ))
    manager.ingest("chassis", _event(mission, 1, "base_hold_confirmed"))
    call_press = manager.request_green_light("arm", "press_call_button")
    manager.ingest("arm", _event(
        mission, 1, "call_button_segment_completed",
        action_token=call_press["permit_token"],
        evidence={"returned_to_origin": True},
        target="down",
    ))

    with pytest.raises(MissionProtocolError, match="不满足"):
        manager.request_green_light("navigation", "board_elevator")
    manager.ingest("elevator", _event(mission, 1, "elevator_door_open"))
    board = manager.request_green_light("chassis", "board_elevator")
    manager.ingest("chassis", _event(
        mission, 2, "boarding_confirmed", action_token=board["permit_token"]
    ))
    # Boarding invalidates the old stopped-at-panel observation.
    with pytest.raises(MissionProtocolError, match="不满足"):
        manager.request_green_light("arm", "press_target_button")
    manager.ingest("chassis", _event(mission, 3, "base_hold_confirmed"))
    target_press = manager.request_green_light("arm", "press_target_button")
    manager.ingest("arm", _event(
        mission, 2, "target_floor_button_segment_completed",
        action_token=target_press["permit_token"],
        evidence={"returned_to_origin": True},
        target=1,
    ))

    with pytest.raises(MissionProtocolError, match="不满足"):
        manager.request_green_light("chassis", "exit_elevator")
    manager.ingest("chassis", _event(mission, 4, "base_hold_confirmed"))
    manager.ingest("elevator", _event(mission, 2, "elevator_door_open"))
    exit_permit = manager.request_green_light("chassis", "exit_elevator")
    manager.ingest("chassis", _event(
        mission, 5, "elevator_exit_confirmed", action_token=exit_permit["permit_token"]
    ))
    with pytest.raises(MissionProtocolError, match="不满足"):
        manager.request_green_light("navigation", "navigate_to_destination")
    manager.ingest("chassis", _event(mission, 6, "base_hold_confirmed"))
    destination = manager.request_green_light("navigation", "navigate_to_destination")
    manager.ingest("navigation", _event(
        mission, 2, "destination_arrived", action_token=destination["permit_token"],
        floor=1,
    ))
    assert manager.snapshot()["phase"] == "COMPLETED"


def test_replay_wrong_source_and_permit_timeout_fail_closed():
    manager = MissionCoordinator(permit_ttl_sec=1.0)
    mission = "mission-test-2"
    manager.start(mission, 1, 3)
    base = time.monotonic()
    nav = manager.request_green_light("navigation", "navigate_to_call_panel", now=base)
    event = _event(mission, 1, "nav_call_panel_arrived", action_token=nav["permit_token"])
    assert manager.ingest("navigation", event)["accepted"] is True
    assert manager.ingest("navigation", event)["duplicate"] is True

    manager = MissionCoordinator(permit_ttl_sec=1.0)
    manager.start(mission, 1, 3)
    with pytest.raises(MissionProtocolError, match="只能由 navigation"):
        manager.ingest("chassis", _event(
            mission, 1, "nav_call_panel_arrived", action_token="forged"
        ))
    timeout_base = time.monotonic()
    manager.request_green_light(
        "navigation", "navigate_to_call_panel", now=timeout_base
    )
    with pytest.raises(MissionProtocolError, match="超时"):
        manager.request_green_light(
            "navigation", "navigate_to_call_panel", now=timeout_base + 1.1
        )
    assert manager.snapshot()["phase"] == "PAUSED"
