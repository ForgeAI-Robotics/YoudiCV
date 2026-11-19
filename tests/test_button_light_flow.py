from collections import defaultdict, deque

import pytest

from piper_handeye.button_light_flow import ButtonLightFlow, LightState
from piper_handeye.button_light_runtime import ButtonLightRuntime


class FakeHardware:
    def __init__(self, waiting, near=(), waiting_is_origin=False, raw_frames=False):
        # Existing scenarios describe stable states; supply three independent frames per state.
        if not raw_frames:waiting=[v for state in waiting for v in ([state] if state=="UNCERTAIN" else [state]*3)]
        self.waiting_position_is_origin = waiting_is_origin
        self.states = defaultdict(deque, waiting=deque(map(LightState, waiting)), near=deque(map(LightState, near)))
        self.actions = []

    def observe_light(self, position):
        self.actions.append(("observe", position))
        key = "near" if position == "near_confirmation" else position
        return self.states[key].popleft()
    def wait_for_next_observation(self): self.actions.append(("wait",))
    def refresh_button_target(self): self.actions.append(("refresh",))
    def press_button(self, attempt):
        self.actions.append(("press", attempt))
        return {
            "segment": "guarded_press",
            "result": {
                "status": "contact_retracted",
                "reason": "sustained_rolling_current_limit",
                "feedback_contact_confirmed": True,
            },
            "encoder_error_rad": 0.004,
        }
    def move_to_near_observation(self, distance): self.actions.append(("near", distance))
    def move_to_waiting_observation(self): self.actions.append(("waiting",))
    def move_to_origin(self):
        self.actions.append(("origin",))
        return {"verified": True, "hardware_ok": 1, "encoder_error_rad": 0.003}


@pytest.mark.parametrize(
    "waiting,near,expected,attempts",
    [
        (["ON", "ON", "OFF"], [], "arrived_after_initial_on", 0),
        (["OFF", "OFF"], ["ON", "ON"], "arrived_after_verified_press", 1),
        (["OFF", "OFF"], ["OFF", "OFF", "ON", "ON"], "arrived_after_verified_press", 2),
        (["OFF"], ["OFF", "OFF", "OFF", "OFF", "OFF", "OFF"], "button_not_lit_after_max_attempts", 3),
        (["UNCERTAIN", "OFF", "OFF"], ["UNCERTAIN", "ON", "ON"], "arrived_after_verified_press", 1),
    ],
)
def test_flow_paths(waiting, near, expected, attempts):
    hardware = FakeHardware(waiting, near)
    result = ButtonLightFlow(hardware).run()
    assert result.status == expected
    assert result.press_attempts == attempts
    assert hardware.actions[-1] == ("origin",)
    assert sum(action[0] == "refresh" for action in hardware.actions) == attempts


def test_visible_at_origin_returns_from_near_before_waiting():
    hardware = FakeHardware(["OFF", "OFF"], ["ON", "ON"], waiting_is_origin=True)
    result = ButtonLightFlow(hardware).run()
    assert result.status == "arrived_after_verified_press"
    assert ("near", 0.05) in hardware.actions
    waiting_move = hardware.actions.index(("waiting",))
    waiting_observations = [
        index for index, action in enumerate(hardware.actions)
        if action == ("observe", "waiting")
    ]
    # Initial observations happen at origin. Once ON is confirmed near-panel,
    # the next waiting observation must happen only after the return motion.
    assert len(waiting_observations) == 6
    assert waiting_move < waiting_observations[-1]


def test_two_on_frames_captured_during_press_skip_stationary_retry_decision():
    class EarlyOn(FakeHardware):
        def move_to_near_observation(self, distance):
            self.actions.append(("early-monitor", distance))
            return {
                "position": "press_monitor",
                "confirmed_state": "ON",
                "evidence": {"frames": [101, 102]},
            }

    hardware = EarlyOn(["OFF", "OFF"], [])
    result = ButtonLightFlow(hardware).run()
    assert result.status == "arrived_after_verified_press"
    assert result.press_attempts == 1
    assert not any(action[0] == "observe" and action[1] != "waiting"
                   for action in hardware.actions)
    event = next(
        row for row in result.events
        if row["event"] == "post_press_on_captured_early"
    )
    assert event["position"] == "press_monitor"
    assert event["evidence"] == {"frames": [101, 102]}


def test_uncertain_is_bounded_and_never_presses():
    hardware = FakeHardware(["UNCERTAIN"] * 3)
    result = ButtonLightFlow(hardware).run()
    assert result.status == "initial_observation_uncertain"
    assert not any(action[0] == "press" for action in hardware.actions)
    assert hardware.actions[-1] == ("origin",)


def test_uncertain_after_confirmed_on_waits_until_decisive_off():
    hardware = FakeHardware(
        ["ON", "UNCERTAIN", "UNCERTAIN", "UNCERTAIN", "OFF"]
    )
    result = ButtonLightFlow(hardware).run()
    assert result.status == "arrived_after_initial_on"
    assert any(
        event["event"] == "waiting_observation_uncertain"
        for event in result.events
    )
    assert hardware.actions[-1] == ("origin",)


def test_production_adapter_wires_all_boundaries():
    calls = []
    waiting = deque(["OFF"]*6)
    near = deque(["ON", "ON"])
    runtime = ButtonLightRuntime(
        observe=lambda position: (waiting if position == "waiting" else near).popleft(),
        next_frame=lambda: calls.append(("frame",)),
        refresh_target=lambda: calls.append(("refresh",)),
        press=lambda attempt: calls.append(("press", attempt)),
        move_near=lambda distance: calls.append(("near", distance)),
        move_waiting=lambda: calls.append(("waiting",)),
        move_origin=lambda: calls.append(("origin",)),
    )
    result = ButtonLightFlow(runtime).run()
    assert result.status == "arrived_after_verified_press"
    assert calls == [
        ("frame",), ("frame",),
        ("refresh",),
        ("press", 1),
        ("near", 0.05),
        ("waiting",),
        ("frame",), ("frame",), ("frame",),
        ("origin",),
    ]


def test_one_qualified_stationary_on_skips_second_read_and_waits_for_off():
    hardware = FakeHardware(["OFF", "OFF"], ["ON", "OFF"])
    result = ButtonLightFlow(hardware).run()
    assert result.status == "arrived_after_verified_press"
    assert result.press_attempts == 1
    assert ("waiting",) in hardware.actions
    assert ("observe", "near_confirmation") not in hardware.actions
    assert list(hardware.states["near"]) == [LightState.OFF]
    assert hardware.actions[-1] == ("origin",)


def test_off_retry_remains_at_near_holding_position():
    hardware = FakeHardware(
        ["OFF", "OFF"], ["OFF", "OFF", "ON", "ON"],
        waiting_is_origin=False,
    )
    result = ButtonLightFlow(hardware).run()
    assert result.status == "arrived_after_verified_press"
    assert result.press_attempts == 2
    # The only raised-waiting movement is after verified ON, never between
    # the first failed press and the retry.
    waiting_indices = [
        index for index, action in enumerate(hardware.actions)
        if action == ("waiting",)
    ]
    second_press = [
        index for index, action in enumerate(hardware.actions)
        if action == ("press", 2)
    ][0]
    assert waiting_indices == [index for index in waiting_indices if index > second_press]
    assert len(waiting_indices) == 1
    assert any(event["event"] == "retry_held_at_near_observation"
               for event in result.events)


def test_refresh_failure_returns_origin_before_propagating_root_error():
    class FailingRefresh(FakeHardware):
        def refresh_button_target(self):
            self.actions.append(("refresh",))
            raise RuntimeError("fresh target unavailable")
    hardware = FailingRefresh(["OFF"])
    with pytest.raises(RuntimeError, match="fresh target unavailable"):
        ButtonLightFlow(hardware).run()
    assert hardware.actions[-1] == ("origin",)


def test_press_log_distinguishes_attempt_from_feedback_confirmation():
    hardware = FakeHardware(["OFF", "OFF"], ["ON", "ON"])
    result = ButtonLightFlow(hardware).run()
    event = next(row for row in result.events if row["event"] == "press_attempt_finished")
    assert event == {
        "event": "press_attempt_finished",
        "attempt": 1,
        "outcome_reported": True,
        "bridge_status": "contact_retracted",
        "bridge_reason": "sustained_rolling_current_limit",
        "feedback_contact_confirmed": True,
        "encoder_error_rad": 0.004,
    }
    assert not any(row["event"] == "press_completed" for row in result.events)


def test_origin_log_contains_measured_verification():
    result = ButtonLightFlow(FakeHardware(["ON", "OFF"])).run()
    assert result.events[-1] == {
        "event": "at_origin",
        "verification": {
            "verified": True,
            "hardware_ok": 1,
            "encoder_error_rad": 0.003,
        },
    }


def test_single_on_then_single_off_cannot_complete_without_press():
    # Replay the two labels from 17:28:28, followed by unavailable observations.
    hw=FakeHardware(['ON','OFF','UNCERTAIN','UNCERTAIN','UNCERTAIN'],raw_frames=True)
    result=ButtonLightFlow(hw).run()
    assert result.status=='initial_observation_uncertain' and result.press_attempts==0
    assert not any(e['event']=='button_already_on' for e in result.events)


def test_transient_initial_on_followed_by_three_off_frames_still_presses():
    hw=FakeHardware(['ON','OFF','OFF','OFF','OFF','OFF','OFF'],['ON','ON'],raw_frames=True)
    result=ButtonLightFlow(hw).run()
    assert result.status=='arrived_after_verified_press' and result.press_attempts==1
    assert not any(e['event']=='button_already_on' for e in result.events)


def test_uncertain_and_on_reset_arrival_off_streak():
    frames=['ON']*3+['OFF','OFF','UNCERTAIN','OFF','ON','OFF','OFF','OFF']
    hw=FakeHardware(frames,raw_frames=True);result=ButtonLightFlow(hw).run()
    assert result.status=='arrived_after_initial_on'
    assert len([a for a in hw.actions if a==('observe','waiting')])==len(frames)
    assert next(e for e in result.events if e['event']=='button_became_off')['frames']==3


def test_initial_alternating_frames_are_bounded_and_never_count_as_confirmed():
    hw=FakeHardware(['ON','OFF']*6,raw_frames=True)
    assert ButtonLightFlow(hw).run().status=='initial_observation_uncertain'
    assert not any(a[0]=='press' for a in hw.actions)
