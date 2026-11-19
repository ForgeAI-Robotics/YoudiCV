"""Stationary single-ON adoption must not weaken OFF or moving confirmation."""
import pytest
from piper_handeye.button_light_flow import ButtonLightFlow, LightState
from piper_handeye.button_light_hardware import POST_PRESS_ON_CONFIRMATION_FRAMES
from run_button_light_closed_loop import LIGHT_EVIDENCE_POLICY
from tests.test_button_light_flow import FakeHardware


@pytest.mark.parametrize("near", [["ON"], ["UNCERTAIN", "ON"], ["OFF", "ON"],
                                 ["OFF", "UNCERTAIN", "ON"]])
def test_stationary_single_on_starts_off_wait_without_a_second_on(near):
    hw = FakeHardware(["OFF", "OFF"], near)
    result = ButtonLightFlow(hw).run()
    assert result.status == "arrived_after_verified_press"
    assert result.press_attempts == 1
    assert not hw.states["near"]
    assert sum(a[0] == "press" for a in hw.actions) == 1
    verified = next(e for e in result.events if e["event"] == "press_verified_on")
    assert verified["confirmation_source"] == "paired_single_stationary_observation"
    assert next(e for e in result.events if e["event"] == "button_became_off")["frames"] == 3
    assert hw.actions[-1] == ("origin",)


def test_no_second_static_capture_or_wait_is_inserted_after_on():
    hw = FakeHardware(["OFF", "OFF"], ["ON"])
    result = ButtonLightFlow(hw).run()
    after_near = hw.actions.index(("near", .05))
    assert hw.actions[after_near + 1:after_near + 3] == [("observe", "near"), ("waiting",)]
    assert result.press_attempts == 1


def test_one_off_does_not_authorize_another_press():
    hw = FakeHardware(["OFF"], ["OFF", "UNCERTAIN", "UNCERTAIN", "UNCERTAIN"])
    result = ButtonLightFlow(hw).run()
    assert result.status == "post_press_confirmation_uncertain"
    assert result.press_attempts == 1
    assert not any(e["event"] == "retry_required" for e in result.events)
    assert hw.actions[-1] == ("origin",)


def test_two_off_still_authorize_short_retry_and_target_refresh():
    hw = FakeHardware(["OFF", "OFF"], ["OFF", "OFF", "ON"])
    result = ButtonLightFlow(hw).run()
    assert result.status == "arrived_after_verified_press" and result.press_attempts == 2
    assert hw.actions.count(("refresh",)) == 2
    assert hw.actions.count(("waiting",)) == 1
    assert next(e for e in result.events if e["event"] == "retry_required")["attempt"] == 1


def test_waiting_off_must_still_be_contiguous_three_frames():
    hw = FakeHardware(["OFF"] * 3 + ["OFF", "OFF", "UNCERTAIN", "OFF", "OFF", "OFF"],
                      ["ON"], raw_frames=True)
    result = ButtonLightFlow(hw).run()
    assert result.status == "arrived_after_verified_press"
    assert hw.actions.count(("observe", "waiting")) == 9
    assert not hw.states["waiting"]


def test_one_initial_on_cannot_skip_the_press():
    hw = FakeHardware(["ON", "OFF", "OFF", "OFF", "OFF", "OFF", "OFF"],
                      ["ON"], raw_frames=True)
    result = ButtonLightFlow(hw).run()
    assert result.status == "arrived_after_verified_press" and result.press_attempts == 1


def test_origin_failure_after_single_on_is_not_success():
    hw = FakeHardware(["OFF", "OFF"], ["ON"])
    def broken_return(): raise RuntimeError("origin verification failed")
    hw.move_to_origin = broken_return
    with pytest.raises(RuntimeError, match="origin verification failed"):
        ButtonLightFlow(hw).run()


def test_moving_and_stationary_requirements_are_reported_separately():
    assert POST_PRESS_ON_CONFIRMATION_FRAMES == 2
    policy = LIGHT_EVIDENCE_POLICY["post_press_stationary_confirmation"]
    assert policy["paired_on_frames"] == 1
    assert policy["paired_off_observations_before_retry"] == 2
    assert policy["moving_monitor_on_frames"] == 2
