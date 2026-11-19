"""Recorded unlit frames must recover through new stationary evidence only."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import json
import time

import numpy as np
import pytest

from piper_handeye.button_light_flow import ButtonLightFlow
from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.button_light_runtime import ButtonLightRuntime
from piper_handeye.mission_status import normalize_flow_event, describe_phase_zh
from piper_handeye.paired_button_light import (
    PairedButtonLight, compare_patches, extract_patch, _CENTER,
)
from tests.test_button_light_flow import FakeHardware
from tests.test_paired_button_light import patch_image
from tests.test_panel_background_reference import SCENES as LIT_SCENES, snapshot as lit_snapshot


FIXTURES = Path(__file__).parent / "fixtures/post_press_off_recheck"
SCENE = json.loads((FIXTURES / "manifest.json").read_text())


def snapshot(frame, *, sequence=None, captured=None):
    seq = frame["sequence"] if sequence is None else sequence
    return dict(
        fresh=True, path=str(FIXTURES / frame["file"]),
        frame_sequence=seq, image_size=[640, 360],
        capture_time_unix_sec=frame["time"] if captured is None else captured,
        target_detection=dict(frame_sequence=seq, bbox_xyxy=frame["box"],
                              label="up", confidence=frame["confidence"]),
    )


def observer_for(states, *, frames=None, stationary=True):
    pair = PairedButtonLight(require_stationary_reference=stationary,
                             stationary_check=lambda: stationary)
    pair.begin_attempt()
    for index, frame in enumerate(SCENE["frames"]["before_press"]):
        pair.collect(snapshot(frame, captured=time.time() - .4 + index * .1), "OFF")
    assert len(pair.references) == 2
    pair.stationary_reference_ready = stationary
    pair.freeze()
    samples = deque(frames or [SCENE["frames"]["near"][i % 3] for i in range(len(states))])
    recognitions = deque(states)
    index = 0

    def request(*args, **kwargs):
        nonlocal index
        frame = samples.popleft()
        index += 1
        return snapshot(frame, sequence=2460400 + index, captured=time.time() + .001)

    recognizer = SimpleNamespace(observe=lambda *a, **k: {"state": recognitions.popleft()})
    observer = FreshLightObserver(SimpleNamespace(request=request), recognizer, "up", paired_light=pair)
    return observer, samples


def test_recorded_borderline_images_stay_uncertain_without_temporal_recheck():
    observer, _ = observer_for(["OFF"] * 3)
    refs = [(p.sequence, p.gray.copy()) for p in observer.paired_light.references]
    for _ in range(3):
        assert observer.observe("near") == "UNCERTAIN"
        result = observer.audit[-1]["recognition"]["paired_comparison"]
        assert result["off_recheck_candidate"] is True
        assert result["background_error"] < 10
        assert 8 < result["center_error"] <= 15
    for previous, current in zip(refs, observer.paired_light.references):
        assert previous[0] == current.sequence
        np.testing.assert_array_equal(previous[1], current.gray)


def test_three_additional_recorded_frames_authorize_off_without_replacing_reference():
    observer, samples = observer_for(["OFF"] * 3)
    references = list(observer.paired_light.references)
    result = observer.reobserve_post_press("near")
    assert result["confirmed_state"] == "OFF"
    assert result["samples"] == 3 and len(result["frames"]) == 3
    assert len({e["frame_sequence"] for e in result["frames"]}) == 3
    assert all(e["off_recheck_candidate"] for e in result["frames"])
    assert not samples
    assert list(observer.paired_light.references) == references
    assert all(e["state"] == "UNCERTAIN" for e in observer.audit)


def test_recheck_evidence_survives_an_exhausted_motion_monitor_quota(tmp_path):
    observer, _ = observer_for(["OFF"] * 3)
    observer.paired_light.evidence_dir = tmp_path
    observer.paired_light.saved_current = 12
    observer.paired_light.saved_states = {"ON": 2, "OFF": 2}
    result = observer.reobserve_post_press("near")
    assert result["confirmed_state"] == "OFF"
    for frame in result["frames"]:
        assert Path(frame["image"]).parent == tmp_path
        assert Path(frame["image"]).is_file()


@pytest.mark.parametrize("states", [
    ["OFF", "OFF", "UNCERTAIN", "OFF", "OFF", "UNCERTAIN"],
    ["UNCERTAIN"] * 6,
])
def test_two_off_or_ambiguous_absolute_frames_cannot_authorize_retry(states):
    observer, _ = observer_for(states)
    result = observer.reobserve_post_press("near")
    assert result["confirmed_state"] is None
    assert result["samples"] == 6


@pytest.mark.parametrize("position,stationary", [("press_monitor", True), ("prepress", True), ("near", False)])
def test_only_fixed_stationary_observation_can_use_recheck(position, stationary):
    observer, samples = observer_for(["OFF"] * 3, stationary=stationary)
    assert observer.reobserve_post_press(position) is None
    assert len(samples) == 3


def test_possible_local_glow_vetoes_even_a_mild_center_change():
    before, after = patch_image(), patch_image(2)
    after.gray[_CENTER] -= 12
    after.gray[26:34, 26:34] += 32
    result = compare_patches(before, after, "OFF")
    assert result["state"] == "UNCERTAIN"
    assert result["off_recheck_candidate"] is False
    assert result["off_recheck_weak_rise_fraction"] >= .01


@pytest.mark.parametrize("scene", LIT_SCENES, ids=lambda scene: scene["label"])
def test_recorded_lit_buttons_cannot_authorize_off_even_with_wrong_absolute_off(scene):
    before = extract_patch(lit_snapshot(scene["before"][0]))
    for frame in scene["after"]:
        result = compare_patches(before, extract_patch(lit_snapshot(frame)), "OFF")
        assert result["state"] != "OFF"
        assert not result.get("off_recheck_candidate")


@pytest.mark.parametrize("fault", ["small", "shift", "large_change", "absolute_unknown"])
def test_candidate_keeps_resolution_geometry_and_appearance_guards(fault):
    before = extract_patch(snapshot(SCENE["frames"]["before_press"][0]))
    after = extract_patch(snapshot(SCENE["frames"]["near"][0]))
    if fault == "small":
        before.box = after.box = np.array([0, 0, 40, 40])
    elif fault == "shift":
        after.box += np.array([7, 0, 7, 0])
    elif fault == "large_change":
        after.gray[_CENTER] -= 20
    result = compare_patches(before, after, "UNCERTAIN" if fault == "absolute_unknown" else "OFF")
    assert result["state"] == "UNCERTAIN"
    assert not result.get("off_recheck_candidate")


def test_frame_mismatch_breaks_off_streak():
    observer, _ = observer_for(["OFF"] * 6)
    request = observer.vision.request
    count = 0

    def mismatched(*args, **kwargs):
        nonlocal count
        count += 1
        data = request(*args, **kwargs)
        if count in (3, 6):
            data["target_detection"]["frame_sequence"] -= 1
        return data

    observer.vision.request = mismatched
    assert observer.reobserve_post_press("near")["confirmed_state"] is None


def test_recheck_never_counts_the_same_frame_twice():
    observer, _ = observer_for(["OFF"] * 6)
    original = observer.vision.request
    saved = None

    def repeated(*args, **kwargs):
        nonlocal saved
        if saved is None:
            saved = original(*args, **kwargs)
        return saved

    observer.vision.request = repeated
    with pytest.raises(RuntimeError, match="stale"):
        observer.reobserve_post_press("near")


def test_one_qualified_on_finishes_recheck_before_a_later_off_retry():
    observer, _ = observer_for(["OFF"] * 6)
    original = observer.observe
    count = 0

    def observe(*args, **kwargs):
        nonlocal count
        count += 1
        original(*args, **kwargs)
        return "ON" if count == 1 else "OFF"

    observer.observe = observe
    result = observer.reobserve_post_press("near")
    assert result['confirmed_state'] == 'ON'
    assert result['samples'] == 1
    assert result['source'] == 'single_paired_on_stationary_recheck_v2'


def test_two_confirmed_on_during_recheck_finish_without_retry():
    observer, _ = observer_for(["OFF"] * 2)
    original = observer.observe
    observer.observe = lambda *a, **k: (original(*a, **k), "ON")[1]
    assert observer.reobserve_post_press("near")["confirmed_state"] == "ON"


def test_flow_rechecks_then_refreshes_and_retries_from_same_five_cm_pose():
    observer, _ = observer_for(["OFF"] * 3)

    class RecheckHardware(FakeHardware):
        def reobserve_post_press(self, position):
            self.actions.append(("recheck", position))
            return observer.reobserve_post_press(position)

    hardware = RecheckHardware(["OFF", "OFF"], ["UNCERTAIN"] * 3 + ["ON", "ON"])
    result = ButtonLightFlow(hardware).run()
    assert result.status == "arrived_after_verified_press" and result.press_attempts == 2
    assert hardware.actions.count(("refresh",)) == 2
    assert hardware.actions.count(("recheck", "near")) == 1
    assert hardware.actions.count(("waiting",)) == 1
    assert hardware.actions.index(("recheck", "near")) < hardware.actions.index(("press", 2))
    assert any(e["event"] == "retry_held_at_near_observation" for e in result.events)


@pytest.mark.parametrize("near,status", [
    (["ON", "ON"], "arrived_after_verified_press"),
    (["ON", "OFF"], "arrived_after_verified_press"),
])
def test_existing_decisive_paths_do_not_add_recheck(near, status):
    class NoRecheck(FakeHardware):
        def reobserve_post_press(self, position):
            raise AssertionError("normal or conflicting evidence must keep the old path")

    result = ButtonLightFlow(NoRecheck(["OFF", "OFF"], near)).run()
    assert result.status == status


def test_an_on_before_uncertainty_cannot_be_overridden_by_off_recheck():
    class NoOffRescue(FakeHardware):
        def reobserve_post_press(self, position):
            raise AssertionError("a previously seen paired ON must veto an OFF rescue")

    hardware = NoOffRescue(["OFF", "OFF"], ["ON"] + ["UNCERTAIN"] * 3)
    result = ButtonLightFlow(hardware).run()
    assert result.status == "arrived_after_verified_press"
    assert ("observe", "near_confirmation") not in hardware.actions
    assert result.press_attempts == 1
    assert hardware.actions[-1] == ("origin",)


def test_three_attempt_limit_is_preserved_when_recheck_confirms_off():
    class OffRecheck(FakeHardware):
        def reobserve_post_press(self, position):
            return {"confirmed_state": "OFF", "frames": [1, 2, 3]}

    result = ButtonLightFlow(OffRecheck(["OFF"], ["UNCERTAIN"] * 9)).run()
    assert result.status == "button_not_lit_after_max_attempts"
    assert result.press_attempts == 3


def test_unresolved_recheck_returns_origin_without_another_press():
    class UnknownRecheck(FakeHardware):
        def reobserve_post_press(self, position):
            return {"confirmed_state": None}

    hardware = UnknownRecheck(["OFF"], ["UNCERTAIN"] * 3)
    result = ButtonLightFlow(hardware).run()
    assert result.status == "post_press_confirmation_uncertain" and result.press_attempts == 1
    assert hardware.actions[-1] == ("origin",)


def test_runtime_routes_optional_recheck_to_observer():
    runtime = ButtonLightRuntime(
        observe=lambda p: "OFF", next_frame=lambda: None, refresh_target=lambda: None,
        press=lambda n: None, move_near=lambda d: "near", move_waiting=lambda: None,
        move_origin=lambda: None, post_press_recheck=lambda p: {"confirmed_state": "OFF", "position": p},
    )
    assert runtime.reobserve_post_press("near") == {"confirmed_state": "OFF", "position": "near"}


def test_tangential_protection_has_its_own_status_without_claiming_success():
    class Tangential(FakeHardware):
        def press_button(self, attempt):
            result = super().press_button(attempt)
            result["result"]["reason"] = "sustained_cartesian_tangential_force_kalman"
            return result

    result = ButtonLightFlow(Tangential(["OFF", "OFF"], ["ON", "ON"])).run()
    event = next(e for e in result.events if e["event"] == "press_tangential_protection")
    assert event["light_confirmation_required"] is True
    assert "侧向力保护" in describe_phase_zh(normalize_flow_event(event))
