from pathlib import Path
from types import SimpleNamespace

from run_button_light_closed_loop import (
    append_journal_checkpoint,
    collect_cleanup_errors,
    flow_status_exit_code,
    journal_phase,
    origin_verification,
    persist_journal_row,
    pointing_command,
    press_attempt_mode,
    validate_production_tactics,
    wait_for_visual_clear,
)
from run_online_replan_no_contact import wait_for_target_after_motion

import numpy as np

import yaml


def test_durable_journal_appends_independent_rows(tmp_path):
    path = tmp_path / "journal.jsonl"
    persist_journal_row(path, {"sequence": 1, "kind": "first"})
    persist_journal_row(path, {"sequence": 2, "kind": "第二"})
    assert path.read_text().splitlines() == [
        '{"sequence":1,"kind":"first"}',
        '{"sequence":2,"kind":"第二"}',
    ]


def test_pointing_phase_is_non_contact_and_holds_waiting_pose():
    args = SimpleNamespace(
        hold_report=Path("hold.yaml"),
        config=Path("config.yaml"),
        target_label="15",
    )
    command = pointing_command(args)
    assert "--pointing-only" in command
    assert "--hold-at-pointing" in command
    assert command[command.index("--press-travel-m") + 1] == "0"
    assert "--visual-safety" in command
    assert "--execute" in command
    assert "--return-without-visual-safety" in command
    assert "--inverse-return" in command


def test_pointing_phase_does_not_inherit_near_panel_coverage_policy():
    args = SimpleNamespace(
        hold_report=Path("hold.yaml"), config=Path("config.yaml"),
        target_label="15", pause_depth_near_button=True,
        depth_degraded_entry_distance_m=0.10,
    )
    command = pointing_command(args)
    assert "--final-depth-degraded-commit" not in command
    assert "--pause-depth-near-button" not in command
    assert "--final-depth-degraded-distance-m" not in command


def test_wait_for_visual_clear_requires_three_consecutive_clear_frames():
    class Vision:
        def __init__(self):
            self.values = iter([
                {"ready": True, "blocked": False, "monitoring_available": True,
                 "frame_sequence": 1},
                {"ready": True, "blocked": True, "monitoring_available": True,
                 "frame_sequence": 2},
                {"ready": True, "blocked": False, "monitoring_available": True,
                 "frame_sequence": 3},
                {"ready": True, "blocked": False, "monitoring_available": True,
                 "frame_sequence": 4},
                {"ready": True, "blocked": False, "monitoring_available": True,
                 "frame_sequence": 5},
            ])
        def request(self, command):
            assert command == "safety"
            return {"visual_safety": next(self.values)}
    assert wait_for_visual_clear(
        Vision(), timeout_sec=1.0, required_clear_frames=3
    )["frame_sequence"] == 5


def test_post_motion_target_acquisition_tolerates_missed_yolo_frames():
    class Vision:
        def __init__(self):
            self.requests = 0

        def request(self, command, **kwargs):
            assert command == "visible_buttons"
            self.requests += 1
            detections = []
            if self.requests == 3:
                detections = [{
                    "label": "8", "confidence": 0.91,
                    "bbox_xyxy": [100, 100, 130, 130],
                }]
            return {"frame_sequence": self.requests, "detections": detections}

    vision = Vision()
    response, target = wait_for_target_after_motion(
        vision, target_label="8", min_confidence=0.70, timeout_sec=0.2
    )
    assert response["frame_sequence"] == 3
    assert target["label"] == "8"
    assert vision.requests == 3


def test_receding_control_is_exclusive_to_initial_approach():
    assert press_attempt_mode(1) == "moving_rgbd_receding_approach"
    assert press_attempt_mode(2) == "current_observation_to_prepress"
    assert press_attempt_mode(3) == "current_observation_to_prepress"


def test_production_tactics_lock_current_strategy():
    config = yaml.safe_load(Path("piper_button_pose.yaml").read_text())
    tactics = validate_production_tactics(config)
    assert tactics == {
        "retry_standoff_m": 0.05,
        "retry_camera_alignment": "optical_axis",
        "press_speed_m_s": 0.01,
        "contact_persistence_sec": 0.80,
        "normal_force_total_floor_n": 0.50,
        "max_forward_travel_m": 0.022,
        "press_segment_max_joint_delta_rad": 0.26,
        "contact_arm_travel_m": 0.008,
        "relative_current_hard_stop_enabled": False,
        "absolute_current_hard_stop_enabled": False,
        "contact_estimator": "cartesian_normal_force_kalman",
        "tangential_force_threshold_n": 0.60,
        "tangential_force_sigma_multiplier": 4.0,
        "tangential_force_persistence_sec": 0.40,
        "post_press_continuous_rgb_monitor": True,
        "post_press_on_confirmation_frames": 2,
        "initial_confirmation_frames": 3,
        "arrival_off_confirmation_frames": 3,
        "waiting_visibility_edge_margin_px": 5.0,
        "post_press_visibility_edge_margin_px": 2.0,
        "post_press_retreat_speed_m_s": 0.25,
        "post_press_retreat_joint_velocity_scale": 0.75,
        "prepress_return_prevalidated": True,
        "prepress_return_speed_m_s": 0.25,
    }


def test_production_tactics_reject_old_250ms_window():
    config = yaml.safe_load(Path("piper_button_pose.yaml").read_text())
    config["guarded_press"]["current_guard"]["persistence_sec"] = 0.25
    try:
        validate_production_tactics(config)
    except RuntimeError as exc:
        assert "contact_persistence_sec" in str(exc)
    else:
        raise AssertionError("obsolete 250 ms tactic was accepted")


def test_flow_failures_have_nonzero_process_exit_code():
    assert flow_status_exit_code("arrived_after_verified_press") == 0
    assert flow_status_exit_code("arrived_after_initial_on") == 0
    assert flow_status_exit_code("button_not_lit_after_max_attempts") == 2
    assert flow_status_exit_code("initial_observation_uncertain") == 2


def test_journal_checkpoint_is_ordered_and_preserves_terminal_status():
    report = {"status": "failed", "lifecycle_status": "failed", "journal": []}
    first = append_journal_checkpoint(
        report, "flow_event", {"event": "at_origin"},
        monotonic_sec=2.0, time_unix_sec=3.0,
    )
    second = append_journal_checkpoint(
        report, "cleanup_finished", {},
        monotonic_sec=4.0, time_unix_sec=5.0,
    )
    assert report["status"] == "failed"
    assert report["lifecycle_status"] == "failed"
    assert [first["sequence"], second["sequence"]] == [1, 2]
    assert first["monotonic_sec"] == 2.0
    assert first["time_unix_sec"] == 3.0
    assert journal_phase("flow_event", {"event": "at_origin"}) == "flow:at_origin"


def test_origin_verification_requires_hardware_ok_even_at_exact_pose():
    state = SimpleNamespace(
        joints=np.zeros(6), raw={"hardware_ok": 0},
    )
    result = origin_verification(state, np.zeros(6))
    assert result["encoder_error_rad"] == 0.0
    assert result["verified"] is False


def test_cleanup_runs_all_callbacks_and_records_each_failure():
    calls = []
    def fail():
        calls.append("fail")
        raise RuntimeError("close failed")
    errors = collect_cleanup_errors([
        ("first", fail),
        ("second", lambda: calls.append("second")),
    ])
    assert calls == ["fail", "second"]
    assert errors == [{
        "component": "first",
        "error": "RuntimeError: close failed",
    }]
