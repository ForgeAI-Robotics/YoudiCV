#!/usr/bin/env python3
"""Prepare the closed-loop segmented motion from a verified historical run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from piper_handeye.button_light_hardware import DryValidatedSegmentExecutor
from piper_handeye.realtime_planner import FixedRollFastIkPlanner
from piper_handeye.segmented_button_motion import plan_segmented_press_attempt
from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient
from run_online_replan_no_contact import solve_independent_press_ik


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-report",
        type=Path,
        default=ROOT / "online_replan_no_contact_runs/20260908_131736/online_replan_report.yaml",
    )
    parser.add_argument(
        "--hold-report",
        type=Path,
        default=ROOT / "button_fast_chain_runs/20260826_150725/chain_report.yaml",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--bridge-validate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = yaml.safe_load(args.run_report.read_text())
    hold = yaml.safe_load(args.hold_report.read_text())
    config = yaml.safe_load(args.config.read_text())
    accepted = [row for row in run["online_samples"] if row.get("accepted")]
    if len(accepted) < 6:
        raise RuntimeError("verified report does not contain six accepted target samples")
    button = np.asarray(hold["stage_1"]["base_to_button"], dtype=float)
    button[:3, 3] = np.mean(
        [row["base_to_button_position_m"] for row in accepted[-6:]], axis=0
    )
    # This installation defines panel distance along negative base X.
    distance_frame = button.copy()
    distance_frame[:3, 2] = [-1.0, 0.0, 0.0]
    waiting = np.asarray(run["auto_pointing"]["pointing_joints_rad"], dtype=float)
    prepress_target = button.copy()
    prepress_target[0, 3] -= float(run["tcp_prepress_distance_m"])
    roll_deg = float(hold["stage_1"]["selected_approach_roll_deg"])
    prepress_ik = FixedRollFastIkPlanner(
        roll_deg=roll_deg,
        stand_off_m=0.0,
        max_joint_delta_rad=1.0,
        max_perpendicular_deviation_deg=35.0,
        downstream_base_x_distances_m=(0.01, -0.005),
    ).solve(prepress_target, waiting)
    press_target = button.copy()
    press_target[0, 3] -= -0.005
    press_ik = solve_independent_press_ik(
        prepress_ik.joints, press_target[:3, 3]
    )
    plan = plan_segmented_press_attempt(
        waiting,
        waiting,
        prepress_ik.joints,
        press_ik.joints,
        distance_frame,
        observation_distance_m=0.15,
        approach_speed_m_s=0.10,
        press_speed_m_s=0.01,
        retreat_speed_m_s=0.15,
        return_speed_m_s=0.20,
    )
    validation = None
    if args.bridge_validate:
        trajectory = config["timed_trajectory"]
        executor = DryValidatedSegmentExecutor(
            TimedTrajectoryUdpClient(
                trajectory["host"], int(trajectory["port"]), timeout_sec=30.0
            ),
            robot=None,
            token=trajectory["token"],
            execute_enabled=False,
        )
        executor.load_plan(plan)
        validation = executor.validations
    report = {
        "source_run": str(args.run_report),
        "target_label": str(run["target_label"]),
        "historical_geometry_only": True,
        "robot_commands_sent": False,
        "bridge_dry_validated": bool(args.bridge_validate),
        "observation_distance_m": plan.observation_distance_m,
        "ik": {
            "prepress_solve_ms": prepress_ik.solve_sec * 1000.0,
            "press_solve_ms": press_ik.solve_sec * 1000.0,
            "prepress_position_error_m": prepress_ik.position_error_m,
            "press_position_error_m": press_ik.position_error_m,
        },
        "segments": {
            "approach": plan.approach_audit.duration_sec,
            "press_cycle": plan.press_cycle_audit.duration_sec,
            "retreat_to_observation": plan.retreat_audit.duration_sec,
            "observation_to_waiting": plan.waiting_return_audit.duration_sec,
        },
        "bridge_validation": validation,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
