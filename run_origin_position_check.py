#!/usr/bin/env python3
"""Read the Piper encoders and compare them with the recorded origin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from piper_handeye.robot import PiperUdpRobot
from piper_handeye.arm_gamepad_udp import ArmGamepadUdpClient


ROOT = Path(__file__).resolve().parent
DEFAULT_HOLD_REPORT = ROOT / "button_fast_chain_runs/20260826_150725/chain_report.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-report", type=Path, default=DEFAULT_HOLD_REPORT)
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument(
        "--gamepad-config", type=Path, default=ROOT / "gamepad_teleop.yaml"
    )
    parser.add_argument("--tolerance-rad", type=float, default=0.03)
    args = parser.parse_args()
    robot = None
    live_client = None
    result = {
        "event": "origin_check_result",
        "ok": False,
        "at_origin": False,
        "tolerance_rad": float(args.tolerance_rad),
    }
    try:
        config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
        gamepad = yaml.safe_load(
            args.gamepad_config.resolve().read_text(encoding="utf-8")
        )
        hold = yaml.safe_load(args.hold_report.resolve().read_text(encoding="utf-8"))
        origin = np.asarray(hold["stage_1_states"][0]["joints_rad"], dtype=float)
        robot = PiperUdpRobot(config["robot"])
        robot.open()
        state = robot.state(max_age=2.0)
        arm_output = gamepad["reserved_ports"]["arm_output"]
        live_client = ArmGamepadUdpClient(
            arm_output["host"],
            int(arm_output["port"]),
            arm_output["token"],
            timeout_sec=2.0,
        )
        current = np.asarray(
            live_client.live_joints(max_age_sec=0.25), dtype=float
        )
        errors = np.abs(current - origin)
        maximum = float(np.max(errors))
        hardware_ok = state.raw.get("hardware_ok")
        result.update({
            "ok": True,
            "at_origin": bool(hardware_ok == 1 and maximum <= args.tolerance_rad),
            "hardware_ok": hardware_ok,
            "max_error_rad": maximum,
            "joint_errors_rad": errors.tolist(),
            "current_joints_rad": current.tolist(),
            "origin_joints_rad": origin.tolist(),
            "joint_state_source": "orange_pi_ros_joint_states",
        })
        return 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        if robot is not None:
            robot.close()
        if live_client is not None:
            live_client.close()
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
