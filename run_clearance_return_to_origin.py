#!/usr/bin/env python3
"""Return to the recorded origin using only live MoveIt collision geometry."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import threading
import time

import numpy as np
import yaml

from piper_handeye.retention import prune_old_entries
from piper_handeye.robot import PiperUdpRobot
from piper_handeye.safety_stop import SafetyStopUdpClient, VisualSafetySupervisor
from piper_handeye.timed_trajectory import (
    TimedTrajectoryUdpClient,
    generate_timed_joint_trajectory,
)
from piper_handeye.vision_rpc import VisionRpcClient
from run_direct_return_to_origin import project_recovery_start_into_limits


ROOT = Path(__file__).resolve().parent


def build_direct_return_trajectory(start, origin):
    """Build one direct path; the OrangePi bridge supplies geometry checks."""
    planned_start, projection = project_recovery_start_into_limits(start, origin)
    points, audit = generate_timed_joint_trajectory(
        planned_start,
        origin,
        cartesian_speed_m_s=0.15,
        joint_velocity_scale=0.50,
        point_period_sec=0.10,
        duration_margin=1.02,
    )
    return points, audit, projection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-clear-workspace", action="store_true")
    args = parser.parse_args()
    if args.execute != args.acknowledge_clear_workspace:
        raise RuntimeError("return requires both physical acknowledgements")

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    hold = yaml.safe_load(args.hold_report.resolve().read_text(encoding="utf-8"))
    origin = np.asarray(hold["stage_1_states"][0]["joints_rad"], dtype=float)
    output = ROOT / "clearance_return_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    prune_old_entries(output.parent)
    report = {
        "status": "initializing",
        "returned_to_origin": False,
        "legacy_height_proxy_removed": True,
        "legacy_elbow_swing_limit_removed": True,
        "collision_policy": "moveit_closed_top_chassis_1mm_and_camera_mesh",
    }
    robot = PiperUdpRobot(config["robot"])
    supervisor = None
    pose_stop = threading.Event()
    pose_thread = None
    stop_cfg = config["safety_stop"]
    stop_client = SafetyStopUdpClient(
        stop_cfg["host"], int(stop_cfg["port"]), timeout_sec=2.0
    )
    try:
        robot.open()
        state = robot.state(max_age=2.0)
        if state.raw.get("hardware_ok") != 1:
            raise RuntimeError(f"hardware_ok={state.raw.get('hardware_ok')!r}")
        start = state.joints.copy()
        points, audit, projection = build_direct_return_trajectory(start, origin)

        trajectory_cfg = config["timed_trajectory"]
        token = str(trajectory_cfg["token"])
        client = TimedTrajectoryUdpClient(
            trajectory_cfg["host"], int(trajectory_cfg["port"]), timeout_sec=30.0
        )
        validation = client.execute(
            points,
            token=token,
            task_id=f"moveit-return-validate-{time.time_ns()}",
            dry_run=True,
        )
        report.update(
            validation=validation,
            duration_sec=float(audit.duration_sec),
            start_joints_rad=start.tolist(),
            planned_start_joints_rad=points[0]["positions"],
            start_limit_projection_rad=projection.tolist(),
            origin_joints_rad=origin.tolist(),
        )
        if not args.execute:
            report["status"] = "moveit_validated_no_motion"
            return 0

        # This independent live monitor protects against people and new
        # objects; it is unrelated to the removed fixed-height proxy.
        service = config["vision_service"]
        vision = VisionRpcClient(
            service["host"], int(service["port"]), service["token"], timeout=0.5
        )
        calibration_path = Path(config["calibration"]["handeye_yaml"])
        if not calibration_path.is_absolute():
            calibration_path = args.config.resolve().parent / calibration_path
        handeye = np.asarray(
            yaml.safe_load(calibration_path.read_text())["gripper_to_camera"]
        )
        vision.request("safety_set_depth_degraded_commit", enabled=False)
        vision.request("safety_set_mode", mode="absolute_near_field")
        pose_errors = []

        def publish_pose():
            while not pose_stop.is_set():
                try:
                    sample = robot.state(max_age=0.25)
                    if sample.raw.get("hardware_ok") != 1:
                        raise RuntimeError("hardware not OK during return")
                    vision.request(
                        "safety_camera_pose",
                        timestamp_monotonic=float(sample.timestamp),
                        base_to_camera=(sample.base_to_gripper @ handeye).tolist(),
                    )
                except Exception as exc:
                    pose_errors.append(str(exc))
                    stop_client.stop(
                        token=str(stop_cfg["token"]),
                        reason="return_pose_monitor_failed",
                        frame_sequence=0,
                    )
                    return
                pose_stop.wait(0.02)

        pose_thread = threading.Thread(target=publish_pose, daemon=True)
        pose_thread.start()
        supervisor = VisualSafetySupervisor(
            vision, stop_client, str(stop_cfg["token"]), auto_resume=False
        )
        preflight = supervisor.preflight(3.0)
        fresh = robot.state(max_age=0.25)
        if pose_errors or np.max(np.abs(fresh.joints - start)) > 0.01:
            raise RuntimeError("pose changed or monitoring failed during preparation")
        supervisor.start()
        supervisor.arm()
        result = client.execute(
            points,
            token=token,
            task_id=f"moveit-return-execute-{time.time_ns()}",
        )
        if supervisor.triggered.is_set() or supervisor.stop_active.is_set() or pose_errors:
            raise RuntimeError("guarded return stopped; no automatic retry")
        final = robot.state(max_age=2.0)
        error = float(np.max(np.abs(final.joints - origin)))
        if final.raw.get("hardware_ok") != 1 or error > 0.03:
            raise RuntimeError(f"origin return tracking failed: {error:.4f} rad")
        report.update(
            status="returned_to_origin_moveit_collision_validated",
            returned_to_origin=True,
            contact_attempted=False,
            final_joints_rad=final.joints.tolist(),
            origin_error_rad=error,
            visual_safety_preflight=preflight,
            result=result,
        )
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if args.execute:
            try:
                report["failure_stop"] = stop_client.stop(
                    token=str(stop_cfg["token"]),
                    reason="moveit_guarded_return_failed",
                    frame_sequence=0,
                )
            except Exception as stop_exc:
                report["failure_stop_error"] = str(stop_exc)
        raise
    finally:
        if supervisor is not None:
            supervisor.close()
            report["visual_safety_events"] = supervisor.events
        pose_stop.set()
        if pose_thread is not None:
            pose_thread.join(timeout=2.0)
        robot.close()
        (output / "report.yaml").write_text(
            yaml.safe_dump(report, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), end="")


if __name__ == "__main__":
    raise SystemExit(main())
