#!/usr/bin/env python3
"""Recover from a latched near-button stop by moving monotonically away."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import numpy as np
import yaml

from piper_handeye.kinematics import forward_kinematics_batch
from piper_handeye.robot import PiperUdpRobot
from piper_handeye.safety_stop import SafetyStopUdpClient
from piper_handeye.trajectory_splice import PreparedSpliceUdpClient
from piper_handeye.timed_trajectory import (
    TimedTrajectoryUdpClient,
    generate_timed_joint_trajectory,
)


ROOT = Path(__file__).resolve().parent
WAITING_JOINTS = np.asarray([0.04, 0.536, -0.812, 0.132, 0.201, -0.075])
BUTTON_X_M = 0.5296723678940777


def make_segment(start, end, speed):
    return generate_timed_joint_trajectory(
        start,
        end,
        cartesian_speed_m_s=speed,
        joint_velocity_scale=0.50,
        point_period_sec=0.10,
        duration_margin=1.02,
    )


def panel_audit(points):
    joints = np.asarray([point["positions"] for point in points], dtype=float)
    xyz = forward_kinematics_batch(joints)[:, :3, 3]
    distance = BUTTON_X_M - xyz[:, 0]
    return {
        "start_m": float(distance[0]),
        "end_m": float(distance[-1]),
        "minimum_step_m": float(np.min(np.diff(distance))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--hold-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-clear-workspace", action="store_true")
    args = parser.parse_args()
    if args.execute != args.acknowledge_clear_workspace:
        raise RuntimeError("execution requires both physical acknowledgements")

    cfg = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    hold = yaml.safe_load(args.hold_report.resolve().read_text(encoding="utf-8"))
    origin = np.asarray(hold["stage_1_states"][0]["joints_rad"], dtype=float)
    out_dir = ROOT / "button_recovery_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    report_path = out_dir / "recovery_report.yaml"
    report = {"status": "initializing", "returned_to_origin": False}

    robot = PiperUdpRobot(cfg["robot"])
    stop_cfg = cfg["safety_stop"]
    stop = SafetyStopUdpClient(stop_cfg["host"], int(stop_cfg["port"]), timeout_sec=2.0)
    trajectory_cfg = cfg["timed_trajectory"]
    trajectory = TimedTrajectoryUdpClient(
        trajectory_cfg["host"], int(trajectory_cfg["port"]), timeout_sec=30.0
    )
    splice = PreparedSpliceUdpClient(
        trajectory_cfg["host"], 5011, timeout_sec=5.0
    )
    stop_token = str(stop_cfg["token"])
    trajectory_token = str(trajectory_cfg["token"])
    monitor_stop = threading.Event()
    monitor_errors = []
    monitor_thread = None

    try:
        report["trajectory_bridge_health"] = trajectory.health(token=trajectory_token)
        robot.open()
        initial = robot.state(max_age=2.0)
        if initial.raw.get("hardware_ok") != 1:
            raise RuntimeError("robot hardware is not OK")
        segments = []
        for name, first, last, speed in (
            ("retreat_to_waiting", initial.joints, WAITING_JOINTS, 0.10),
            ("waiting_to_origin", WAITING_JOINTS, origin, 0.12),
        ):
            points, audit = make_segment(first, last, speed)
            distance = panel_audit(points)
            if distance["minimum_step_m"] < -1e-6:
                raise RuntimeError(f"{name} is not monotonic away from the panel")
            validation = trajectory.execute(
                points,
                token=trajectory_token,
                task_id=f"button-recovery-{name}-validate-{time.time_ns()}",
                dry_run=True,
            )
            segments.append((name, points, last, audit, distance, validation))
        report.update(
            status="dry_run_validated_no_motion",
            initial_joints_rad=initial.joints.tolist(),
            waiting_joints_rad=WAITING_JOINTS.tolist(),
            origin_joints_rad=origin.tolist(),
            segments=[
                {
                    "name": name,
                    "duration_sec": float(audit.duration_sec),
                    "panel_distance": distance,
                    "validation": validation,
                }
                for name, _points, _last, audit, distance, validation in segments
            ],
        )
        if not args.execute:
            return 0

        health = stop.health(token=stop_token)
        if not health.get("latched"):
            raise RuntimeError("expected recovery to begin behind the safety latch")

        # A visual stop cancels the ROS goal but the prepared-splice bridge
        # intentionally caches its remaining path. Clearing the latch without
        # discarding that path makes it resume concurrently with recovery.
        splice_status = splice.execution_status(token=trajectory_token)
        report["prepared_splice_before_recovery"] = splice_status
        if splice_status.get("execution_status") not in {
            "idle", "replacement_complete", "fallback_complete", "failed", "aborted"
        }:
            abort = splice.abort_active(
                token=trajectory_token,
                active_id=splice_status.get("active_id"),
            )
            if abort.get("execution_status") != "aborted":
                raise RuntimeError(f"prepared trajectory was not aborted: {abort}")
            report["prepared_splice_abort"] = abort

        def monitor():
            while not monitor_stop.is_set():
                try:
                    sample = robot.state(max_age=0.25)
                    if sample.raw.get("hardware_ok") != 1:
                        raise RuntimeError("hardware_ok changed during recovery")
                except Exception as exc:
                    monitor_errors.append(f"{type(exc).__name__}: {exc}")
                    try:
                        stop.stop(token=stop_token, reason="recovery_encoder_monitor_failed", frame_sequence=0)
                    finally:
                        return
                monitor_stop.wait(0.02)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        before_unlock = robot.state(max_age=0.25)
        if np.max(np.abs(before_unlock.joints - initial.joints)) > 0.01:
            raise RuntimeError("robot moved between validation and latch reset")
        stop.reset_latch(token=stop_token)

        executions = []
        for name, points, target, _audit, _distance, _validation in segments:
            if monitor_errors:
                raise RuntimeError(monitor_errors[-1])
            before = robot.state(max_age=0.25)
            start_error = float(
                np.max(np.abs(before.joints - np.asarray(points[0]["positions"])))
            )
            if start_error > 0.03:
                raise RuntimeError(f"{name} start error {start_error:.4f} rad")
            result = trajectory.execute(
                points,
                token=trajectory_token,
                task_id=f"button-recovery-{name}-execute-{time.time_ns()}",
            )
            after = robot.state(max_age=2.0)
            end_error = float(np.max(np.abs(after.joints - target)))
            if monitor_errors or after.raw.get("hardware_ok") != 1 or end_error > 0.03:
                raise RuntimeError(f"{name} end check failed: {end_error:.4f} rad")
            executions.append(
                {"name": name, "start_error_rad": start_error,
                 "end_error_rad": end_error, "result": result}
            )

        final = robot.state(max_age=2.0)
        origin_error = float(np.max(np.abs(final.joints - origin)))
        report.update(
            status="returned_to_origin",
            returned_to_origin=True,
            depth_safety_policy=(
                "paused_for_prevalidated_monotonic_retreat_from_occluded_panel"
            ),
            executions=executions,
            final_joints_rad=final.joints.tolist(),
            origin_error_rad=origin_error,
        )
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if args.execute:
            try:
                report["failure_stop"] = stop.stop(
                    token=stop_token,
                    reason="button_recovery_failed",
                    frame_sequence=0,
                )
            except Exception as stop_exc:
                report["failure_stop_error"] = f"{type(stop_exc).__name__}: {stop_exc}"
        raise
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)
        robot.close()
        report_path.write_text(
            yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        print(json.dumps({"report": str(report_path), **report}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
