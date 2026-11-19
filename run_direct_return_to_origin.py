#!/usr/bin/env python3
"""Return the stopped Piper arm directly to a recorded origin without vision."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import yaml

from piper_handeye.retention import prune_old_entries
from piper_handeye.arm_gamepad_udp import ArmGamepadUdpClient
from piper_handeye.bag_task_timing import timed_stage
from piper_handeye.cycle_recovery import require_unlatched_generation

from piper_handeye.robot import PIPER_JOINT_LIMITS, PiperUdpRobot
from piper_handeye.safety_stop import SafetyStopUdpClient
from piper_handeye.timed_trajectory import (
    PIPER_JOINT_NAMES,
    PIPER_UDP_MAX_PAYLOAD_BYTES,
    PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
    TimedTrajectoryUdpClient,
    encode_trajectory_transport_payload,
    generate_timed_joint_trajectory,
    TrajectoryAudit,
)


ROOT = Path(__file__).resolve().parent
GRIPPER_CLOSED_THRESHOLD_RAD = 0.35
ORIGIN_RETURN_JOINT_VELOCITY_SCALE = 0.25
ORIGIN_RETURN_MINIMUM_DURATION_SEC = 1.5
from piper_handeye.trajectory_transport import MAX_TRAJECTORY_BYTES

ORANGEPI_SSH = (
    "ssh",
    "-i",
    "/home/qianjue/.ssh/orangepi_codex_ed25519",
    "-o",
    "BatchMode=yes",
    "orangepi@192.168.168.120",
)
PIPER_MODE_CTRL_FRAME = "151#0101320000000000"


def build_origin_return_trajectory(
    start_joints, origin_joints, *, speed_m_s=0.08,
    joint_velocity_scale=ORIGIN_RETURN_JOINT_VELOCITY_SCALE,
):
    """Build a deliberately slow direct return to avoid controller path aborts."""
    return generate_timed_joint_trajectory(
        start_joints,
        origin_joints,
        cartesian_speed_m_s=float(speed_m_s),
        joint_velocity_scale=float(joint_velocity_scale),
        point_period_sec=0.05,
        minimum_duration_sec=ORIGIN_RETURN_MINIMUM_DURATION_SEC,
        cartesian_speed_cap_m_s=PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S,
    )


def build_transportable_origin_return_trajectory(
    start_joints, origin_joints, *, requested_speed_m_s=0.08,
    requested_joint_velocity_scale=ORIGIN_RETURN_JOINT_VELOCITY_SCALE,
    token="",
):
    """Choose the slowest safe preset whose full trajectory fits the bounded bulk transport."""
    candidates = [
        (float(requested_speed_m_s), float(requested_joint_velocity_scale)),
        (0.10, 0.35),
        (0.12, 0.40),
        (0.15, 0.50),
    ]
    seen = set()
    last_error = None
    budget = int(MAX_TRAJECTORY_BYTES)
    for speed, joint_scale in candidates:
        key = (speed, joint_scale)
        if key in seen:
            continue
        seen.add(key)
        points, audit = build_origin_return_trajectory(
            start_joints, origin_joints, speed_m_s=speed,
            joint_velocity_scale=joint_scale,
        )
        payload = {
            "protocol": "piper_timed_trajectory_v1",
            "command": "validate",
            "joint_names": PIPER_JOINT_NAMES,
            "token": str(token),
            "task_id": "direct-return-payload-fit-" + ("x" * 48),
            "points": points,
        }
        try:
            encoded, _, _ = encode_trajectory_transport_payload(
                payload, max_bytes=budget
            )
        except ValueError as exc:
            last_error = exc
            continue
        return points, audit, {
            "selected_speed_m_s": speed,
            "selected_joint_velocity_scale": joint_scale,
            "estimated_payload_bytes": len(encoded),
            "payload_budget_bytes": budget,
            "candidate_count": len(seen),
        }
    raise ValueError(
        "no safe direct-return trajectory fits the trajectory transport and safety bounds; "
        + str(last_error)
    )


def build_clearance_return_trajectory(start, origin, lift_rad, *, speed_m_s,
                                      joint_velocity_scale):
    """Stop smoothly at two raised waypoints before turning and lowering.

    This builds candidates only. Complete MoveIt validation is mandatory
    before any segment may be executed.
    """
    start, origin = np.asarray(start, dtype=float), np.asarray(origin, dtype=float)
    raised_start, raised_origin = start.copy(), origin.copy()
    shoulder = max(start[1], origin[1])+float(lift_rad)
    elbow = min(start[2], origin[2])-float(lift_rad)
    for q in (raised_start, raised_origin):
        q[1], q[2] = shoulder, elbow
        if np.any(q < PIPER_JOINT_LIMITS[:, 0]) or np.any(q > PIPER_JOINT_LIMITS[:, 1]):
            raise ValueError('raised return waypoint exceeds joint limits')
    waypoints = [start, raised_start, raised_origin, origin]
    points, audits = [], []
    offset = 0.
    for first, last in zip(waypoints, waypoints[1:]):
        segment, audit = generate_timed_joint_trajectory(
            first, last, cartesian_speed_m_s=float(speed_m_s),
            joint_velocity_scale=float(joint_velocity_scale), point_period_sec=.10,
            minimum_duration_sec=1.5,
            cartesian_speed_cap_m_s=PIPER_RETURN_CARTESIAN_SPEED_CAP_M_S)
        for p in segment if not points else segment[1:]:
            points.append(dict(p, time_from_start_sec=p['time_from_start_sec']+offset))
        offset += audit.duration_sec
        audits.append(audit)
    audit = TrajectoryAudit(
        duration_sec=offset, point_count=len(points),
        cartesian_path_m=sum(a.cartesian_path_m for a in audits),
        cartesian_peak_speed_m_s=max(a.cartesian_peak_speed_m_s for a in audits),
        joint_peak_velocity_rad_s=np.max([a.joint_peak_velocity_rad_s for a in audits], axis=0),
        joint_velocity_fraction=np.max([a.joint_velocity_fraction for a in audits], axis=0))
    return points, audit, [q.tolist() for q in waypoints]


def select_collision_checked_origin_return(start, origin, direct_candidate, validate, *, token):
    """Preserve a safe direct path; otherwise preflight complete raised paths."""
    points, audit, selection = direct_candidate
    rejected = []
    try:
        result = validate(points)
        return points, audit, dict(selection, path_kind='direct'), result
    except RuntimeError as exc:
        if 'MoveIt collision at sample' not in str(exc):
            raise
        rejected.append(dict(path_kind='direct', reason=str(exc)))
    for lift in (.02, .04, .06, .08, .12, .18, .25, .35):
        timing_errors=[]
        for speed,joint_scale in dict.fromkeys([
                (selection['selected_speed_m_s'],selection['selected_joint_velocity_scale']),
                (.10,.35),(.12,.40),(.15,.50)]):
            try:
                candidate, candidate_audit, waypoints = build_clearance_return_trajectory(
                    start, origin, lift, speed_m_s=speed,
                    joint_velocity_scale=joint_scale)
                encoded, _, _ = encode_trajectory_transport_payload(dict(
                    protocol='piper_timed_trajectory_v1', command='validate',
                    joint_names=PIPER_JOINT_NAMES, token=token,
                    task_id='direct-return-payload-fit-'+('x'*48), points=candidate),
                    max_bytes=int(selection['payload_budget_bytes']))
                break
            except (ValueError,RuntimeError) as exc:
                timing_errors.append(str(exc))
        else:
            rejected.append(dict(lift_rad=lift,reason=str(timing_errors)))
            continue
        try:
            result = validate(candidate)
        except RuntimeError as exc:
            if 'MoveIt collision at sample' not in str(exc):
                raise
            rejected.append(dict(lift_rad=lift, reason=str(exc)))
            continue
        return candidate, candidate_audit, dict(
            selection, path_kind='raise_turn_lower', lift_rad=lift,
            selected_speed_m_s=speed, selected_joint_velocity_scale=joint_scale,
            waypoints_rad=waypoints, estimated_payload_bytes=len(encoded),
            rejected_paths=rejected, collision_candidate_count=len(rejected)+1), result
    raise RuntimeError('no fully collision-free origin return path; candidates='+str(rejected))


def enable_arm_while_holding_feedback(robot, live_client, *, max_drift_rad=0.02):
    """Enable the vendor hardware tree and report encoder drift.

    The vendor tree switches back to the trajectory controller, so this must
    run before starting the live-pose position hold used for mode switching.
    """
    before = np.asarray(live_client.live_joints(max_age_sec=0.25), dtype=float)
    robot.enable()
    after = np.asarray(live_client.live_joints(max_age_sec=0.25), dtype=float)
    drift = float(np.max(np.abs(after - before)))
    if drift > float(max_drift_rad):
        raise RuntimeError(
            "arm moved while enabling motors: max encoder drift="
            f"{drift:.6f} rad"
        )
    return {
        "vendor_enable_tree_completed": True,
        "before_joints_rad": before.tolist(),
        "after_joints_rad": after.tolist(),
        "max_enable_drift_rad": drift,
        "max_allowed_drift_rad": float(max_drift_rad),
    }


def prepare_active_return_state(robot, live_client):
    """Activate Piper before the guarded chassis-boundary recovery check.

    The recovery bridge deliberately refuses to operate unless Piper reports
    Active.  Enabling through the vendor tree first is safe only while the
    measured encoder pose is held and its drift is checked immediately.
    """
    hardware_state = robot.state(max_age=2.0)
    hardware_ok = hardware_state.raw.get("hardware_ok")
    if hardware_ok != 1:
        raise RuntimeError(f"hardware_ok={hardware_ok!r}")
    motor_enable = enable_arm_while_holding_feedback(robot, live_client)
    recovery_response = live_client.recover_top_chassis_boundary()
    recovery = recovery_response["status"].get("last_boundary_recovery")
    start_joints = np.asarray(
        live_client.live_joints(max_age_sec=0.25), dtype=float
    )
    return hardware_state, motor_enable, recovery, start_joints


class ReturnSshCommandError(RuntimeError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__("SSH %s failed: rc=%s; %s" % (
            diagnostic['operation'], diagnostic.get('return_code'),
            diagnostic.get('stderr') or diagnostic['failure_kind']))


def _ssh_text(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return str(value or '')[-4096:]


def _run_orangepi_command(command, *, timeout_sec=4.0):
    operation = ('can_mode_change' if command.startswith('cansend ') else
                 'can_mode_feedback' if 'candump ' in command else 'read_only_probe')
    try:
        return subprocess.run(
            [*ORANGEPI_SSH, command],
            check=True, capture_output=True, stdin=subprocess.DEVNULL,
            text=True, timeout=float(timeout_sec),
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        diagnostic = dict(operation=operation, return_code=getattr(exc,'returncode',None),
            stderr=_ssh_text(getattr(exc,'stderr',None)),
            stdout=_ssh_text(getattr(exc,'stdout',None)), timeout_sec=float(timeout_sec),
            failure_kind=('ssh_timeout' if isinstance(exc,subprocess.TimeoutExpired)
                          else 'ssh_launch_error' if isinstance(exc,OSError)
                          else 'ssh_command_exit'), exception_type=type(exc).__name__)
        if isinstance(exc,OSError):diagnostic['stderr']=_ssh_text(exc)
        raise ReturnSshCommandError(diagnostic) from exc


def prepare_return_mode(live_client, bridge_health, *, bridge_handoff=False):
    if not bridge_handoff:
        return set_can_joint_mode_while_holding(live_client)
    if bridge_health.get('status')!='ok' or bridge_health.get('busy'):
        raise RuntimeError('归位轨迹桥未就绪')
    if bridge_health.get('safety_handoff_v2') is not True:
        raise RuntimeError('轨迹桥未具备所需的停止与模式交接保护')
    # The deployed bridge already reconciles ROS controllers, reads fresh CAN
    # feedback, holds the measured pose if necessary, and checks stop generation
    # on both sides of send_goal. It runs this on every execute request.
    return {'control_mode':'bridge_handoff_pending_execution','delegated_to_bridge':True,
            'local_ssh_mode_switch':False}


def set_can_joint_mode_while_holding(live_client, *, max_drift_rad=0.02):
    """Switch Piper to CAN joint mode while ROS holds live encoder positions.

    Uses Piper's documented 0x151 mode-control frame and verifies three fresh
    0x2A1 feedback frames before allowing the trajectory controller to run.
    A zero-velocity heartbeat keeps the position-hold watchdog alive while
    the CAN mode change and feedback verification complete.
    """
    before = np.asarray(live_client.live_joints(max_age_sec=0.25), dtype=float)
    live_client.enable()
    hold_status = live_client.status()["status"]
    held_target = hold_status.get("joint_target_rad")
    if (
        not hold_status.get("active")
        or not isinstance(held_target, list)
        or len(held_target) != 6
        or float(np.max(np.abs(np.asarray(held_target) - before))) > 0.005
    ):
        live_client.disable()
        raise RuntimeError("live-pose hold was not confirmed; mode switch aborted")

    stop = threading.Event()
    heartbeat_errors = []

    def heartbeat():
        while not stop.wait(0.10):
            try:
                live_client.update_joints(velocities_rad_s=(0.0,) * 6)
            except Exception as exc:  # propagate after the bounded mode operation
                heartbeat_errors.append(f"{type(exc).__name__}: {exc}")
                stop.set()

    thread = threading.Thread(target=heartbeat, name="arm-hold-heartbeat", daemon=True)
    thread.start()
    try:
        # ModeCtrl: CAN command mode, joint mode, 50% speed, position-speed
        # mode, no offline dwell, default installation position.
        _run_orangepi_command(f"cansend can0 {PIPER_MODE_CTRL_FRAME}")
        feedback = _run_orangepi_command(
            "timeout 1 candump -L -n 50 can0,2A1:7FF", timeout_sec=4.0
        )
        frames = []
        for line in feedback.splitlines():
            if "2A1#" not in line:
                continue
            payload = line.rsplit("2A1#", 1)[1].strip().split()[0]
            if len(payload) >= 2:
                frames.append(int(payload[:2], 16))
        if len(frames) < 3 or any(mode != 0x01 for mode in frames[-3:]):
            raise RuntimeError(
                f"Piper did not confirm CAN joint mode; 0x2A1 ctrl_mode={frames}"
            )
        after = np.asarray(live_client.live_joints(max_age_sec=0.25), dtype=float)
        drift = float(np.max(np.abs(after - before)))
        if heartbeat_errors:
            raise RuntimeError("position-hold heartbeat failed: " + heartbeat_errors[0])
        if drift > float(max_drift_rad):
            raise RuntimeError(
                "arm moved during mode switch: max encoder drift="
                f"{drift:.6f} rad"
            )
        return {
            "control_mode": "can_joint",
            "control_mode_feedback": frames[-3:],
            "mode_control_can_id": "0x151",
            "mode_control_frame": PIPER_MODE_CTRL_FRAME,
            "hold_controller": "joint_group_position_controller",
            "before_joints_rad": before.tolist(),
            "after_joints_rad": after.tolist(),
            "max_mode_switch_drift_rad": drift,
            "max_allowed_drift_rad": float(max_drift_rad),
        }
    finally:
        stop.set()
        thread.join(timeout=0.5)
        live_client.disable()


def ensure_origin_clamp_closed(arm_output, *, client_factory=ArmGamepadUdpClient,
                               clamp_timeout_sec=None):
    """Put the gripper in the defined origin state and verify live feedback."""
    client = client_factory(
        str(arm_output["host"]),
        int(arm_output["port"]),
        str(arm_output["token"]),
        timeout_sec=5.0,
    )
    try:
        before = client.status()["status"]
        position = before.get("gripper_position")
        if (
            before.get("clamp_open") is False
            and position is not None
            and float(position) < GRIPPER_CLOSED_THRESHOLD_RAD
        ):
            return {"command_sent": False, "status": before}
        response = (client.close_clamp() if clamp_timeout_sec is None
                    else client.close_clamp(timeout_sec=clamp_timeout_sec))
        status = response["status"]
        position = status.get("gripper_position")
        if (
            status.get("clamp_open") is not False
            or position is None
            or float(position) >= GRIPPER_CLOSED_THRESHOLD_RAD
        ):
            raise RuntimeError(
                "origin gripper close was not confirmed by live feedback"
            )
        return {"command_sent": True, "status": status}
    finally:
        client.close()


def project_recovery_start_into_limits(
    measured_joints, target_joints, *, tolerance_rad: float = 0.005
):
    """Project only a small measured boundary overrun onto the legal boundary.

    Encoder quantisation can report a joint a few milliradians beyond a hard
    software limit while the controller is healthy. Recovery may begin there
    only when the target is legal and moves that joint into the allowed range.
    Ordinary target and trajectory validation remains strict.
    """
    measured = np.asarray(measured_joints, dtype=float)
    target = np.asarray(target_joints, dtype=float)
    if measured.shape != (6,) or target.shape != (6,):
        raise ValueError("recovery joints must contain six values")
    if not np.isfinite(measured).all() or not np.isfinite(target).all():
        raise ValueError("recovery joints must be finite")
    lower = PIPER_JOINT_LIMITS[:, 0]
    upper = PIPER_JOINT_LIMITS[:, 1]
    if np.any(target < lower) or np.any(target > upper):
        raise ValueError("recovery target exceeds Piper joint limits")
    below = np.maximum(lower - measured, 0.0)
    above = np.maximum(measured - upper, 0.0)
    overrun = np.maximum(below, above)
    if np.any(overrun > float(tolerance_rad)):
        raise ValueError(
            f"measured recovery start exceeds joint limits by "
            f"{float(np.max(overrun)):.6f} rad"
        )
    projected = np.clip(measured, lower, upper)
    # The legal target plus a straight interpolation from the projected
    # boundary guarantees every commanded point moves within the limit box.
    return projected, measured - projected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument(
        "--gamepad-config", type=Path, default=ROOT / "gamepad_teleop.yaml"
    )
    parser.add_argument("--speed-m-s", type=float, default=0.08)
    parser.add_argument(
        "--joint-velocity-scale", type=float, default=None,
        help="maximum fraction of Piper joint speed limits for the return path",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--bridge-mode-handoff", action="store_true",
                        help="let the existing trajectory bridge verify ROS/CAN handoff before sending the goal")
    parser.add_argument("--acknowledge-clear-workspace", action="store_true")
    parser.add_argument('--require-unlatched-generation', type=int, default=None,
                        help='automatic recovery only: require this unchanged unlatched stop generation; never reset it')
    args = parser.parse_args()
    if not (args.execute and args.acknowledge_clear_workspace):
        raise RuntimeError("return requires both physical acknowledgements")

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    gamepad_config = yaml.safe_load(
        args.gamepad_config.resolve().read_text(encoding="utf-8")
    )
    arm_output = gamepad_config["reserved_ports"]["arm_output"]
    hold = yaml.safe_load(args.hold_report.resolve().read_text(encoding="utf-8"))
    origin = np.asarray(hold["stage_1_states"][0]["joints_rad"], dtype=float)
    trajectory_cfg = config["timed_trajectory"]
    stop_cfg = config["safety_stop"]
    token = str(trajectory_cfg["token"])
    output = ROOT / "direct_return_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    prune_old_entries(output.parent)
    report = {"status": "initializing", "returned_to_origin": False,
              "trajectory_execution_started": False,
              "mode_handoff_policy": "trajectory_bridge" if args.bridge_mode_handoff else "ssh"}
    stage = "initializing"
    robot = PiperUdpRobot(config["robot"])
    live_client = ArmGamepadUdpClient(
        str(arm_output["host"]),
        int(arm_output["port"]),
        str(arm_output["token"]),
        timeout_sec=8.0,
    )
    def check_recovery_permission():
        if args.require_unlatched_generation is not None:
            return require_unlatched_generation(SafetyStopUdpClient(
                stop_cfg['host'], int(stop_cfg['port']), timeout_sec=.5
            ).health(token=str(stop_cfg['token'])), args.require_unlatched_generation)
    try:
        check_recovery_permission()
        robot.open()
        stage = "preparing_active_return_state"
        with timed_stage(report,'active_return_preparation'):
            (hardware_state,motor_enable,recovery,start_joints) = prepare_active_return_state(robot,live_client)
        check_recovery_permission()
        report.update(
            start_joints_rad=start_joints.tolist(),
            origin_joints_rad=origin.tolist(),
            start_hardware_ok=hardware_state.raw.get("hardware_ok"),
            motor_enable=motor_enable,
            joint_state_source="orange_pi_ros_joint_states",
            top_chassis_boundary_recovery=recovery,
        )
        stage = "preparing_trajectory"
        report["start_joints_after_enable_rad"] = start_joints.tolist()
        # Close before moving so the origin path uses the compact gripper
        # geometry.  A failure here is recorded and retried after the arm has
        # safely returned; it must not strand the arm away from origin.
        try:
            with timed_stage(report,'pre_return_clamp'):
                report["pre_return_clamp"] = ensure_origin_clamp_closed(arm_output)
        except Exception as exc:
            report["pre_return_clamp_error"] = f"{type(exc).__name__}: {exc}"
        planned_start, start_projection = project_recovery_start_into_limits(
            start_joints, origin
        )
        report["start_limit_projection_rad"] = start_projection.tolist()
        if args.joint_velocity_scale is None:
            points, audit, trajectory_selection = (
                build_transportable_origin_return_trajectory(
                    planned_start, origin,
                    requested_speed_m_s=float(args.speed_m_s),
                    token=token,
                )
            )
        else:
            points, audit = build_origin_return_trajectory(
                planned_start, origin, speed_m_s=float(args.speed_m_s),
                joint_velocity_scale=float(args.joint_velocity_scale),
            )
            payload = {
                "protocol": "piper_timed_trajectory_v1",
                "command": "validate",
                "joint_names": PIPER_JOINT_NAMES,
                "token": token,
                "task_id": "direct-return-payload-fit-" + ("x" * 48),
                "points": points,
            }
            encoded, _, _ = encode_trajectory_transport_payload(payload)
            trajectory_selection = {
                "selected_speed_m_s": float(args.speed_m_s),
                "selected_joint_velocity_scale": float(args.joint_velocity_scale),
                "estimated_payload_bytes": len(encoded),
                "payload_budget_bytes": MAX_TRAJECTORY_BYTES,
                "candidate_count": 1,
            }
        timed = TimedTrajectoryUdpClient(
            trajectory_cfg["host"], int(trajectory_cfg["port"]), timeout_sec=30.0
        )
        report["trajectory_bridge_health"] = timed.health(token=token)
        with timed_stage(report,'origin_path_preflight'):
            points, audit, trajectory_selection, validation = select_collision_checked_origin_return(
                planned_start, origin, (points, audit, trajectory_selection),
                lambda candidate: timed.execute(
                    candidate, token=token,
                    task_id=f"direct-return-validate-{time.time_ns()}", dry_run=True),
                token=token)
        report["return_trajectory_selection"] = trajectory_selection
        report.update(
            trajectory_duration_sec=audit.duration_sec,
            trajectory_point_count=audit.point_count,
            trajectory_peak_cartesian_speed_m_s=audit.cartesian_peak_speed_m_s,
            trajectory_peak_joint_velocity_rad_s=audit.joint_peak_velocity_rad_s.tolist(),
            return_speed_m_s=trajectory_selection["selected_speed_m_s"],
            return_joint_velocity_scale=(
                trajectory_selection["selected_joint_velocity_scale"]
            ),
        )
        report["validation"] = validation
        # Do not alter control mode until packet sizing and the complete MoveIt
        # collision preflight have passed. The live-pose gate in the trajectory
        # bridge performs and verifies the final ROS/CAN handoff before motion.
        stage = "switching_piper_to_can_joint_mode"
        check_recovery_permission()
        with timed_stage(report,'mode_handoff_preparation'):
            report["mode_switch"] = prepare_return_mode(live_client,report['trajectory_bridge_health'],
                                                        bridge_handoff=args.bridge_mode_handoff)
        start_after_mode = np.asarray(
            live_client.live_joints(max_age_sec=0.25), dtype=float
        )
        report["start_joints_after_mode_switch_rad"] = start_after_mode.tolist()
        if float(np.max(np.abs(start_after_mode - planned_start))) > 0.03:
            raise RuntimeError("arm moved away from the validated return start")
        stage = "resetting_safety_latch"
        if args.require_unlatched_generation is None:
            reset = SafetyStopUdpClient(
                stop_cfg["host"], int(stop_cfg["port"]), timeout_sec=2.0
            ).reset_latch(token=str(stop_cfg["token"]))
        else:
            reset = dict(check_recovery_permission(), reset_performed=False,
                         policy='unchanged_unlatched_generation_only')
        report["safety_reset"] = reset
        stage = "executing_return_trajectory"
        check_recovery_permission()
        with timed_stage(report,'origin_motion',planned_duration_sec=audit.duration_sec):
            report['trajectory_execution_started'] = True
            result = timed.execute(points,token=token,
                                   task_id=f"direct-return-execute-{time.time_ns()}")
        final_joints = np.asarray(
            live_client.live_joints(max_age_sec=0.25), dtype=float
        )
        report["final_joints_rad"] = final_joints.tolist()
        stage = "verifying_return"
        check_recovery_permission()
        final_hardware = robot.state(max_age=2.0)
        error = float(np.max(np.abs(final_joints - origin)))
        if final_hardware.raw.get("hardware_ok") != 1 or error > 0.03:
            raise RuntimeError(f"return tracking failed: {error:.4f} rad")
        origin_clamp = ensure_origin_clamp_closed(arm_output)
        report.update(
            status="returned_to_origin",
            returned_to_origin=True,
            final_hardware_ok=final_hardware.raw.get("hardware_ok"),
            visual_safety_enabled=False,
            final_joints_rad=final_joints.tolist(),
            origin_error_rad=error,
            origin_clamp=origin_clamp,
            duration_sec=audit.duration_sec,
            peak_speed_m_s=audit.cartesian_peak_speed_m_s,
            validation=validation,
            safety_reset=reset,
            result=result,
        )
        return 0
    except Exception as exc:
        report.update(
            status="failed",
            failed_stage=stage,
            error=f"{type(exc).__name__}: {exc}",
        )
        diagnostic = getattr(exc, "diagnostic", None)
        if isinstance(diagnostic, dict):
            report["ssh_failure_detail"] = diagnostic
        # Capture encoder state after an aborted action so a path-tracking
        # failure can be distinguished from a controller or telemetry failure.
        try:
            report["failure_boundary_recovery"] = live_client.status()["status"].get("last_boundary_recovery")
            report["failure_live_joints_rad"] = live_client.live_joints(
                max_age_sec=0.25
            )
            report["failure_hardware_ok"] = robot.state(max_age=2.0).raw.get(
                "hardware_ok"
            )
        except Exception as diagnostic_exc:
            report["failure_state_read_error"] = (
                f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
            )
        raise
    finally:
        robot.close()
        live_client.close()
        (output / "report.yaml").write_text(
            yaml.safe_dump(report, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), end="")
        print(json.dumps({
            "event": "origin_return_result",
            "ok": bool(report.get("returned_to_origin")),
            "at_origin": bool(report.get("returned_to_origin")),
            "hardware_ok": (
                report.get("final_hardware_ok", report.get("start_hardware_ok"))
            ),
            "max_error_rad": report.get("origin_error_rad"),
            "tolerance_rad": 0.03,
            "error": report.get("error"),
            "report_path": str(output / "report.yaml"),
        }, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
