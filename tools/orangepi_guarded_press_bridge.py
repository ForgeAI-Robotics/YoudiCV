#!/usr/bin/env python3
"""OrangePi-local current-guarded Piper press executor.

The forward trajectory endpoint is only a hard geometric bound.  Normal
termination is a sustained motor-current residual followed by an immediate
cancel and a zero-boundary retreat to the prepress pose.
"""

import argparse
import gc
import json
import os
import socket
import struct
import subprocess
import threading
import time

import actionlib
from actionlib_msgs.msg import GoalStatus
import numpy as np
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from controller_manager_msgs.srv import ListControllers, SwitchController, SwitchControllerRequest
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

from orangepi_timed_trajectory_bridge import JOINT_NAMES, validate
from piper_handeye.timed_trajectory import generate_blended_guarded_retract
from orangepi_arm_mode import PIPER_MODE_CTRL_FRAME, activate_actuation_mode
from piper_action_watchdog import wait_for_action_result, require_safety_generation, send_goal_with_safety
from piper_moveit_collision import MoveItCollisionValidator
from piper_handeye.guarded_press import (
    GuardedPressConfig,
    GuardedPressDecision,
    GuardedPressMonitor,
    MotorCurrentSample,
    PiperCurrentFrameAssembler,
    RobustCurrentBaseline,
    generate_bounded_retract_trajectory,
)
from piper_handeye.kinematics import forward_kinematics_batch
from piper_handeye.joint_convergence import wait_for_joint_convergence, JointConvergenceError
from piper_handeye.current_baseline import collect_stable_current_baseline, baseline_audit_summary
from piper_handeye.contact_prevention import CandidateGuardedPressMonitor
from piper_handeye.press_numeric_trace import NumericPressTrace


MAX_VELOCITIES = np.asarray([0.30, 0.45, 0.45, 0.45, 0.45, 0.60])
MAX_ACCELERATIONS = np.full(6, 1.0)
CAN_FRAME = struct.Struct("=IB3x8s")
GUARDED_PRESS_TRACE_DIRECTORY = '/home/orangepi/mobile_arm_demo/guarded_press_traces'
TERMINAL_STATES = {
    GoalStatus.PREEMPTED, GoalStatus.SUCCEEDED, GoalStatus.ABORTED,
    GoalStatus.REJECTED, GoalStatus.RECALLED, GoalStatus.LOST,
}


class CurrentReceiver:
    def __init__(self, interface):
        self.interface = str(interface)
        self.assembler = PiperCurrentFrameAssembler()
        self.condition = threading.Condition()
        self.sequence = 0
        self.latest = None
        self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            sock.bind((self.interface,))
            while not rospy.is_shutdown():
                packet = sock.recv(CAN_FRAME.size)
                can_id, length, payload = CAN_FRAME.unpack(packet)
                can_id &= socket.CAN_EFF_MASK
                if length != 8 or not 0x251 <= can_id <= 0x256:
                    continue
                sample = self.assembler.add(can_id, payload, time.monotonic())
                if sample is not None:
                    with self.condition:
                        self.sequence += 1
                        self.latest = sample
                        self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def wait_next(self, previous_sequence, timeout_sec=0.05):
        deadline = time.monotonic() + timeout_sec
        with self.condition:
            while self.sequence <= previous_sequence and self.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no fresh complete motor-current sample")
                self.condition.wait(remaining)
            if self.error is not None:
                raise RuntimeError("CAN current receiver failed: %s" % self.error)
            return self.sequence, self.latest


class JointFeedbackReceiver:
    """Non-blocking cache of measured encoder state for the 200 Hz guard."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_joints = None
        self.latest_monotonic = None
        self.sequence = 0
        self.subscriber = rospy.Subscriber(
            "/piper_arm/joint_states", JointState, self._update, queue_size=1
        )

    def _update(self, state):
        try:
            values = dict(zip(state.name, state.position))
            joints = np.asarray([values[name] for name in JOINT_NAMES], dtype=float)
            if not np.isfinite(joints).all():
                return
        except (KeyError, TypeError, ValueError):
            return
        with self.lock:
            self.latest_joints = joints
            self.latest_monotonic = time.monotonic()
            self.sequence += 1

    def snapshot(self):
        with self.lock:
            if self.latest_joints is None:
                return None
            return self.sequence, self.latest_monotonic, self.latest_joints.tolist()

    def sample(self, max_age_sec=0.10):
        with self.lock:
            if self.latest_joints is None:
                return None, None
            joints = self.latest_joints.copy()
            age = time.monotonic() - self.latest_monotonic
        if age > max_age_sec:
            return None, float(age)
        return joints, float(age)


class GuardedBridge:
    def __init__(self, args):
        self.args = args
        self.busy = threading.Lock()
        rospy.wait_for_service("/piper_arm/controller_manager/switch_controller", timeout=10.0)
        rospy.wait_for_service("/piper_arm/controller_manager/list_controllers", timeout=10.0)
        self.switch = rospy.ServiceProxy(
            "/piper_arm/controller_manager/switch_controller", SwitchController
        )
        self.list_controllers = rospy.ServiceProxy(
            "/piper_arm/controller_manager/list_controllers", ListControllers
        )
        self.position_command_pub = rospy.Publisher(
            "/piper_arm/joint_group_position_controller/command",
            Float64MultiArray, queue_size=1,
        )
        self.client = actionlib.SimpleActionClient(
            "/piper_arm/joint_trajectory_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError("joint trajectory action server unavailable")
        self.collision = MoveItCollisionValidator(JOINT_NAMES)
        self.current = CurrentReceiver(args.can_interface)
        self.joint_feedback = JointFeedbackReceiver()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((args.bind_ip, args.port))
        self.sock.settimeout(0.5)

    def response(self, address, task_id, status, **extra):
        value = {"task_id": task_id, "status": status}
        value.update(extra)
        self.sock.sendto(json.dumps(value, separators=(",", ":")).encode(), address)

    @staticmethod
    def goal(points, lead_sec=0.15):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(lead_sec)
        for value in points:
            point = JointTrajectoryPoint()
            point.positions = value["positions"]
            point.velocities = value["velocities"]
            point.accelerations = value["accelerations"]
            point.time_from_start = rospy.Duration.from_sec(value["time_from_start_sec"])
            goal.trajectory.points.append(point)
        return goal

    @staticmethod
    def current_joints():
        state = rospy.wait_for_message("/piper_arm/joint_states", JointState, timeout=1.0)
        values = dict(zip(state.name, state.position))
        return np.asarray([values[name] for name in JOINT_NAMES], dtype=float)

    def switch_to_trajectory(self):
        states = {item.name: item.state for item in self.list_controllers().controller}
        if states.get("joint_trajectory_controller") == "running" and states.get("joint_group_position_controller") != "running":
            return
        request = SwitchControllerRequest()
        request.start_controllers = (
            [] if states.get("joint_trajectory_controller") == "running"
            else ["joint_trajectory_controller"]
        )
        request.stop_controllers = ["joint_group_position_controller"] if states.get("joint_group_position_controller") == "running" else []
        request.strictness = SwitchControllerRequest.STRICT
        request.start_asap = True
        request.timeout = 2.0
        if not self.switch(request).ok:
            raise RuntimeError("controller switch failed")
        after = {item.name: item.state for item in self.list_controllers().controller}
        if (
            after.get("joint_trajectory_controller") != "running"
            or after.get("joint_group_position_controller") == "running"
        ):
            raise RuntimeError("trajectory controller handoff failed: %r" % after)

    def controller_states(self):
        return {item.name: item.state for item in self.list_controllers().controller}

    def switch_to_position_hold(self, joints):
        states = self.controller_states()
        trajectory = "joint_trajectory_controller"
        position = "joint_group_position_controller"
        command = Float64MultiArray(data=np.asarray(joints, dtype=float).tolist())
        for _ in range(5):
            self.position_command_pub.publish(command)
            rospy.sleep(0.01)
        if states.get(position) != "running" or states.get(trajectory) == "running":
            request = SwitchControllerRequest()
            request.start_controllers = (
                [] if states.get(position) == "running" else [position]
            )
            request.stop_controllers = (
                [trajectory] if states.get(trajectory) == "running" else []
            )
            request.strictness = SwitchControllerRequest.STRICT
            request.start_asap = True
            request.timeout = 2.0
            if not self.switch(request).ok:
                raise RuntimeError("could not activate position hold: %r" % states)
        for _ in range(5):
            self.position_command_pub.publish(command)
            rospy.sleep(0.01)
        after = self.controller_states()
        if after.get(position) != "running" or after.get(trajectory) == "running":
            raise RuntimeError("position hold handoff failed: %r" % after)

    @staticmethod
    def read_piper_ctrl_modes():
        result = subprocess.run(
            ["timeout", "1", "candump", "-L", "-n", "10", "can0,2A1:7FF"],
            check=True, capture_output=True, text=True, timeout=3.0,
        )
        modes = []
        for line in result.stdout.splitlines():
            if "2A1#" not in line:
                continue
            payload = line.rsplit("2A1#", 1)[1].strip().split()[0]
            if len(payload) >= 2:
                modes.append(int(payload[:2], 16))
        if len(modes) < 3:
            raise RuntimeError("fewer than three fresh Piper status frames received")
        return modes

    def activate_actuation_mode(self, expected_start):
        return activate_actuation_mode(
            np.asarray(expected_start, dtype=float).tolist(),
            controller_states=self.controller_states,
            current_joints=lambda: self.current_joints().tolist(),
            validate_state=lambda joints: self.collision.validate([
                {"positions": list(joints)}, {"positions": list(joints)}
            ]),
            hold_current_pose=self.switch_to_position_hold,
            send_can_joint_mode=lambda: subprocess.run(
                ["cansend", "can0", PIPER_MODE_CTRL_FRAME],
                check=True, capture_output=True, text=True, timeout=2.0,
            ),
            read_mode_feedback=self.read_piper_ctrl_modes,
            switch_to_trajectory=self.switch_to_trajectory,
        )

    def baseline(self, duration_sec=0.30, *, safety_generation, expected_start, audit):
        def check_safety():
            if rospy.is_shutdown():
                raise RuntimeError('ROS shutdown during current baseline collection')
            require_safety_generation(
                safety_generation=lambda: int(rospy.get_param('/piper_visual_safety_stop/generation', 0)),
                safety_latched=lambda: bool(rospy.get_param('/piper_visual_safety_stop/latched', False)),
                initial_generation=safety_generation)
        return collect_stable_current_baseline(
            self.current.wait_next, self.joint_feedback.snapshot, check_safety,
            initial_sequence=self.current.sequence, expected_start=expected_start,
            stable_sec=duration_sec, timeout_sec=5.0, audit=audit)

    @staticmethod
    def parse_config(payload):
        cfg = payload["current_guard"]
        return GuardedPressConfig(
            contact_delta_a=np.asarray(cfg["contact_delta_a"], dtype=float),
            hard_delta_a=np.asarray(cfg["hard_delta_a"], dtype=float),
            absolute_current_limit_a=np.asarray(cfg["absolute_current_limit_a"], dtype=float),
            noise_sigma_multiplier=float(cfg.get("noise_sigma_multiplier", 5.0)),
            hard_noise_sigma_multiplier=float(
                cfg.get("hard_noise_sigma_multiplier", 8.0)
            ),
            median_window=int(cfg.get("median_window", 5)),
            persistence_samples=int(cfg.get("persistence_samples", 5)),
            persistence_sec=(
                None if cfg.get("persistence_sec") is None
                else float(cfg["persistence_sec"])
            ),
            dynamic_reference_window_samples=int(
                cfg.get("dynamic_reference_window_samples", 40)
            ),
            contact_arm_travel_m=float(cfg.get("contact_arm_travel_m", 0.0)),
            max_forward_travel_m=float(payload["max_forward_travel_m"]),
            max_press_time_sec=float(cfg.get("max_press_time_sec", 5.0)),
            max_sample_age_sec=float(cfg.get("max_sample_age_sec", 0.03)),
            contact_joint_indices=tuple(
                int(value) for value in cfg.get("contact_joint_indices", range(6))
            ),
            hard_relative_joint_indices=tuple(
                int(value) for value in cfg.get(
                    "hard_relative_joint_indices", range(6)
                )
            ),
            contact_estimator=str(cfg.get("contact_estimator", "joint_threshold")),
            normal_force_threshold_n=float(cfg.get("normal_force_threshold_n", 0.35)),
            normal_force_total_floor_n=float(cfg.get("normal_force_total_floor_n", 0.0)),
            normal_force_sigma_multiplier=float(
                cfg.get("normal_force_sigma_multiplier", 4.0)
            ),
            normal_force_min_alignment=float(
                cfg.get("normal_force_min_alignment", 0.60)
            ),
            force_process_noise_n2_per_sec=float(
                cfg.get("force_process_noise_n2_per_sec", 80.0)
            ),
            tangential_force_threshold_n=float(
                cfg.get("tangential_force_threshold_n", 0.60)
            ),
            tangential_force_sigma_multiplier=float(
                cfg.get("tangential_force_sigma_multiplier", 4.0)
            ),
            tangential_force_persistence_sec=float(
                cfg.get("tangential_force_persistence_sec", 0.10)
            ),
            relative_current_hard_stop_enabled=bool(
                cfg.get("relative_current_hard_stop_enabled", True)
            ),
            absolute_current_hard_stop_enabled=bool(
                cfg.get("absolute_current_hard_stop_enabled", True)
            ),
        )

    def validate_request(self, payload):
        points = validate({
            "protocol": "piper_timed_trajectory_v1", "command": "validate",
            "joint_names": JOINT_NAMES, "points": payload["forward_points"],
        })
        distance = float(payload["max_forward_travel_m"])
        if not 0.010 <= distance <= 0.025:
            raise ValueError("max forward travel must be in 0.010..0.025 m")
        config = self.parse_config(payload)
        if config.contact_estimator == "cartesian_normal_force_kalman":
            normal = np.asarray(payload.get("contact_normal_base"), dtype=float)
            if normal.shape != (3,) or not np.isfinite(normal).all():
                raise ValueError("Cartesian force estimator requires contact_normal_base")
        collision = self.collision.validate(points)
        retreat_points = payload.get("retreat_to_observation")
        retract_target = payload.get("retract_target_joints")
        if (retreat_points is None) != (retract_target is None):
            raise ValueError("observation retract requires both points and endpoint")
        if retreat_points is not None:
            retreat_points = validate({
                "protocol": "piper_timed_trajectory_v1", "command": "validate",
                "joint_names": JOINT_NAMES, "points": retreat_points,
            })
            retract_target = np.asarray(retract_target, dtype=float)
            if retract_target.shape != (6,) or not np.isfinite(retract_target).all():
                raise ValueError("retract target must contain six finite joints")
            if not np.allclose(retreat_points[0]["positions"], points[0]["positions"], atol=1e-6):
                raise ValueError("observation retreat does not start at prepress")
            if not np.allclose(retreat_points[-1]["positions"], retract_target, atol=1e-6):
                raise ValueError("observation retreat endpoint does not match target")
            self.collision.validate(retreat_points)
            normal = np.asarray(payload.get("contact_normal_base"), dtype=float)
            origin = np.asarray(payload.get("contact_panel_origin_base"), dtype=float)
            if normal.shape != (3,) or origin.shape != (3,) or not np.isfinite(np.r_[normal, origin]).all():
                raise ValueError("blended retreat requires finite panel origin and normal")
        return points, config, collision

    def retract(self, prepress, observation_target=None, retreat_to_observation=None,
                panel_origin=None, panel_normal=None, safety_generation=None):
        if safety_generation is None:
            raise RuntimeError("guarded retract requires the original safety generation")
        current = self.current_joints()
        destination = np.asarray(prepress, dtype=float)
        retract_mode = "bounded_prepress_fallback"
        fallback_reason = None
        collision_audit = None
        points = None
        if observation_target is not None and retreat_to_observation is not None:
            try:
                target = np.asarray(observation_target, dtype=float)
                normal = np.asarray(panel_normal, dtype=float)
                origin = np.asarray(panel_origin, dtype=float)
                normal_norm = float(np.linalg.norm(normal))
                if (target.shape != (6,) or normal.shape != (3,) or origin.shape != (3,)
                        or not np.isfinite(np.r_[target, normal, origin]).all()
                        or normal_norm < 0.9):
                    raise ValueError("invalid near-observation retract geometry")
                normal = normal / normal_norm
                points, audit = generate_blended_guarded_retract(
                    current, prepress, retreat_to_observation,
                    release_speed_m_s=0.25,
                    return_joint_velocity_scale=0.75,
                    point_period_sec=0.02,
                    duration_margin=1.02,
                )
                validate({"protocol": "piper_timed_trajectory_v1", "command": "validate",
                          "joint_names": JOINT_NAMES, "points": points})
                if not np.allclose(points[-1]["positions"], target, atol=1e-6):
                    raise RuntimeError("blended retreat ended at unexpected joints")
                path_joints = np.asarray([point["positions"] for point in points], dtype=float)
                tcp_positions = forward_kinematics_batch(path_joints)[:, :3, 3]
                panel_distances = (tcp_positions - origin) @ normal
                if float(np.min(np.diff(panel_distances))) < -0.002:
                    raise RuntimeError("blended retreat moves toward the panel")
                if float(panel_distances[-1]) < float(panel_distances[0]) + 0.015:
                    raise RuntimeError("blended retreat does not clear the panel")
                if not 0.03 <= float(panel_distances[-1]) <= 0.07:
                    raise RuntimeError("blended retreat misses 5 cm observation band")
                collision_audit = self.collision.validate(points)
                destination = target
                retract_mode = "blended_to_observation"
                blend_audit = {
                    "duration_sec": audit.duration_sec,
                    "point_count": audit.point_count,
                    "cartesian_peak_speed_m_s": audit.cartesian_peak_speed_m_s,
                    "joint_velocity_fraction": audit.joint_velocity_fraction.tolist(),
                    "prepress_joint_speed_rad_s": audit.prepress_joint_speed_rad_s,
                    "prepress_cartesian_speed_m_s": audit.prepress_cartesian_speed_m_s,
                    "inverse_join_index": audit.inverse_join_index,
                }
            except Exception as exc:
                fallback_reason = "%s: %s" % (type(exc).__name__, exc)
                points = None
        if points is None:
            points = generate_bounded_retract_trajectory(
                current, prepress, max_velocities=MAX_VELOCITIES,
                max_accelerations=MAX_ACCELERATIONS,
            )
            validate({"protocol": "piper_timed_trajectory_v1", "command": "validate",
                      "joint_names": JOINT_NAMES, "points": points})
        goal = self.goal(points, lead_sec=0.03)
        send_goal_with_safety(self.client, goal,
            safety_generation=lambda: int(rospy.get_param("/piper_visual_safety_stop/generation", 0)),
            safety_latched=lambda: bool(rospy.get_param("/piper_visual_safety_stop/latched", False)),
            initial_generation=safety_generation)
        duration = float(points[-1]["time_from_start_sec"])
        wait_for_action_result(self.client,
            safety_generation=lambda: int(rospy.get_param("/piper_visual_safety_stop/generation", 0)),
            initial_generation=safety_generation, duration_sec=duration,
            duration_factory=rospy.Duration, grace_sec=2.0)
        result = self.client.get_result()
        if result is None or result.error_code != result.SUCCESSFUL:
            raise RuntimeError("guarded retract trajectory failed")
        # actionlib SUCCESSFUL uses controller tolerances, not our stricter
        # 0.03-rad endpoint gate. Keep holding the same validated destination
        # while distinct fresh measured encoders settle; never resend motion.
        try:
            convergence = wait_for_joint_convergence(
                destination.tolist(), self.joint_feedback.snapshot,
                lambda: require_safety_generation(
                    safety_generation=lambda: int(rospy.get_param('/piper_visual_safety_stop/generation', 0)),
                    safety_latched=lambda: bool(rospy.get_param('/piper_visual_safety_stop/latched', False)),
                    initial_generation=safety_generation))
        except JointConvergenceError as exc:
            exc.audit['safety_generation'] = int(safety_generation)
            raise
        error = convergence['final_error_rad']
        return {
            "duration_sec": duration,
            "end_error_rad": error,
            "destination": retract_mode,
            "fallback_reason": fallback_reason,
            "collision_validation": collision_audit,
            "blend_audit": locals().get("blend_audit"),
            "convergence": convergence,
        }

    def execute(self, points, config, contact_normal_base, task_id,
                observation_target=None, retreat_to_observation=None,
                panel_origin=None):
        trace = []
        failure = None
        result = None
        baseline_audit = None
        self.numeric_contact_trace = (NumericPressTrace()
            if getattr(getattr(self, 'args', None), 'full_rate_trace', False) is True else None)
        directory = GUARDED_PRESS_TRACE_DIRECTORY
        safe_id = ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(task_id))
        path = os.path.join(directory, safe_id + '.jsonl')
        try:
            result = self._execute(points, config, contact_normal_base, task_id,
                observation_target, retreat_to_observation, panel_origin, trace)
            baseline_audit = result.get('baseline_acquisition')
            if baseline_audit is not None:
                result['baseline_acquisition'] = baseline_audit_summary(baseline_audit)
            result.update(full_feedback_trace_path=path, full_feedback_trace_samples=len(trace))
            return result
        except BaseException as exc:
            failure = exc
            baseline_audit = getattr(exc, 'baseline_acquisition_audit', None)
            exc.guarded_press_trace_path = path
            raise
        finally:
            # Keep original failure evidence even if retraction throws. Never
            # let diagnostic IO replace the original motion/safety exception.
            try:
                os.makedirs(directory, exist_ok=True)
                with open(path + '.tmp', 'w') as output:
                    for row in trace:
                        output.write(json.dumps(row, separators=(',', ':')) + '\n')
                os.replace(path + '.tmp', path)
                audit = dict(task_id=str(task_id), full_feedback_trace_path=path,
                    full_feedback_trace_samples=len(trace),
                    error=None if failure is None else '%s: %s' % (type(failure).__name__, failure),
                    baseline_acquisition=baseline_audit,
                    retract_convergence=(result or {}).get('retract_convergence') if failure is None
                        else getattr(failure, 'audit', getattr(failure, 'convergence_audit', None)))
                if self.numeric_contact_trace is not None:
                    numeric_audit = self.numeric_contact_trace.save(path+'.fullrate.npz',
                        task_id=str(task_id), historical_reason=(result or {}).get('reason'),
                        error=None if failure is None else str(failure))
                    audit['full_rate_numeric_trace'] = numeric_audit
                    if result is not None:
                        result['full_rate_numeric_trace'] = numeric_audit
                with open(path + '.retract.json.tmp', 'w') as output:
                    json.dump(audit, output)
                os.replace(path + '.retract.json.tmp', path + '.retract.json')
            except Exception as exc:
                if failure is not None:
                    failure.feedback_trace_write_error = str(exc)
                elif result is not None:
                    result['feedback_trace_write_error'] = str(exc)
                rospy.logerr('guarded press diagnostic write failed: %s', exc)

    def _execute(self, points, config, contact_normal_base, task_id,
                 observation_target, retreat_to_observation, panel_origin, full_feedback_trace):
        generation = int(rospy.get_param("/piper_visual_safety_stop/generation", 0))
        if rospy.get_param("/piper_visual_safety_stop/latched", False):
            raise RuntimeError("visual safety stop is latched")
        prepress = np.asarray(points[0]["positions"], dtype=float)
        if float(np.max(np.abs(self.current_joints() - prepress))) > 0.03:
            raise RuntimeError("guarded press start does not match prepress")
        mode_audit = self.activate_actuation_mode(prepress)
        require_safety_generation(
            safety_generation=lambda: int(rospy.get_param("/piper_visual_safety_stop/generation", 0)),
            safety_latched=lambda: bool(rospy.get_param("/piper_visual_safety_stop/latched", False)),
            initial_generation=generation)
        if float(np.max(np.abs(self.current_joints() - prepress))) > 0.03:
            raise RuntimeError("arm start changed during actuation-mode handoff")
        baseline_audit = {}
        baseline = self.baseline(safety_generation=generation, expected_start=prepress,
                                 audit=baseline_audit)
        candidate = getattr(getattr(self, 'args', None), 'contact_observer', 'candidate') == 'candidate'
        observer_class = CandidateGuardedPressMonitor if candidate else GuardedPressMonitor
        monitor = observer_class(baseline, config, contact_normal_base=contact_normal_base)
        numeric = getattr(self, 'numeric_contact_trace', None)
        if numeric is not None:
            numeric.configure(baseline, config, contact_normal_base,
                              'candidate' if candidate else 'original', monitor=monitor)
        goal = self.goal(points)
        start = goal.trajectory.header.stamp
        duration = float(points[-1]["time_from_start_sec"])
        send_goal_with_safety(self.client, goal,
            safety_generation=lambda: int(rospy.get_param("/piper_visual_safety_stop/generation", 0)),
            safety_latched=lambda: bool(rospy.get_param("/piper_visual_safety_stop/latched", False)),
            initial_generation=generation)
        sequence = self.current.sequence
        final = None
        peak_ratio = np.zeros(6)
        feedback_trace = []
        elapsed = 0.0
        travel = 0.0
        point_times = np.asarray([p["time_from_start_sec"] for p in points], dtype=float)
        point_joints = np.asarray([p["positions"] for p in points], dtype=float)
        point_velocities = np.asarray([p["velocities"] for p in points], dtype=float)
        point_accelerations = np.asarray(
            [p["accelerations"] for p in points], dtype=float
        )
        sample_index = 0
        path_delta = point_joints[-1] - point_joints[0]
        path_norm_sq = float(path_delta @ path_delta)
        if path_norm_sq <= 1e-12:
            raise ValueError("guarded press forward path has no motion")
        while not rospy.is_shutdown():
            if int(rospy.get_param("/piper_visual_safety_stop/generation", 0)) > generation:
                self.client.cancel_all_goals()
                raise RuntimeError("guarded press cancelled by visual safety")
            try:
                sequence, sample = self.current.wait_next(sequence)
            except TimeoutError:
                sample = MotorCurrentSample(time.monotonic() - 1.0, baseline.median_a)
            elapsed = max(0.0, (rospy.Time.now() - start).to_sec())
            desired = np.asarray([
                np.interp(min(elapsed, duration), point_times, point_joints[:, joint])
                for joint in range(6)
            ])
            progress = float(np.clip(
                ((desired - point_joints[0]) @ path_delta) / path_norm_sq,
                0.0, 1.0,
            ))
            travel = config.max_forward_travel_m * progress
            measured_joints, measured_joint_age_sec = self.joint_feedback.sample()
            estimator_joints = desired if measured_joints is None else measured_joints
            decision_now = time.monotonic()
            if candidate:
                decision = monitor.update(sample, now=decision_now, elapsed_sec=elapsed,
                    forward_travel_m=travel, joints_rad=measured_joints,
                    encoder_age_sec=measured_joint_age_sec)
            else:
                decision = monitor.update(sample, now=decision_now, elapsed_sec=elapsed,
                    forward_travel_m=travel, joints_rad=estimator_joints)
            if numeric is not None:
                numeric.record(sample, decision_now, elapsed, travel, measured_joints,
                               measured_joint_age_sec, desired, monitor, decision)
            peak_ratio = np.maximum(peak_ratio, decision.contact_ratio)
            # Record at 20 Hz, while the safety decision itself remains at the
            # full CAN rate.  Building JSON-shaped Python objects at 200 Hz
            # caused a 51 ms GIL/GC pause on RK3588 and made fresh feedback look
            # stale. Full-rate decision scalars remain in the terminal result;
            # the lower-rate vectors are for trajectory identification.
            record_sample = (
                sample_index % 10 == 0
                or decision.decision is not GuardedPressDecision.CONTINUE
            )
            sample_index += 1
            if record_sample:
                desired_velocity = np.asarray([
                    np.interp(min(elapsed, duration), point_times,
                              point_velocities[:, joint])
                    for joint in range(6)
                ])
                desired_acceleration = np.asarray([
                    np.interp(min(elapsed, duration), point_times,
                              point_accelerations[:, joint])
                    for joint in range(6)
                ])
                trace_row = {
                    "elapsed_sec": float(elapsed),
                    "forward_travel_m": float(travel),
                    "sample_timestamp_monotonic": float(sample.timestamp),
                    "raw_current_a": sample.currents_a.tolist(),
                    "filtered_current_a": decision.filtered_current_a.tolist(),
                    "predicted_free_current_a": monitor.relative_reference_a.tolist(),
                    "predicted_free_sigma_a": monitor.relative_sigma_a.tolist(),
                    "current_residual_a": decision.residual_a.tolist(),
                    "desired_joints_rad": desired.tolist(),
                    "measured_joints_rad": (
                        None if measured_joints is None else measured_joints.tolist()
                    ),
                    "measured_joint_age_sec": measured_joint_age_sec,
                    "joint_tracking_error_rad": (
                        None if measured_joints is None
                        else (measured_joints - desired).tolist()
                    ),
                    "desired_joint_velocity_rad_s": desired_velocity.tolist(),
                    "desired_joint_acceleration_rad_s2": desired_acceleration.tolist(),
                    "persistence": int(decision.persistence),
                    "persistence_sec": float(decision.persistence_sec),
                    "decision": decision.decision.value,
                    "reason": decision.reason,
                    "estimated_normal_force_n": decision.estimated_normal_force_n,
                    "normal_force_sigma_n": decision.normal_force_sigma_n,
                    "normal_force_score": decision.normal_force_score,
                    "normal_force_alignment": decision.normal_force_alignment,
                    "estimated_tangential_force_n": (
                        decision.estimated_tangential_force_n
                    ),
                    "tangential_force_sigma_n": decision.tangential_force_sigma_n,
                    "tangential_force_score": decision.tangential_force_score,
                    "estimated_total_force_n": decision.estimated_total_force_n,
                }
                feedback_trace.append({
                    "elapsed_sec": trace_row["elapsed_sec"],
                    "forward_travel_m": trace_row["forward_travel_m"],
                    "persistence_sec": trace_row["persistence_sec"],
                    "decision": trace_row["decision"],
                    "reason": trace_row["reason"],
                    "estimated_normal_force_n": trace_row["estimated_normal_force_n"],
                    "normal_force_sigma_n": trace_row["normal_force_sigma_n"],
                    "normal_force_score": trace_row["normal_force_score"],
                    "normal_force_alignment": trace_row["normal_force_alignment"],
                    "estimated_tangential_force_n": trace_row[
                        "estimated_tangential_force_n"
                    ],
                    "tangential_force_sigma_n": trace_row[
                        "tangential_force_sigma_n"
                    ],
                    "tangential_force_score": trace_row[
                        "tangential_force_score"
                    ],
                    "estimated_total_force_n": trace_row[
                        "estimated_total_force_n"
                    ],
                })
                full_feedback_trace.append(trace_row)
                feedback_trace = feedback_trace[-8:]
            if decision.decision is not GuardedPressDecision.CONTINUE:
                final = decision
                break
            if self.client.get_state() in TERMINAL_STATES:
                final = decision
                break
        self.client.cancel_all_goals()
        rospy.sleep(0.05)
        if final.decision in (
            GuardedPressDecision.CONTACT,
            GuardedPressDecision.CONTINUE,
        ):
            retract_result = self.retract(
                prepress,
                observation_target=observation_target,
                retreat_to_observation=retreat_to_observation,
                panel_origin=panel_origin,
                panel_normal=contact_normal_base,
                safety_generation=generation,
            )
        else:
            retract_result = self.retract(prepress, safety_generation=generation)
        status = "contact_retracted" if final.decision is GuardedPressDecision.CONTACT else "limit_retracted"
        if self.client.get_state() == GoalStatus.SUCCEEDED and final.decision is GuardedPressDecision.CONTINUE:
            status = "maximum_travel_retracted"
        return {
            "status": status,
            "piper_mode": mode_audit,
            "reason": final.reason,
            "feedback_contact_confirmed": bool(
                final.decision is GuardedPressDecision.CONTACT
            ),
            "trigger_joint_index": final.trigger_joint_index,
            "baseline_current_a": baseline.median_a.tolist(),
            "baseline_sigma_a": baseline.sigma_a.tolist(),
            "baseline_acquisition": baseline_audit,
            "contact_observer": 'candidate' if candidate else 'legacy',
            "contact_prevention_audit": monitor.candidate_audit if candidate else None,
            "armed_reference_current_a": monitor.relative_reference_a.tolist(),
            "armed_reference_sigma_a": monitor.relative_sigma_a.tolist(),
            "filtered_current_a": final.filtered_current_a.tolist(),
            "current_residual_a": final.residual_a.tolist(),
            "peak_contact_ratio": peak_ratio.tolist(),
            "contact_estimator": config.contact_estimator,
            "relative_current_hard_stop_enabled": bool(
                config.relative_current_hard_stop_enabled
            ),
            "absolute_current_hard_stop_enabled": bool(
                config.absolute_current_hard_stop_enabled
            ),
            "contact_normal_base": np.asarray(contact_normal_base, dtype=float).tolist(),
            "estimated_normal_force_n": final.estimated_normal_force_n,
            "normal_force_sigma_n": final.normal_force_sigma_n,
            "normal_force_score": final.normal_force_score,
            "normal_force_alignment": final.normal_force_alignment,
            "estimated_tangential_force_n": final.estimated_tangential_force_n,
            "tangential_force_sigma_n": final.tangential_force_sigma_n,
            "tangential_force_score": final.tangential_force_score,
            "estimated_total_force_n": final.estimated_total_force_n,
            "rolling_reference_window_samples": int(
                config.dynamic_reference_window_samples
            ),
            "required_persistence_sec": config.persistence_sec,
            "reference_update_mode": "causal_linear_rolling_frozen_at_arm_boundary",
            "feedback_trace_tail": feedback_trace,
            "trigger_elapsed_sec": float(elapsed),
            "estimated_forward_travel_m": float(travel),
            "retract_duration_sec": retract_result["duration_sec"],
            "retract_end_error_rad": retract_result["end_error_rad"],
            "retract_destination": (
                "observation"
                if retract_result["destination"] == "blended_to_observation"
                else "prepress"
            ),
            "retract_mode": retract_result["destination"],
            "retract_fallback_reason": retract_result["fallback_reason"],
            "retract_collision_validation": retract_result["collision_validation"],
            "retract_blend_audit": retract_result["blend_audit"],
            "retract_convergence": retract_result["convergence"],
        }

    def _process(self, address, task_id, payload):
        try:
            points, config, collision = self.validate_request(payload)
            if payload["command"] == "validate":
                self.response(address, task_id, "validated", collision_validation=collision)
            else:
                # Diagnostic row allocation previously caused an occasional
                # cyclic-GC pause longer than the 30 ms freshness limit.  Keep
                # the bounded motion loop deterministic; collect again after
                # execute() has cancelled/retracted and returned.
                gc.collect()
                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    result = self.execute(
                        points, config, payload.get("contact_normal_base"), task_id,
                        observation_target=payload.get("retract_target_joints"),
                        retreat_to_observation=payload.get("retreat_to_observation"),
                        panel_origin=payload.get("contact_panel_origin_base"),
                    )
                finally:
                    if gc_was_enabled:
                        gc.enable()
                self.response(address, task_id, result.pop("status"),
                              collision_validation=collision, **result)
        except BaseException as exc:
            extra = {}
            if isinstance(exc, JointConvergenceError):
                extra.update(failure_code='GUARDED_RETRACT_CONVERGENCE_FAILED', retract_convergence=exc.audit)
            if hasattr(exc, 'guarded_press_trace_path'):
                extra['full_feedback_trace_path'] = exc.guarded_press_trace_path
            if hasattr(exc, 'feedback_trace_write_error'):
                extra['feedback_trace_write_error'] = exc.feedback_trace_write_error
            if hasattr(exc, 'baseline_acquisition_audit'):
                if isinstance(exc, TimeoutError) and str(exc).startswith('no stable current baseline within'):
                    extra['failure_code'] = 'PRESS_CURRENT_BASELINE_NOT_READY'
                extra['baseline_acquisition'] = baseline_audit_summary(exc.baseline_acquisition_audit)
            self.response(address, task_id, "failed", error="%s: %s" % (type(exc).__name__, exc), **extra)
        finally:
            self.busy.release()

    def spin(self):
        while not rospy.is_shutdown():
            try:
                packet, address = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                payload = json.loads(packet.decode())
                task_id = str(payload.get("task_id", ""))
                if address[0] != self.args.allowed_client or payload.get("token") != self.args.token:
                    continue
                if payload.get("protocol") != "piper_guarded_press_v1" or payload.get("command") not in ("health", "validate", "execute"):
                    raise ValueError("unsupported guarded press request")
                if payload["command"] == "health":
                    self.response(address, task_id, "ok", busy=self.busy.locked(),
                                  current_feedback_available=self.current.latest is not None,
                                  safety_handoff_v2=True, retract_convergence_v1=True,
                                  stable_current_baseline_v1=True, current_baseline_timeout_sec=5.0,
                                  contact_observer=getattr(self.args,'contact_observer','candidate'),
                                  full_rate_trace_enabled=bool(getattr(self.args,'full_rate_trace',False)))
                    continue
                if not self.busy.acquire(False):
                    self.response(address, task_id, "busy")
                    continue
                threading.Thread(target=self._process, args=(address, task_id, payload), daemon=True).start()
            except BaseException as exc:
                self.response(address, locals().get("task_id", ""), "failed",
                              error="%s: %s" % (type(exc).__name__, exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-ip", required=True)
    parser.add_argument("--port", type=int, default=5014)
    parser.add_argument("--allowed-client", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--can-interface", default="can0")
    parser.add_argument('--contact-observer', choices=['legacy','candidate'], default='candidate',
                        help='User-enabled contact prevention; legacy explicitly selects the previous observer')
    parser.add_argument('--full-rate-trace', action='store_true', default=True,
                        help='Preallocated numeric capture; writes only after the press ends')
    parser.add_argument('--no-full-rate-trace', dest='full_rate_trace', action='store_false')
    args = parser.parse_args()
    rospy.init_node("piper_guarded_press_bridge", anonymous=False)
    GuardedBridge(args).spin()


if __name__ == "__main__":
    main()
