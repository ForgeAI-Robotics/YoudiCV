#!/usr/bin/env python3
"""Watchdog-guarded, MoveIt-checked single-joint DualSense teleoperation."""

import argparse
import json
import socket
import subprocess
import threading
import time

import actionlib
from actionlib_msgs.msg import GoalStatus, GoalStatusArray
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, ListControllers
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint
from piper_msgs.msg import PiperState
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest

from piper_moveit_collision import MoveItCollisionValidator
from gamepad_joint_safety import allowed_by_soft_limit
from piper_escape_geometry import EscapeGeometry
from orangepi_arm_mode import confirm_escape_actuation_mode, PIPER_MODE_CTRL_FRAME, EscapeModeDriftError
from top_chassis_recovery import (
    RECOVERY_COMMAND_PERIOD_SEC,
    interpolate_j2_j3_escape,
    select_geometric_j2_j3_escape_target,
    require_expected_contacts,
    geometry_score,
    ESCAPE_REQUIRED_CLEARANCE_M,
    validate_escape_transition_pose,
)


ACTIVE_GOAL_STATES = {
    GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.PREEMPTING, GoalStatus.RECALLING
}

ARM_JOINT_NAMES = ["piper_joint%d" % index for index in range(1, 7)]
JOINT_LIMITS = [
    (-2.618, 2.618), (0.0, 3.14), (-2.967, 0.03490658503988659),
    (-1.745, 1.745), (-1.22, 1.22), (-2.0944, 2.0944),
]
GRIPPER_CLOSED_RAD = 0.0
GRIPPER_OPEN_RAD = 1.396
GRIPPER_STATE_THRESHOLD_RAD = 0.35
GRIPPER_MIN_CONFIRMED_TRAVEL_RAD = 0.08


def _bounded(value, limit):
    return max(-limit, min(limit, float(value)))


class GamepadServoBridge:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.active = False
        self.last_packet = 0.0
        self.last_update = time.monotonic()
        self.joint_velocities = [0.0] * 6
        self.joint_target = None
        self.actual_joints = None
        self.joint_feedback_monotonic = None
        self.blocked_reason = None
        self.blocked_detail = None
        self.last_collision_check_sec = None
        self.sequence = -1
        self.clamp_open = True
        self.gripper_position = None
        self.gripper_feedback_monotonic = None
        self.last_gripper_target = None
        self.gripper_command_active = False
        self.other_motion_active = False
        self.trajectory_motion_active = False
        self.foreign_qp_active = False
        self.fault_reason = None
        self.last_boundary_recovery = None
        self.escape_geometry = EscapeGeometry()

        self.position_command_pub = rospy.Publisher(
            "/piper_arm/joint_group_position_controller/command",
            Float64MultiArray,
            queue_size=1,
        )
        self.collision = MoveItCollisionValidator(
            ARM_JOINT_NAMES, max_step_rad=args.collision_max_step
        )
        self.switch = rospy.ServiceProxy(
            "/piper_arm/controller_manager/switch_controller", SwitchController
        )
        self.gripper = actionlib.SimpleActionClient(
            "/piper_arm/piper_gripper_trajectory_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        rospy.Subscriber(
            "/execute_trajectory/status", GoalStatusArray, self._motion_status,
            queue_size=1,
        )
        rospy.Subscriber(
            "/piper_qp_servo/piper_servo/status", GoalStatusArray,
            self._foreign_qp_status, queue_size=1,
        )
        rospy.Subscriber(
            "/piper_arm/joint_states", JointState, self._joint_state,
            queue_size=1,
        )
        self.timer = rospy.Timer(rospy.Duration(1.0 / args.joint_update_hz), self._tick)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((args.bind_ip, args.port))
        self.sock.settimeout(0.1)

    def _motion_status(self, message):
        self.trajectory_motion_active = any(
            item.status in ACTIVE_GOAL_STATES for item in message.status_list
        )
        self._update_other_motion_active()

    def _foreign_qp_status(self, message):
        self.foreign_qp_active = any(
            item.status in ACTIVE_GOAL_STATES for item in message.status_list
        )
        self._update_other_motion_active()

    def _update_other_motion_active(self):
        self.other_motion_active = bool(
            self.trajectory_motion_active or self.foreign_qp_active
        )

    def _joint_state(self, message):
        values = dict(zip(message.name, message.position))
        if "piper_gripper_motor_joint" in values:
            self.gripper_position = float(values["piper_gripper_motor_joint"])
            self.clamp_open = (
                self.gripper_position >= GRIPPER_STATE_THRESHOLD_RAD
            )
            self.gripper_feedback_monotonic = time.monotonic()
        if all(name in values for name in ARM_JOINT_NAMES):
            self.actual_joints = [float(values[name]) for name in ARM_JOINT_NAMES]
            self.joint_feedback_monotonic = time.monotonic()

    def _switch_controller(self, position_mode):
        request = SwitchControllerRequest()
        request.start_controllers = [
            "joint_group_position_controller" if position_mode
            else "joint_trajectory_controller"
        ]
        request.stop_controllers = [
            "joint_trajectory_controller" if position_mode
            else "joint_group_position_controller"
        ]
        request.strictness = SwitchControllerRequest.STRICT
        request.start_asap = True
        request.timeout = 2.0
        response = self.switch(request)
        if not response.ok:
            raise RuntimeError("controller switch failed")

    @staticmethod
    def _current_joints():
        state = rospy.wait_for_message(
            "/piper_arm/joint_states", JointState, timeout=1.0
        )
        values = dict(zip(state.name, state.position))
        if not all(name in values for name in ARM_JOINT_NAMES):
            raise RuntimeError("joint state does not contain all Piper arm joints")
        return [float(values[name]) for name in ARM_JOINT_NAMES]

    def _seed_position_controller(self, joints, cycles=5):
        command = Float64MultiArray(data=list(joints))
        for _ in range(cycles):
            self.position_command_pub.publish(command)
            rospy.sleep(0.01)

    def enable(self):
        with self.lock:
            if self.active:
                return
            if self.other_motion_active:
                raise RuntimeError("another MoveIt trajectory is active")
            rospy.wait_for_service(
                "/piper_arm/controller_manager/switch_controller", timeout=2.0
            )
            current_joints = self._current_joints()
            # Fail closed before taking control.  This also applies and verifies
            # the 5 mm top-chassis padding in the shared MoveIt scene.
            self.collision.validate([
                {"positions": current_joints}, {"positions": current_joints}
            ])
            # Seed the absolute position command from live joint feedback before
            # and immediately after handover so controller switching cannot
            # consume any command left by an earlier session.
            try:
                self._seed_position_controller(current_joints)
                self._switch_controller(True)
                self._seed_position_controller(current_joints)
                self.last_packet = time.monotonic()
                self.last_update = self.last_packet
                self.joint_target = list(current_joints)
                self.joint_velocities = [0.0] * 6
                self.blocked_reason = None
                self.blocked_detail = None
                self.fault_reason = None
                self.active = True
            except Exception:
                try:
                    self._switch_controller(False)
                except Exception as restore_error:
                    rospy.logerr(
                        "failed to restore trajectory controller after enable error: %s",
                        restore_error,
                    )
                raise

    def disable(self):
        with self.lock:
            if not self.active:
                return
            self.joint_velocities = [0.0] * 6
            rospy.sleep(0.05)
            try:
                self._switch_controller(False)
            finally:
                self.active = False

    def update_joints(self, values):
        with self.lock:
            if not self.active:
                raise RuntimeError("joint control is not enabled")
            requested = values.get("velocities_rad_s")
            if not isinstance(requested, list) or len(requested) != 6:
                raise ValueError("velocities_rad_s must contain six values")
            self.joint_velocities = [
                _bounded(value, self.args.max_joint_speed) for value in requested
            ]
            self.last_packet = time.monotonic()

    @staticmethod
    def _controller_states():
        rospy.wait_for_service('/piper_arm/controller_manager/list_controllers', timeout=2.)
        response = rospy.ServiceProxy('/piper_arm/controller_manager/list_controllers', ListControllers)()
        return {c.name: c.state for c in response.controller}

    @staticmethod
    def _read_can_modes():
        result = subprocess.run(
            ['timeout', '1', 'candump', '-L', '-n', '10', 'can0,2A1:7FF'],
            check=True, capture_output=True, text=True, timeout=3.)
        return [int(line.rsplit('2A1#', 1)[1][:2], 16)
                for line in result.stdout.splitlines() if '2A1#' in line]

    def recover_top_chassis_boundary(self):
        # One replan only, exclusively after a verified CAN mode transition.
        # The failed attempt restores/holds live pose in its finally block.
        with self.lock:
            try:
                return self._recover_top_chassis_boundary_once()
            except EscapeModeDriftError as exc:
                first_failure = dict(self.last_boundary_recovery or {})
                try:
                    recovery_start = list(exc.audit['held_joints_rad'])
                    initial_score = geometry_score(self.escape_geometry.measure(recovery_start))
                    deadline = time.monotonic()+2.0
                    window_start = None
                    window_pose = None
                    observations = []
                    while time.monotonic() < deadline:
                        modes = self._read_can_modes()
                        if len(modes)<3 or any(m != 1 for m in modes[-3:]):
                            raise RuntimeError('escape replan CAN mode became invalid') from exc
                        q = self._current_joints()
                        validate_escape_transition_pose(recovery_start, q)
                        require_expected_contacts(self.collision.check_state(q))
                        metric = self.escape_geometry.measure(q)
                        score = geometry_score(metric)
                        if score < initial_score-.0005:
                            raise RuntimeError('escape mode settling worsened chassis clearance') from exc
                        initial_score = max(initial_score, score)
                        now = time.monotonic()
                        observations.append(dict(joints_rad=q,geometry=metric))
                        if window_pose is None or max(abs(a-b) for a,b in zip(q,window_pose))>.001:
                            window_pose,window_start=q,now
                        elif now-window_start>=.30:
                            audit = self._recover_top_chassis_boundary_once()
                            audit['activation_replan'] = dict(
                                count=1, reason='confirmed_mode_transition_changed_start',
                                first_failure=dict(
                                    failed_stage=first_failure.get('failed_stage'),
                                    error=first_failure.get('error'),
                                    ctrl_mode_before=exc.audit['ctrl_mode_before'],
                                    ctrl_mode_after=exc.audit['ctrl_mode_after'],
                                    max_hold_drift_rad=exc.audit['max_hold_drift_rad']),
                                stable_start_rad=q, settling_sample_count=len(observations),
                                settling_first=observations[0], settling_last=observations[-1])
                            self.last_boundary_recovery=audit
                            return audit
                        rospy.sleep(.05)
                    raise RuntimeError('escape mode settling did not stabilize within 2 seconds') from exc
                except Exception as replan_error:
                    self.last_boundary_recovery = dict(
                        status='failed', failed_stage='mode_settling_or_repreflight',
                        error=str(replan_error), first_failure=first_failure,
                        actual_joints_rad=self.actual_joints)
                    raise

    def _recover_top_chassis_boundary_once(self):
        """Escape one narrowly defined chassis/J2 boundary condition.

        Ordinary control remains fail-closed.  This command is accepted only
        when MoveIt reports the sole piper_link2/top_chassis_link pair, and it
        may lift J2 and fold J3 by at most 0.080 rad at 0.020 rad/s. Once the
        endpoint becomes valid, the normal trajectory controller is restored.
        """
        with self.lock:
            if self.active:
                raise RuntimeError("disable joint control before boundary recovery")
            if self.other_motion_active:
                raise RuntimeError("another MoveIt trajectory is active")
            hardware = rospy.wait_for_message(
                "/piper_arm/piper_hw_interface/state_machine",
                PiperState,
                timeout=2.0,
            )
            if int(hardware.state) != 2:
                raise RuntimeError("Piper hardware is not Active")
            start = self._current_joints()
            plan = select_geometric_j2_j3_escape_target(
                start, self.collision.check_state, self.escape_geometry.measure)
            if not plan["needed"]:
                audit = {
                    "status": "already_clear",
                    "moved": False,
                    "start_joints_rad": start,
                    "final_joints_rad": start,
                    "initial_state": plan["initial"],
                }
                self.last_boundary_recovery = audit
                return audit

            target = plan["target"]
            samples = interpolate_j2_j3_escape(start, target)
            # Confirm that "rise" really lifts both elbow and distal arm in
            # the current configuration, rather than assuming a joint sign.
            rospy.wait_for_service('/compute_fk', timeout=2.0)
            fk = rospy.ServiceProxy('/compute_fk', GetPositionFK)
            previous_heights = None
            start_heights = None
            for joints in [start]+[p['joints_rad'] for p in plan['checked_path']]:
                request = GetPositionFKRequest()
                request.header.frame_id = 'world'
                request.fk_link_names = ['piper_link3', 'piper_link6']
                request.robot_state.joint_state = JointState(
                    name=ARM_JOINT_NAMES, position=joints)
                request.robot_state.is_diff = True
                result = fk(request)
                if result.error_code.val != 1 or len(result.pose_stamped) != 2:
                    raise RuntimeError('escape forward kinematics unavailable')
                heights = [p.pose.position.z for p in result.pose_stamped]
                if previous_heights is not None and any(
                        h < p-.00001 for h, p in zip(heights, previous_heights)):
                    raise RuntimeError('J2/J3 escape would lower the elbow or distal arm')
                if start_heights is None:
                    start_heights = heights
                previous_heights = heights
            if any(h-s < .001 for h, s in zip(previous_heights, start_heights)):
                raise RuntimeError('J2/J3 escape produces insufficient lift')
            fresh_start = self._current_joints()
            if max(abs(a-b) for a,b in zip(fresh_start,start)) > .003:
                raise RuntimeError("escape start changed during preflight; recheck required")
            switched = False
            started = time.monotonic()
            last_checked = 0.0
            last_geometry_score = geometry_score(plan['initial_geometry'])
            recovery_stage = "holding_current_pose"
            mode_audit = None
            try:
                self._seed_position_controller(start, cycles=8)
                self._switch_controller(True)
                switched = True
                self._seed_position_controller(start, cycles=8)
                recovery_stage = "confirming_can_actuation_mode"
                mode_audit = confirm_escape_actuation_mode(
                    start, controller_states=self._controller_states,
                    current_joints=self._current_joints,
                    send_can_joint_mode=lambda: subprocess.run(
                        ["cansend", "can0", PIPER_MODE_CTRL_FRAME],
                        check=True, capture_output=True, text=True, timeout=2.),
                    read_mode_feedback=self._read_can_modes)
                recovery_stage = "lifting_j2_j3"
                for sample in samples:
                    self.position_command_pub.publish(
                        Float64MultiArray(data=sample)
                    )
                    rospy.sleep(RECOVERY_COMMAND_PERIOD_SEC)
                    actual = self.actual_joints
                    if (actual is None or self.joint_feedback_monotonic is None
                            or time.monotonic()-self.joint_feedback_monotonic > .25):
                        raise RuntimeError("joint feedback lost during recovery")
                    if max(
                        abs(actual[index] - start[index])
                        for index in (0, 3, 4, 5)
                    ) > 0.010:
                        raise RuntimeError("non-J2/J3 joint moved during recovery")
                    if max(abs(actual[i]-sample[i]) for i in (1, 2)) > .006:
                        raise RuntimeError('J2/J3 recovery tracking diverged: command=%r actual=%r' % (sample, actual))
                    if time.monotonic()-last_checked >= .10:
                        require_expected_contacts(self.collision.check_state(actual))
                        score = geometry_score(self.escape_geometry.measure(actual))
                        if score < last_geometry_score-.0005:
                            raise RuntimeError('actual full-mesh escape clearance decreased')
                        last_geometry_score = max(last_geometry_score, score)
                        last_checked = time.monotonic()
                for _ in range(15):
                    self.position_command_pub.publish(
                        Float64MultiArray(data=target)
                    )
                    rospy.sleep(RECOVERY_COMMAND_PERIOD_SEC)
                final = self._current_joints()
                if max(abs(final[i]-target[i]) for i in (1, 2)) > 0.005:
                    raise RuntimeError(
                        "J2/J3 recovery tracking failed: target=%r actual=%r"
                        % (target[1:3], final[1:3])
                    )
                final_state = self.collision.check_state(final)
                if not final_state["valid"]:
                    raise RuntimeError(
                        "boundary recovery endpoint is still invalid: %r"
                        % final_state["contact_pairs"]
                    )
                final_geometry = self.escape_geometry.measure(final)
                if (final_geometry['overlap'] or
                        final_geometry['clearance_m'] < ESCAPE_REQUIRED_CLEARANCE_M):
                    raise RuntimeError('actual escape clearance is below 5mm')
                audit = {
                    "status": "recovered",
                    "moved": True,
                    "start_joints_rad": start,
                    "target_joints_rad": target,
                    "final_joints_rad": final,
                    "j2_travel_rad": target[1] - start[1],
                    "j3_travel_rad": target[2] - start[2],
                    "moving_joints": ['J2', 'J3'],
                    "start_heights_m": start_heights,
                    "final_planned_heights_m": previous_heights,
                    "duration_sec": time.monotonic() - started,
                    "probe_count": plan["probe_count"],
                    "initial_state": plan["initial"],
                    "final_state": final_state,
                    "initial_geometry": plan['initial_geometry'],
                    "planned_geometry": plan['target_geometry'],
                    "final_geometry": final_geometry,
                    "metric_kind": plan['metric_kind'],
                    "actuation_mode": mode_audit,
                }
                self.last_boundary_recovery = audit
                return audit
            except Exception as exc:
                self.last_boundary_recovery = dict(
                    status="failed", failed_stage=recovery_stage, error=str(exc),
                    mode_transition=getattr(exc,"audit",None),
                    start_joints_rad=start, target_joints_rad=target,
                    actual_joints_rad=self.actual_joints, actuation_mode=mode_audit)
                rospy.logerr("boundary recovery failed: %s", self.last_boundary_recovery)
                raise
            finally:
                if switched:
                    current = self._current_joints()
                    self._seed_position_controller(current, cycles=8)
                    self._switch_controller(False)

    def toggle_clamp(self):
        return self._run_clamp_command(target=None)

    def close_clamp(self):
        return self._run_clamp_command(target=GRIPPER_CLOSED_RAD)

    def _run_clamp_command(self, target):
        # A gripper action takes longer than the arm heartbeat timeout.  Hold
        # the arm at zero joint velocity and keep its lease alive while the
        # synchronous action completes, then resume the same control session.
        with self.lock:
            self.gripper_command_active = True
            self.joint_velocities = [0.0] * 6
            self.last_packet = time.monotonic()
        try:
            return self._command_clamp_impl(target=target)
        finally:
            with self.lock:
                self.gripper_command_active = False
                self.joint_velocities = [0.0] * 6
                self.last_packet = time.monotonic()
                self.last_update = self.last_packet

    def _command_clamp_impl(self, target=None):
        if not self.gripper.wait_for_server(rospy.Duration(1.0)):
            raise RuntimeError("gripper action server unavailable")
        start = self.gripper_position
        if start is None:
            state = rospy.wait_for_message(
                "/piper_arm/joint_states", JointState, timeout=1.0
            )
            values = dict(zip(state.name, state.position))
            if "piper_gripper_motor_joint" not in values:
                raise RuntimeError("gripper joint feedback unavailable")
            start = float(values["piper_gripper_motor_joint"])
        # This driver exposes the gripper *motor angle*, not the finger gap in
        # metres.  The installed Piper behavior trees use 0.0 rad for closed
        # and 1.396 rad for fully open.  A former 0.04 target confused these
        # units and was too small to create a visible opening.
        if target is None:
            target = (
                GRIPPER_CLOSED_RAD
                if start >= GRIPPER_STATE_THRESHOLD_RAD
                else GRIPPER_OPEN_RAD
            )
        target = float(target)
        if target not in {GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD}:
            raise ValueError("unsupported gripper target")
        opening = target == GRIPPER_OPEN_RAD
        self.last_gripper_target = target
        sent_at = time.monotonic()
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["piper_gripper_motor_joint"]
        point = JointTrajectoryPoint()
        point.positions = [target]
        point.time_from_start = rospy.Duration(1.5)
        goal.trajectory.points = [point]
        self.gripper.send_goal(goal)
        if not self.gripper.wait_for_result(rospy.Duration(3.0)):
            self.gripper.cancel_goal()
            raise RuntimeError("gripper command timed out")
        if self.gripper.get_state() != GoalStatus.SUCCEEDED:
            raise RuntimeError(
                "gripper action failed with state %d" % self.gripper.get_state()
            )
        # wait_for_result can return before the subscriber callback has
        # delivered the final hardware sample.  Require a fresh sample and
        # allow up to one second for the measured motor angle to settle.
        deadline = time.monotonic() + 1.0
        final = self.gripper_position
        while time.monotonic() < deadline and not rospy.is_shutdown():
            feedback_is_fresh = (
                self.gripper_feedback_monotonic is not None
                and self.gripper_feedback_monotonic >= sent_at
            )
            moved = final is not None and (
                (opening and final >= start + GRIPPER_MIN_CONFIRMED_TRAVEL_RAD)
                or (
                    not opening
                    and final <= start - GRIPPER_MIN_CONFIRMED_TRAVEL_RAD
                )
            )
            at_requested_state = final is not None and (
                (opening and final >= GRIPPER_STATE_THRESHOLD_RAD)
                or (not opening and final < GRIPPER_STATE_THRESHOLD_RAD)
            )
            if feedback_is_fresh and (moved or at_requested_state):
                self.clamp_open = opening
                return
            rospy.sleep(0.02)
            final = self.gripper_position
        if final is None or not (
            (opening and final >= GRIPPER_STATE_THRESHOLD_RAD)
            or (not opening and final < GRIPPER_STATE_THRESHOLD_RAD)
        ):
            raise RuntimeError(
                "gripper feedback did not move: start=%.4f target=%.4f final=%s"
                % (start, target, "none" if final is None else "%.4f" % final)
            )
        self.clamp_open = opening

    def _tick(self, _event):
        with self.lock:
            if not self.active:
                return
            now = time.monotonic()
            if self.gripper_command_active:
                self.joint_velocities = [0.0] * 6
                self.last_packet = now
                self.last_update = now
                return
            dt = min(0.10, max(0.0, now - self.last_update))
            self.last_update = now
            age = now - self.last_packet
            if age > self.args.disable_timeout:
                rospy.logwarn("gamepad servo heartbeat expired; restoring trajectory controller")
                try:
                    self.disable()
                except Exception as exc:
                    rospy.logerr("servo watchdog restore failed: %s", exc)
                return
            velocities = (
                self.joint_velocities
                if age <= self.args.command_timeout else [0.0] * 6
            )
            if not any(abs(value) > 1e-8 for value in velocities) or self.joint_target is None:
                return
            candidate = list(self.joint_target)
            for index, velocity in enumerate(velocities):
                candidate[index] += velocity * dt
                if abs(velocity) <= 1e-8:
                    continue
                low, high = JOINT_LIMITS[index]
                low += self.args.joint_limit_margin
                high -= self.args.joint_limit_margin
                current = self.joint_target[index]
                if not allowed_by_soft_limit(current, candidate[index], low, high):
                    self.blocked_reason = "joint_limit"
                    self.blocked_detail = "J%d limit [%.4f, %.4f]" % (
                        index + 1, low, high
                    )
                    self.joint_velocities = [0.0] * 6
                    return
            started = time.monotonic()
            try:
                self.collision.validate([
                    {"positions": self.joint_target}, {"positions": candidate}
                ])
            except Exception as exc:
                self.blocked_reason = "moveit_collision_or_unavailable"
                self.blocked_detail = str(exc)
                self.joint_velocities = [0.0] * 6
                rospy.logwarn("joint jog stopped before unsafe state: %s", exc)
                return
            self.last_collision_check_sec = time.monotonic() - started
            self.blocked_reason = None
            self.blocked_detail = None
            self.joint_target = candidate
            self.position_command_pub.publish(Float64MultiArray(data=candidate))

    def status(self):
        with self.lock:
            return {
                "active": self.active,
                "control_mode": "direct_joint_moveit_checked",
                "clamp_open": self.clamp_open,
                "gripper_position": self.gripper_position,
                "last_gripper_target": self.last_gripper_target,
                "gripper_command_active": self.gripper_command_active,
                "other_motion_active": self.other_motion_active,
                "joint_velocities_rad_s": [
                    round(value, 5) for value in self.joint_velocities
                ],
                "joint_target_rad": self.joint_target,
                "actual_joints_rad": self.actual_joints,
                "joint_feedback_age_sec": (
                    None if self.joint_feedback_monotonic is None
                    else round(time.monotonic() - self.joint_feedback_monotonic, 4)
                ),
                "blocked_reason": self.blocked_reason,
                "blocked_detail": self.blocked_detail,
                "collision_check_sec": self.last_collision_check_sec,
                "fault_reason": self.fault_reason,
                "last_boundary_recovery": self.last_boundary_recovery,
                "packet_age_sec": (
                    None if self.last_packet <= 0
                    else round(time.monotonic() - self.last_packet, 4)
                ),
            }

    def _handle(self, request, address):
        if address[0] != self.args.allowed_client:
            raise PermissionError("client not allowed")
        if request.get("token") != self.args.token:
            raise PermissionError("invalid token")
        sequence = int(request.get("seq", -1))
        command = str(request.get("command", ""))
        if command != "status" and sequence <= self.sequence:
            raise ValueError("stale sequence")
        if command != "status":
            self.sequence = sequence
        if command == "enable":
            self.enable()
        elif command == "update_joints":
            self.update_joints(request)
        elif command == "toggle_clamp":
            self.toggle_clamp()
        elif command == "close_clamp":
            self.close_clamp()
            self.last_packet = time.monotonic()
        elif command == "recover_top_chassis_boundary":
            self.recover_top_chassis_boundary()
        elif command == "disable":
            self.disable()
        elif command != "status":
            raise ValueError("unknown command")
        return {"ok": True, "seq": sequence, "status": self.status()}

    def serve(self):
        rospy.loginfo(
            "gamepad joint bridge listening on %s:%d", self.args.bind_ip, self.args.port
        )
        while not rospy.is_shutdown():
            try:
                payload, address = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            try:
                request = json.loads(payload.decode("utf-8"))
                response = self._handle(request, address)
            except Exception as exc:
                response = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
            self.sock.sendto(
                json.dumps(response, separators=(",", ":")).encode("utf-8"), address
            )

    def close(self):
        try:
            self.disable()
        finally:
            self.escape_geometry.close()
            self.sock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-ip", default="192.168.168.120")
    parser.add_argument("--port", type=int, default=5015)
    parser.add_argument("--allowed-client", default="192.168.168.130")
    parser.add_argument("--token", required=True)
    parser.add_argument("--command-timeout", type=float, default=0.20)
    parser.add_argument("--disable-timeout", type=float, default=0.60)
    parser.add_argument("--joint-update-hz", type=float, default=20.0)
    parser.add_argument("--max-joint-speed", type=float, default=0.35)
    parser.add_argument("--joint-limit-margin", type=float, default=0.02)
    parser.add_argument("--collision-max-step", type=float, default=0.015)
    args = parser.parse_args()
    if not 10.0 <= args.joint_update_hz <= 50.0:
        parser.error("--joint-update-hz must be in [10, 50]")
    if not 0.02 <= args.max_joint_speed <= 0.5:
        parser.error("--max-joint-speed must be in [0.02, 0.5]")
    rospy.init_node("piper_gamepad_servo_bridge", disable_signals=True)
    bridge = GamepadServoBridge(args)
    try:
        bridge.serve()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
