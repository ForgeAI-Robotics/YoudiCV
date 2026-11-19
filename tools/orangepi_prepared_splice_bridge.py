#!/usr/bin/env python3
"""Prepared two-goal splice bridge for a guarded, continuous Piper trajectory."""

import argparse
import json
import subprocess
import socket
import threading
import time

import actionlib
import numpy as np
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from controller_manager_msgs.srv import (
    ListControllers,
    SwitchController,
    SwitchControllerRequest,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

from orangepi_timed_trajectory_bridge import (
    JOINT_NAMES,
    MAX_VELOCITIES,
    REJOIN_ACCELERATIONS,
    VELOCITY_CAP_SCALE,
    validate,
)
from piper_trajectory_resume import (
    accept_near_goal_tolerance_violation,
    generate_rejoin_trajectory,
)
from piper_moveit_collision import MoveItCollisionValidator
from orangepi_arm_mode import PIPER_MODE_CTRL_FRAME, activate_actuation_mode


class ActiveTrajectoryAborted(RuntimeError):
    pass


class PreparedSpliceBridge:
    def __init__(self, args):
        self.args = args
        self.busy = threading.Lock()
        self.state_lock = threading.Lock()
        self.active_context = None
        self.prepared = {}
        rospy.wait_for_service(
            "/piper_arm/controller_manager/switch_controller", timeout=10.0
        )
        rospy.wait_for_service(
            "/piper_arm/controller_manager/list_controllers", timeout=10.0
        )
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
            raise RuntimeError("joint trajectory action server is unavailable")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((args.bind_ip, args.port))
        self.sock.settimeout(0.5)
        self.collision = MoveItCollisionValidator(JOINT_NAMES)

    def response(self, address, task_id, status, **extra):
        value = {"task_id": task_id, "status": status}
        value.update(extra)
        self.sock.sendto(
            json.dumps(value, separators=(",", ":")).encode("utf-8"), address
        )

    def current_joints(self):
        state = rospy.wait_for_message(
            "/piper_arm/joint_states", JointState, timeout=2.0
        )
        values = dict(zip(state.name, state.position))
        if not all(name in values for name in JOINT_NAMES):
            raise RuntimeError("joint state does not contain all Piper joints")
        return np.asarray([values[name] for name in JOINT_NAMES], dtype=float)

    def controller_states(self):
        return {item.name: item.state for item in self.list_controllers().controller}

    def switch_to_trajectory(self):
        states = self.controller_states()
        trajectory = "joint_trajectory_controller"
        position = "joint_group_position_controller"
        if states.get(trajectory) == "running" and states.get(position) != "running":
            return
        request = SwitchControllerRequest()
        request.start_controllers = [] if states.get(trajectory) == "running" else [trajectory]
        request.stop_controllers = [position] if states.get(position) == "running" else []
        request.strictness = SwitchControllerRequest.STRICT
        request.start_asap = True
        request.timeout = 2.0
        if not self.switch(request).ok:
            raise RuntimeError("controller switch failed from states %r" % states)
        after = self.controller_states()
        if (
            after.get(trajectory) != "running"
            or after.get(position) == "running"
        ):
            raise RuntimeError("trajectory controller handoff failed: %r" % after)

    def switch_to_position_hold(self, joints):
        states = self.controller_states()
        trajectory = "joint_trajectory_controller"
        position = "joint_group_position_controller"
        command = Float64MultiArray(data=np.asarray(joints, dtype=float).tolist())
        for _ in range(5):
            self.position_command_pub.publish(command)
            rospy.sleep(0.01)
        if (
            states.get(position) != "running"
            or states.get(trajectory) == "running"
        ):
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

    @staticmethod
    def goal(points, start_stamp):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.header.stamp = start_stamp
        for value in points:
            point = JointTrajectoryPoint()
            point.positions = value["positions"]
            point.velocities = value["velocities"]
            point.accelerations = value["accelerations"]
            point.time_from_start = rospy.Duration.from_sec(
                float(value["time_from_start_sec"])
            )
            goal.trajectory.points.append(point)
        return goal

    def prune(self):
        cutoff = time.monotonic() - 120.0
        self.prepared = {
            key: value for key, value in self.prepared.items()
            if value["received_monotonic"] >= cutoff
        }

    def prepare(self, payload):
        trajectory_id = str(payload.get("trajectory_id", ""))
        if not trajectory_id or len(trajectory_id) > 80:
            raise ValueError("invalid trajectory_id")
        with self.state_lock:
            context = self.active_context
            if (
                context is not None
                and context["status"] in (
                    "running_fallback", "replacement_scheduled"
                )
                and trajectory_id == context["active_id"]
            ):
                raise RuntimeError(
                    "cannot overwrite the active trajectory while it is running"
                )
        validation = dict(payload)
        validation["protocol"] = "piper_timed_trajectory_v1"
        validation["command"] = "validate"
        points = validate(validation)
        collision_audit = self.collision.validate(points)
        self.prune()
        self.prepared[trajectory_id] = {
            "points": points,
            "received_monotonic": time.monotonic(),
            "collision_validation": collision_audit,
        }
        return trajectory_id, len(points)

    def splice_inputs(self, payload):
        active_id = str(payload["active_id"])
        replacement_id = str(payload["replacement_id"])
        if active_id not in self.prepared or replacement_id not in self.prepared:
            raise ValueError("one or both prepared trajectories are missing")
        active = self.prepared[active_id]["points"]
        replacement = self.prepared[replacement_id]["points"]
        handoff = float(payload["handoff_active_time_sec"])
        matches = [
            point for point in active
            if abs(float(point["time_from_start_sec"]) - handoff) <= 1e-6
        ]
        if len(matches) != 1:
            raise ValueError("handoff must match exactly one active point")
        active_point = matches[0]
        replacement_point = replacement[0]
        position_error = float(np.max(np.abs(
            np.asarray(active_point["positions"]) - np.asarray(replacement_point["positions"])
        )))
        velocity_error = float(np.max(np.abs(
            np.asarray(active_point["velocities"]) - np.asarray(replacement_point["velocities"])
        )))
        acceleration_error = float(np.max(np.abs(
            np.asarray(active_point["accelerations"]) - np.asarray(replacement_point["accelerations"])
        )))
        if position_error > 1e-6 or velocity_error > 1e-5 or acceleration_error > 1e-4:
            raise ValueError(
                "splice discontinuity %.3g/%.3g/%.3g" % (
                    position_error, velocity_error, acceleration_error
                )
            )
        if handoff < 0.5 or handoff >= float(active[-1]["time_from_start_sec"]):
            raise ValueError("handoff is outside the replaceable active interval")
        return active, replacement, handoff, {
            "position_error_rad": position_error,
            "velocity_error_rad_s": velocity_error,
            "acceleration_error_rad_s2": acceleration_error,
        }

    def execute_splice(self, active, replacement, handoff):
        if rospy.get_param("/piper_visual_safety_stop/latched", False):
            raise RuntimeError("visual safety stop is latched")
        start_error = float(np.max(np.abs(
            self.current_joints() - np.asarray(active[0]["positions"], dtype=float)
        )))
        if start_error > 0.03:
            raise RuntimeError("active trajectory start mismatch %.4f rad" % start_error)
        mode_audit = self.activate_actuation_mode(active[0]["positions"])
        active_start = rospy.Time.now() + rospy.Duration(0.30)
        handoff_stamp = active_start + rospy.Duration.from_sec(handoff)
        self.client.send_goal(self.goal(active, active_start))
        while not rospy.is_shutdown():
            remaining = (handoff_stamp - rospy.Time.now()).to_sec()
            if remaining <= 0.02:
                break
            rospy.sleep(min(0.02, max(0.001, remaining - 0.02)))
        self.client.send_goal(self.goal(replacement, handoff_stamp))
        duration = float(replacement[-1]["time_from_start_sec"])
        if not self.client.wait_for_result(rospy.Duration(duration + 5.0)):
            self.client.cancel_goal()
            raise RuntimeError("replacement trajectory timed out")
        result = self.client.get_result()
        if result is None or result.error_code != result.SUCCESSFUL:
            code = None if result is None else result.error_code
            text = "" if result is None else result.error_string
            raise RuntimeError("replacement failed code=%r %s" % (code, text))
        end_error = float(np.max(np.abs(
            self.current_joints() - np.asarray(replacement[-1]["positions"], dtype=float)
        )))
        if end_error > 0.03:
            raise RuntimeError("replacement final mismatch %.4f rad" % end_error)
        return (
            start_error, end_error, active_start.to_sec(),
            handoff_stamp.to_sec(), mode_audit,
        )

    def start_active(self, payload):
        if rospy.get_param("/piper_visual_safety_stop/latched", False):
            raise RuntimeError("visual safety stop is latched")
        active_id = str(payload["active_id"])
        if active_id not in self.prepared:
            raise ValueError("prepared active trajectory is missing")
        if not self.busy.acquire(False):
            raise RuntimeError("bridge is busy")
        try:
            active = self.prepared[active_id]["points"]
            start_error = float(np.max(np.abs(
                self.current_joints()
                - np.asarray(active[0]["positions"], dtype=float)
            )))
            if start_error > 0.03:
                raise RuntimeError(
                    "active trajectory start mismatch %.4f rad" % start_error
                )
            mode_audit = self.activate_actuation_mode(active[0]["positions"])
            active_start = rospy.Time.now() + rospy.Duration(0.30)
            context = {
                "active_id": active_id,
                "active": active,
                "active_start": active_start,
                "active_end": active_start + rospy.Duration.from_sec(
                    float(active[-1]["time_from_start_sec"])
                ),
                "start_error_rad": start_error,
                "mode_audit": mode_audit,
                "status": "running_fallback",
                "replacement": None,
                "error": None,
                "end_error_rad": None,
                "safety_generation": self.safety_generation(),
                "safety_resumes": [],
                "abort_requested": False,
            }
            with self.state_lock:
                self.active_context = context
            self.client.send_goal(self.goal(active, active_start))
            thread = threading.Thread(
                target=self.active_supervisor, args=(context,), daemon=True
            )
            thread.start()
            return context
        except Exception:
            self.client.cancel_all_goals()
            self.busy.release()
            raise

    def schedule_replacement(self, payload):
        with self.state_lock:
            context = self.active_context
        if context is None or context["status"] not in (
            "running_fallback", "replacement_scheduled", "safety_paused"
        ):
            raise RuntimeError("there is no replaceable active trajectory")
        active_id = str(payload["active_id"])
        if active_id != context["active_id"]:
            raise ValueError("active_id does not match the running trajectory")
        active, replacement, handoff, continuity = self.splice_inputs(payload)
        if active is not context["active"]:
            # The prepared object should remain the same throughout execution.
            raise RuntimeError("running active trajectory storage changed")
        handoff_stamp = context["active_start"] + rospy.Duration.from_sec(handoff)
        remaining = (handoff_stamp - rospy.Time.now()).to_sec()
        if remaining < 0.25 and context["status"] != "safety_paused":
            raise RuntimeError(
                "replacement arrived too late; %.3f s remain" % remaining
            )
        with self.state_lock:
            if context["replacement"] is not None:
                raise RuntimeError("a replacement is already scheduled")
            was_safety_paused = context["status"] == "safety_paused"
            context["replacement"] = {
                "replacement_id": str(payload["replacement_id"]),
                "points": replacement,
                "handoff_active_time_sec": handoff,
                "handoff_stamp": handoff_stamp,
                "continuity": continuity,
                "scheduled_remaining_sec": remaining,
            }
            # The safety-resume thread owns the state while held.  Preserve
            # safety_paused so status polling and timeout accounting remain
            # correct; resume_after_safety will compose this replacement after
            # the latch clears.
            if not was_safety_paused:
                context["status"] = "replacement_scheduled"
        return context["replacement"]

    @staticmethod
    def safety_generation():
        return int(rospy.get_param("/piper_visual_safety_stop/generation", 0))

    @staticmethod
    def composed_path(context):
        """The geometric path selected so far, on the active time axis."""
        replacement = context["replacement"]
        if replacement is None:
            return [dict(point) for point in context["active"]], "fallback_complete"
        handoff = float(replacement["handoff_active_time_sec"])
        points = [
            dict(point) for point in context["active"]
            if float(point["time_from_start_sec"]) <= handoff + 1e-9
        ]
        for point in replacement["points"][1:]:
            value = dict(point)
            value["time_from_start_sec"] = (
                handoff + float(point["time_from_start_sec"])
            )
            points.append(value)
        return points, "replacement_complete"

    def resume_after_safety(self, context, elapsed_sec):
        """Wait held, then blend from measured joints into the cached path."""
        self.client.cancel_all_goals()
        with self.state_lock:
            context["status"] = "safety_paused"
        current_points = None
        final_status = None
        current_elapsed = float(elapsed_sec)
        while not rospy.is_shutdown():
            while rospy.get_param("/piper_visual_safety_stop/latched", False):
                if context.get("abort_requested"):
                    raise ActiveTrajectoryAborted("active trajectory aborted")
                rospy.sleep(0.02)
            if context.get("abort_requested"):
                raise ActiveTrajectoryAborted("active trajectory aborted")
            generation = self.safety_generation()
            if current_points is None:
                # A live replacement may arrive while the arm is held. Build
                # the path only after clearance so it is never lost to a race
                # with the online planner.
                with self.state_lock:
                    current_points, final_status = self.composed_path(context)
            current_points, audit = generate_rejoin_trajectory(
                current_points,
                self.current_joints(),
                elapsed_sec=current_elapsed,
                max_velocities=MAX_VELOCITIES * VELOCITY_CAP_SCALE,
                max_accelerations=REJOIN_ACCELERATIONS,
            )
            validate({
                "protocol": "piper_timed_trajectory_v1",
                "command": "validate",
                "joint_names": JOINT_NAMES,
                "points": current_points,
            })
            audit["collision_validation"] = self.collision.validate(
                current_points
            )
            with self.state_lock:
                context["safety_resumes"].append(audit)
                context["status"] = "safety_resuming"
            audit["piper_mode"] = self.activate_actuation_mode(
                current_points[0]["positions"]
            )
            start = rospy.Time.now() + rospy.Duration(0.15)
            self.client.send_goal(self.goal(current_points, start))
            duration = float(current_points[-1]["time_from_start_sec"])
            if not self.client.wait_for_result(rospy.Duration(duration + 5.0)):
                self.client.cancel_goal()
                raise RuntimeError("safety-resumed trajectory timed out")
            result = self.client.get_result()
            if context.get("abort_requested"):
                raise ActiveTrajectoryAborted("active trajectory aborted")
            current_generation = self.safety_generation()
            if current_generation > generation:
                # A second stop may race with the first resumed goal's success
                # callback. The safety generation wins over actionlib state;
                # freeze this resumed timeline and create another connector.
                current_elapsed = max(0.0, (rospy.Time.now() - start).to_sec())
                with self.state_lock:
                    context["status"] = "safety_paused"
                continue
            endpoint = np.asarray(current_points[-1]["positions"], dtype=float)
            end_error = float(np.max(np.abs(self.current_joints() - endpoint)))
            succeeded = result is not None and result.error_code == result.SUCCESSFUL
            accepted_near_goal = result is not None and accept_near_goal_tolerance_violation(
                result.error_code, end_error
            )
            if succeeded or accepted_near_goal:
                if end_error > 0.03:
                    raise RuntimeError(
                        "safety-resumed final mismatch %.4f rad" % end_error
                    )
                with self.state_lock:
                    if accepted_near_goal:
                        context["safety_resumes"][-1][
                            "accepted_goal_tolerance_violation"
                        ] = True
                        context["safety_resumes"][-1][
                            "accepted_end_error_rad"
                        ] = end_error
                    context["end_error_rad"] = end_error
                    context["status"] = final_status
                    context["safety_generation"] = self.safety_generation()
                return
            code = None if result is None else result.error_code
            text = "" if result is None else result.error_string
            raise RuntimeError(
                "safety-resumed trajectory failed code=%r %s" % (code, text)
            )
        raise RuntimeError("shutdown while resuming visual safety stop")

    def active_supervisor(self, context):
        try:
            while not rospy.is_shutdown():
                if context.get("abort_requested"):
                    raise ActiveTrajectoryAborted("active trajectory aborted")
                if self.safety_generation() > context["safety_generation"]:
                    elapsed = max(
                        0.0, (rospy.Time.now() - context["active_start"]).to_sec()
                    )
                    self.resume_after_safety(context, elapsed)
                    return
                if rospy.get_param("/piper_visual_safety_stop/latched", False):
                    rospy.sleep(0.01)
                    continue
                with self.state_lock:
                    replacement = context["replacement"]
                if replacement is not None:
                    handoff_stamp = replacement["handoff_stamp"]
                    while not rospy.is_shutdown():
                        if context.get("abort_requested"):
                            raise ActiveTrajectoryAborted("active trajectory aborted")
                        if self.safety_generation() > context["safety_generation"]:
                            break
                        remaining = (handoff_stamp - rospy.Time.now()).to_sec()
                        if remaining <= 0.02:
                            break
                        rospy.sleep(min(0.02, max(0.001, remaining - 0.02)))
                    if self.safety_generation() > context["safety_generation"]:
                        continue
                    self.client.send_goal(
                        self.goal(replacement["points"], handoff_stamp)
                    )
                    duration = float(
                        replacement["points"][-1]["time_from_start_sec"]
                    )
                    if not self.client.wait_for_result(rospy.Duration(duration + 5.0)):
                        self.client.cancel_goal()
                        raise RuntimeError("scheduled replacement timed out")
                    result = self.client.get_result()
                    if context.get("abort_requested"):
                        raise ActiveTrajectoryAborted("active trajectory aborted")
                    if self.safety_generation() > context["safety_generation"]:
                        elapsed = max(
                            0.0,
                            (rospy.Time.now() - context["active_start"]).to_sec(),
                        )
                        self.resume_after_safety(context, elapsed)
                        return
                    if result is None or result.error_code != result.SUCCESSFUL:
                        code = None if result is None else result.error_code
                        text = "" if result is None else result.error_string
                        raise RuntimeError(
                            "scheduled replacement failed code=%r %s" % (code, text)
                        )
                    end_error = float(np.max(np.abs(
                        self.current_joints()
                        - np.asarray(
                            replacement["points"][-1]["positions"], dtype=float
                        )
                    )))
                    if end_error > 0.03:
                        raise RuntimeError(
                            "scheduled replacement final mismatch %.4f rad" % end_error
                        )
                    with self.state_lock:
                        context["end_error_rad"] = end_error
                        context["status"] = "replacement_complete"
                    return
                if rospy.Time.now() >= context["active_end"]:
                    if not self.client.wait_for_result(rospy.Duration(5.0)):
                        self.client.cancel_goal()
                        raise RuntimeError("fallback active trajectory timed out")
                    result = self.client.get_result()
                    if context.get("abort_requested"):
                        raise ActiveTrajectoryAborted("active trajectory aborted")
                    if self.safety_generation() > context["safety_generation"]:
                        elapsed = max(
                            0.0,
                            (rospy.Time.now() - context["active_start"]).to_sec(),
                        )
                        self.resume_after_safety(context, elapsed)
                        return
                    if result is None or result.error_code != result.SUCCESSFUL:
                        code = None if result is None else result.error_code
                        text = "" if result is None else result.error_string
                        raise RuntimeError(
                            "fallback active failed code=%r %s" % (code, text)
                        )
                    end_error = float(np.max(np.abs(
                        self.current_joints()
                        - np.asarray(context["active"][-1]["positions"], dtype=float)
                    )))
                    if end_error > 0.03:
                        raise RuntimeError(
                            "fallback final mismatch %.4f rad" % end_error
                        )
                    with self.state_lock:
                        context["end_error_rad"] = end_error
                        context["status"] = "fallback_complete"
                    return
                rospy.sleep(0.01)
        except ActiveTrajectoryAborted:
            self.client.cancel_all_goals()
            with self.state_lock:
                context["error"] = None
                context["status"] = "aborted"
            rospy.logwarn("active splice trajectory aborted before recovery")
        except Exception as exc:
            self.client.cancel_all_goals()
            with self.state_lock:
                context["error"] = "%s: %s" % (type(exc).__name__, exc)
                context["status"] = "failed"
            rospy.logerr("active splice supervisor failed: %s", context["error"])
        finally:
            self.busy.release()

    def abort_active(self, payload):
        """Cancel and discard the cached asynchronous path before latch reset."""
        with self.state_lock:
            context = self.active_context
            if context is None:
                return "idle", None
            requested = payload.get("active_id")
            if requested is not None and str(requested) != context["active_id"]:
                raise ValueError("active_id does not match the running trajectory")
            if context["status"] in (
                "replacement_complete", "fallback_complete", "failed", "aborted"
            ):
                return context["status"], context["active_id"]
            context["abort_requested"] = True
            context["status"] = "aborting"
            active_id = context["active_id"]
        self.client.cancel_all_goals()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with self.state_lock:
                status = context["status"]
            if status in ("aborted", "failed"):
                return status, active_id
            time.sleep(0.01)
        raise RuntimeError("active trajectory abort did not reach terminal state")

    def context_status(self):
        with self.state_lock:
            context = self.active_context
            if context is None:
                return {"execution_status": "idle"}
            value = {
                "execution_status": context["status"],
                "active_id": context["active_id"],
                "active_start_ros_sec": context["active_start"].to_sec(),
                "active_end_ros_sec": context["active_end"].to_sec(),
                "start_error_rad": context["start_error_rad"],
                "end_error_rad": context["end_error_rad"],
                "error": context["error"],
                "safety_resume_count": len(context.get("safety_resumes", [])),
                "safety_resumes": list(context.get("safety_resumes", [])),
            }
            if context["replacement"] is not None:
                replacement = context["replacement"]
                value.update(
                    {
                        "replacement_id": replacement["replacement_id"],
                        "handoff_ros_sec": replacement["handoff_stamp"].to_sec(),
                        "scheduled_remaining_sec": replacement[
                            "scheduled_remaining_sec"
                        ],
                        "continuity": replacement["continuity"],
                    }
                )
            return value

    def spin(self):
        rospy.loginfo(
            "prepared splice bridge listening on %s:%d",
            self.args.bind_ip, self.args.port,
        )
        while not rospy.is_shutdown():
            try:
                data, address = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            task_id = "unknown"
            try:
                if address[0] != self.args.allowed_client:
                    raise ValueError("client IP is not allowed")
                payload = json.loads(data.decode("utf-8"))
                task_id = str(payload.get("task_id", "unknown"))
                if payload.get("protocol") != "piper_prepared_splice_v1":
                    raise ValueError("unsupported protocol")
                if payload.get("token") != self.args.token:
                    raise ValueError("invalid token")
                command = payload.get("command")
                if command == "prepare":
                    trajectory_id, count = self.prepare(payload)
                    self.response(
                        address, task_id, "prepared",
                        trajectory_id=trajectory_id, point_count=count,
                    )
                    continue
                if command == "execution_status":
                    self.response(address, task_id, "ok", **self.context_status())
                    continue
                if command == "abort_active":
                    execution_status, active_id = self.abort_active(payload)
                    self.response(
                        address, task_id, "ok",
                        execution_status=execution_status, active_id=active_id,
                    )
                    continue
                if command == "start_active":
                    context = self.start_active(payload)
                    self.response(
                        address, task_id, "started",
                        active_id=context["active_id"],
                        active_start_ros_sec=context["active_start"].to_sec(),
                        active_end_ros_sec=context["active_end"].to_sec(),
                        start_error_rad=context["start_error_rad"],
                        piper_mode=context["mode_audit"],
                    )
                    continue
                if command == "schedule_replacement":
                    replacement = self.schedule_replacement(payload)
                    self.response(
                        address, task_id, "scheduled",
                        replacement_id=replacement["replacement_id"],
                        handoff_ros_sec=replacement["handoff_stamp"].to_sec(),
                        scheduled_remaining_sec=replacement[
                            "scheduled_remaining_sec"
                        ],
                        continuity=replacement["continuity"],
                    )
                    continue
                active, replacement, handoff, continuity = self.splice_inputs(payload)
                if command == "validate_splice":
                    self.response(
                        address, task_id, "splice_validated",
                        active_point_count=len(active),
                        replacement_point_count=len(replacement),
                        handoff_active_time_sec=handoff,
                        continuity=continuity,
                    )
                    continue
                if command != "execute_splice":
                    raise ValueError("unsupported command")
                if not self.busy.acquire(False):
                    raise RuntimeError("bridge is busy")
                try:
                    values = self.execute_splice(active, replacement, handoff)
                except Exception:
                    self.client.cancel_all_goals()
                    raise
                finally:
                    self.busy.release()
                self.response(
                    address, task_id, "complete",
                    start_error_rad=values[0], end_error_rad=values[1],
                    active_start_ros_sec=values[2], handoff_ros_sec=values[3],
                    piper_mode=values[4],
                    continuity=continuity,
                )
            except Exception as exc:
                rospy.logerr("splice request %s failed: %s: %s", task_id, type(exc).__name__, exc)
                self.response(
                    address, task_id, "failed",
                    error="%s: %s" % (type(exc).__name__, exc),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-ip", default="192.168.168.120")
    parser.add_argument("--port", type=int, default=5011)
    parser.add_argument("--allowed-client", default="192.168.168.130")
    parser.add_argument("--token", required=True)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("piper_prepared_splice_bridge", anonymous=False)
    PreparedSpliceBridge(args).spin()


if __name__ == "__main__":
    main()
