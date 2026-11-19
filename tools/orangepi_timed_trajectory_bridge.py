#!/usr/bin/env python3
"""ROS Noetic UDP bridge for strictly validated Piper timed trajectories.

Install this file on the OrangePi.  It accepts commands only from one configured
Jetson IP and requires a shared token.  Validation requests never switch a
controller or send a trajectory.
"""

import argparse
import json
import hashlib
from collections import OrderedDict
import socket
import subprocess
import threading

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

from piper_trajectory_resume import (
    accept_near_goal_tolerance_violation,
    generate_rejoin_trajectory,
)
from piper_moveit_collision import MoveItCollisionValidator
from piper_action_watchdog import wait_for_action_result, require_safety_generation, send_goal_with_safety
from orangepi_arm_mode import activate_actuation_mode
from piper_trajectory_transport import FramedTrajectoryServer, MAX_TRAJECTORY_BYTES


JOINT_NAMES = ["piper_joint%d" % i for i in range(1, 7)]
JOINT_LIMITS = np.asarray(
    [
        [-2.6179, 2.6179], [0.0, 3.14], [-2.967, 0.03490658503988659],
        [-1.745, 1.745], [-1.22, 1.22], [-2.09439, 2.09439],
    ],
    dtype=float,
)
MAX_VELOCITIES = np.asarray([0.6, 1.0, 1.0, 1.0, 1.0, 2.0])
VELOCITY_CAP_SCALE = 0.75
REJOIN_ACCELERATIONS = np.full(6, 2.0)
PIPER_MODE_CTRL_FRAME = "151#0101320000000000"


def validate(payload):
    if payload.get("protocol") != "piper_timed_trajectory_v1":
        raise ValueError("unsupported protocol")
    if payload.get("command") not in ("execute", "validate"):
        raise ValueError("unsupported command")
    if payload.get("joint_names") != JOINT_NAMES:
        raise ValueError("unexpected joint ordering")
    points = payload.get("points")
    if not isinstance(points, list) or not 2 <= len(points) <= 500:
        raise ValueError("requires 2..500 points")
    previous_time = -1.0
    previous_positions = None
    for index, point in enumerate(points):
        stamp = float(point["time_from_start_sec"])
        positions = np.asarray(point["positions"], dtype=float)
        velocities = np.asarray(point["velocities"], dtype=float)
        accelerations = np.asarray(point["accelerations"], dtype=float)
        if positions.shape != (6,) or velocities.shape != (6,) or accelerations.shape != (6,):
            raise ValueError("point %d does not contain six values" % index)
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all() or not np.isfinite(accelerations).all():
            raise ValueError("point %d contains non-finite values" % index)
        if np.any(positions < JOINT_LIMITS[:, 0]) or np.any(positions > JOINT_LIMITS[:, 1]):
            raise ValueError("point %d exceeds joint position limits" % index)
        if not np.isfinite(stamp) or stamp <= previous_time or stamp > 15.0:
            raise ValueError("invalid trajectory time at point %d" % index)
        if np.any(np.abs(velocities) > MAX_VELOCITIES * VELOCITY_CAP_SCALE + 1e-9):
            raise ValueError("point %d exceeds bridge velocity cap" % index)
        if previous_positions is not None:
            average = np.abs(positions - previous_positions) / (stamp - previous_time)
            if np.any(average > MAX_VELOCITIES * VELOCITY_CAP_SCALE + 1e-9):
                raise ValueError("segment %d exceeds bridge average velocity cap" % index)
        previous_time = stamp
        previous_positions = positions
    if abs(float(points[0]["time_from_start_sec"])) > 1e-9:
        raise ValueError("first point must be at time zero")
    return points


class Bridge:
    def __init__(self, args):
        self.args = args
        self.busy = threading.Lock()
        rospy.wait_for_service("/piper_arm/controller_manager/switch_controller", timeout=10.0)
        rospy.wait_for_service("/piper_arm/controller_manager/list_controllers", timeout=10.0)
        self.switch = rospy.ServiceProxy(
            "/piper_arm/controller_manager/switch_controller", SwitchController
        )
        self.position_command_pub = rospy.Publisher(
            "/piper_arm/joint_group_position_controller/command",
            Float64MultiArray,
            queue_size=1,
        )
        self.list_controllers = rospy.ServiceProxy(
            "/piper_arm/controller_manager/list_controllers", ListControllers
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
        # A finite timeout lets rospy shutdown and systemd restarts complete
        # without waiting for an incoming UDP packet.
        self.sock.settimeout(0.5)
        self.collision = MoveItCollisionValidator(JOINT_NAMES)
        # Health requests must remain independent of a controller-manager RPC
        # that may itself be stalled by a cancelled action.
        self.last_controller_states = {}
        self.tcp_results = OrderedDict()
        self.tcp_server = FramedTrajectoryServer(
            (args.bind_ip,args.port),args.allowed_client,self._handle_tcp_request)
        rospy.on_shutdown(self.tcp_server.close)

    def response(self, address, task_id, status, **extra):
        value = {"task_id": task_id, "status": status}
        value.update(extra)
        self.sock.sendto(json.dumps(value, separators=(",", ":")).encode("utf-8"), address)

    def current_joints(self):
        state = rospy.wait_for_message("/piper_arm/joint_states", JointState, timeout=2.0)
        values = dict(zip(state.name, state.position))
        if not all(name in values for name in JOINT_NAMES):
            raise RuntimeError("joint state does not contain all Piper joints")
        return np.asarray([values[name] for name in JOINT_NAMES], dtype=float)

    def controller_states(self):
        states = {item.name: item.state for item in self.list_controllers().controller}
        self.last_controller_states = dict(states)
        return states

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
        result = self.switch(request)
        if not result.ok:
            raise RuntimeError("controller switch failed from states %r" % states)
        after = self.controller_states()
        if after.get(trajectory) != "running" or after.get(position) == "running":
            raise RuntimeError("controller state invalid after switch: %r" % after)

    def switch_to_position_hold(self, joints):
        """Seed current feedback before handing trajectory control to position hold."""
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
            request.stop_controllers = [trajectory] if states.get(trajectory) == "running" else []
            request.strictness = SwitchControllerRequest.STRICT
            request.start_asap = True
            request.timeout = 2.0
            if not self.switch(request).ok:
                raise RuntimeError("could not activate live-pose position hold")
        for _ in range(5):
            self.position_command_pub.publish(command)
            rospy.sleep(0.01)
        after = self.controller_states()
        if after.get(position) != "running" or after.get(trajectory) == "running":
            raise RuntimeError("position hold did not take control: %r" % after)

    @staticmethod
    def read_piper_ctrl_modes():
        result = subprocess.run(
            ["timeout", "1", "candump", "-L", "-n", "10", "can0,2A1:7FF"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
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
    def goal(points):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.15)
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

    def execute(self, points):
        generation = int(rospy.get_param("/piper_visual_safety_stop/generation", 0))
        if rospy.get_param("/piper_visual_safety_stop/latched", False):
            raise RuntimeError("visual safety stop is latched")
        expected = np.asarray(points[0]["positions"], dtype=float)
        mode_audit = self.activate_actuation_mode(expected)
        require_safety_generation(
            safety_generation=lambda: int(rospy.get_param("/piper_visual_safety_stop/generation", 0)),
            safety_latched=lambda: bool(rospy.get_param("/piper_visual_safety_stop/latched", False)),
            initial_generation=generation)
        start_error = float(np.max(np.abs(self.current_joints() - expected)))
        if start_error > 0.03:
            raise RuntimeError("trajectory start mismatch %.4f rad" % start_error)
        endpoint = np.asarray(points[-1]["positions"], dtype=float)
        current_points = points
        resumes = []
        while True:
            goal = self.goal(current_points)
            goal_start = goal.trajectory.header.stamp
            send_goal_with_safety(self.client, goal,
                safety_generation=lambda: int(rospy.get_param("/piper_visual_safety_stop/generation", 0)),
                safety_latched=lambda: bool(rospy.get_param("/piper_visual_safety_stop/latched", False)),
                initial_generation=generation)
            duration = float(current_points[-1]["time_from_start_sec"])
            wait_for_action_result(
                self.client,
                safety_generation=lambda: int(
                    rospy.get_param("/piper_visual_safety_stop/generation", 0)
                ),
                initial_generation=generation,
                duration_sec=duration,
                duration_factory=rospy.Duration,
            )
            result = self.client.get_result()
            current_generation = int(
                rospy.get_param("/piper_visual_safety_stop/generation", 0)
            )
            if current_generation <= generation:
                if result is not None and result.error_code == result.SUCCESSFUL:
                    break
                measured_error = float(
                    np.max(
                        np.abs(
                            self.current_joints()
                            - np.asarray(current_points[-1]["positions"], dtype=float)
                        )
                    )
                )
                if result is not None and accept_near_goal_tolerance_violation(
                    result.error_code, measured_error
                ):
                    if resumes:
                        resumes[-1]["accepted_goal_tolerance_violation"] = True
                        resumes[-1]["accepted_end_error_rad"] = measured_error
                    break
                code = None if result is None else result.error_code
                text = "" if result is None else result.error_string
                raise RuntimeError("trajectory failed code=%r %s" % (code, text))
            # Safety generation is authoritative.  This application uses a
            # latched stop policy: a cancelled task must end immediately and
            # leave recovery to the separately validated retreat path.  Do
            # not wait for the latch inside this bridge thread; doing so kept
            # ``busy`` true and made a later recovery health check depend on
            # a wedged ROS/actionlib call.
            self.client.cancel_all_goals()
            raise RuntimeError(
                "trajectory cancelled by visual safety generation %d" % current_generation
            )
        end_error = float(
            np.max(
                np.abs(
                    self.current_joints()
                    - endpoint
                )
            )
        )
        if end_error > 0.03:
            raise RuntimeError("trajectory final mismatch %.4f rad" % end_error)
        return start_error, end_error, resumes, mode_audit

    def _process_request(self, address, task_id, command, points, respond=None):
        """Run validation and motion without blocking UDP health replies."""
        respond = respond or self.response
        try:
            # MoveIt may take seconds to inspect a finely sampled path.  It
            # belongs in this worker too: the UDP loop must always remain able
            # to report a busy bridge and accept a health probe.
            collision_audit = self.collision.validate(points)
            if command == "validate":
                respond(
                    address, task_id, "validated", point_count=len(points),
                    collision_validation=collision_audit,
                )
                return
            start_error, end_error, resumes, mode_audit = self.execute(points)
            respond(
                address, task_id, "complete", point_count=len(points),
                start_error_rad=start_error, end_error_rad=end_error,
                safety_resume_count=len(resumes), safety_resumes=resumes,
                piper_mode=mode_audit,
                collision_validation=collision_audit,
            )
        except Exception as exc:
            rospy.logerr("trajectory request %s failed: %s: %s", task_id, type(exc).__name__, exc)
            respond(address, task_id, "failed", error="%s: %s" % (type(exc).__name__, exc))
        finally:
            self.busy.release()

    def _handle_tcp_request(self, payload, address):
        if address[0]!=self.args.allowed_client or payload.get('token')!=self.args.token:
            raise ValueError('TCP trajectory authentication failed')
        points=validate(payload)
        task_id=str(payload.get('task_id','unknown'))
        if not task_id or task_id=='unknown' or len(task_id)>200:
            raise ValueError('TCP trajectory needs bounded task_id')
        if not self.busy.acquire(False):raise RuntimeError('bridge is busy')
        # Lock remains owned until the existing request processor releases it.
        try:
            digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            cached=self.tcp_results.get(task_id)
            if cached is not None:
                if cached[0]!=digest:raise ValueError('TCP task_id reused with different trajectory')
        except Exception:
            self.busy.release();raise
        if cached is not None:
            self.busy.release();return dict(cached[1],replayed=True)
        result={}
        def collect(_address,_task,status,**values):
            result.update(task_id=_task,status=status,**values)
            if payload['command']=='execute':
                self.tcp_results[task_id]=(digest,dict(result))
                while len(self.tcp_results)>256:self.tcp_results.popitem(last=False)
        self._process_request(address,task_id,payload['command'],points,respond=collect)
        # Validation is always fresh; cache execute outcomes only. No automatic
        # retry on transport failure, and repeats within this cache never move.
        return result

    def spin(self):
        rospy.loginfo("timed trajectory UDP bridge listening on %s:%d", self.args.bind_ip, self.args.port)
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
                if payload.get("token") != self.args.token:
                    raise ValueError("invalid token")
                if payload.get("command") == "health":
                    self.response(
                        address, task_id, "ok", busy=self.busy.locked(),
                        safety_handoff_v2=True,
                        controllers=dict(self.last_controller_states),
                        trajectory_tcp_v1=True, trajectory_max_bytes=MAX_TRAJECTORY_BYTES,
                    )
                    continue
                points = validate(payload)
                if not self.busy.acquire(False):
                    raise RuntimeError("bridge is busy")
                threading.Thread(
                    target=self._process_request,
                    args=(address, task_id, payload["command"], points),
                    daemon=True,
                ).start()
            except Exception as exc:
                rospy.logerr("trajectory request %s failed: %s: %s", task_id, type(exc).__name__, exc)
                self.response(address, task_id, "failed", error="%s: %s" % (type(exc).__name__, exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-ip", default="192.168.168.120")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--allowed-client", default="192.168.168.130")
    parser.add_argument("--token", required=True)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("piper_timed_trajectory_bridge", anonymous=False)
    Bridge(args).spin()


if __name__ == "__main__":
    main()
