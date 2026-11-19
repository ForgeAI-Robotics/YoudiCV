#!/usr/bin/env python3
"""Independent authenticated cancel channel for visual safety stops.

This process is deliberately separate from both trajectory bridges, so it can
publish an actionlib cancel even while either bridge is blocked waiting for a
trajectory result.  It does not resume motion; reset_latch only clears status.
"""

import argparse
import json
import socket
import time

from piper_handeye.cycle_fault_recovery import require_reset_identity

import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction
from sensor_msgs.msg import JointState


class SafetyStopBridge:
    def __init__(self, args):
        self.args = args
        self.client = actionlib.SimpleActionClient(
            "/piper_arm/joint_trajectory_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError("joint trajectory action server is unavailable")
        self.latched = bool(
            rospy.get_param("/piper_visual_safety_stop/latched", False)
        )
        rospy.set_param("/piper_visual_safety_stop/latched", self.latched)
        self.stop_generation = int(
            rospy.get_param("/piper_visual_safety_stop/generation", 0)
        )
        rospy.set_param(
            "/piper_visual_safety_stop/generation", self.stop_generation
        )
        self.last_stop = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((args.bind_ip, args.port))
        self.sock.settimeout(0.5)

    def response(self, address, task_id, status, **extra):
        value = {"task_id": task_id, "status": status}
        value.update(extra)
        self.sock.sendto(
            json.dumps(value, separators=(",", ":")).encode("utf-8"), address
        )

    def current_joints(self):
        state = rospy.wait_for_message(
            "/piper_arm/joint_states", JointState, timeout=1.0
        )
        positions = dict(zip(state.name, state.position))
        names = ["piper_joint%d" % index for index in range(1, 7)]
        return [float(positions[name]) for name in names]

    def stop(self, payload):
        received = time.time()
        # GoalID stamp zero semantics cancel every goal on this action server,
        # including goals owned by the timed and prepared-splice bridge nodes.
        self.client.cancel_all_goals()
        self.latched = True
        self.stop_generation += 1
        rospy.set_param("/piper_visual_safety_stop/latched", True)
        rospy.set_param(
            "/piper_visual_safety_stop/generation", self.stop_generation
        )
        self.last_stop = {
            "received_unix_sec": received,
            "reason": str(payload.get("reason", "visual_safety")),
            "frame_sequence": int(payload.get("frame_sequence", 0)),
        }
        rospy.logwarn("VISUAL SAFETY STOP: %r", self.last_stop)
        return self.current_joints()

    def spin(self):
        rospy.loginfo(
            "visual safety stop bridge listening on %s:%d",
            self.args.bind_ip,
            self.args.port,
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
                if payload.get("protocol") != "piper_visual_safety_v1":
                    raise ValueError("unsupported protocol")
                if payload.get("token") != self.args.token:
                    raise ValueError("invalid token")
                command = payload.get("command")
                if command == "health":
                    self.response(
                        address, task_id, "ok", latched=self.latched,
                        last_stop=self.last_stop, generation=self.stop_generation,
                        conditional_reset_v1=True,
                    )
                elif command == "reset_latch_if_generation":
                    require_reset_identity(payload, self.stop_generation, self.last_stop)
                    self.latched = False
                    rospy.set_param("/piper_visual_safety_stop/latched", False)
                    self.response(address, task_id, "ok", latched=False,
                                  generation=self.stop_generation)
                elif command == "reset_latch":
                    self.latched = False
                    rospy.set_param("/piper_visual_safety_stop/latched", False)
                    self.response(address, task_id, "ok", latched=False)
                elif command == "stop":
                    joints = self.stop(payload)
                    self.response(
                        address, task_id, "stopped", latched=True,
                        joints_rad=joints, last_stop=self.last_stop,
                        generation=self.stop_generation,
                    )
                else:
                    raise ValueError("unsupported command")
            except Exception as exc:
                rospy.logerr("safety request failed: %s: %s", type(exc).__name__, exc)
                self.response(
                    address, task_id, "failed",
                    error="%s: %s" % (type(exc).__name__, exc),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-ip", default="192.168.168.120")
    parser.add_argument("--port", type=int, default=5013)
    parser.add_argument("--allowed-client", default="192.168.168.130")
    parser.add_argument("--token", required=True)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("piper_visual_safety_stop_bridge", anonymous=False)
    SafetyStopBridge(args).spin()


if __name__ == "__main__":
    main()
