#!/usr/bin/env python3
"""Submit the live Piper motor joint state to MoveIt without modification."""

import json

import rospy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from sensor_msgs.msg import JointState


JOINT_NAMES = ["piper_joint%d" % index for index in range(1, 7)]


def main():
    rospy.init_node("codex_live_motor_moveit_check", anonymous=True)
    message = rospy.wait_for_message(
        "/piper_arm/joint_states", JointState, timeout=5.0
    )
    by_name = dict(zip(message.name, message.position))
    joints = [float(by_name[name]) for name in JOINT_NAMES]
    state = RobotState()
    state.joint_state = JointState(name=JOINT_NAMES, position=joints)
    state.is_diff = True
    rospy.wait_for_service("/check_state_validity", timeout=5.0)
    response = rospy.ServiceProxy(
        "/check_state_validity", GetStateValidity
    )(GetStateValidityRequest(robot_state=state, group_name="piper_arm"))
    print(json.dumps({
        "joint_names": JOINT_NAMES,
        "live_motor_joints_rad": joints,
        "moveit_valid": bool(response.valid),
        "contacts": [{
            "body_1": item.contact_body_1,
            "body_2": item.contact_body_2,
            "depth_m": float(item.depth),
            "position": [item.position.x, item.position.y, item.position.z],
            "normal": [item.normal.x, item.normal.y, item.normal.z],
        } for item in response.contacts],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
