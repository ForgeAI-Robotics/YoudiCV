#!/usr/bin/env python3
"""Read-only contact-depth probe for a candidate Piper recovery line."""

import json

import numpy as np
import rospy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from sensor_msgs.msg import JointState


JOINT_NAMES = ["piper_joint%d" % index for index in range(1, 7)]
START = np.array([-0.02, 0.07, 0.0, 0.111, 0.256, 0.065])
ORIGIN = np.array([0.04, 0.105, 0.0, 0.051, 0.204, 0.0])


def main():
    rospy.init_node("codex_moveit_contact_depth_probe", anonymous=True)
    rospy.wait_for_service("/check_state_validity", timeout=5.0)
    check = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
    for fraction in np.linspace(0.0, 1.0, 41):
        joints = START + float(fraction) * (ORIGIN - START)
        state = RobotState()
        state.joint_state = JointState(name=JOINT_NAMES, position=joints.tolist())
        state.is_diff = True
        response = check(GetStateValidityRequest(
            robot_state=state, group_name="piper_arm"
        ))
        contacts = [{
            "pair": sorted([item.contact_body_1, item.contact_body_2]),
            "depth_m": float(item.depth),
        } for item in response.contacts]
        print(json.dumps({
            "fraction": float(fraction),
            "valid": bool(response.valid),
            "contacts": contacts,
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
