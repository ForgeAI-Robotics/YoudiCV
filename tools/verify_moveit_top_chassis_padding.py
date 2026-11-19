#!/usr/bin/env python3
"""Apply/verify top-chassis padding and collision-check a legal origin pose."""

import json

import rospy

from piper_moveit_collision import (
    MoveItCollisionValidator,
    TOP_CHASSIS_LINK,
    TOP_CHASSIS_PADDING_M,
)


JOINT_NAMES = [f"piper_joint{index}" for index in range(1, 7)]
ORIGIN_JOINTS = [0.04, 0.105, 0.0, 0.051, 0.204, 0.0]
BUTTON8_POINTING_JOINTS = [0.04, 0.537, -0.811, 0.138, 0.197, -0.077]


def main():
    rospy.init_node("verify_moveit_top_chassis_padding", anonymous=True)
    validator = MoveItCollisionValidator(JOINT_NAMES)
    def point(positions, stamp):
        return {
            "positions": positions,
            "velocities": [0.0] * 6,
            "accelerations": [0.0] * 6,
            "time_from_start_sec": stamp,
        }

    audit = validator.validate([
        point(ORIGIN_JOINTS, 0.0),
        point(BUTTON8_POINTING_JOINTS, 1.0),
        point(ORIGIN_JOINTS, 2.0),
    ])
    if audit.get("moveit_padded_link") != TOP_CHASSIS_LINK:
        raise RuntimeError("unexpected padded link in collision audit")
    if audit.get("moveit_link_padding_m") != TOP_CHASSIS_PADDING_M:
        raise RuntimeError("unexpected top-chassis padding in collision audit")
    print(json.dumps({
        "status": "verified",
        "physical_motion": False,
        "path": "button8_recorded_origin_to_pointing_and_return",
        **audit,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
