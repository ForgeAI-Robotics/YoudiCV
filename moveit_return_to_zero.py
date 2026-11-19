#!/usr/bin/env python3
"""Plan, revalidate, and execute a Piper return to six-axis zero with MoveIt."""

import json
import sys

import moveit_commander
import rospy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from sensor_msgs.msg import JointState


JOINTS = ["piper_joint%d" % index for index in range(1, 7)]
EXPECTED_START = [0.041, 0.106, 0.0, 0.052, 0.204, 0.0]
GOAL = [0.0] * 6


def state(values):
    result = RobotState()
    result.joint_state = JointState(name=JOINTS, position=list(values))
    result.is_diff = True
    return result


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("moveit_return_to_six_axis_zero", anonymous=True)
    group = moveit_commander.MoveGroupCommander("piper_arm")
    group.set_planning_time(5.0)
    group.set_num_planning_attempts(5)
    group.set_max_velocity_scaling_factor(0.20)
    group.set_max_acceleration_scaling_factor(0.20)
    current = list(group.get_current_joint_values()[:6])
    start_error = max(abs(a - b) for a, b in zip(current, EXPECTED_START))
    if start_error > 0.03:
        raise RuntimeError("live start differs from recorded state: %.4f rad" % start_error)
    group.set_start_state_to_current_state()
    group.set_joint_value_target(dict(zip(JOINTS, GOAL)))
    planned = group.plan()
    if isinstance(planned, tuple):
        success, trajectory = bool(planned[0]), planned[1]
    else:
        trajectory = planned
        success = bool(trajectory.joint_trajectory.points)
    points = list(trajectory.joint_trajectory.points)
    if not success or not points:
        raise RuntimeError("MoveIt could not plan a nonempty zero trajectory")

    rospy.wait_for_service("/check_state_validity", timeout=10.0)
    validity = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
    invalid = []
    names = list(trajectory.joint_trajectory.joint_names)
    for index, point in enumerate(points):
        positions = dict(zip(names, point.positions))
        values = [positions[name] for name in JOINTS]
        response = validity(
            GetStateValidityRequest(robot_state=state(values), group_name="piper_arm")
        )
        if not response.valid:
            invalid.append(index)
    if invalid:
        raise RuntimeError("planned trajectory contains invalid points: %r" % invalid)

    executed = bool(group.execute(trajectory, wait=True))
    group.stop()
    final = list(group.get_current_joint_values()[:6])
    final_error = max(abs(value) for value in final)
    report = {
        "status": "returned_to_six_axis_zero" if executed and final_error <= 0.03 else "failed",
        "moveit_execution_success": executed,
        "start_joints_rad": current,
        "goal_joints_rad": GOAL,
        "trajectory_points": len(points),
        "trajectory_invalid_points": invalid,
        "velocity_scaling": 0.20,
        "acceleration_scaling": 0.20,
        "final_joints_rad": final,
        "max_zero_error_rad": final_error,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not executed or final_error > 0.03:
        raise RuntimeError("zero return tracking failed: %.4f rad" % final_error)


if __name__ == "__main__":
    main()
