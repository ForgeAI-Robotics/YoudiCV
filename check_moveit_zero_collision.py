#!/usr/bin/env python3
"""Read-only MoveIt audit from a recorded Piper state to six-axis zero."""

import json
import sys
import xml.etree.ElementTree as ET

import rospy
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PlanningSceneComponents,
    RobotState,
)
from moveit_msgs.srv import (
    GetMotionPlan,
    GetMotionPlanRequest,
    GetPlanningScene,
    GetPlanningSceneRequest,
    GetStateValidity,
    GetStateValidityRequest,
)
from sensor_msgs.msg import JointState


JOINTS = ["piper_joint%d" % index for index in range(1, 7)]


def robot_state(values):
    state = RobotState()
    state.joint_state = JointState(name=JOINTS, position=list(values))
    state.is_diff = True
    return state


def contact_pairs(response):
    return sorted(
        set(
            tuple(sorted((contact.contact_body_1, contact.contact_body_2)))
            for contact in response.contacts
        )
    )


def main():
    if len(sys.argv) != 7:
        raise SystemExit("expected six current joint values")
    start = [float(value) for value in sys.argv[1:]]
    goal = [0.0] * 6
    rospy.init_node("moveit_zero_collision_audit", anonymous=True, disable_signals=True)
    rospy.wait_for_service("/check_state_validity", timeout=10.0)
    rospy.wait_for_service("/plan_kinematic_path", timeout=10.0)
    rospy.wait_for_service("/get_planning_scene", timeout=10.0)
    validity = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
    planner = rospy.ServiceProxy("/plan_kinematic_path", GetMotionPlan)
    scene_service = rospy.ServiceProxy("/get_planning_scene", GetPlanningScene)

    def check(values):
        request = GetStateValidityRequest(
            robot_state=robot_state(values), group_name="piper_arm"
        )
        return validity(request)

    start_check = check(start)
    goal_check = check(goal)
    interpolation_invalid = []
    for index in range(101):
        fraction = index / 100.0
        values = [a + fraction * (b - a) for a, b in zip(start, goal)]
        response = check(values)
        if not response.valid:
            interpolation_invalid.append(
                {"index": index, "fraction": fraction, "contacts": contact_pairs(response)}
            )

    plan_request = GetMotionPlanRequest()
    motion = plan_request.motion_plan_request
    motion.group_name = "piper_arm"
    motion.start_state = robot_state(start)
    motion.num_planning_attempts = 5
    motion.allowed_planning_time = 5.0
    constraints = Constraints()
    for name, position in zip(JOINTS, goal):
        constraints.joint_constraints.append(
            JointConstraint(
                joint_name=name,
                position=position,
                tolerance_above=0.001,
                tolerance_below=0.001,
                weight=1.0,
            )
        )
    motion.goal_constraints = [constraints]
    planned = planner(plan_request).motion_plan_response
    trajectory_points = list(planned.trajectory.joint_trajectory.points)
    planned_invalid = []
    for index, point in enumerate(trajectory_points):
        response = check(point.positions[:6])
        if not response.valid:
            planned_invalid.append(
                {"index": index, "contacts": contact_pairs(response)}
            )

    components = PlanningSceneComponents()
    components.components = (
        PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        | PlanningSceneComponents.OCTOMAP
        | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        | PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
    )
    scene = scene_service(GetPlanningSceneRequest(components=components)).scene

    urdf = ET.fromstring(rospy.get_param("/robot_description"))
    links = [element.attrib["name"] for element in urdf.findall("link")]
    collision_links = [
        element.attrib["name"]
        for element in urdf.findall("link")
        if element.find("collision") is not None
    ]
    result = {
        "mode": "planning_and_collision_check_only_no_execution",
        "start_joints_rad": start,
        "goal_joints_rad": goal,
        "start_valid": bool(start_check.valid),
        "start_contacts": contact_pairs(start_check),
        "goal_valid": bool(goal_check.valid),
        "goal_contacts": contact_pairs(goal_check),
        "linear_interpolation_samples": 101,
        "linear_interpolation_invalid_count": len(interpolation_invalid),
        "linear_interpolation_invalid": interpolation_invalid,
        "moveit_plan_success": planned.error_code.val == MoveItErrorCodes.SUCCESS,
        "moveit_error_code": int(planned.error_code.val),
        "moveit_planning_time_sec": float(planned.planning_time),
        "moveit_trajectory_points": len(trajectory_points),
        "moveit_trajectory_invalid_count": len(planned_invalid),
        "moveit_trajectory_invalid": planned_invalid,
        "urdf_link_count": len(links),
        "urdf_collision_link_count": len(collision_links),
        "urdf_collision_links": collision_links,
        "urdf_links_without_collision": sorted(set(links) - set(collision_links)),
        "planning_scene_world_collision_objects": [
            item.id for item in scene.world.collision_objects
        ],
        "planning_scene_attached_collision_objects": [
            item.object.id for item in scene.robot_state.attached_collision_objects
        ],
        "planning_scene_octomap_bytes": len(scene.world.octomap.octomap.data),
        "allowed_collision_matrix_entries": len(
            scene.allowed_collision_matrix.entry_names
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
