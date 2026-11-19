"""Fail-closed MoveIt collision validation shared by OrangePi bridges."""

from __future__ import annotations

import math

import numpy as np
import rospy
from moveit_msgs.msg import (
    LinkPadding,
    PlanningScene,
    PlanningSceneComponents,
    RobotState,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    ApplyPlanningSceneRequest,
    GetPlanningScene,
    GetPlanningSceneRequest,
    GetStateValidity,
    GetStateValidityRequest,
)
from sensor_msgs.msg import JointState

TOP_CHASSIS_LINK = "top_chassis_link"
# The calibrated URDF already places the physical chassis surface so that the
# recorded origin has about 5 mm real clearance.  A second 5 mm expansion put
# that origin directly on the numerical collision boundary.  Keep only a
# small modelling/encoder allowance here; operational clearance is enforced
# by geometry and by the escape-only boundary recovery.
TOP_CHASSIS_PADDING_M = 0.001
REQUIRED_COLLISION_PAIRS = (
    ("piper_link1", TOP_CHASSIS_LINK),
    ("piper_link2", TOP_CHASSIS_LINK),
)


class MoveItCollisionValidator:
    def __init__(self, joint_names, *, group_name="piper_arm", max_step_rad=0.03):
        self.joint_names = list(joint_names)
        self.group_name = str(group_name)
        self.max_step_rad = float(max_step_rad)
        self._service = None
        self._padding_applied = False
        self._required_pairs_enabled = False

    @staticmethod
    def _acm_pair(matrix, first, second):
        names = list(matrix.entry_names)
        if first not in names or second not in names:
            raise RuntimeError(
                "MoveIt ACM is missing required links %s/%s" % (first, second)
            )
        i, j = names.index(first), names.index(second)
        if len(matrix.entry_values) != len(names):
            raise RuntimeError("MoveIt ACM has inconsistent row count")
        if any(len(row.enabled) != len(names) for row in matrix.entry_values):
            raise RuntimeError("MoveIt ACM has inconsistent column count")
        return i, j

    def _ensure_required_collision_pairs(self):
        """Undo unsafe SRDF exemptions for arm-to-case collision pairs."""
        if self._required_pairs_enabled:
            return
        try:
            rospy.wait_for_service("/apply_planning_scene", timeout=3.0)
            rospy.wait_for_service("/get_planning_scene", timeout=3.0)
        except rospy.ROSException as exc:
            raise RuntimeError(
                "MoveIt planning-scene service unavailable; trajectory rejected"
            ) from exc
        request = GetPlanningSceneRequest()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        get_scene = rospy.ServiceProxy("/get_planning_scene", GetPlanningScene)
        apply_scene = rospy.ServiceProxy("/apply_planning_scene", ApplyPlanningScene)
        try:
            matrix = get_scene(request).scene.allowed_collision_matrix
            for first, second in REQUIRED_COLLISION_PAIRS:
                i, j = self._acm_pair(matrix, first, second)
                matrix.entry_values[i].enabled[j] = False
                matrix.entry_values[j].enabled[i] = False
            update = PlanningScene()
            update.is_diff = True
            update.robot_state.is_diff = True
            update.allowed_collision_matrix = matrix
            applied = apply_scene(ApplyPlanningSceneRequest(scene=update))
            if not applied.success:
                raise RuntimeError("MoveIt rejected required collision-pair update")
            verified = get_scene(request).scene.allowed_collision_matrix
        except rospy.ServiceException as exc:
            raise RuntimeError(
                "failed to enforce required MoveIt collision pairs"
            ) from exc
        for first, second in REQUIRED_COLLISION_PAIRS:
            i, j = self._acm_pair(verified, first, second)
            if verified.entry_values[i].enabled[j] or verified.entry_values[j].enabled[i]:
                raise RuntimeError(
                    "MoveIt retained unsafe collision exemption %s/%s"
                    % (first, second)
                )
        self._required_pairs_enabled = True

    def _ensure_top_chassis_padding(self):
        """Apply and verify the small chassis model-error allowance."""
        if self._padding_applied:
            return
        try:
            rospy.wait_for_service("/apply_planning_scene", timeout=3.0)
            rospy.wait_for_service("/get_planning_scene", timeout=3.0)
        except rospy.ROSException as exc:
            raise RuntimeError(
                "MoveIt planning-scene service unavailable; trajectory rejected"
            ) from exc

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.link_padding = [LinkPadding(
            link_name=TOP_CHASSIS_LINK,
            padding=TOP_CHASSIS_PADDING_M,
        )]
        apply_scene = rospy.ServiceProxy("/apply_planning_scene", ApplyPlanningScene)
        try:
            applied = apply_scene(ApplyPlanningSceneRequest(scene=scene))
        except rospy.ServiceException as exc:
            raise RuntimeError(
                "failed to apply top-chassis collision padding; trajectory rejected"
            ) from exc
        if not applied.success:
            raise RuntimeError(
                "MoveIt rejected top-chassis collision padding; trajectory rejected"
            )

        request = GetPlanningSceneRequest()
        request.components.components = PlanningSceneComponents.LINK_PADDING_AND_SCALING
        get_scene = rospy.ServiceProxy("/get_planning_scene", GetPlanningScene)
        try:
            returned = get_scene(request).scene
        except rospy.ServiceException as exc:
            raise RuntimeError(
                "failed to verify top-chassis collision padding; trajectory rejected"
            ) from exc
        padding = {
            item.link_name: float(item.padding) for item in returned.link_padding
        }.get(TOP_CHASSIS_LINK)
        if padding is None or not math.isclose(
            padding, TOP_CHASSIS_PADDING_M, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError(
                "top-chassis collision padding was not retained by MoveIt; "
                "trajectory rejected"
            )
        self._padding_applied = True

    def _proxy(self):
        self._ensure_top_chassis_padding()
        self._ensure_required_collision_pairs()
        if self._service is None:
            try:
                rospy.wait_for_service("/check_state_validity", timeout=3.0)
            except rospy.ROSException as exc:
                raise RuntimeError(
                    "MoveIt collision service unavailable; trajectory rejected"
                ) from exc
            self._service = rospy.ServiceProxy(
                "/check_state_validity", GetStateValidity, persistent=True
            )
        return self._service

    @staticmethod
    def _contact_pairs(response):
        return sorted(set(
            tuple(sorted((item.contact_body_1, item.contact_body_2)))
            for item in response.contacts
        ))

    def check_state(self, values):
        """Return validity/contact details for one exact joint state."""
        values = np.asarray(values, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("collision state must contain six finite joints")
        state = RobotState()
        state.joint_state = JointState(
            name=self.joint_names, position=values.tolist()
        )
        state.is_diff = True
        try:
            response = self._proxy()(GetStateValidityRequest(
                robot_state=state, group_name=self.group_name
            ))
        except rospy.ServiceException as exc:
            self._service = None
            self._padding_applied = False
            self._required_pairs_enabled = False
            raise RuntimeError(
                "MoveIt collision query failed; state rejected"
            ) from exc
        return {
            "valid": bool(response.valid),
            "contact_pairs": [list(pair) for pair in self._contact_pairs(response)],
            "max_contact_depth_m": max(
                (float(item.depth) for item in response.contacts), default=0.0
            ),
        }

    def validate(self, points):
        positions = [np.asarray(item["positions"], dtype=float) for item in points]
        if len(positions) < 2 or any(value.shape != (6,) for value in positions):
            raise ValueError("collision validation requires an Nx6 trajectory")
        samples = [positions[0]]
        for start, end in zip(positions, positions[1:]):
            divisions = max(
                1,
                int(math.ceil(
                    float(np.max(np.abs(end - start))) / self.max_step_rad
                )),
            )
            for index in range(1, divisions + 1):
                samples.append(start + (end - start) * (index / divisions))
        service = self._proxy()
        for index, values in enumerate(samples):
            try:
                result = self.check_state(values)
            except RuntimeError as exc:
                raise RuntimeError(
                    "MoveIt collision query failed; trajectory rejected"
                ) from exc
            if not result["valid"]:
                raise RuntimeError(
                    "MoveIt collision at sample %d/%d contacts=%s"
                    % (index, len(samples) - 1, result["contact_pairs"])
                )
        return {
            "moveit_collision_validated": True,
            "moveit_collision_samples": len(samples),
            "moveit_max_interpolation_step_rad": self.max_step_rad,
            "moveit_padded_link": TOP_CHASSIS_LINK,
            "moveit_link_padding_m": TOP_CHASSIS_PADDING_M,
            "moveit_required_collision_pairs": [
                list(pair) for pair in REQUIRED_COLLISION_PAIRS
            ],
        }
