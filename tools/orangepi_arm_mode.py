#!/usr/bin/env python3
"""Fail-closed handoff from manual ROS joint control to arm actuation."""

import math


TRAJECTORY_CONTROLLER = "joint_trajectory_controller"
POSITION_CONTROLLER = "joint_group_position_controller"
PIPER_MODE_CTRL_FRAME = "151#0101320000000000"


def _max_error(first, second):
    if len(first) != 6 or len(second) != 6:
        raise ValueError("arm mode handoff requires six joint positions")
    values = [abs(float(a) - float(b)) for a, b in zip(first, second)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("arm mode handoff received non-finite joint positions")
    return max(values)


def activate_actuation_mode(
    expected_start,
    *,
    controller_states,
    current_joints,
    validate_state,
    hold_current_pose,
    send_can_joint_mode,
    read_mode_feedback,
    switch_to_trajectory,
    start_tolerance_rad=0.03,
    hold_drift_tolerance_rad=0.02,
):
    """Reconcile ROS controllers and Piper CAN mode before an action goal.

    A handoff is required whenever ROS is in position/manual control, the
    trajectory controller is not exclusively active, or Piper reports a
    non-joint command mode.  During every detected handoff this explicitly
    seeds a collision-checked live-pose hold, sends the CAN joint-mode command
    even when the last reported CAN mode was already joint mode, verifies
    fresh mode feedback and encoder drift, and only then activates trajectory
    control.  On any failed check the caller must not submit an action goal.
    """
    states_before = dict(controller_states())
    modes_before = list(read_mode_feedback())
    if len(modes_before) < 3:
        raise RuntimeError("fewer than three fresh Piper mode frames received")

    manual_controller_active = states_before.get(POSITION_CONTROLLER) == "running"
    trajectory_controller_inactive = (
        states_before.get(TRAJECTORY_CONTROLLER) != "running"
    )
    can_mode_inactive = any(int(mode) != 0x01 for mode in modes_before[-3:])
    transition_required = (
        manual_controller_active
        or trajectory_controller_inactive
        or can_mode_inactive
    )

    held = list(current_joints())
    start_error = _max_error(held, expected_start)
    if start_error > float(start_tolerance_rad):
        raise RuntimeError(
            "actuation start differs from live arm pose: %.6f rad" % start_error
        )
    if not transition_required:
        return {
            "transition": "already_in_actuation_mode",
            "changed": False,
            "controller_states": states_before,
            "ctrl_mode_feedback": [int(value) for value in modes_before[-3:]],
            "start_error_rad": start_error,
        }

    validate_state(held)
    hold_current_pose(held)
    # This explicit command is intentional on every manual -> actuation
    # handoff.  The ROS controller state and Piper's CAN mode are independent.
    send_can_joint_mode()
    modes_after = list(read_mode_feedback())
    if len(modes_after) < 3 or any(
        int(mode) != 0x01 for mode in modes_after[-3:]
    ):
        raise RuntimeError(
            "Piper did not confirm joint actuation mode; 0x2A1 ctrl_mode=%r"
            % modes_after[-3:]
        )

    held_after = list(current_joints())
    hold_drift = _max_error(held_after, held)
    start_error_after_hold = _max_error(held_after, expected_start)
    if hold_drift > float(hold_drift_tolerance_rad):
        raise RuntimeError(
            "arm drifted during manual-to-actuation handoff: %.6f rad"
            % hold_drift
        )
    if start_error_after_hold > float(start_tolerance_rad):
        raise RuntimeError(
            "arm moved away from action start during mode handoff: %.6f rad"
            % start_error_after_hold
        )

    switch_to_trajectory()
    states_after = dict(controller_states())
    if (
        states_after.get(TRAJECTORY_CONTROLLER) != "running"
        or states_after.get(POSITION_CONTROLLER) == "running"
    ):
        raise RuntimeError(
            "ROS trajectory controller handoff was not confirmed: %r"
            % states_after
        )
    final_joints = list(current_joints())
    final_error = _max_error(final_joints, expected_start)
    if final_error > float(start_tolerance_rad):
        raise RuntimeError(
            "arm moved away from action start when enabling trajectory control: %.6f rad"
            % final_error
        )
    return {
        "transition": "manual_to_actuation",
        "changed": True,
        "controller_states_before": states_before,
        "controller_states_after": states_after,
        "ctrl_mode_feedback_before": [int(value) for value in modes_before[-3:]],
        "ctrl_mode_feedback": [int(value) for value in modes_after[-3:]],
        "can_joint_mode_command_sent": True,
        "start_error_rad": start_error,
        "hold_drift_rad": hold_drift,
        "final_start_error_rad": final_error,
    }


class EscapeModeDriftError(RuntimeError):
    """Confirmed CAN transition, but the preflight start became stale."""
    def __init__(self, audit):
        self.audit = audit
        super().__init__('escape arm drifted during CAN mode switch: %.6f rad' % audit['max_hold_drift_rad'])


def confirm_escape_actuation_mode(expected_start, *, controller_states,
                                 current_joints, send_can_joint_mode,
                                 read_mode_feedback, max_drift_rad=.003):
    """Called only after the full escape preflight and live position hold.

    Keep the position controller active throughout escape. This does not
    validate or exempt collisions: the caller already checked the escape
    path, and must refuse to publish moving targets on any failure here.
    """
    states = dict(controller_states())
    if (states.get(POSITION_CONTROLLER) != 'running'
            or states.get(TRAJECTORY_CONTROLLER) == 'running'):
        raise RuntimeError('escape live-pose position hold not confirmed')
    if _max_error(current_joints(), expected_start) > max_drift_rad:
        raise RuntimeError('escape start changed before CAN mode switch')
    before = list(read_mode_feedback())
    if len(before) < 3:
        raise RuntimeError('escape needs three fresh CAN mode frames')
    changed = any(int(m) != 1 for m in before[-3:])
    if changed:
        send_can_joint_mode()
    after = list(read_mode_feedback())
    if len(after) < 3 or any(int(m) != 1 for m in after[-3:]):
        raise RuntimeError('escape CAN joint mode not confirmed: %r' % after[-3:])
    actual = list(current_joints())
    drift = _max_error(actual, expected_start)
    audit = dict(ctrl_mode_before=before[-3:], ctrl_mode_after=after[-3:],
                 can_joint_mode_command_sent=changed, max_hold_drift_rad=drift,
                 held_joints_rad=list(expected_start), actual_joints_rad=actual)
    if drift > max_drift_rad:
        if changed:
            raise EscapeModeDriftError(audit)
        raise RuntimeError('escape arm moved despite unchanged CAN mode: %.6f rad' % drift)
    return audit
