import pytest

from tools.orangepi_arm_mode import activate_actuation_mode


ORIGIN = [0.1, 0.2, -0.3, 0.0, 0.4, 0.0]


def make_gate(*, manual=True, modes=None, drift=0.0):
    calls = []
    state = {
        "controllers": {
            "joint_trajectory_controller": "stopped" if manual else "running",
            "joint_group_position_controller": "running" if manual else "stopped",
        },
        "joints": list(ORIGIN),
        "modes": list(modes or (1, 1, 1)),
    }

    def read_controllers():
        return dict(state["controllers"])

    def read_joints():
        return list(state["joints"])

    def hold(joints):
        calls.append(("hold", list(joints)))
        state["controllers"] = {
            "joint_trajectory_controller": "stopped",
            "joint_group_position_controller": "running",
        }
        state["joints"] = list(joints)
        state["joints"][0] += drift

    def send_mode():
        calls.append(("send_can_mode",))
        state["modes"] = [1, 1, 1]

    def switch_trajectory():
        calls.append(("switch_trajectory",))
        state["controllers"] = {
            "joint_trajectory_controller": "running",
            "joint_group_position_controller": "stopped",
        }

    result = activate_actuation_mode(
        ORIGIN,
        controller_states=read_controllers,
        current_joints=read_joints,
        validate_state=lambda _joints: calls.append(("validate",)),
        hold_current_pose=hold,
        send_can_joint_mode=send_mode,
        read_mode_feedback=lambda: list(state["modes"]),
        switch_to_trajectory=switch_trajectory,
    )
    return result, calls


def test_manual_ros_mode_always_resends_piper_actuation_mode():
    result, calls = make_gate(manual=True, modes=(1, 1, 1))
    assert result["transition"] == "manual_to_actuation"
    assert result["can_joint_mode_command_sent"] is True
    assert [item[0] for item in calls] == [
        "validate", "hold", "send_can_mode", "switch_trajectory"
    ]


def test_non_actuation_can_mode_is_reconciled_even_if_ros_is_in_trajectory_mode():
    result, calls = make_gate(manual=False, modes=(2, 1, 1))
    assert result["transition"] == "manual_to_actuation"
    assert ("send_can_mode",) in calls
    assert calls[-1] == ("switch_trajectory",)


def test_existing_actuation_mode_is_a_noop():
    result, calls = make_gate(manual=False, modes=(1, 1, 1))
    assert result["transition"] == "already_in_actuation_mode"
    assert result["changed"] is False
    assert calls == []


def test_handoff_stops_before_mode_command_when_live_pose_drifts():
    with pytest.raises(RuntimeError, match="drifted"):
        make_gate(manual=True, drift=0.03)


def test_escape_confirms_can_before_moving_without_trajectory_switch():
    from tools.orangepi_arm_mode import confirm_escape_actuation_mode
    events=[]
    modes=iter([[2]*3,[1]*3])
    result=confirm_escape_actuation_mode([0.]*6,
        controller_states=lambda:dict(joint_group_position_controller='running',joint_trajectory_controller='stopped'),
        current_joints=lambda:[0.]*6,
        send_can_joint_mode=lambda:events.append('send'),
        read_mode_feedback=lambda: next(modes))
    assert events==['send']
    assert result['ctrl_mode_before']==[2]*3
    assert result['ctrl_mode_after']==[1]*3


def test_escape_refuses_unconfirmed_can_mode():
    from tools.orangepi_arm_mode import confirm_escape_actuation_mode
    with pytest.raises(RuntimeError,match='CAN joint mode not confirmed'):
        confirm_escape_actuation_mode([0.]*6,
            controller_states=lambda:dict(joint_group_position_controller='running',joint_trajectory_controller='stopped'),
            current_joints=lambda:[0.]*6,send_can_joint_mode=lambda:None,
            read_mode_feedback=lambda:[2]*3)


def test_escape_does_not_send_can_without_position_hold():
    from tools.orangepi_arm_mode import confirm_escape_actuation_mode
    with pytest.raises(RuntimeError,match='hold not confirmed'):
        confirm_escape_actuation_mode([0.]*6,
            controller_states=lambda:dict(joint_group_position_controller='stopped',joint_trajectory_controller='running'),
            current_joints=lambda:[0.]*6,send_can_joint_mode=lambda:pytest.fail('mode sent without hold'),
            read_mode_feedback=lambda:[2]*3)


def test_escape_rejects_drift_during_mode_switch():
    from tools.orangepi_arm_mode import confirm_escape_actuation_mode
    poses=iter([[0.]*6,[.004]*6])
    with pytest.raises(RuntimeError,match='drifted|moved despite unchanged'):
        confirm_escape_actuation_mode([0.]*6,
            controller_states=lambda:dict(joint_group_position_controller='running',joint_trajectory_controller='stopped'),
            current_joints=lambda:next(poses),send_can_joint_mode=lambda:None,
            read_mode_feedback=lambda:[1]*3)


def test_escape_already_active_does_not_resend_mode():
    from tools.orangepi_arm_mode import confirm_escape_actuation_mode
    r=confirm_escape_actuation_mode([0.]*6,
        controller_states=lambda:dict(joint_group_position_controller='running',joint_trajectory_controller='stopped'),
        current_joints=lambda:[0.]*6,send_can_joint_mode=lambda:pytest.fail('already active'),
        read_mode_feedback=lambda:[1]*3)
    assert not r['can_joint_mode_command_sent']


def test_mode_drift_replan_requires_confirmed_transition():
    from tools.orangepi_arm_mode import confirm_escape_actuation_mode,EscapeModeDriftError
    poses=iter([[0.]*6,[0.,0.,-.005,0.,0.,0.]])
    modes=iter([[2]*3,[1]*3])
    with pytest.raises(EscapeModeDriftError) as caught:
        confirm_escape_actuation_mode([0.]*6,
            controller_states=lambda:dict(joint_group_position_controller='running',joint_trajectory_controller='stopped'),
            current_joints=lambda:next(poses),send_can_joint_mode=lambda:None,
            read_mode_feedback=lambda:next(modes))
    assert caught.value.audit['ctrl_mode_after']==[1]*3
