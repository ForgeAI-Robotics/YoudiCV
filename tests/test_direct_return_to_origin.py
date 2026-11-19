import numpy as np
import pytest

from run_direct_return_to_origin import (
    build_origin_return_trajectory,
    build_transportable_origin_return_trajectory,
    enable_arm_while_holding_feedback,
    ensure_origin_clamp_closed,
    prepare_active_return_state,
    project_recovery_start_into_limits,
    set_can_joint_mode_while_holding,
    build_clearance_return_trajectory,
    select_collision_checked_origin_return,
)


ORIGIN = np.array([0.04, 0.105, 0.0, 0.051, 0.204, 0.0])
J3_UPPER = np.deg2rad(2.)


def test_j3_positive_two_degrees_is_legal_without_projecting_encoder_pose():
    from piper_handeye.robot import PIPER_JOINT_LIMITS
    assert PIPER_JOINT_LIMITS[2,1] == pytest.approx(J3_UPPER)
    measured=ORIGIN.copy();measured[2]=0.03145083312093782
    projected,correction=project_recovery_start_into_limits(measured,ORIGIN)
    assert np.allclose(projected,measured)
    assert np.max(np.abs(correction))==0.
    points,audit=build_origin_return_trajectory(projected,ORIGIN)
    assert points[0]['positions'][2]==pytest.approx(measured[2])


def test_clearance_path_keeps_base_fixed_until_raised_and_lowers_after_turn():
    start=ORIGIN.copy();start[0]=1.;start[1]=.05;start[2]=-.015
    points,audit,waypoints=build_clearance_return_trajectory(
        start,ORIGIN,.04,speed_m_s=.12,joint_velocity_scale=.4)
    assert waypoints[1][0]==start[0]
    assert waypoints[2][0]==ORIGIN[0]
    assert waypoints[1][1:3]==waypoints[2][1:3]
    assert waypoints[1][1]>max(start[1],ORIGIN[1])
    assert waypoints[1][2]<min(start[2],ORIGIN[2])
    assert np.allclose(points[0]['positions'],start)
    assert np.allclose(points[-1]['positions'],ORIGIN)
    assert all(b['time_from_start_sec']>a['time_from_start_sec'] for a,b in zip(points,points[1:]))
    from piper_handeye.timed_trajectory import validate_trajectory_payload,PIPER_JOINT_NAMES
    validate_trajectory_payload(dict(protocol='piper_timed_trajectory_v1',command='validate',joint_names=PIPER_JOINT_NAMES,points=points))
    for waypoint in waypoints[1:3]:
        p=next(p for p in points if np.allclose(p['positions'],waypoint))
        assert np.max(np.abs(p['velocities']))<1e-9


def test_selector_checks_entire_fallback_and_records_direct_collision():
    start=ORIGIN.copy();start[0]=1.;start[1]=.05;start[2]=-.015
    direct=build_transportable_origin_return_trajectory(start,ORIGIN,token='test')
    calls=[]
    def validate(points):
        calls.append(points)
        if len(calls)==1:raise RuntimeError('MoveIt collision at sample 34/175')
        return dict(status='validated')
    points,audit,selection,result=select_collision_checked_origin_return(start,ORIGIN,direct,validate,token='test')
    assert len(calls)==2
    assert selection['path_kind']=='raise_turn_lower'
    assert selection['rejected_paths'][0]['path_kind']=='direct'
    assert calls[-1] is points
    assert result['status']=='validated'


def test_selector_never_masks_unavailable_bridge_or_collision_service():
    direct=build_transportable_origin_return_trajectory(ORIGIN,ORIGIN,token='test')
    def unavailable(points):raise RuntimeError('MoveIt collision query failed')
    with pytest.raises(RuntimeError,match='query failed'):
        select_collision_checked_origin_return(ORIGIN,ORIGIN,direct,unavailable,token='test')


def test_selector_keeps_previously_valid_direct_return():
    direct=build_transportable_origin_return_trajectory(ORIGIN,ORIGIN,token='test')
    points,_,selection,_=select_collision_checked_origin_return(ORIGIN,ORIGIN,direct,lambda p:dict(status='validated'),token='test')
    assert points is direct[0]
    assert selection['path_kind']=='direct'


def test_selector_rejects_when_all_complete_paths_collide():
    start=ORIGIN.copy();start[0]=1.;start[1]=.05;start[2]=-.015
    direct=build_transportable_origin_return_trajectory(start,ORIGIN,token='test')
    calls=[]
    def collide(points):
        calls.append(points)
        raise RuntimeError('MoveIt collision at sample 5/100')
    with pytest.raises(RuntimeError,match='no fully collision-free'):
        select_collision_checked_origin_return(start,ORIGIN,direct,collide,token='test')
    assert len(calls)>1


def test_small_encoder_overrun_is_projected_to_boundary_for_inward_recovery():
    measured = ORIGIN.copy()
    measured[2] = J3_UPPER + 0.002
    projected, correction = project_recovery_start_into_limits(measured, ORIGIN)
    assert projected[2] == pytest.approx(J3_UPPER)
    assert correction[2] == pytest.approx(0.002)
    assert np.allclose(projected[[0, 1, 3, 4, 5]], measured[[0, 1, 3, 4, 5]])


def test_material_encoder_overrun_remains_rejected():
    measured = ORIGIN.copy()
    measured[2] = J3_UPPER + 0.006
    with pytest.raises(ValueError, match="exceeds joint limits"):
        project_recovery_start_into_limits(measured, ORIGIN)


def test_illegal_recovery_target_is_never_projected():
    target = ORIGIN.copy()
    target[2] = J3_UPPER + 0.001
    with pytest.raises(ValueError, match="target exceeds"):
        project_recovery_start_into_limits(ORIGIN, target)


def test_origin_return_uses_conservative_timing_for_joint_path_tolerance():
    start = np.array([0.0294437, 0.3338989, -0.1823869, -0.0467574, 0.1275487, -0.0903557])
    points, audit = build_origin_return_trajectory(start, ORIGIN)
    assert audit.duration_sec >= 1.5
    assert audit.point_count >= 31
    assert audit.cartesian_peak_speed_m_s <= 0.08 + 1e-9
    assert np.max(audit.joint_peak_velocity_rad_s / np.array([0.6, 1.0, 1.0, 1.0, 1.0, 2.0])) <= 0.25 + 1e-9
    assert np.allclose(points[0]["positions"], start)
    assert np.allclose(points[-1]["positions"], ORIGIN)


def test_long_origin_return_can_fit_udp_packet_with_bounded_joint_speed():
    from piper_handeye.timed_trajectory import (
        PIPER_JOINT_NAMES,
        encode_trajectory_udp_payload,
    )

    start = np.array([
        0.0041015, 1.7649817, -1.2168436,
        0.0567930, -0.7326718, 0.0347495,
    ])
    points, audit = build_origin_return_trajectory(
        start, ORIGIN, speed_m_s=0.15, joint_velocity_scale=0.40
    )
    payload = {
        "protocol": "piper_timed_trajectory_v1",
        "command": "validate",
        "joint_names": PIPER_JOINT_NAMES,
        "points": points,
        "task_id": "large-return-packet-test",
    }
    encoded, _, _ = encode_trajectory_udp_payload(payload)
    assert audit.point_count < 200
    assert len(encoded) < 60_000
    assert np.max(
        audit.joint_peak_velocity_rad_s / np.array([0.6, 1.0, 1.0, 1.0, 1.0, 2.0])
    ) <= 0.40 + 1e-9


def test_gui_return_keeps_requested_speed_with_bulk_transport():
    start = np.array([
        0.0041015, 1.7649817, -1.2168436,
        0.0567930, -0.7326718, 0.0347495,
    ])
    recorded_origin = np.array([
        -0.0321490, 0.1065524, -0.0004538,
        0.0, 0.1151743, 0.0977559,
    ])
    from piper_handeye.timed_trajectory import (
        PIPER_JOINT_NAMES,
        encode_trajectory_transport_payload,
    )

    points, audit, selection = build_transportable_origin_return_trajectory(
        start, recorded_origin, requested_speed_m_s=0.08,
        requested_joint_velocity_scale=0.25, token="test-token",
    )
    payload = {
        "protocol": "piper_timed_trajectory_v1",
        "command": "validate",
        "joint_names": PIPER_JOINT_NAMES,
        "token": "test-token",
        "task_id": "direct-return-payload-fit-" + ("x" * 48),
        "points": points,
    }
    encoded, _, _ = encode_trajectory_transport_payload(payload)
    assert selection["selected_joint_velocity_scale"] == 0.25
    assert selection["selected_speed_m_s"] == 0.08
    assert len(encoded) <= selection["payload_budget_bytes"]
    assert audit.point_count > 200
    assert len(encoded)>60_000


class FakeEnableRobot:
    def __init__(self):
        self.enable_calls = 0

    def enable(self):
        self.enable_calls += 1


class FakeHoldClient:
    def __init__(self, before, after=None):
        self.before = list(before)
        self.after = list(after if after is not None else before)
        self.calls = []

    def live_joints(self, *, max_age_sec):
        self.calls.append(("read", max_age_sec))
        values = self.before if self.calls.count(("read", max_age_sec)) == 1 else self.after
        return list(values)

    def enable(self):
        self.calls.append(("hold_enable",))

    def disable(self):
        self.calls.append(("hold_disable",))


def test_vendor_enable_completes_before_returning_encoder_pose():
    robot = FakeEnableRobot()
    live = FakeHoldClient(ORIGIN)
    report = enable_arm_while_holding_feedback(robot, live)
    assert robot.enable_calls == 1
    assert report["vendor_enable_tree_completed"] is True
    assert report["max_enable_drift_rad"] == pytest.approx(0.0)
    assert [call[0] for call in live.calls] == ["read", "read"]


def test_motor_enable_aborts_and_restores_trajectory_mode_if_pose_drifts():
    robot = FakeEnableRobot()
    moved = ORIGIN.copy()
    moved[1] += 0.03
    live = FakeHoldClient(ORIGIN, moved)
    with pytest.raises(RuntimeError, match="moved while enabling"):
        enable_arm_while_holding_feedback(robot, live)
    assert robot.enable_calls == 1
    assert [call[0] for call in live.calls] == ["read", "read"]


def test_return_enables_piper_before_guarded_boundary_recovery():
    order = []

    class Robot(FakeEnableRobot):
        def state(self, *, max_age):
            order.append("read_hardware_state")
            return type("State", (), {"raw": {"hardware_ok": 1}})()

        def enable(self):
            super().enable()
            order.append("vendor_enable")

    class Live(FakeHoldClient):
        def recover_top_chassis_boundary(self):
            order.append("recover_boundary")
            return {"status": {"last_boundary_recovery": {"status": "already_clear"}}}

        def live_joints(self, *, max_age_sec):
            order.append("read_live_joints")
            return list(ORIGIN)

    robot = Robot()
    hardware, enabled, recovery, joints = prepare_active_return_state(
        robot, Live(ORIGIN)
    )
    assert hardware.raw["hardware_ok"] == 1
    assert enabled["vendor_enable_tree_completed"] is True
    assert recovery["status"] == "already_clear"
    assert joints == pytest.approx(ORIGIN)
    assert order.index("vendor_enable") < order.index("recover_boundary")


def test_return_does_not_enable_when_hardware_health_fails():
    order = []

    class Robot(FakeEnableRobot):
        def state(self, *, max_age):
            return type("State", (), {"raw": {"hardware_ok": 0}})()

        def enable(self):
            order.append("enable")

    class Live:
        def recover_top_chassis_boundary(self):
            order.append("recover")

    with pytest.raises(RuntimeError, match="hardware_ok=0"):
        prepare_active_return_state(Robot(), Live())
    assert order == []


class FakeClampClient:
    initial_status = {"clamp_open": True, "gripper_position": 1.39}
    close_status = {"clamp_open": False, "gripper_position": 0.02}
    close_calls = 0

    def __init__(self, *_args, **_kwargs):
        self.closed = False

    def status(self):
        return {"status": dict(self.initial_status)}

    def close_clamp(self):
        type(self).close_calls += 1
        return {"status": dict(self.close_status)}

    def close(self):
        self.closed = True


class FakeLiveClient:
    def __init__(self, status):
        self._status = status

    def status(self):
        return {"status": self._status}


class FakeModeHoldClient:
    def __init__(self, joints):
        self.joints = list(joints)
        self.active = False
        self.calls = []

    def live_joints(self, *, max_age_sec):
        self.calls.append(("live", max_age_sec))
        return list(self.joints)

    def enable(self):
        self.active = True
        self.calls.append(("enable",))

    def status(self):
        return {"status": {
            "active": self.active,
            "joint_target_rad": list(self.joints),
        }}

    def update_joints(self, *, velocities_rad_s):
        self.calls.append(("heartbeat", tuple(velocities_rad_s)))

    def disable(self):
        self.active = False
        self.calls.append(("disable",))


def test_can_mode_switch_is_verified_while_live_pose_is_held(monkeypatch):
    import subprocess
    import run_direct_return_to_origin as module

    client = FakeModeHoldClient(ORIGIN)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command[-1])
        if command[-1].startswith("timeout 1 candump"):
            lines = [
                f"(1.000000) can0 2A1#0100010200000000"
                for _ in range(3)
            ]
            return subprocess.CompletedProcess(command, 0, "\n".join(lines), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = set_can_joint_mode_while_holding(client)
    assert result["control_mode"] == "can_joint"
    assert result["control_mode_feedback"] == [1, 1, 1]
    assert calls[0] == "cansend can0 151#0101320000000000"
    assert client.calls[0] == ("live", 0.25)
    assert client.calls[1] == ("enable",)
    assert client.calls[-1] == ("disable",)


def test_can_mode_switch_fails_closed_when_feedback_stays_in_teach_mode(monkeypatch):
    import subprocess
    import run_direct_return_to_origin as module

    client = FakeModeHoldClient(ORIGIN)

    def fake_run(command, **kwargs):
        output = ""
        if command[-1].startswith("timeout 1 candump"):
            output = "\n".join(
                "(1.000000) can0 2A1#0200010200000000" for _ in range(3)
            )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="did not confirm CAN joint mode"):
        set_can_joint_mode_while_holding(client)
    assert client.calls[-1] == ("disable",)


def test_live_joint_reader_requires_fresh_six_axis_feedback():
    from piper_handeye.arm_gamepad_udp import ArmGamepadUdpClient

    client = FakeLiveClient({
        "actual_joints_rad": ORIGIN.tolist(),
        "joint_feedback_age_sec": 0.02,
    })
    assert ArmGamepadUdpClient.live_joints(client) == pytest.approx(ORIGIN)


def test_live_joint_reader_rejects_stale_feedback():
    from piper_handeye.arm_gamepad_udp import ArmGamepadUdpClient

    client = FakeLiveClient({
        "actual_joints_rad": ORIGIN.tolist(),
        "joint_feedback_age_sec": 0.40,
    })
    with pytest.raises(RuntimeError, match="stale"):
        ArmGamepadUdpClient.live_joints(client, max_age_sec=0.25)


def test_origin_clamp_uses_deterministic_close_when_open():
    FakeClampClient.initial_status = {
        "clamp_open": True, "gripper_position": 1.39,
    }
    FakeClampClient.close_status = {
        "clamp_open": False, "gripper_position": 0.02,
    }
    FakeClampClient.close_calls = 0
    result = ensure_origin_clamp_closed(
        {"host": "test", "port": 1, "token": "x"},
        client_factory=FakeClampClient,
    )
    assert result["command_sent"] is True
    assert FakeClampClient.close_calls == 1


def test_origin_clamp_does_not_toggle_when_already_closed():
    FakeClampClient.initial_status = {
        "clamp_open": False, "gripper_position": 0.01,
    }
    FakeClampClient.close_calls = 0
    result = ensure_origin_clamp_closed(
        {"host": "test", "port": 1, "token": "x"},
        client_factory=FakeClampClient,
    )
    assert result["command_sent"] is False
    assert FakeClampClient.close_calls == 0


def test_origin_clamp_rejects_unconfirmed_close():
    FakeClampClient.initial_status = {
        "clamp_open": True, "gripper_position": 1.39,
    }
    FakeClampClient.close_status = {
        "clamp_open": True, "gripper_position": 1.20,
    }
    with pytest.raises(RuntimeError, match="not confirmed"):
        ensure_origin_clamp_closed(
            {"host": "test", "port": 1, "token": "x"},
            client_factory=FakeClampClient,
        )
