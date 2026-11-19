from types import SimpleNamespace

import pytest

from piper_handeye.rospy_twist import RospyTwistPublisher


class FakeMaster:
    def __init__(self, state):
        self.state = state

    def getSystemState(self):
        return 1, "ok", self.state


SAFE_PARAMS = {
    "/velocity_smoother/speed_lim_v": 0.1,
    "/velocity_smoother/speed_lim_w": 0.4,
    "/velocity_smoother/accel_lim_v": 0.1,
    "/velocity_smoother/use_calmanFilter": False,
    "/velocity_smoother/use_movingFilter": False,
    "/velocity_smoother/use_limit_speedup": False,
}


def publisher_with_graph(state, params=None):
    publisher = object.__new__(RospyTwistPublisher)
    values = dict(SAFE_PARAMS if params is None else params)
    publisher.rospy = SimpleNamespace(
        get_master=lambda: FakeMaster(state),
        get_param=lambda name, default=None: values.get(name, default),
    )
    return publisher


def test_verified_vendor_chassis_pipeline():
    state = (
        [
            ["/cmd_vel", ["/dualsense_chassis_gui_1"]],
            ["/plt_cmd_vel", ["/velocity_smoother"]],
        ],
        [
            ["/cmd_vel", ["/velocity_smoother", "/bag_server"]],
            ["/plt_cmd_vel", ["/chassis_server_node"]],
        ],
        [],
    )
    graph = publisher_with_graph(state).verify_vendor_chassis_pipeline()
    assert graph["cmd_vel_subscribers"] == (
        "/velocity_smoother", "/bag_server"
    )
    assert graph["cmd_vel_publishers"] == ("/dualsense_chassis_gui_1",)
    assert graph["plt_cmd_vel_publishers"] == ("/velocity_smoother",)
    assert graph["plt_cmd_vel_subscribers"] == ("/chassis_server_node",)


def test_pipeline_rejects_missing_chassis_driver():
    state = (
        [["/plt_cmd_vel", ["/velocity_smoother"]]],
        [["/cmd_vel", ["/velocity_smoother"]]],
        [],
    )
    with pytest.raises(RuntimeError, match="chassis_server_node"):
        publisher_with_graph(state).verify_vendor_chassis_pipeline()


def test_pipeline_rejects_keyboard_teleop_that_overwrites_gamepad():
    state = (
        [
            ["/cmd_vel", ["/dualsense_chassis_gui_1", "/RosAria"]],
            ["/plt_cmd_vel", ["/velocity_smoother"]],
        ],
        [
            ["/cmd_vel", ["/velocity_smoother"]],
            ["/plt_cmd_vel", ["/chassis_server_node"]],
        ],
        [],
    )
    with pytest.raises(RuntimeError, match="100 Hz"):
        publisher_with_graph(state).verify_vendor_chassis_pipeline()


def test_pipeline_rejects_delayed_filter_configuration():
    state = (
        [
            ["/cmd_vel", ["/dualsense_chassis_gui_1"]],
            ["/plt_cmd_vel", ["/velocity_smoother"]],
        ],
        [
            ["/cmd_vel", ["/velocity_smoother"]],
            ["/plt_cmd_vel", ["/chassis_server_node"]],
        ],
        [],
    )
    params = dict(SAFE_PARAMS)
    params["/velocity_smoother/use_calmanFilter"] = True
    with pytest.raises(RuntimeError, match="use_calmanFilter"):
        publisher_with_graph(state, params).verify_vendor_chassis_pipeline()


def test_refresh_subscriber_node_accepts_bounded_handshake_timeout():
    publisher = object.__new__(RospyTwistPublisher)

    class Connections:
        def __init__(self):
            self.calls = 0

        @property
        def connections(self):
            self.calls += 1
            if self.calls < 2:
                return ()
            return (SimpleNamespace(_connection=SimpleNamespace(
                endpoint_id="/gamepad_smoother_runtime_manager"
            )),)

    publisher._publisher = SimpleNamespace(impl=Connections())
    names = publisher.refresh_subscriber_node(
        "/gamepad_smoother_runtime_manager", timeout_sec=0.2
    )
    assert names == ("/gamepad_smoother_runtime_manager",)
