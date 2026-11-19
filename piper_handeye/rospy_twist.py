"""Standard rospy Twist publisher for the live DualSense chassis GUI."""

from __future__ import annotations

import os
import threading
import time


_INIT_LOCK = threading.Lock()


class RospyTwistPublisher:
    def __init__(
        self,
        *,
        master_uri: str,
        bind_ip: str,
        topic: str = "/cmd_vel",
    ) -> None:
        os.environ["ROS_MASTER_URI"] = master_uri
        os.environ["ROS_IP"] = bind_ip
        import rospy
        from geometry_msgs.msg import Twist

        self.rospy = rospy
        self.Twist = Twist
        self.topic = topic
        self._closed = False
        with _INIT_LOCK:
            if not rospy.core.is_initialized():
                rospy.init_node(
                    "dualsense_chassis_gui", anonymous=True, disable_signals=True
                )
        self._publisher = rospy.Publisher(
            topic, Twist, queue_size=1, tcp_nodelay=True
        )

    @property
    def subscriber_names(self) -> tuple[str, ...]:
        names = []
        impl = getattr(self._publisher, "impl", None)
        for queued in list(getattr(impl, "connections", ())):
            connection = getattr(queued, "_connection", queued)
            name = str(getattr(connection, "endpoint_id", "unknown"))
            if name not in names:
                names.append(name)
        return tuple(names)

    @property
    def subscriber_count(self) -> int:
        return int(self._publisher.get_num_connections())

    def graph_nodes(self, *, topic: str, direction: str) -> tuple[str, ...]:
        """Return ROS-master node names for one topic without shelling out."""
        code, message, state = self.rospy.get_master().getSystemState()
        if code != 1:
            raise RuntimeError(f"ROS graph query failed: {message}")
        publishers, subscribers, _services = state
        table = publishers if direction == "publishers" else subscribers
        if direction not in {"publishers", "subscribers"}:
            raise ValueError("direction must be publishers or subscribers")
        for registered_topic, nodes in table:
            if registered_topic == topic:
                return tuple(str(node) for node in nodes)
        return ()

    def verify_vendor_chassis_pipeline(self) -> dict[str, object]:
        """Verify the complete rosaria launch graph used by the chassis."""
        cmd_publishers = self.graph_nodes(
            topic="/cmd_vel", direction="publishers"
        )
        cmd_subscribers = self.graph_nodes(
            topic="/cmd_vel", direction="subscribers"
        )
        plt_publishers = self.graph_nodes(
            topic="/plt_cmd_vel", direction="publishers"
        )
        plt_subscribers = self.graph_nodes(
            topic="/plt_cmd_vel", direction="subscribers"
        )
        missing = []
        if "/velocity_smoother" not in cmd_subscribers:
            missing.append("/velocity_smoother 未订阅 /cmd_vel")
        if "/velocity_smoother" not in plt_publishers:
            missing.append("/velocity_smoother 未发布 /plt_cmd_vel")
        if "/chassis_server_node" not in plt_subscribers:
            missing.append("/chassis_server_node 未订阅 /plt_cmd_vel")
        if "/RosAria" in cmd_publishers:
            missing.append("/RosAria 键盘节点正在以 100 Hz 覆盖手柄指令")
        safety_parameters = {
            "speed_lim_v": float(self.rospy.get_param(
                "/velocity_smoother/speed_lim_v", float("inf")
            )),
            "speed_lim_w": float(self.rospy.get_param(
                "/velocity_smoother/speed_lim_w", float("inf")
            )),
            "accel_lim_v": float(self.rospy.get_param(
                "/velocity_smoother/accel_lim_v", float("inf")
            )),
            "use_calmanFilter": bool(self.rospy.get_param(
                "/velocity_smoother/use_calmanFilter", True
            )),
            "use_movingFilter": bool(self.rospy.get_param(
                "/velocity_smoother/use_movingFilter", True
            )),
            "use_limit_speedup": bool(self.rospy.get_param(
                "/velocity_smoother/use_limit_speedup", True
            )),
        }
        if safety_parameters["speed_lim_v"] > 0.10:
            missing.append("底盘线速度限制高于 0.10 m/s")
        if safety_parameters["speed_lim_w"] > 0.40:
            missing.append("底盘角速度限制高于 0.40 rad/s")
        if safety_parameters["accel_lim_v"] > 0.10:
            missing.append("底盘线加速度限制高于 0.10 m/s²")
        for name in ("use_calmanFilter", "use_movingFilter", "use_limit_speedup"):
            if safety_parameters[name]:
                missing.append(f"{name} 必须关闭以避免延迟或松手后峰值")
        if missing:
            raise RuntimeError("；".join(missing))
        return {
            "cmd_vel_publishers": cmd_publishers,
            "cmd_vel_subscribers": cmd_subscribers,
            "plt_cmd_vel_publishers": plt_publishers,
            "plt_cmd_vel_subscribers": plt_subscribers,
            "safety_parameters": safety_parameters,
        }

    def refresh_subscriber_node(
        self, subscriber_node: str, *, timeout_sec: float = 1.0
    ) -> tuple[str, ...]:
        """Wait for a named TCPROS subscriber to finish its handshake.

        A publisher created during a mode switch can be registered with the
        master before its TCPROS connection is visible in ``connections``.
        Keep the old one-second default for callers that only need a quick
        probe, while allowing safety-critical mode transitions to use a
        longer bounded wait.
        """
        timeout_sec = max(0.0, float(timeout_sec))
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            names = self.subscriber_names
            if subscriber_node in names:
                return names
            time.sleep(0.02)
        return self.subscriber_names

    def publish(self, linear_x: float, angular_z: float) -> None:
        if self._closed:
            raise RuntimeError("publisher is closed")
        message = self.Twist()
        message.linear.x = float(linear_x)
        message.angular.z = float(angular_z)
        self._publisher.publish(message)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._publisher.unregister()


class RospyTwistStampedPublisher(RospyTwistPublisher):
    """TwistStamped publisher used by the guarded chassis command mux."""

    def __init__(self, *, master_uri: str, bind_ip: str,
                 topic: str = "/gamepad/plt_cmd_vel") -> None:
        os.environ["ROS_MASTER_URI"] = master_uri
        os.environ["ROS_IP"] = bind_ip
        import rospy
        from geometry_msgs.msg import TwistStamped

        self.rospy = rospy
        self.TwistStamped = TwistStamped
        self.topic = topic
        self._closed = False
        with _INIT_LOCK:
            if not rospy.core.is_initialized():
                rospy.init_node(
                    "dualsense_chassis_gui", anonymous=True, disable_signals=True
                )
        self._publisher = rospy.Publisher(
            topic, TwistStamped, queue_size=1, tcp_nodelay=True
        )

    def publish(self, linear_x: float, angular_z: float) -> None:
        if self._closed:
            raise RuntimeError("publisher is closed")
        message = self.TwistStamped()
        message.header.stamp = self.rospy.Time.now()
        message.header.frame_id = "gamepad"
        message.twist.linear.x = float(linear_x)
        message.twist.angular.z = float(angular_z)
        self._publisher.publish(message)
