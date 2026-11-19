"""AGX adapter for the existing chassis SendGoal interface.

No ROS publisher or service call is created on import. Navigation acceptance
and physical arrival are separate facts. Business point roles must be confirmed
before a request can be built.
"""
from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ROLES = ("initial", "call_panel", "floor_panel", "arrival")
SERVICE = "/send_goal"
SERVICE_MD5 = "ef481e1565c122492c68e00caf063141"
GENERATED = Path(__file__).resolve().parents[1] / "vendor_interfaces/custom_msgs_srvs/_SendGoal.py"


@dataclass(frozen=True)
class NavigationPoint:
    point_id: str
    map_id: str
    x: float
    y: float
    qz: float
    qw: float

    def __post_init__(self):
        if not self.point_id or not self.map_id:
            raise ValueError("点位名和地图号不能为空")
        if not all(math.isfinite(v) for v in (self.x, self.y, self.qz, self.qw)):
            raise ValueError("点位坐标必须为有限数值")
        if abs(math.hypot(self.qz, self.qw) - 1.0) > 0.01:
            raise ValueError("点位朝向必须是归一化的平面四元数")

    @property
    def yaw(self):
        return 2.0 * math.atan2(self.qz, self.qw)


def confirmed_route(config: Mapping, active_map: str):
    """Reject missing roles, stale exports and accidental cross-map requests."""
    route = {}
    for role in ROLES:
        entry = config.get("points", {}).get(role)
        if not isinstance(entry, dict) or entry.get("verified") is not True:
            raise ValueError(f"尚未确认导航点位：{role}")
        point = NavigationPoint(**{key: entry[key] for key in
                                  ("point_id", "map_id", "x", "y", "qz", "qw")})
        if point.map_id != active_map:
            raise ValueError(f"点位 {role} 属于 {point.map_id}，当前地图是 {active_map}")
        route[role] = point
    return route


def bind_ordered_goals(initial: NavigationPoint,
                       destinations: list[NavigationPoint], active_map: str):
    """Bind the user-defined scene order to three *business* navigation goals.

    Preserve the navigation system's explicit order. PathNode/GlobalPath
    transit nodes and duplicate arrival notifications are not destinations.
    """
    if len(destinations) != 3:
        raise ValueError("布景路线必须提供外呼、内呼、目标三个有序导航目标")
    points = [initial, *destinations]
    if any(not isinstance(point, NavigationPoint) for point in points):
        raise ValueError("路线必须包含完整导航目标姿态")
    if any(point.map_id != active_map for point in points):
        raise ValueError("布景路线点位必须属于当前地图")
    if len({point.point_id for point in points}) != 4:
        raise ValueError("四个业务点位的标识不能重复")
    return dict(zip(ROLES, points))


def load_service_class():
    spec = importlib.util.spec_from_file_location("agx_vendor_send_goal", GENERATED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SendGoal._md5sum != SERVICE_MD5:
        raise RuntimeError("厂商导航服务定义校验失败")
    return module.SendGoal


def build_request(point: NavigationPoint, stamp=None):
    """Android ArmPose z/w are quaternion values, not position altitude."""
    import rospy
    request = load_service_class()._request_class()
    request.goal.header.frame_id = "map"
    request.goal.header.stamp = rospy.Time.now() if stamp is None else stamp
    request.goal.pose.position.x = point.x
    request.goal.pose.position.y = point.y
    request.goal.pose.position.z = 0.0
    request.goal.pose.orientation.z = point.qz
    request.goal.pose.orientation.w = point.qw
    return request


class VendorNavigationClient:
    def __init__(self, *, execute=False):
        self.execute = execute

    def submit(self, point: NavigationPoint, active_map: str):
        if point.map_id != active_map:
            raise ValueError("禁止向错误地图发送点位")
        if not self.execute:
            return {"status": "prepared", "point_id": point.point_id,
                    "service": SERVICE, "motion_command_sent": False}
        import rospy
        service_class = load_service_class()
        rospy.wait_for_service(SERVICE, timeout=5.0)
        response = rospy.ServiceProxy(SERVICE, service_class)(build_request(point))
        if not response.res:
            raise RuntimeError("底盘拒绝导航目标")
        return {"status": "accepted", "point_id": point.point_id,
                "motion_command_sent": True, "arrived": False}


class ArrivalGate:
    """Only fresh pose + odometry can prove arrival and continuous stationarity.

    Inputs use AGX receipt monotonic time. A vendor Bool is advisory because it
    has no goal ID and has multiple publishers. Continuous stillness resets on
    stale data, any motion or position/orientation outside the goal tolerance.
    """
    def __init__(self, point, *, xy_tolerance=0.05, yaw_tolerance=0.10,
                 linear_tolerance=0.01, angular_tolerance=0.02,
                 hold_sec=0.5, freshness_sec=0.5):
        self.point = point
        self.xy_tolerance = xy_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.linear_tolerance = linear_tolerance
        self.angular_tolerance = angular_tolerance
        self.hold_sec = hold_sec
        self.freshness_sec = freshness_sec
        self._still_since = None
        self._last_observation = None

    def observe(self, *, now, pose_received, odom_received, x, y, yaw,
                linear_speed, angular_speed):
        values = (now, pose_received, odom_received, x, y, yaw,
                  linear_speed, angular_speed)
        valid = all(math.isfinite(v) for v in values)
        valid = valid and 0 <= now - pose_received <= self.freshness_sec
        valid = valid and 0 <= now - odom_received <= self.freshness_sec
        if self._last_observation is not None:
            valid = valid and 0 <= now - self._last_observation <= self.freshness_sec
        self._last_observation = now
        dyaw = math.atan2(math.sin(yaw-self.point.yaw), math.cos(yaw-self.point.yaw))
        valid = valid and math.hypot(x-self.point.x, y-self.point.y) <= self.xy_tolerance
        valid = valid and abs(dyaw) <= self.yaw_tolerance
        valid = valid and abs(linear_speed) <= self.linear_tolerance
        valid = valid and abs(angular_speed) <= self.angular_tolerance
        if not valid:
            self._still_since = None
            return False
        if self._still_since is None:
            self._still_since = now
        return now-self._still_since >= self.hold_sec
