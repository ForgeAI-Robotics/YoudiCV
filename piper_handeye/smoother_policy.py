"""Watchdog lease for exclusive gamepad chassis control.

The companion manager pauses the unmodified vendor smoother for the lifetime
of the lease. Lease loss sends zero and resumes the same vendor process.
"""

from __future__ import annotations

import atexit
import threading
from typing import Callable

LEASE_SERVICE = "/gamepad_smoother_runtime_manager/set_enabled"
HEARTBEAT_TOPIC = "/gamepad_smoother_runtime_manager/heartbeat"


class VelocitySmootherRuntimeLease:
    """Acquire the chassis manager lease and keep its watchdog alive."""

    def __init__(
        self,
        *,
        timeout_sec: float = 5.0,
        heartbeat_interval_sec: float | None = 0.10,
        transport: Callable[[bool], dict] | None = None,
        heartbeat_transport: Callable[[], None] | None = None,
    ) -> None:
        if transport is None or heartbeat_transport is None:
            ros_transport, ros_heartbeat = self._make_ros_transport(timeout_sec)
            transport = transport or ros_transport
            heartbeat_transport = heartbeat_transport or ros_heartbeat
        self._transport = transport
        self._heartbeat_transport = heartbeat_transport
        self._heartbeat_interval_sec = heartbeat_interval_sec
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._active = False
        self._details: dict = {}
        atexit.register(self._restore_at_exit)

    @staticmethod
    def _make_ros_transport(
        timeout_sec: float,
    ) -> tuple[Callable[[bool], dict], Callable[[], None]]:
        import rospy
        from dynamic_reconfigure.msg import BoolParameter
        from dynamic_reconfigure.srv import Reconfigure, ReconfigureRequest
        from std_msgs.msg import Empty

        rospy.wait_for_service(LEASE_SERVICE, timeout=timeout_sec)
        service = rospy.ServiceProxy(LEASE_SERVICE, Reconfigure, persistent=False)
        heartbeat_publisher = rospy.Publisher(
            HEARTBEAT_TOPIC, Empty, queue_size=1, tcp_nodelay=True
        )

        def call(enabled: bool) -> dict:
            request = ReconfigureRequest()
            request.config.bools = [
                BoolParameter(name="enabled", value=bool(enabled))
            ]
            response = service(request)
            result = {}
            for field in ("bools", "ints", "doubles", "strs"):
                for item in getattr(response.config, field):
                    result[item.name] = item.value
            if bool(result.get("active")) != bool(enabled):
                raise RuntimeError(f"底盘平滑器租约状态校验失败：{result}")
            return result

        def heartbeat() -> None:
            heartbeat_publisher.publish(Empty())

        return call, heartbeat

    @property
    def active(self) -> bool:
        return self._active

    @property
    def details(self) -> dict:
        return dict(self._details)

    def acquire(self) -> dict:
        if self._active:
            return dict(self._details)
        details = self._transport(True)
        if not bool(details.get("active")):
            raise RuntimeError(f"底盘平滑器租约未生效：{details}")
        if details.get("topology") != "vendor_pause_lease":
            raise RuntimeError(f"底盘控制管理器版本或拓扑不正确：{details}")
        if not bool(details.get("vendor_smoother_paused")):
            raise RuntimeError(f"厂商速度发布器未暂停，拒绝开启手柄控制：{details}")
        self._details = dict(details)
        self._active = True
        self._start_heartbeat_thread()
        self.heartbeat()
        return dict(details)

    def _start_heartbeat_thread(self) -> None:
        if self._heartbeat_interval_sec is None:
            return
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()

        def run() -> None:
            while not self._heartbeat_stop.wait(self._heartbeat_interval_sec):
                if not self._active:
                    return
                try:
                    self._heartbeat_transport()
                except Exception:
                    # The chassis watchdog is the authority: if transport is
                    # really lost it restores the vendor policy automatically.
                    pass

        self._heartbeat_thread = threading.Thread(
            target=run, name="smoother-lease-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat_thread(self) -> None:
        self._heartbeat_stop.set()
        thread, self._heartbeat_thread = self._heartbeat_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)

    def heartbeat(self) -> None:
        if self._active:
            self._heartbeat_transport()

    def restore(self) -> dict | None:
        if not self._active:
            return None
        self._stop_heartbeat_thread()
        details = self._transport(False)
        if bool(details.get("active")):
            raise RuntimeError(f"底盘平滑器租约恢复校验失败：{details}")
        self._active = False
        self._details = dict(details)
        return dict(details)

    def _restore_at_exit(self) -> None:
        try:
            self.restore()
        except Exception:
            pass


VelocitySmootherPolicyLease = VelocitySmootherRuntimeLease
