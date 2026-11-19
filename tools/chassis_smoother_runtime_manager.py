#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fail-safe lease that pauses the vendor smoother during gamepad control.

The vendor roslaunch owns ``/velocity_smoother`` and may respawn it. Starting a
second, remapped copy creates a race: the original smoother can return and send
zero commands beside the gamepad. This manager leaves vendor files untouched,
SIGSTOPs the live smoother during a lease, and SIGCONTs it on every exit path.
"""
from __future__ import print_function

import glob
import io
import json
import os
import signal
import threading
import time

import rosgraph
import rosnode
import rospy
try:
    import xmlrpclib
except ImportError:
    import xmlrpc.client as xmlrpclib

from dynamic_reconfigure.msg import BoolParameter, Config, DoubleParameter, IntParameter, StrParameter
from dynamic_reconfigure.srv import Reconfigure, ReconfigureResponse
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Empty

NODE_NAME = "/velocity_smoother"
PARAM_NAMESPACE = "/velocity_smoother"
LEASE_SERVICE = "/gamepad_smoother_runtime_manager/set_enabled"
HEARTBEAT_TOPIC = "/gamepad_smoother_runtime_manager/heartbeat"
GAMEPAD_TOPIC = "/gamepad/plt_cmd_vel"
OUTPUT_TOPIC = "/plt_cmd_vel"
WATCHDOG_TIMEOUT_SEC = 0.60
INITIAL_HEARTBEAT_GRACE_SEC = 3.0
COMMAND_TIMEOUT_SEC = 0.25
STATE_PATH = os.path.expanduser("~/codex_chassis_control/smoother_runtime_state.json")
VENDOR_EXECUTABLE = "/home/ubuntu/ud_ws/install/lib/velocity_smoother/velocity_smoother_node"


class RuntimeManager(object):
    def __init__(self):
        self._lock = threading.RLock()
        self._active = False
        self._generation = 0
        self._last_heartbeat = 0.0
        self._lease_started = 0.0
        self._heartbeat_seen = False
        self._last_gamepad_command = 0.0
        self._paused_pids = set()
        self._output = rospy.Publisher(
            OUTPUT_TOPIC, TwistStamped, queue_size=1, tcp_nodelay=True
        )
        self._gamepad_subscriber = rospy.Subscriber(
            GAMEPAD_TOPIC, TwistStamped, self._gamepad_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self._heartbeat_subscriber = rospy.Subscriber(
            HEARTBEAT_TOPIC, Empty, self._heartbeat_callback, queue_size=1
        )
        # A manager or GUI crash must never leave the vendor controller paused.
        self._resume_all_vendor_processes()
        self._recover_stale_lease()
        self._service = rospy.Service(
            LEASE_SERVICE, Reconfigure, self._service_callback
        )
        self._timer = rospy.Timer(rospy.Duration(0.05), self._watchdog_callback)

    def _write_state(self, phase):
        directory = os.path.dirname(STATE_PATH)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        payload = {
            "phase": phase,
            "active": self._active,
            "generation": self._generation,
            "topology": "vendor_pause_lease",
            "paused_pids": sorted(self._paused_pids),
            "updated_unix": time.time(),
        }
        temporary = STATE_PATH + ".tmp"
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
        if not isinstance(encoded, bytes):
            encoded = encoded.encode("ascii")
        with open(temporary, "wb") as stream:
            stream.write(encoded)
        os.chmod(temporary, 0o600)
        os.rename(temporary, STATE_PATH)

    def _recover_stale_lease(self):
        stale = False
        if os.path.exists(STATE_PATH):
            try:
                with io.open(STATE_PATH, "r", encoding="utf-8") as stream:
                    stale = bool(json.load(stream).get("active"))
            except Exception as exc:
                rospy.logerr("cannot read lease state: %s", exc)
        if stale:
            rospy.logwarn("cleared stale gamepad lease and resumed vendor smoother")
        self._active = False
        self._write_state("stale_lease_cleared" if stale else "ready")

    def _publish_zero(self):
        message = TwistStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "gamepad_lease_stop"
        for _ in range(5):
            self._output.publish(message)
            time.sleep(0.01)

    @staticmethod
    def _node_present():
        try:
            return bool(rosnode.rosnode_ping(
                NODE_NAME, max_count=1, verbose=False, skip_cache=True
            ))
        except Exception:
            return False

    @staticmethod
    def _is_vendor_pid(pid):
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as stream:
                cmdline = stream.read().replace(b"\x00", b" ")
            return VENDOR_EXECUTABLE.encode("ascii") in cmdline
        except (IOError, OSError):
            return False

    @staticmethod
    def _proc_state(pid):
        try:
            with open("/proc/%d/status" % pid, "r") as stream:
                for line in stream:
                    if line.startswith("State:"):
                        return line.split()[1]
        except (IOError, OSError):
            pass
        return None

    def _node_pid(self):
        try:
            master = rosgraph.Master(rospy.get_name())
            uri = master.lookupNode(NODE_NAME)
            code, message, pid = xmlrpclib.ServerProxy(uri).getPid(rospy.get_name())
            if code != 1:
                raise RuntimeError(message)
            pid = int(pid)
            return pid if self._is_vendor_pid(pid) else None
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "cannot resolve vendor smoother pid: %s", exc)
            return None

    def _all_vendor_pids(self):
        result = []
        for path in glob.glob("/proc/[0-9]*/cmdline"):
            try:
                pid = int(path.split("/")[2])
            except (ValueError, IndexError):
                continue
            if self._is_vendor_pid(pid):
                result.append(pid)
        return result

    def _resume_all_vendor_processes(self):
        for pid in self._all_vendor_pids():
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        self._paused_pids.clear()

    def _pause_current_vendor(self):
        for paused_pid in list(self._paused_pids):
            if self._is_vendor_pid(paused_pid) and self._proc_state(paused_pid) == "T":
                return paused_pid
            self._paused_pids.discard(paused_pid)
        pid = self._node_pid()
        if pid is None:
            raise RuntimeError("找不到厂商 velocity_smoother 进程")
        if pid not in self._paused_pids or self._proc_state(pid) != "T":
            os.kill(pid, signal.SIGSTOP)
            deadline = time.time() + 0.5
            while time.time() < deadline and self._proc_state(pid) != "T":
                time.sleep(0.01)
            if self._proc_state(pid) != "T":
                raise RuntimeError("厂商 velocity_smoother 未进入暂停状态")
            self._paused_pids.add(pid)
        return pid

    def _gamepad_callback(self, message):
        with self._lock:
            self._last_gamepad_command = time.time()
            if self._active:
                message.header.stamp = rospy.Time.now()
                message.header.frame_id = "gamepad_lease"
                self._output.publish(message)

    def _enable(self):
        if self._active:
            self._last_heartbeat = time.time()
            return self._status("already_active")
        pid = self._pause_current_vendor()
        try:
            self._publish_zero()
            self._active = True
            self._generation += 1
            self._lease_started = time.time()
            self._last_heartbeat = self._lease_started
            self._last_gamepad_command = self._lease_started
            self._heartbeat_seen = False
            self._write_state("active")
            rospy.loginfo("gamepad lease enabled; paused vendor smoother pid=%d", pid)
            return self._status("enabled")
        except Exception:
            self._resume_all_vendor_processes()
            raise

    def _restore(self, reason):
        if self._active:
            self._publish_zero()
        self._active = False
        self._heartbeat_seen = False
        self._resume_all_vendor_processes()
        self._write_state(reason)
        rospy.loginfo("gamepad lease restored vendor smoother: %s", reason)
        return self._status(reason)

    def _status(self, reason):
        params = rospy.get_param(PARAM_NAMESPACE, {})
        if self._active and self._paused_pids:
            pid = next(iter(self._paused_pids))
        else:
            pid = self._node_pid()
        return {
            "active": self._active,
            "generation": self._generation,
            "reason": reason,
            "use_limit_speedup": params.get("use_limit_speedup", True),
            "use_limit_back": params.get("use_limit_back", True),
            "watchdog_timeout_sec": WATCHDOG_TIMEOUT_SEC,
            "initial_heartbeat_grace_sec": INITIAL_HEARTBEAT_GRACE_SEC,
            "command_timeout_sec": COMMAND_TIMEOUT_SEC,
            "topology": "vendor_pause_lease",
            "vendor_smoother_alive": pid is not None,
            "vendor_smoother_paused": pid in self._paused_pids and self._proc_state(pid) == "T",
        }

    def _service_callback(self, request):
        with self._lock:
            requested = [item.value for item in request.config.bools if item.name == "enabled"]
            if len(requested) != 1:
                raise rospy.ServiceException("请求必须包含一个 enabled 布尔值")
            result = self._enable() if requested[0] else self._restore("disabled")
            config = Config()
            config.bools = [
                BoolParameter(name="active", value=bool(result["active"])),
                BoolParameter(name="use_limit_speedup", value=bool(result["use_limit_speedup"])),
                BoolParameter(name="use_limit_back", value=bool(result["use_limit_back"])),
                BoolParameter(name="vendor_smoother_alive", value=bool(result["vendor_smoother_alive"])),
                BoolParameter(name="vendor_smoother_paused", value=bool(result["vendor_smoother_paused"])),
            ]
            config.ints = [IntParameter(name="generation", value=int(result["generation"]))]
            config.doubles = [
                DoubleParameter(name="watchdog_timeout_sec", value=float(result["watchdog_timeout_sec"])),
                DoubleParameter(name="initial_heartbeat_grace_sec", value=float(result["initial_heartbeat_grace_sec"])),
                DoubleParameter(name="command_timeout_sec", value=float(result["command_timeout_sec"])),
            ]
            config.strs = [
                StrParameter(name="reason", value=str(result["reason"])),
                StrParameter(name="topology", value=str(result["topology"])),
            ]
            return ReconfigureResponse(config=config)

    def _heartbeat_callback(self, _message):
        with self._lock:
            if self._active:
                self._last_heartbeat = time.time()
                self._heartbeat_seen = True

    def _watchdog_callback(self, _event):
        with self._lock:
            if not self._active:
                return
            now = time.time()
            if not self._heartbeat_seen:
                heartbeat_expired = now - self._lease_started > INITIAL_HEARTBEAT_GRACE_SEC
            else:
                heartbeat_expired = now - self._last_heartbeat > WATCHDOG_TIMEOUT_SEC
            if heartbeat_expired:
                rospy.logwarn("gamepad heartbeat expired; resuming vendor smoother")
                self._restore("heartbeat_timeout_restored")
                return
            try:
                self._pause_current_vendor()
            except Exception as exc:
                rospy.logerr_throttle(1.0, "cannot keep vendor smoother paused: %s", exc)
                self._restore("vendor_pause_failed")
                return
            if now - self._last_gamepad_command > COMMAND_TIMEOUT_SEC:
                self._publish_zero()

    def shutdown(self):
        with self._lock:
            self._restore("manager_shutdown")


def main():
    rospy.init_node("gamepad_smoother_runtime_manager", disable_signals=True)
    manager = RuntimeManager()

    def stop(_signum, _frame):
        manager.shutdown()
        rospy.signal_shutdown("signal")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    rospy.spin()


if __name__ == "__main__":
    main()
