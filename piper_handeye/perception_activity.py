"""Keep models loaded, throttling only inference when nobody needs it."""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time


class PerceptionActivity:
    def __init__(self, idle_interval_sec=5.0, active_hold_sec=10.0, motion_lock=None):
        if idle_interval_sec <= 0 or active_hold_sec <= 0:
            raise ValueError("activity intervals must be positive")
        self.idle_interval_sec = float(idle_interval_sec)
        self.active_hold_sec = float(active_hold_sec)
        self.motion_lock = Path(motion_lock or os.environ.get(
            "YOUXIAOMEI_ROBOT_MOTION_LOCK", "/tmp/youxiaomei_robot_motion.lock"))
        self._condition = threading.Condition()
        self._active_until = 0.0
        self._last_started = float('-inf')
        self._closed = False

    def touch(self):
        with self._condition:
            self._active_until = time.monotonic() + self.active_hold_sec
            self._condition.notify_all()

    def motion_busy(self):
        # Inspect the existing flock without acquiring it: a probe must never
        # briefly take execution permission away from the task dispatcher.
        try:
            stat = self.motion_lock.stat()
            major, minor = os.major(stat.st_dev), os.minor(stat.st_dev)
            with open('/proc/locks') as stream:
                for line in stream:
                    fields = line.split()
                    if len(fields) < 6 or fields[1] != 'FLOCK':
                        continue
                    device = fields[5].split(':')
                    if len(device) == 3 and (int(device[0], 16), int(device[1], 16),
                                            int(device[2])) == (major, minor, stat.st_ino):
                        return True
            return False
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            # If activity cannot be determined, keep full inference frequency.
            return True

    def active(self):
        return time.monotonic() < self._active_until or self.motion_busy()

    def wait_for_turn(self):
        with self._condition:
            while not self._closed:
                now = time.monotonic()
                if self.active() or now - self._last_started >= self.idle_interval_sec:
                    self._last_started = now
                    return True
                # A task acquiring the motion lock wakes us within 100 ms,
                # even before its first vision RPC arrives.
                self._condition.wait(min(.1, self.idle_interval_sec - (now - self._last_started)))
            return False

    def status(self):
        with self._condition:
            return {"mode": "active" if self.active() else "idle",
                    "idle_interval_sec": self.idle_interval_sec,
                    "active_hold_sec": self.active_hold_sec,
                    "models_resident": True}

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
