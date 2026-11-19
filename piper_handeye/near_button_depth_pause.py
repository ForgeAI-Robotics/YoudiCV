"""Bounded, encoder-renewed depth-guard pause for an explicit contact test."""
import numpy as np


NEAR_BUTTON_DEPTH_LEASE_SEC = 25.0


class NearButtonDepthPause:
    def __init__(self, entry_distance_m: float = 0.05):
        if not 0.01 <= float(entry_distance_m) <= 0.15:
            raise ValueError("entry_distance_m must be in 0.01..0.15 m")
        self.entry_distance_m = float(entry_distance_m)
        self.started = None
        self.finished = False
        self.active = False
        self.events = []

    def begin_cycle(self):
        """Start a new bounded near-panel allowance for one press attempt."""
        self.started = None
        self.finished = False
        self.active = False

    def update(self, tcp, target, sample_time, now, *, enabled, hardware_ok):
        tcp = np.asarray(tcp, dtype=float)
        target = np.asarray(target, dtype=float)
        valid = (tcp.shape == (4, 4) and target.shape == (4, 4)
                 and np.isfinite(tcp).all() and np.isfinite(target).all()
                 and np.isfinite([sample_time, now]).all())
        distance = float(target[0, 3] - tcp[0, 3]) if valid else float('nan')
        lateral = float(np.linalg.norm(target[1:3, 3] - tcp[1:3, 3])) if valid else float('nan')
        # Match the proven receding planner: enter on panel-normal distance.
        # A lateral gate can leave a diagonal approach unprotected after it
        # has already crossed the configured near-depth boundary.
        inside = (valid and enabled and hardware_ok and not self.finished
                  and 0 <= now - sample_time <= 0.25
                  and -0.01 <= distance <= self.entry_distance_m)
        if self.started is not None and now >= self.started + NEAR_BUTTON_DEPTH_LEASE_SEC:
            self.finished = True
            inside = False
        if inside and self.started is None:
            self.started = now
        if self.active and valid and distance > self.entry_distance_m:
            self.finished = True
        if inside != self.active:
            self.events.append({'active': bool(inside), 'distance_m': distance,
                                'lateral_m': lateral, 'monotonic': now})
        self.active = bool(inside)
        return min(sample_time + 0.25, self.started + NEAR_BUTTON_DEPTH_LEASE_SEC) if inside else 0.0
