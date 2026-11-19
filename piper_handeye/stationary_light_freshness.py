"""Continuous encoder evidence for stationary numeric and call-button lamps."""
import math
import time

import numpy as np
from .numeric_layout_handoff import selected_button_region, region_is_inferred

NORMAL_RGB_MAX_AGE_SEC = .35
STATIONARY_RGB_MAX_AGE_SEC = .7
STATIONARY_RETRY_RGB_MAX_AGE_SEC = 1.0
_PHASES = {'waiting': 'waiting', 'before_press': 'reference',
           'near': 'near', 'near_confirmation': 'near', 'near_recheck': 'near',
           'prepress': 'prepress', 'prepress_confirmation': 'prepress'}


class StationaryRetryLightGuard:
    """Never accepts motion input or controls a robot. No new thread or RPC.

    Each observation poll reads the task's already receiving encoder cache.
    Evidence begins after entering the phase, resets on motion/feedback gaps,
    and must cover the original image capture through the current check.
    """
    def __init__(self, read_state, check_safety, target_label, *, clock=time):
        self.read_state, self.check_safety = read_state, check_safety
        self.target = str(target_label)
        self.clock = clock
        self.call_button = self.target in {'up', 'down'}
        self.enabled = self.call_button or self.target in tuple(map(str, range(1, 17)))
        self.reset()

    def reset(self):
        self.phase = None
        self.first = self.last = None
        self.offset = None
        self.low = self.high = None
        self.count = 0

    def allows_phase(self, position):
        return self.enabled and position in _PHASES

    def poll(self, position):
        if not self.allows_phase(position):
            self.reset()
            return
        self.check_safety()  # A real stop is never swallowed as visual uncertainty.
        now, unix = self.clock.monotonic(), self.clock.time()
        phase = _PHASES[position]
        try:
            state = self.read_state()
            q = np.asarray(state.joints, float)
            stamp = float(state.timestamp)
            if (q.shape != (6,) or not np.isfinite(q).all()
                    or not math.isfinite(stamp) or not 0 <= now-stamp <= .25
                    or state.raw.get('hardware_ok') != 1):
                raise ValueError('unhealthy or stale encoders')
        except (TimeoutError, RuntimeError, ValueError, TypeError, AttributeError):
            self.reset()
            return
        offset = unix-now
        if (self.phase != phase or self.last is None or stamp < self.last
                or stamp-self.last > .25 or abs(offset-self.offset) > .05
                or np.max(np.maximum(self.high, q)-np.minimum(self.low, q)) > .003):
            self.reset()
            self.phase, self.offset = phase, offset
            # A cached pre-entry packet cannot prove the camera was stationary.
            self.first = max(now, stamp)
            self.low = q.copy(); self.high = q.copy()
        if self.last is None or stamp > self.last:
            self.low = np.minimum(self.low, q); self.high = np.maximum(self.high, q)
            self.count += 1
            self.last = stamp

    def admission(self, snapshot, position):
        if not self.allows_phase(position):
            return None
        try:
            captured = float(snapshot['capture_time_unix_sec'])
            age = self.clock.time()-captured
            detection = selected_button_region(snapshot, target_label=self.target,
                                               max_frame_age_sec=STATIONARY_RGB_MAX_AGE_SEC)
            spans = (snapshot.get('perception_timing') or {}).get('spans', [])
            retry = any(s.get('name') == 'vision.detection' and s.get('status') == 'ok'
                        and type(s.get('retry_inference_sec')) in (int, float)
                        and math.isfinite(s['retry_inference_sec'])
                        and s['retry_inference_sec'] > 0
                        and not s.get('numeric_retry_skipped_by_layout', False) for s in spans)
            limit = STATIONARY_RETRY_RGB_MAX_AGE_SEC if retry and not region_is_inferred(detection) else STATIONARY_RGB_MAX_AGE_SEC
            if (not snapshot.get('fresh') or detection is None or not math.isfinite(age)
                    or not -.05 <= age <= limit
                    or detection['label'] != self.target
                    or type(snapshot['frame_sequence']) is not int
                    or detection['frame_sequence'] != snapshot['frame_sequence']
                    or (not region_is_inferred(detection) and
                        (not math.isfinite(float(detection['confidence']))
                         or float(detection['confidence']) < .70))):
                return None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        self.poll(position)
        now = self.clock.monotonic()
        if (self.phase != _PHASES[position] or self.first is None or self.last is None
                or self.count < 2 or self.last-self.first < .08
                or not 0 <= now-self.last <= .25
                or not self.first+self.offset <= captured <= self.last+self.offset):
            return None
        family = 'call' if self.call_button else 'numeric'
        return dict(policy=(f'stationary_{family}_local_retry_rgb_1s_v1' if limit == 1.0
                            else f'stationary_{family}_lamp_rgb_700ms_v1'),
                    region_source='panel_layout_inferred' if region_is_inferred(detection) else 'yolo',
                    max_rgb_age_sec=limit,
                    image_age_sec=age, stationary_since_unix_sec=self.first+self.offset,
                    latest_encoder_unix_sec=self.last+self.offset,
                    encoder_samples=self.count, max_joint_spread_rad=float(np.max(self.high-self.low)),
                    frame_sequence=snapshot['frame_sequence'])
