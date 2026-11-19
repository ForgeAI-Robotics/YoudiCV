"""Contact prevention observer shared by the bridge and offline replay.

The production GuardedPressMonitor is intentionally not edited. This candidate
adds independently switchable continuity, measured travel and bounded local
joint-pose current prediction, so each change can be evaluated separately.
"""
from collections import deque

import numpy as np

from .guarded_press import (GuardedPressMonitor, CartesianNormalForceKalman,
                           CartesianForceKalman)
from .kinematics import forward_kinematics


class JointPoseCurrentReference:
    """Learn a rank-one local pose/current relation before the contact gate.

    Only the observed press-path direction is identified; a short movement
    cannot identify a general six-joint dynamics model. Coefficients freeze at
    the gate, while prediction continues from measured joints. Extrapolation
    and correction are bounded. Out-of-support inputs use the legacy reference.
    """
    def __init__(self, window_samples, correction_limit_a, extrapolation_spans=2.):
        self.window = deque(maxlen=window_samples)
        self.limit = np.asarray(correction_limit_a, dtype=float)
        self.extrapolation_spans = float(extrapolation_spans)
        self.frozen = False
        self.model = None
        self.last_fit_q = None
        self.audit = dict(status='warming_up')

    def observe(self, q, filtered_current):
        if not self.frozen:
            self.window.append((np.asarray(q).copy(), np.asarray(filtered_current).copy()))

    def fit(self):
        if len(self.window) < 20:
            return
        q = np.asarray([v[0] for v in self.window])
        currents = np.asarray([v[1] for v in self.window])
        if self.last_fit_q is not None and np.array_equal(q[-1], self.last_fit_q):
            return
        self.last_fit_q = q[-1].copy()
        if len(np.unique(q, axis=0)) < 4:
            self.audit = dict(status='insufficient_distinct_encoder_poses')
            return
        center = np.mean(q, axis=0)
        _, _, directions = np.linalg.svd(q-center, full_matrices=False)
        direction = directions[0]
        x = (q-center) @ direction
        span = float(np.ptp(x))
        if span < .002:
            self.audit = dict(status='insufficient_joint_path_span', span_rad=span)
            return
        design = np.column_stack((np.ones(len(x)), x))
        coefficients = np.linalg.lstsq(design, currents, rcond=None)[0]
        # One robust trimming pass limits spikes without using ON/OFF labels.
        residual = currents-design @ coefficients
        med = np.median(residual, axis=0)
        sigma = np.maximum(1.4826*np.median(np.abs(residual-med), axis=0), .005)
        inliers = np.max(np.abs(residual-med)/sigma, axis=1) <= 4.
        if np.count_nonzero(inliers) >= 20:
            coefficients = np.linalg.lstsq(design[inliers], currents[inliers], rcond=None)[0]
            residual = currents[inliers]-design[inliers] @ coefficients
            med = np.median(residual, axis=0)
            sigma = np.maximum(1.4826*np.median(np.abs(residual-med), axis=0), .005)
        self.model = dict(center=center, direction=direction, coefficients=coefficients,
                          sigma=sigma, minimum=float(np.min(x)), maximum=float(np.max(x)),
                          span=span, transverse_limit=max(.002, float(np.max(np.linalg.norm(
                              q-center-x[:, None]*direction, axis=1)))*2.))
        self.audit = dict(status='identified', samples=len(q), span_rad=span)

    def freeze(self):
        if not self.frozen:
            self.fit()
            self.frozen = True

    def predict(self, joints, legacy):
        if not self.frozen:
            self.fit()
        model = self.model
        if model is None or legacy is None:
            return None
        delta = np.asarray(joints)-model['center']
        x = float(delta @ model['direction'])
        transverse = float(np.linalg.norm(delta-x*model['direction']))
        lower = model['minimum']-self.extrapolation_spans*model['span']
        upper = model['maximum']+self.extrapolation_spans*model['span']
        if not lower <= x <= upper or transverse > model['transverse_limit']:
            self.audit = dict(status='legacy_fallback_outside_identified_path',
                              coordinate_rad=x, transverse_rad=transverse,
                              lower_rad=lower, upper_rad=upper)
            return None
        expected = np.asarray([1., x]) @ model['coefficients']
        reference, sigma = legacy
        correction = np.clip(expected-reference, -self.limit, self.limit)
        self.audit = dict(status='bounded_pose_prediction', frozen=self.frozen,
                          coordinate_rad=x, correction_a=correction.tolist(),
                          correction_limited=bool(np.any(np.abs(expected-reference) > self.limit)))
        return reference+correction, np.maximum(sigma, model['sigma'])


class CandidateGuardedPressMonitor(GuardedPressMonitor):
    """Bounded contact observer. Continuing alone never implies press success."""
    def __init__(self, baseline, config, *, contact_normal_base,
                 continuous_evidence=True, measured_progress=True,
                 motion_reference=True, encoder_max_age_sec=.10):
        super().__init__(baseline, config, contact_normal_base=contact_normal_base)
        self.continuous_evidence = bool(continuous_evidence)
        self.measured_progress = bool(measured_progress)
        self.motion_reference_enabled = bool(motion_reference)
        self.encoder_max_age_sec = float(encoder_max_age_sec)
        self._previous_sample_time = None
        self._previous_update_time = None
        self._origin = None
        self._current_q = None
        self._ever_armed = False
        self._frozen_legacy = None
        self.current_reference = JointPoseCurrentReference(
            config.dynamic_reference_window_samples, config.contact_delta_a)
        self.continuity_resets = 0
        self.candidate_audit = {}

    def _reset_contact_evidence(self):
        self._persistence, self._persistence_started_at = 0, None
        self._tangential_persistence, self._tangential_persistence_started_at = 0, None
        self._window.clear()
        if self._force_observer is not None:
            self._force_observer = CartesianNormalForceKalman(
                process_noise_n2_per_sec=self.config.force_process_noise_n2_per_sec)
            self._cartesian_force_observer = CartesianForceKalman(
                process_noise_n2_per_sec=self.config.force_process_noise_n2_per_sec)
        if not self._ever_armed:
            self._free_space_window.clear()
            self.current_reference = JointPoseCurrentReference(
                self.config.dynamic_reference_window_samples, self.config.contact_delta_a)
        self.continuity_resets += 1

    def _predict_rolling_reference(self):
        legacy = (self._frozen_legacy if self._ever_armed and self._frozen_legacy is not None
                  else super()._predict_rolling_reference())
        if self.motion_reference_enabled and self._ever_armed and self._current_q is not None:
            prediction = self.current_reference.predict(self._current_q, legacy)
            if prediction is not None:
                return prediction
        return legacy

    def update(self, sample, *, now, elapsed_sec, forward_travel_m,
               joints_rad=None, encoder_age_sec=None):
        currents = np.asarray(sample.currents_a, dtype=float)
        stamp = float(sample.timestamp)
        if self.continuous_evidence:
            if (not np.isfinite(stamp) or not np.isfinite(now)
                    or not 0 <= now-stamp <= self.config.max_sample_age_sec):
                return self._hard_stop('invalid_or_stale_current_sample', currents)
            if self._previous_sample_time is not None:
                if stamp <= self._previous_sample_time or now < self._previous_update_time:
                    return self._hard_stop('current_clock_or_sequence_not_increasing', currents)
                if max(stamp-self._previous_sample_time, now-self._previous_update_time) > self.config.max_sample_age_sec:
                    self._reset_contact_evidence()
            self._previous_sample_time, self._previous_update_time = stamp, float(now)
        planned = float(forward_travel_m)
        if not np.isfinite(planned) or not np.isfinite(elapsed_sec):
            return self._hard_stop('invalid_press_progress_or_time', currents)
        if planned > self.config.max_forward_travel_m:
            return self._hard_stop('maximum_forward_travel', currents)
        q = None if joints_rad is None else np.asarray(joints_rad, dtype=float)
        if self.measured_progress or self.motion_reference_enabled:
            if (q is None or q.shape != (6,) or not np.isfinite(q).all()
                    or encoder_age_sec is None or not np.isfinite(encoder_age_sec)
                    or not 0 <= encoder_age_sec <= self.encoder_max_age_sec):
                return self._hard_stop('measured_joint_feedback_missing_or_stale', currents)
        self._current_q = q
        distance = planned
        if self.measured_progress:
            position = forward_kinematics(q)[:3, 3]
            if self._origin is None:
                self._origin = position.copy()
            normal = self._contact_normal_base/np.linalg.norm(self._contact_normal_base)
            distance = max(0., float((position-self._origin) @ -normal))
        if distance >= self.config.contact_arm_travel_m and not self._ever_armed:
            self._frozen_legacy = super()._predict_rolling_reference()
            if self._frozen_legacy is None and self.continuous_evidence:
                return self._hard_stop('precontact_motion_reference_not_ready', currents)
            self.current_reference.freeze()
            self._ever_armed = True
        result = super().update(sample, now=now, elapsed_sec=elapsed_sec,
                                forward_travel_m=distance, joints_rad=q)
        if (self.motion_reference_enabled and not self._ever_armed
                and len(self._window) == self.config.median_window):
            self.current_reference.observe(q, result.filtered_current_a)
        self.candidate_audit = dict(planned_travel_m=planned, measured_travel_m=distance,
            continuity_resets=self.continuity_resets,
            motion_reference=dict(self.current_reference.audit),
            normal_persistence_sec=result.persistence_sec,
            tangential_persistence_sec=(0. if self._tangential_persistence_started_at is None
                else max(0., now-self._tangential_persistence_started_at)))
        return result
