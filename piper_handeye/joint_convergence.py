"""Bounded confirmation of distinct measured encoder samples, without ROS."""
from __future__ import annotations

import math
import time


class JointConvergenceError(RuntimeError):
    def __init__(self, reason, audit):
        self.audit = dict(audit, outcome=reason)
        super().__init__("guarded_retract_convergence_failed: " + reason)


def wait_for_joint_convergence(target, snapshot, check_safety, *, timeout_sec=1.0,
                               tolerance_rad=.03, max_age_sec=.10,
                               stable_sec=.08, clock=time.monotonic, sleep=time.sleep):
    """Observe the held endpoint; never send/resend a goal or relax tolerance.

    A sample is (sequence, receive_monotonic, six measured joint positions).
    Confirmation starts after action completion, using only subsequent samples.
    Safety exceptions propagate unchanged, including their diagnostic audit.
    """
    target = list(target)
    if (len(target) != 6 or not all(math.isfinite(q) for q in target)
            or not 0 < stable_sec < timeout_sec or tolerance_rad <= 0 or max_age_sec <= 0):
        raise ValueError("invalid joint convergence parameters")
    started = clock()
    audit = dict(target_joints_rad=target, tolerance_rad=tolerance_rad,
                 timeout_sec=timeout_sec, stable_sec=stable_sec, samples=[],
                 distinct_samples=0, elapsed_sec=0.)
    previous = None
    previous_received = None
    stable_since = None
    good_count = 0
    try:
        while True:
            check_safety()
            now = clock()
            audit['elapsed_sec'] = now-started
            if now-started >= timeout_sec:
                raise JointConvergenceError('timeout', audit)
            value = snapshot()
            if value is None:
                raise JointConvergenceError('feedback_unavailable', audit)
            sequence, received, joints = value
            if (not isinstance(sequence, int) or not isinstance(received, (int, float))
                    or not math.isfinite(received) or not 0 <= now-received <= max_age_sec
                    or len(joints) != 6 or not all(math.isfinite(q) for q in joints)):
                raise JointConvergenceError('feedback_invalid_or_stale', audit)
            if previous is not None and (sequence < previous or received < previous_received):
                raise JointConvergenceError('feedback_clock_or_sequence_reversed', audit)
            if sequence != previous:
                # A scheduler/feedback gap cannot be bridged into a claim of
                # continuous convergence by two samples on either side of it.
                if previous_received is not None and received-previous_received > max_age_sec:
                    stable_since, good_count = None, 0
                    audit['confirmation_gap_resets'] = audit.get('confirmation_gap_resets', 0)+1
                previous, previous_received = sequence, received
                # A fresh but pre-completion cache entry is not arrival evidence.
                if received >= started:
                    error = max(abs(a-b) for a, b in zip(joints, target))
                    row = dict(sequence=sequence, received_monotonic=received,
                               age_sec=now-received, joints_rad=list(joints),
                               max_error_rad=error, elapsed_sec=now-started)
                    audit['samples'].append(row)
                    audit['distinct_samples'] += 1
                    audit['final_error_rad'] = error
                    audit['first_error_rad'] = audit.get('first_error_rad', error)
                    audit['peak_error_rad'] = max(audit.get('peak_error_rad', 0.), error)
                    if error <= tolerance_rad:
                        stable_since = received if stable_since is None else stable_since
                        good_count += 1
                        if good_count >= 2 and received-stable_since >= stable_sec:
                            check_safety()
                            audit.update(outcome='converged', elapsed_sec=clock()-started)
                            return audit
                    else:
                        stable_since, good_count = None, 0
            sleep(.01)
    except BaseException as exc:
        if not isinstance(exc, JointConvergenceError):
            exc.convergence_audit = dict(audit, outcome='safety_or_callback_failure', error=str(exc))
        raise
