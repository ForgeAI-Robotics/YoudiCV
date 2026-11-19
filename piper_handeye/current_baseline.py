"""Bounded pre-contact current collection; never sends a robot command.

The current estimator and its MAD noise floor are unchanged. Only samples
collected during one continuous stationary encoder window enter the fit.
Movement discards the candidate window, not the whole press attempt.
"""
from __future__ import annotations

import time

import numpy as np

from .guarded_press import RobustCurrentBaseline


def collect_stable_current_baseline(wait_next, joint_snapshot, check_safety, *,
                                    initial_sequence=0, expected_start=None,
                                    timeout_sec=5.0, stable_sec=0.30,
                                    minimum_samples=40, joint_tolerance_rad=0.003,
                                    start_tolerance_rad=0.03,
                                    joint_max_age_sec=0.10, current_max_age_sec=0.03,
                                    audit=None, clock=time.monotonic):
    """Wait up to five seconds for a valid stationary current baseline.

    `wait_next` is the existing bounded CAN receiver. Encoder snapshots are
    distinct measured packets, not repeated synchronous ROS subscriptions.
    Timeout/old CAN samples discard only the unfinished candidate. Hardware,
    safety, invalid/stale encoders and receiver exceptions still propagate.
    """
    if (not 0 < stable_sec < timeout_sec or minimum_samples < 40
            or min(joint_tolerance_rad, start_tolerance_rad,
                   joint_max_age_sec, current_max_age_sec) <= 0):
        raise ValueError('invalid current baseline collection parameters')
    target = None if expected_start is None else np.asarray(expected_start, dtype=float)
    if target is not None and (target.shape != (6,) or not np.isfinite(target).all()):
        raise ValueError('baseline start requires six finite joints')
    audit = {} if audit is None else audit
    started = clock()
    deadline = started + timeout_sec
    audit.update(policy='stationary_current_baseline_within_5s_v1',
                 timeout_sec=timeout_sec, stable_sec=stable_sec,
                 minimum_samples=minimum_samples, joint_tolerance_rad=joint_tolerance_rad,
                 start_tolerance_rad=start_tolerance_rad,
                 joint_max_age_sec=joint_max_age_sec, current_max_age_sec=current_max_age_sec,
                 started_monotonic_sec=started, outcome='collecting',
                 joint_samples=[], discarded_windows=[], total_current_samples=0,
                 current_wait_timeouts=0)
    currents = []
    first_current_at = None
    previous_current_at = None
    previous_joint_sequence = None
    previous_joint_at = None
    window_first_joint_at = None
    joint_low = joint_high = None
    first_joints = last_joints = None
    sequence = initial_sequence

    def discard(reason):
        nonlocal currents, first_current_at, window_first_joint_at, joint_low, joint_high, first_joints
        row = dict(reason=reason, elapsed_sec=clock()-started, current_samples=len(currents))
        if first_joints is not None:
            spread = joint_high-joint_low
            row.update(first_joints_rad=first_joints.tolist(),
                       last_joints_rad=last_joints.tolist(),
                       joint_spread_rad=spread.tolist(),
                       max_joint_index_1based=int(np.argmax(spread))+1,
                       max_joint_spread_rad=float(np.max(spread)))
        if currents:
            row.update(first_current_a=currents[0].tolist(), last_current_a=currents[-1].tolist())
        audit['discarded_windows'].append(row)
        currents, first_current_at, window_first_joint_at = [], None, None
        joint_low = joint_high = first_joints = None

    def read_joints():
        nonlocal previous_joint_sequence, previous_joint_at, window_first_joint_at
        nonlocal joint_low, joint_high, first_joints, last_joints
        value = joint_snapshot()
        now = clock()
        if value is None:
            raise RuntimeError('current baseline joint feedback unavailable')
        seq, received, q = value
        q = np.asarray(q, dtype=float)
        if (not isinstance(seq, int) or not np.isfinite(received)
                or not 0 <= now-received <= joint_max_age_sec
                or q.shape != (6,) or not np.isfinite(q).all()):
            raise RuntimeError('current baseline joint feedback invalid or stale')
        if previous_joint_sequence is not None and (
                seq < previous_joint_sequence or received < previous_joint_at
                or (seq == previous_joint_sequence and received != previous_joint_at)):
            raise RuntimeError('current baseline joint feedback clock or sequence reversed')
        if target is not None and float(np.max(np.abs(q-target))) > start_tolerance_rad:
            raise RuntimeError('guarded press start changed during current baseline collection')
        last_joints = q
        if seq == previous_joint_sequence:
            return
        if previous_joint_at is not None and received-previous_joint_at > joint_max_age_sec:
            discard('joint_feedback_gap')
        previous_joint_sequence, previous_joint_at = seq, received
        audit['joint_samples'].append(dict(sequence=seq, received_monotonic_sec=received,
                                          age_sec=now-received, joints_rad=q.tolist()))
        if joint_low is None:
            joint_low, joint_high, first_joints = q.copy(), q.copy(), q.copy()
            window_first_joint_at = received
        else:
            joint_low, joint_high = np.minimum(joint_low, q), np.maximum(joint_high, q)
            if float(np.max(joint_high-joint_low)) > joint_tolerance_rad+1e-12:
                discard('joint_window_moved')
                joint_low, joint_high, first_joints = q.copy(), q.copy(), q.copy()
                window_first_joint_at = received

    try:
        while True:
            check_safety()
            now = clock()
            audit['elapsed_sec'] = now-started
            if now >= deadline:
                raise TimeoutError('no stable current baseline within %.1fs' % timeout_sec)
            read_joints()
            remaining = deadline-clock()
            if remaining <= 0:
                raise TimeoutError('no stable current baseline within %.1fs' % timeout_sec)
            try:
                sequence, sample = wait_next(sequence, timeout_sec=min(0.05, remaining))
            except TimeoutError:
                audit['current_wait_timeouts'] += 1
                discard('current_wait_timeout')
                continue
            check_safety()
            now = clock()
            if now >= deadline:
                raise TimeoutError('no stable current baseline within %.1fs' % timeout_sec)
            # Check again after waiting: motion during the wait must not be
            # hidden by a current packet that happened to arrive afterwards.
            read_joints()
            now = clock()
            stamp = float(sample.timestamp)
            values = np.asarray(sample.currents_a, dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all() or not np.isfinite(stamp):
                raise RuntimeError('current baseline contains invalid motor-current data')
            audit['total_current_samples'] += 1
            if not 0 <= now-stamp <= current_max_age_sec:
                discard('current_sample_stale')
                continue
            if previous_current_at is not None and stamp <= previous_current_at:
                discard('current_sample_not_new')
                continue
            if previous_current_at is not None and stamp-previous_current_at > current_max_age_sec:
                discard('current_sample_gap')
            previous_current_at = stamp
            # After a discarded CAN window, seed from a *new* encoder packet
            # before accumulating reference currents again.
            if joint_low is None:
                continue
            if first_current_at is None:
                first_current_at = stamp
            currents.append(values.copy())
            if (len(currents) >= minimum_samples and stamp-first_current_at >= stable_sec
                    and previous_joint_at-window_first_joint_at >= stable_sec):
                fitted = RobustCurrentBaseline.fit(currents, minimum_samples=minimum_samples)
                check_safety()
                read_joints()
                if joint_low is None or len(currents) < minimum_samples:
                    continue
                if clock() >= deadline:
                    raise TimeoutError('no stable current baseline within %.1fs' % timeout_sec)
                audit.update(outcome='stable', elapsed_sec=clock()-started,
                             accepted_samples=fitted.sample_count,
                             accepted_current_span_sec=stamp-first_current_at,
                             first_joints_rad=first_joints.tolist(), last_joints_rad=last_joints.tolist(),
                             joint_spread_rad=(joint_high-joint_low).tolist(),
                             baseline_current_a=fitted.median_a.tolist(),
                             baseline_sigma_a=fitted.sigma_a.tolist())
                return fitted
    except BaseException as exc:
        audit.update(outcome='failed', elapsed_sec=clock()-started,
                     error='%s: %s' % (type(exc).__name__, exc),
                     unfinished_current_samples=len(currents))
        if first_joints is not None:
            audit.update(first_joints_rad=first_joints.tolist(), last_joints_rad=last_joints.tolist(),
                         joint_spread_rad=(joint_high-joint_low).tolist())
        exc.baseline_acquisition_audit = audit
        raise


def baseline_audit_summary(audit):
    """Keep UDP results bounded; full encoder/window evidence stays on disk."""
    if not isinstance(audit, dict):
        return None
    result = {key: value for key, value in audit.items()
              if key not in ('joint_samples', 'discarded_windows')}
    result['joint_samples_count'] = len(audit.get('joint_samples', []))
    result['discarded_window_count'] = len(audit.get('discarded_windows', []))
    result['discarded_windows_tail'] = audit.get('discarded_windows', [])[-8:]
    return result
