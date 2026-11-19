"""One shared photo-pose tolerance and bounded read-only encoder confirmation."""
from __future__ import annotations

import math
import time

OBSERVATION_JOINT_TOLERANCE_RAD = .008
OBSERVATION_FEEDBACK_MAX_AGE_SEC = .25
OBSERVATION_STABLE_SEC = .08
OBSERVATION_CONFIRM_TIMEOUT_SEC = 1.0
OBSERVATION_STABILITY_DELTA_RAD = .002
# Independent of the short pre-photo stability window above. A reference may
# only pass if its separate .008-rad endpoint and all image checks also pass.
OBSERVATION_CAPTURE_DRIFT_RAD = .006
OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD = 1e-12


def capture_drift_exceeds_limit(value):
    """Compare a finite max joint change, tolerating only floating-point noise."""
    return value > OBSERVATION_CAPTURE_DRIFT_RAD + OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD


def observation_joint_error(joints, target):
    measured, desired = list(joints), list(target)
    if (len(measured) != 6 or len(desired) != 6
            or not all(math.isfinite(float(q)) for q in measured + desired)):
        raise ValueError("observation pose requires six finite joints")
    return max(abs(float(a)-float(b)) for a, b in zip(measured, desired))


class ObservationPoseError(RuntimeError):
    def __init__(self, reason, audit):
        self.audit = dict(audit, outcome=reason)
        super().__init__("observation pose confirmation failed: " + reason)


def wait_for_stable_observation_pose(target, read_state, check_safety, *,
        clock=time.monotonic, sleep=time.sleep):
    """Observe distinct post-entry samples; never command/resend a movement.

    read_state must be nonblocking (PiperUdpRobot.latest_state in production).
    Both target error and measured motion across the stable window are bounded.
    """
    target = list(target)
    observation_joint_error(target, target)
    started = clock()
    audit = dict(target_joints_rad=target, tolerance_rad=OBSERVATION_JOINT_TOLERANCE_RAD,
        stable_sec=OBSERVATION_STABLE_SEC, timeout_sec=OBSERVATION_CONFIRM_TIMEOUT_SEC,
        stability_delta_rad=OBSERVATION_STABILITY_DELTA_RAD, samples=[],
        distinct_samples=0, elapsed_sec=0.)
    previous = None
    stable_since = None
    low = high = None
    count = 0
    try:
        while True:
            check_safety()
            now = clock()
            audit['elapsed_sec'] = now-started
            if now-started >= OBSERVATION_CONFIRM_TIMEOUT_SEC:
                raise ObservationPoseError('timeout', audit)
            sample = read_state()
            now = clock()
            if now-started >= OBSERVATION_CONFIRM_TIMEOUT_SEC:
                raise ObservationPoseError('timeout', audit)
            stamp = sample.timestamp
            joints = list(sample.joints)
            try:
                error = observation_joint_error(joints, target)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ObservationPoseError('feedback_invalid_or_stale', audit) from exc
            if (sample.raw.get('hardware_ok') != 1
                    or type(stamp) not in (int, float) or not math.isfinite(stamp)
                    or not 0 <= now-stamp <= OBSERVATION_FEEDBACK_MAX_AGE_SEC):
                raise ObservationPoseError('feedback_invalid_or_stale', audit)
            if previous is not None and stamp < previous:
                raise ObservationPoseError('feedback_timestamp_reversed', audit)
            if stamp != previous:
                if previous is not None and stamp-previous > OBSERVATION_FEEDBACK_MAX_AGE_SEC:
                    stable_since, count = None, 0
                    audit['gap_resets'] = audit.get('gap_resets', 0)+1
                previous = stamp
                # A fresh cached packet received before this check is not a
                # second independent confirmation of the completed endpoint.
                if stamp >= started:
                    row = dict(timestamp=stamp, age_sec=now-stamp,
                        joints_rad=joints, max_error_rad=error, elapsed_sec=now-started)
                    audit['samples'].append(row)
                    audit['distinct_samples'] += 1
                    audit['final_error_rad'] = error
                    if error <= OBSERVATION_JOINT_TOLERANCE_RAD:
                        if stable_since is None:
                            stable_since, count = stamp, 0
                            low, high = list(joints), list(joints)
                        next_low = [min(a,b) for a,b in zip(low,joints)]
                        next_high = [max(a,b) for a,b in zip(high,joints)]
                        spread = max(b-a for a,b in zip(next_low,next_high))
                        if spread > OBSERVATION_STABILITY_DELTA_RAD:
                            stable_since, count = stamp, 0
                            low, high = list(joints), list(joints)
                            audit['motion_resets'] = audit.get('motion_resets',0)+1
                        else:
                            low, high = next_low, next_high
                        count += 1
                        if count >= 2 and stamp-stable_since >= OBSERVATION_STABLE_SEC:
                            check_safety()
                            if clock()-started >= OBSERVATION_CONFIRM_TIMEOUT_SEC:
                                raise ObservationPoseError('timeout', audit)
                            audit.update(outcome='stable', elapsed_sec=clock()-started,
                                stable_window_sec=stamp-stable_since,
                                stable_window_spread_rad=max(b-a for a,b in zip(low,high)))
                            return sample, audit
                    else:
                        stable_since, count = None, 0
            sleep(.01)
    except BaseException as exc:
        if not isinstance(exc, ObservationPoseError):
            exc.observation_pose_audit = dict(audit, outcome='feedback_or_safety_failure', error=str(exc))
        raise


def stationary_reference_failure_audit(
        before, after, target_joints, *, capture_started, reference_ready,
        feedback_checked, failed_check_stage, final_error, capture_drift,
        reference, exception):
    """Failure-only evidence; never decides whether a pose may move or pass.

    Nonfinite values become null with explicit invalid-element indices so the
    journal is strict JSON and preserves which readings could not be computed.
    Only whitelisted feedback fields are recorded, never the full raw packet.
    """
    def number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError, OverflowError):
            return None

    def vector(value):
        try:
            return [number(q) for q in value]
        except (TypeError, ValueError):
            return None

    def feedback(sample, observed):
        if sample is None:
            return None
        joints = vector(getattr(sample, 'joints', None))
        stamp = number(getattr(sample, 'timestamp', None))
        raw = getattr(sample, 'raw', {})
        flag = raw.get('hardware_ok') if isinstance(raw, dict) else None
        # Preserve string-vs-integer flags: original hardware_ok == 1 semantics
        # must not silently be changed by diagnostic numeric conversion.
        safe_flag = flag if type(flag) in (str, int, float, bool, type(None)) else str(flag)[:120]
        if type(safe_flag) is float and not math.isfinite(safe_flag):
            safe_flag = None
        return dict(joints_rad=joints, hardware_ok=safe_flag,
            hardware_ok_type=type(flag).__name__, timestamp_monotonic_sec=stamp,
            age_sec=None if stamp is None else number(observed-stamp),
            joint_count=None if joints is None else len(joints),
            nonfinite_or_invalid_joint_indices_1based=([] if joints is None else
                [j+1 for j,q in enumerate(joints) if q is None]))

    start = feedback(before, capture_started)
    end = feedback(after, feedback_checked)
    target = vector(target_joints)
    def differences(left, right):
        if left is None or right is None or len(left) != len(right):
            return None
        return [None if a is None or b is None else number(a-b)
                for a,b in zip(left,right)]
    end_joints = None if end is None else end['joints_rad']
    start_joints = None if start is None else start['joints_rad']
    drift = differences(end_joints, start_joints)
    error = differences(end_joints, target)
    def absolute(values):
        return None if values is None else [None if q is None else abs(q) for q in values]
    def maximum_joint(values):
        if values is None or not values or any(q is None for q in values):
            return None
        return max(range(len(values)), key=lambda j:abs(values[j]))+1

    final = number(final_error)
    change = number(capture_drift)
    raw = getattr(after, 'raw', {})
    hardware_ok = raw.get('hardware_ok') if isinstance(raw, dict) else None
    reasons = []
    if failed_check_stage == 'pose_guard':
        if hardware_ok != 1:
            reasons.append('hardware_not_ok')
        if final is None:
            reasons.append('target_error_nonfinite')
        elif final > OBSERVATION_JOINT_TOLERANCE_RAD:
            reasons.append('target_error_exceeded')
        if change is None:
            reasons.append('capture_drift_nonfinite')
        elif capture_drift_exceeds_limit(change):
            reasons.append('capture_drift_exceeded')
    else:
        reasons.append(dict(feedback_read='feedback_read_failed',
            error_calculation='joint_error_calculation_failed',
            safety_check='safety_check_failed').get(failed_check_stage, 'unknown_check_failed'))
    ref = reference if isinstance(reference, dict) else {}
    return dict(version=1, diagnostic_only=True, stage='after_capture',
        failed_check_stage=failed_check_stage, guard_evaluated=failed_check_stage=='pose_guard',
        rejection_reasons=reasons, primary_rejection_reason=reasons[0] if reasons else None,
        exception_type=type(exception).__name__, exception_message=str(exception),
        before=start, after=end, target_joints_rad=target,
        joint_drift_signed_rad=drift, joint_drift_abs_rad=absolute(drift),
        joint_target_error_signed_rad=error, joint_target_error_abs_rad=absolute(error),
        max_drift_joint_1based=maximum_joint(drift), max_target_error_joint_1based=maximum_joint(error),
        capture_drift_rad=change, final_error_rad=final,
        capture_drift_is_finite=change is not None, final_error_is_finite=final is not None,
        thresholds=dict(capture_drift_rad=OBSERVATION_CAPTURE_DRIFT_RAD,
            capture_drift_numeric_epsilon_rad=OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD,
            stability_window_delta_rad=OBSERVATION_STABILITY_DELTA_RAD,
            target_error_rad=OBSERVATION_JOINT_TOLERANCE_RAD,
            feedback_max_age_sec=OBSERVATION_FEEDBACK_MAX_AGE_SEC,
            comparison='capture drift > limit + 1e-12 rad; target error retains strict greater-than'),
        capture_started_monotonic_sec=number(capture_started),
        reference_ready_monotonic_sec=number(reference_ready),
        feedback_checked_monotonic_sec=number(feedback_checked),
        capture_duration_sec=number(reference_ready-capture_started),
        reference=dict(mode=ref.get('mode'), reference_count=ref.get('reference_count'),
            sequences=ref.get('sequences'), capture_barrier_unix=ref.get('capture_barrier_unix')))
