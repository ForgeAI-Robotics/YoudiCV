"""Narrow automatic recovery eligibility, independent of hardware and ROS."""
import math


def require_unlatched_generation(health, expected):
    if (type(expected) is not int or expected < 0 or health.get('status') != 'ok'
            or health.get('latched') is not False or type(health.get('generation')) is not int
            or health['generation'] != expected):
        raise RuntimeError('停止锁或停止代号已变化，禁止自动归位；请人工检查')
    return health


def recoverable_retract_failure(response):
    """Only a measured convergence timeout, never stale feedback or a stop."""
    if not isinstance(response, dict) or response.get('failure_code') != 'GUARDED_RETRACT_CONVERGENCE_FAILED':
        return None
    audit = response.get('retract_convergence') or {}
    error, generation = audit.get('final_error_rad'), audit.get('safety_generation')
    if (audit.get('outcome') != 'timeout' or type(generation) is not int or generation < 0
            or type(error) not in (int, float) or not math.isfinite(error)
            or audit.get('distinct_samples', 0) < 2):
        return None
    return dict(safety_generation=generation, final_error_rad=error,
                failure_code=response['failure_code'])


def cycle_parent_recovery_required(exc, enabled):
    return enabled is True and recoverable_retract_failure(getattr(exc, 'response', None)) is not None
