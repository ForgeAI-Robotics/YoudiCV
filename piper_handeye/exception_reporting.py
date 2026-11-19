"""Exception reporting only; no motion, latch reset, power or retry decisions."""
from __future__ import annotations

MANUAL_INTERVENTION_REQUIRED = 'MANUAL_INTERVENTION_REQUIRED'
TEST_RECOVERY_PENDING = 'TEST_RECOVERY_PENDING'
_COMM_CODES = frozenset({'ARM_COMMUNICATION_LOST', 'VISION_COMMUNICATION_LOST',
    'ARM_FEEDBACK_LOST', 'ROBOT_STATE_TIMEOUT', 'COMMUNICATION_LOST'})
_RETURN_CODES = frozenset({'ARM_ORIGIN_RETURN_FAILED', 'ORIGIN_RETURN_FAILED'})


def exception_report(report, *, cycle_test=False):
    """Report positively evidenced level-one faults without guessing from age.

    A low-confidence image, missing origin verification or generic TimeoutError
    does not prove communication loss or a failed return. Test recovery remains
    owned by the existing parent; ordinary tasks report human attention.
    """
    failure = report.get('failure') or {}
    code = str(report.get('failure_code') or failure.get('code') or '')
    phase = str(report.get('failure_stage') or failure.get('stage') or '')
    origin = report.get('origin_verification') or {}
    failed_return = bool(report.get('failure_recovery_error')) or code in _RETURN_CODES
    failed_return = failed_return or (
        origin.get('verified') is False and 'return' in phase)
    communication_lost = code in _COMM_CODES or failure.get('type') in {
        'ConnectionRefusedError', 'ConnectionResetError', 'BrokenPipeError'}
    if not failed_return and not communication_lost:
        return {}
    return {
        'exception_level': 1,
        'exception_status_code': TEST_RECOVERY_PENDING if cycle_test else MANUAL_INTERVENTION_REQUIRED,
        'production_exception_status_code': MANUAL_INTERVENTION_REQUIRED,
        'exception_reason_code': 'ARM_ORIGIN_RETURN_FAILED' if failed_return else 'COMMUNICATION_LOST',
        'manual_intervention_required': not cycle_test,
        'recovery_owner': 'existing_cycle_test_parent' if cycle_test else 'robot_supervisor_and_operator',
        'report_only': True,
    }
