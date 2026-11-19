"""Execution evidence and report I/O only; no robot control or success policy."""
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
import os
import sys
import tempfile

import yaml


def plain_data(value):
    """Preserve numeric types instead of stringifying NumPy measurements."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Enum):
        return plain_data(value.value)
    if isinstance(value, Path):
        return str(value)
    if type(value).__module__.split('.')[0] == 'numpy' and hasattr(value, 'tolist'):
        converted = value.tolist()
        if type(converted) is type(value):
            raise TypeError(f'unsupported report scalar: {type(value).__name__}')
        return plain_data(converted)
    if isinstance(value, Mapping):
        return {plain_data(k): plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_data(v) for v in value]
    raise TypeError(f'unsupported report value: {type(value).__name__}')


def report_yaml(report):
    return yaml.safe_dump(plain_data(report), sort_keys=False, allow_unicode=True)


def write_report_atomic(path, report):
    """Serialize first; replace only a fully written report in the same folder."""
    path = Path(path)
    encoded = report_yaml(report)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture_failure(report, exc, traceback_text):
    # Recovery checkpoints may change phase later. Freeze the failing stage now.
    stage = report.get('active_stage') or report.get('phase') or 'unknown'
    message = str(exc)
    report.update(status='failed', lifecycle_status='failed', process_exit_code=1,
                  failure_stage=stage,
                  failure={'type': type(exc).__name__, 'message': message,
                           'stage': stage, 'traceback': traceback_text})
    pose_audit = getattr(exc, 'stationary_reference_pose_audit', None)
    if isinstance(pose_audit, dict):
        try:
            report['stationary_reference_pose_failure'] = plain_data(pose_audit)
        except (TypeError, ValueError) as reporting_exc:
            report['stationary_reference_pose_diagnostics_error'] = f'{type(reporting_exc).__name__}: {reporting_exc}'
    diagnostics_error = getattr(exc, 'stationary_reference_pose_diagnostics_error', None)
    if diagnostics_error is not None:
        report['stationary_reference_pose_diagnostics_error'] = str(diagnostics_error)
    if 'no stable fresh button target before motion' in message:
        report['failure_code'] = 'BUTTON_LOCALIZATION_TIMEOUT'
        diagnostics = getattr(exc, 'stationary_target_failure', None)
        if isinstance(diagnostics, dict):
            try:
                report['stationary_target_failure'] = plain_data(diagnostics)
            except (TypeError, ValueError) as reporting_exc:
                report['stationary_target_diagnostics_error'] = f'{type(reporting_exc).__name__}: {reporting_exc}'


def update_report_checkpoint(report, kind, value, phase):
    """Mirror recorded facts without commanding or verifying any new motion."""
    report['active_stage'] = phase
    if (kind in {'origin_return_completed', 'origin_return_verified',
                 'origin_return_failed', 'final_origin_verification'}
            and 'verified' in value and 'hardware_ok' in value):
        origin = plain_data(value)
        report['origin_verification'] = origin
        report['returned_to_origin'] = bool(
            origin.get('verified') is True and origin.get('hardware_ok') == 1)
    if kind == 'motion_segment_started' and value.get('segment') == 'guarded_press':
        report['press_execution_started'] = True


def execution_evidence(report):
    """Use encoder verification, never returned_to_origin alone, as proof."""
    journal = report.get('journal') or []
    flow_events = list(report.get('events') or []) + [r.get('value') or {}
        for r in journal if r.get('kind') == 'flow_event']
    report['opencv_fallback_used'] = bool(report.get('opencv_fallback_used') or
        any(e.get('event') == 'post_press_opencv_adopted' for e in flow_events))
    on_events = [e for e in flow_events if e.get('event') == 'press_verified_on']
    if on_events:
        report['on_confirmation_source'] = on_events[-1].get('confirmation_source')
    guarded = [r for r in journal if r.get('kind') == 'motion_segment_started'
               and (r.get('value') or {}).get('segment') == 'guarded_press']
    executed = [r for r in report.get('motion_executions') or []
                if r.get('segment') == 'guarded_press']
    report['press_execution_started'] = bool(
        report.get('press_execution_started') or guarded or executed)
    report.setdefault('press_attempts', max(len(guarded), len(executed)))
    origin = report.get('origin_verification')
    if not isinstance(origin, dict):
        origin = (report.get('failure_recovery') or {}).get('origin_verification')
    if not isinstance(origin, dict):
        origin = report.get('failure_robot_state')
    if isinstance(origin, dict):
        report['origin_verification'] = origin
        report['returned_to_origin'] = bool(origin.get('verified') == True
                                            and origin.get('hardware_ok') == 1)
    failure = report.get('failure') or {}
    from .exception_reporting import exception_report
    attention = exception_report(report,
        cycle_test=os.environ.get('YOUXIAOMEI_CYCLE_PARENT_RECOVERY') == '1')
    if attention:
        report['exception_report'] = attention
    return plain_data({
        **({'exception_report': attention} if attention else {}),
        'flow_status': report.get('flow_status'),
        'external_call_off_wait_timeout': report.get('external_call_off_wait_timeout'),
        'numeric_cycle_on_wait_timeout': report.get('numeric_cycle_on_wait_timeout'),
        'process_exit_code': report.get('process_exit_code'),
        'origin_verification': report.get('origin_verification'),
        'returned_to_origin': report.get('returned_to_origin'),
        'press_execution_started': report['press_execution_started'],
        'press_attempts': report['press_attempts'],
        'opencv_fallback_used': report['opencv_fallback_used'],
        'on_confirmation_source': report.get('on_confirmation_source'),
        'failure_stage': report.get('failure_stage'),
        'failure_code': report.get('failure_code'),
        'failure_type': failure.get('type'),
        'failure_reason': failure.get('message'),
        'reporting_errors': report.get('reporting_errors'),
    })


def terminal_event(report):
    evidence = execution_evidence(report)
    success = int(report.get('process_exit_code', 1)) == 0
    aborted = not success and evidence['failure_type'] == 'TerminationRequested'
    from .button_light_flow import NUMERIC_CYCLE_TIMEOUT_FLOWS
    degraded = success and evidence['flow_status'] in NUMERIC_CYCLE_TIMEOUT_FLOWS
    phase = ({'call': 'call_button_segment_completed',
              'target_floor': 'target_floor_button_segment_completed'}.get(
                  report.get('segment_id'), 'arm_button_task_completed') if success
             else 'aborted' if aborted else 'failed')
    return plain_data({
        **(report.get('exception_report') or {}),
        **{k: report.get(k) for k in ('task_id', 'mission_id', 'segment_id',
                                     'current_floor', 'destination_floor')},
        'target': report.get('target_label'), 'producer': 'arm_pressing',
        'phase': 'numeric_cycle_on_timeout_skip' if degraded else phase,
        'outcome': 'degraded' if degraded else 'completed' if success else 'aborted' if aborted else 'failed',
        'reason_code': ('NUMERIC_CYCLE_ON_TIMEOUT_SKIP' if degraded else None if success else 'TASK_ABORTED' if aborted else
                        report.get('failure_code') or str(evidence['failure_type'] or 'ARM_TASK_FAILED').upper()),
        'evidence': evidence, 'raw_event': 'task_terminal',
    })


def finalize_report(report, path, publish):
    """Keep the primary result even if a secondary reporting operation fails.

    Return reporting errors; callers fail an otherwise successful process if
    persistence fails, but never replace an existing motion/vision exception.
    """
    errors = []

    def failed(component, exc):
        item = {'component': component, 'error': f'{type(exc).__name__}: {exc}'}
        errors.append(item)
        report.setdefault('reporting_errors', []).append(item)
        print(f'arm reporting error: {item}', file=sys.stderr, flush=True)
        if int(report.get('process_exit_code', 1)) == 0:
            capture_failure(report, RuntimeError(item['error']), '')
            report['failure_code'] = 'REPORTING_FAILED'

    try:
        execution_evidence(report)
        write_report_atomic(path, report)
    except Exception as exc:
        failed('execution_report', exc)
        # A bad optional diagnostic cannot prevent saving the core result.
        try:
            core = {k: report.get(k) for k in ('schema_version', 'status', 'lifecycle_status',
                    'phase', 'task_id', 'mission_id', 'segment_id', 'target_label',
                    'current_floor', 'destination_floor', 'failure', 'failure_stage', 'failure_code')}
            core.update(execution_evidence(report), report_incomplete=True)
            write_report_atomic(path, core)
        except Exception as fallback_exc:
            failed('minimal_execution_report', fallback_exc)
    try:
        publish(terminal_event(report))
    except Exception as exc:
        failed('terminal_status', exc)
        try:
            write_report_atomic(path, report)
        except Exception:
            pass
    return errors


def arm_failure_message(evidence):
    """Human-readable failure facts; no inference that a failed task succeeded."""
    evidence = evidence if isinstance(evidence, dict) else {}
    reason = evidence.get('failure_reason')
    code = evidence.get('failure_code')
    if not reason and not code:
        return '任务失败'
    if code == 'BUTTON_LOCALIZATION_TIMEOUT' or 'no stable fresh button target before motion' in str(reason):
        reason = '按压前定位采样超时' if evidence.get('press_execution_started') is False else '定位采样超时'
    elif code == 'REPORTING_FAILED':
        reason = '任务报告保存或状态反馈失败'
    parts = ['任务失败：' + str(reason)[:240]]
    if evidence.get('press_execution_started') is False:
        parts.append('未执行按压')
    elif evidence.get('press_execution_started') is True:
        parts.append('已进入按压阶段')
    origin = evidence.get('origin_verification') or {}
    if origin.get('verified') is True and origin.get('hardware_ok') == 1:
        parts.append('机械臂已归位')
    elif origin.get('verified') is False and origin.get('hardware_ok') == 1:
        parts.append('机械臂尚未归位')
    else:
        parts.append('机械臂归位状态未核验')
    return '；'.join(parts)
