"""Recovery policy gates only: elapsed time never authorizes movement."""
import math

RECOVERY_STABLE_SEC = .5
RETURN_REASSESS_SEC = 10.
FAULT_RECHECK_SEC = 1.
RECOVERY_POLL_SEC = .1
# Explicit GUI/voice/user stops are intentionally absent.
CLEARABLE_STOP_REASONS = frozenset({
    'cycle_physical_estop', 'cycle_feedback_unavailable',
    'matrix_prediction_coverage_insufficient', 'camera_pose_stale',
    'camera_pose_unavailable', 'depth_stream_timeout',
    'absolute_near_field_object', 'matrix_residual_obstruction',
    'unexpected_depth_matrix_residual', 'matrix_monitor_failure_latched',
    'visual_safety_monitor_failure',
})


class EmergencyFeedbackError(RuntimeError):
    def __init__(self, code, message, sample):
        self.code = code
        self.audit = dict(decision='stop', emergency_reason_code=code,
            emergency_feedback=dict(sample), topic='/emergency', reason=message)
        super().__init__(message)


def require_emergency_released(sample):
    age = sample.get('age_sec')
    if sample.get('available') is not True:
        raise EmergencyFeedbackError('feedback_unavailable', '尚未收到有效急停反馈，等待恢复', sample)
    if sample.get('error'):
        raise EmergencyFeedbackError('feedback_error', '急停反馈通信异常，等待恢复：'+str(sample['error']), sample)
    if type(age) not in (int,float) or not math.isfinite(age) or age < 0:
        raise EmergencyFeedbackError('feedback_invalid', '急停反馈时间无效，等待恢复', sample)
    if age > 1.5:
        raise EmergencyFeedbackError('feedback_stale', '急停反馈已过期（%.3f秒，允许1.5秒），等待恢复' % age, sample)
    if sample.get('engaged') is True:
        raise EmergencyFeedbackError('emergency_engaged', '实体急停已按下，等待释放', sample)
    if sample.get('engaged') is not False:
        raise EmergencyFeedbackError('feedback_invalid', '急停反馈状态无效，等待恢复', sample)
    return sample


def emergency_stop_reason(error):
    return ('cycle_physical_estop' if isinstance(error, EmergencyFeedbackError)
            and error.code=='emergency_engaged' else 'cycle_feedback_unavailable')


def require_reset_identity(payload, generation, last_stop):
    """Atomic server-side compare before clearing; never clear a newer stop."""
    expected = payload.get('expected_generation')
    reason = payload.get('expected_reason')
    if (type(expected) is not int or expected != generation
            or reason not in CLEARABLE_STOP_REASONS
            or not isinstance(last_stop,dict) or last_stop.get('reason') != reason):
        raise RuntimeError('停止代号或原因已变化，不允许自动解除')


def recovery_stop_identity(health):
    generation = health.get('generation')
    if (health.get('status') != 'ok' or type(generation) is not int or generation < 0
            or type(health.get('latched')) is not bool):
        raise RuntimeError('停止桥状态无效，等待恢复')
    if health['latched']:
        reason = (health.get('last_stop') or {}).get('reason')
        if reason not in CLEARABLE_STOP_REASONS:
            raise RuntimeError('当前为人工停止或未知停止原因，不自动解除：'+str(reason))
        if health.get('conditional_reset_v1') is not True:
            raise RuntimeError('停止桥尚不支持按代号恢复，保持等待')
    else:reason=None
    return dict(generation=generation, latched=health['latched'], reason=reason)


def require_fresh_clear_visual(status, now):
    stamp=((status.get('pose_timing') or {}).get('last_frame') or {}).get('frame_received_monotonic')
    if (status.get('ready') is not True or status.get('blocked') is not False
            or status.get('monitoring_available') is not True
            or type(stamp) not in (int,float) or not math.isfinite(stamp) or not 0<=now-stamp<=.35
            or (status.get('metrics') or {}).get('matrix_residual_detected_raw') is True):
        raise RuntimeError('原视觉安全条件尚未取得新鲜明确恢复证据，保持等待')
    return status
