"""Stable, robot-neutral mission status events for the arm task pipeline.

The JSONL outbox is deliberately separate from the vendor Android UDP protocol
and ROS. A navigation adapter can consume this contract without gaining motion
authority or relying on undocumented vendor commands.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import time
import uuid
from datetime import datetime
from typing import Any
from .arm_report import arm_failure_message


SCHEMA_VERSION = 1
PRESS_RESULT_UNCERTAIN_MESSAGE = "已执行按压，但亮灯确认不确定"
UNCERTAIN_PRESS_FLOWS = frozenset({
    "post_press_confirmation_uncertain", "post_press_confirmation_disagreement",
})
DEFAULT_OUTBOX = (
    Path(__file__).resolve().parents[1]
    / "mission_status_runs"
    / f"mission_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
)

_FLOW_MAP = {
    "light_observed": ("observing_button", "observation"),
    "waiting_observation_uncertain": ("waiting_for_floor", "degraded"),
    "waiting_button_still_on": ("waiting_for_floor", "active"),
    "button_already_on": ("button_on_confirmed", "active"),
    "initial_state_confirmed": ("observing_button", "observation"),
    "initial_light_recheck": ("initial_light_recheck", "degraded"),
    "initial_light_unresolved": ("initial_light_unresolved", "degraded"),
    "initial_observation_reacquiring": ("initial_observation_reacquiring", "degraded"),
    "initial_view_search_started": ("initial_view_search", "degraded"),
    "initial_view_search_finished": ("initial_observation_reacquiring", "degraded"),
    "initial_observation_waiting": ("initial_observation_waiting", "degraded"),
    "initial_observation_recovered": ("initial_observation_recovered", "active"),
    "button_target_refresh_failed": ("failed", "failed"),
    "button_target_refreshed": ("relocalizing_button", "active"),
    "press_attempt_finished": ("press_feedback_received", "active"),
    "press_tangential_protection": ("press_tangential_protection", "degraded"),
    "post_press_recheck_started": ("post_press_rechecking", "degraded"),
    "post_press_recheck_finished": ("post_press_rechecking", "active"),
    "post_press_opencv_adopted": ("post_press_opencv_adopted", "degraded"),
    "at_post_press_observation": ("observing_press_result", "active"),
    "post_press_on_captured_early": ("button_on_confirmed", "active"),
    "post_press_observation_fallback": ("observing_press_result", "degraded"),
    "post_press_confirmation_uncertain": ("press_result_uncertain", "degraded"),
    "post_press_confirmation_disagreement": ("press_result_uncertain", "degraded"),
    "post_press_state_confirmed": ("button_state_confirmed", "active"),
    "retry_moved_to_near_observation": ("retry_repositioning", "active"),
    "retry_held_at_near_observation": ("retry_ready", "active"),
    "retry_required": ("retry_required", "active"),
    "press_verified_on": ("button_on_confirmed", "active"),
    "at_waiting_observation": ("waiting_for_floor", "active"),
    "button_became_off": ("floor_arrival_observed", "active"),
    "external_call_off_wait_timeout": ("external_call_timeout_release", "degraded"),
    "numeric_cycle_on_wait_timeout": ("numeric_cycle_on_timeout_skip", "degraded"),
    "at_origin": ("origin_verified", "active"),
}

PHASE_ZH = {
    "numeric_cycle_on_timeout_skip": "按钮持续亮10秒，疑似内呼机异常；归位后跳过本项",
    "accepted": "任务已接收",
    "arm_task_started": "机械臂任务已启动",
    "preflight": "正在做执行前检查",
    "approaching_button": "正在接近按钮",
    "button_observation_ready": "已到按钮观察位置",
    "approach_result": "接近阶段已结束",
    "approach_completed": "按钮接近动作已完成",
    "approach_target_staged": "接近轨迹已准备",
    "motion_plan_validated": "机械臂轨迹已通过碰撞检查",
    "motion_segment_started": "机械臂正在执行运动段",
    "motion_segment_completed": "机械臂运动段已完成",
    "press_readiness_checked": "正在确认按压条件",
    "pressing": "正在按压",
    "press_feedback_received": "已收到按压反馈",
    "press_tangential_protection": "侧向力保护触发回撤，正在检查按钮是否按亮",
    "post_press_rechecking": "按后灯态待确认，正在原位复核",
    "post_press_opencv_adopted": "对照未确认，按OpenCV结果继续任务",
    "button_on_adopted": "已按OpenCV采用亮灯结果，正在等待熄灭",
    "press_feedback_unconfirmed": "按压接触反馈未确认",
    "observing_button": "正在观察按钮灯态",
    "observation_reacquiring": "等待重新观测按钮",
    "initial_light_recheck": "观察视角已调整，正在重新确认按钮灯态",
    "initial_light_unresolved": "灯态无法可靠确认，未执行按压",
    "initial_observation_reacquiring": "正在重新识别按钮并确认灯态",
    "initial_view_search": "正在小幅调整观察视角，重新寻找按钮",
    "initial_observation_waiting": "暂未识别到按钮，保持静止等待恢复",
    "initial_observation_recovered": "按钮识别已恢复，继续任务",
    "numeric_visibility_waiting": "数字按钮识别暂不稳定，正在静止复核",
    "numeric_visibility_recovered": "数字按钮识别已恢复，正在确认定位",
    "numeric_target_recovered": "数字按钮定位已恢复，继续执行前检查",
    "numeric_visibility_failed": "数字按钮识别复核超时，未执行按压",
    "observing_press_result": "正在快速确认按压结果",
    "press_result_uncertain": PRESS_RESULT_UNCERTAIN_MESSAGE,
    "button_state_confirmed": "按钮灯态已确认",
    "button_on_confirmed": "按钮已亮，正在等待楼层到达",
    "relocalizing_button": "正在重新定位按钮",
    "retry_repositioning": "正在调整重试位置",
    "retry_ready": "重试位置已就绪",
    "retry_required": "按钮未亮，尝试再次按压",
    "waiting_for_floor": "按钮保持亮起，等待目标楼层",
    "floor_arrival_observed": "按钮已熄灭，检测到楼层到达",
    "external_call_timeout_release": "外呼已确认按亮，等待熄灭满30秒，按测试规则超时放行；未确认熄灭",
    "returning_origin": "正在返回机械臂原位",
    "origin_verified": "机械臂原位已验证",
    "origin_return_failed": "返回原位失败",
    "motion_segment_failed": "机械臂运动段失败",
    "motion_segment_cleanup_failed": "机械臂运动段安全清理失败",
    "safety_paused": "安全监测已暂停机械臂任务",
    "safety_resumed": "安全条件恢复，机械臂任务已继续",
    "press_failed": "按压失败",
    "completed": "机械臂任务已完成",
    "failed": "机械臂任务失败",
    "abort_requested": "已请求中止，正在执行安全停止",
    "aborted": "机械臂任务已中止",
    "awaiting_operator_inside_elevator": "呼梯按钮已完成，请人工驶入电梯后继续",
    "call_button_segment_completed": "呼梯按钮操作段完成",
    "target_floor_button_segment_completed": "目标楼层按钮操作段完成",
    "arm_button_task_completed": "机械臂按钮任务已完成",
}


def press_result_is_uncertain(event: dict[str, Any]) -> bool:
    """Read the task's recorded result, never infer success from a bright frame."""
    records = [event]
    for record in event.get("mission_status_events") or []:
        if not isinstance(record, dict):
            continue
        if event.get("task_id") and record.get("task_id") != event["task_id"]:
            continue
        records.append(record)
    for record in records:
        evidence = record.get("evidence") or {}
        phase = str(record.get("phase") or "").removeprefix("flow:")
        if (phase == "press_result_uncertain" or phase in UNCERTAIN_PRESS_FLOWS
                or record.get("flow_status") in UNCERTAIN_PRESS_FLOWS
                or evidence.get("flow_status") in UNCERTAIN_PRESS_FLOWS):
            return True
    return False


def describe_phase_zh(event: dict[str, Any]) -> str:
    phase = str(event.get("phase") or "")
    if phase == 'failed':
        failure = arm_failure_message(event.get('evidence'))
        if failure != '任务失败':
            return failure
    detail = PHASE_ZH.get(phase, phase or "机械臂状态更新")
    if phase == 'initial_observation_waiting':
        pending_reason = (event.get('evidence') or {}).get('reason')
        if pending_reason == 'target_localization_pending':
            detail = '按钮定位尚未稳定，保持静止等待恢复'
        elif pending_reason in {'light_state_unconfirmed', 'local_highlight_conflict'}:
            detail = '按钮灯态尚未确认，保持静止等待恢复'
    attempt = event.get("attempt")
    if attempt is not None:
        detail += f"（第 {attempt} 次）"
    reason = event.get("reason_code")
    if reason:
        detail += f"；原因码 {reason}"
    return detail


def new_task_id() -> str:
    return str(uuid.uuid4())


def normalize_flow_event(value: dict[str, Any]) -> dict[str, Any]:
    raw_event = str(value.get("event") or "flow_event")
    phase, outcome = _FLOW_MAP.get(raw_event, (raw_event, "active"))
    reason_code = None
    if raw_event == "button_target_refresh_failed":
        reason_code = "BUTTON_RELOCALIZATION_FAILED"
    elif raw_event == 'button_already_on':
        # Stage feedback only: the existing wait-OFF/return flow still owns
        # task completion, so this must not publish a completed outcome.
        reason_code = 'BUTTON_ALREADY_TRIGGERED'
    elif raw_event.startswith("post_press_confirmation_"):
        reason_code = raw_event.upper()
    elif raw_event == "waiting_observation_uncertain":
        reason_code = "LIGHT_STATE_UNCERTAIN"
    elif raw_event == "retry_required":
        reason_code = "BUTTON_REMAINS_OFF"
    elif raw_event == "external_call_off_wait_timeout":
        reason_code = "EXTERNAL_CALL_OFF_WAIT_TIMEOUT"
    elif raw_event == "numeric_cycle_on_wait_timeout":
        reason_code = "NUMERIC_CYCLE_ON_TIMEOUT_SKIP"
    elif raw_event == 'post_press_opencv_adopted':
        reason_code = 'POST_PRESS_OPENCV_ADOPTED'
    elif (raw_event == 'press_verified_on'
            and str(value.get('confirmation_source', '')).startswith('opencv_')):
        phase, outcome = 'button_on_adopted', 'active'
        reason_code = 'POST_PRESS_OPENCV_ADOPTED'
    if ((raw_event == "light_observed" and value.get("state") == "UNCERTAIN")
            or raw_event == "waiting_observation_uncertain"
            or (raw_event == "waiting_button_still_on" and value.get("observation_uncertain"))):
        # Lack of valid visual evidence is neither an OFF verdict nor arrival.
        # The flow keeps its original streak resets, retry rules and budgets.
        phase, outcome = "observation_reacquiring", "degraded"
        reason_code = "LIGHT_OBSERVATION_PENDING"
    evidence = {
        key: item for key, item in value.items()
        if key not in {"event"} and item is not None
    }
    return {
        "phase": phase,
        "outcome": outcome,
        "reason_code": reason_code,
        "attempt": value.get("attempt"),
        "evidence": evidence,
        "raw_event": raw_event,
    }


def normalize_light_observation(value: dict[str, Any]) -> dict[str, Any]:
    """Expose image validity without changing the internal light-state machine."""
    pending = value.get("state") == "UNCERTAIN"
    return {
        "phase": "observation_reacquiring" if pending else "observing_button",
        "outcome": "degraded" if pending else "observation",
        "reason_code": "LIGHT_OBSERVATION_PENDING" if pending else None,
        "evidence": value,
        "raw_event": "light_observation",
    }


def _last_global_sequence(stream) -> int:
    stream.seek(0, os.SEEK_END)
    end = stream.tell()
    if end == 0:
        return 0
    cursor = end - 1
    while cursor >= 0:
        stream.seek(cursor)
        if stream.read(1) == b"\n" and cursor != end - 1:
            break
        cursor -= 1
    stream.seek(cursor + 1 if cursor >= 0 else 0)
    line = stream.readline()
    try:
        return int(json.loads(line).get("sequence", 0))
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0


def append_status(
    path: Path, event: dict[str, Any], *, durable: bool = True
) -> dict[str, Any]:
    """Append one fsynced, ordered event. Safe for dispatcher and child writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        sequence = _last_global_sequence(stream) + 1
        record = {
            **event,
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "event_id": str(uuid.uuid4()),
            "task_id": str(event.get("task_id") or ""),
            "mission_id": event.get("mission_id") or event.get("task_id"),
            "segment_id": event.get("segment_id"),
            "emitted_at_unix_sec": time.time(),
            "emitted_monotonic_sec": time.monotonic(),
            "producer": event.get("producer", "arm_pressing"),
        }
        stream.seek(0, os.SEEK_END)
        stream.write((json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), default=str
        ) + "\n").encode("utf-8"))
        stream.flush()
        # The press process already fsyncs its authoritative journal before
        # calling this live-status mirror. Avoid a second synchronous disk wait
        # on the time-critical press/retreat path; dispatcher milestones remain
        # durable and an abrupt child exit is itself reported by the parent.
        if durable:
            os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return record


def read_events(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read complete appended records from a byte offset, tolerating partial tails."""
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, int(offset)))
            data = stream.read()
            end = stream.tell()
    except FileNotFoundError:
        return [], 0
    last_newline = data.rfind(b"\n")
    if last_newline < 0:
        return [], offset
    complete, tail = data[:last_newline + 1], data[last_newline + 1:]
    records = []
    for line in complete.splitlines():
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records, end - len(tail)
