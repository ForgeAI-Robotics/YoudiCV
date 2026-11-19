"""Map production voice/robot JSON events to stable Chinese chat messages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .mission_status import PRESS_RESULT_UNCERTAIN_MESSAGE, press_result_is_uncertain
from .arm_report import arm_failure_message


def safety_pause_message(evidence: dict | None) -> str:
    """Describe the evidence, without labelling every sensor fault an object."""
    evidence = evidence if isinstance(evidence, dict) else {}
    reason = str(evidence.get('reason') or '').lower()
    underlying = str(evidence.get('underlying_reason') or '').lower()
    if reason in {'unexpected_depth_matrix_residual', 'unexpected_foreground',
                  'matrix_residual_first_frame'}:
        return '检测到异物，暂停任务'
    if 'camera_pose_stale' in reason or 'camera_pose_stale' in underlying:
        return '相机位姿数据过期，安全暂停'
    if 'rgb' in reason or 'scene_change' in reason:
        return '视觉画面变化，尚未确认异物，安全暂停'
    if any(word in reason for word in ('monitor_failure', 'stale', 'coverage', 'unavailable', 'not_ready')):
        return '安全感知数据异常，暂停任务'
    return '安全监测触发暂停，原因待核实'


@dataclass(frozen=True)
class ChatMessage:
    speaker: str
    text: str


def completion_message(event: dict, *, uncertainty_seen: bool = False) -> str:
    code = int(event.get("return_code", 1))
    if code == 0:
        return "机械臂按钮操作已完成"
    if code == 2 and (uncertainty_seen or press_result_is_uncertain(event)):
        return PRESS_RESULT_UNCERTAIN_MESSAGE
    evidence = event.get('arm_report')
    if not isinstance(evidence, dict):
        evidence = next((r.get('evidence') or {} for r in reversed(event.get('mission_status_events') or [])
                         if r.get('producer') == 'arm_pressing' and r.get('raw_event') == 'task_terminal'
                         and (not event.get('task_id') or r.get('task_id') == event['task_id'])), {})
    return arm_failure_message(evidence)


class FloorEstimate:
    """Conservative floor display based only on confirmed button-task events."""

    def __init__(self, *, execute: bool):
        self.execute = bool(execute)
        self.current_floor: int | None = None
        self.last_confirmed_floor: int | None = None
        self.target: int | str | None = None
        self.moving = False
        self.arrived = False
        self.reason = "暂无到站确认"
        self.manual_floor: int | None = None

    def set_manual_floor(self, value) -> str:
        """Set or clear a test-only displayed floor without moving the robot."""
        if value in (None, ""):
            self.manual_floor = None
            return self.consume({})
        floor = self._floor(value)
        if floor is None:
            raise ValueError("测试楼层必须是 1–16 之间的整数")
        self.manual_floor = floor
        self.current_floor = floor
        self.last_confirmed_floor = floor
        self.reason = "手动测试起始楼层"
        return f"当前楼层估计：{floor} 楼（手动测试值）"

    @staticmethod
    def _floor(value):
        if isinstance(value, bool):
            return None
        try:
            floor = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return floor if 1 <= floor <= 16 else None

    @classmethod
    def _target(cls, event: dict):
        for key in ("button", "target"):
            value = event.get(key)
            if str(value).casefold() in {"up", "down"}:
                return str(value).casefold()
            floor = cls._floor(value)
            if floor is not None:
                return floor
        argv = event.get("argv")
        if isinstance(argv, list):
            for index, item in enumerate(argv[:-1]):
                if Path(str(item)).name == "run_button_batch.py":
                    value = str(argv[index + 1]).casefold()
                    if value in {"up", "down"}:
                        return value
                    return cls._floor(value)
        return None

    def consume(self, event: dict) -> str:
        kind = str(event.get("event") or "")
        phase = str(event.get("phase") or "")
        value = event.get("value") if isinstance(event.get("value"), dict) else {}
        if not self.execute:
            if self.manual_floor is not None:
                return f"当前楼层估计：{self.manual_floor} 楼（手动测试值）"
            return "当前楼层估计：未知（预览模式不驱动机械臂）"

        if kind == "motion_started":
            if self.current_floor is not None:
                self.last_confirmed_floor = self.current_floor
            self.current_floor = None
            # The editable floor is a test starting condition; once a trip
            # begins, resume automatic estimation from mission events.
            self.manual_floor = None
            self.target = self._target(event)
            self.moving = True
            self.arrived = False
            self.reason = "正在执行，尚未确认到站"
        elif kind == "flow:button_became_off" or (
            phase == "flow:button_became_off"
            or (kind == "flow_event" and value.get("event") == "button_became_off")
        ):
            self.arrived = True
            self.moving = False
            floor = self._floor(self.target)
            if floor is not None:
                self.current_floor = floor
                self.last_confirmed_floor = floor
                self.reason = "目标楼层按钮熄灭，已确认到站"
            else:
                self.current_floor = None
                self.reason = "上下行呼梯按钮熄灭，但没有具体楼层读数"
        elif kind == "motion_completed":
            self.moving = False
            completed_floor = self._floor(event.get("button", self.target))
            if event.get("trip_stage") == "floor" and int(
                event.get("return_code", 1)
            ) == 0 and completed_floor is not None:
                self.current_floor = completed_floor
                self.last_confirmed_floor = completed_floor
                self.arrived = True
                self.reason = "目标楼层按钮流程确认到站"
                return f"当前楼层估计：{completed_floor} 楼（{self.reason}）"
            if int(event.get("return_code", 1)) != 0 and not self.arrived:
                self.current_floor = None
                self.reason = "任务未确认到站"
            elif int(event.get("return_code", 1)) == 0 and not self.arrived:
                # Process success alone is not a floor sensor. Require the
                # state-machine's explicit target-button OFF checkpoint.
                self.current_floor = None
                self.reason = "任务结束，但未收到目标按钮熄灭确认"
            elif completed_floor is not None:
                self.current_floor = completed_floor
                self.last_confirmed_floor = completed_floor
        elif kind in {"stop", "abort_return_started", "abort_completed_return_started"}:
            if self.moving:
                self.current_floor = None
                self.moving = False
                self.reason = "任务中止，当前位置未确认"
        elif kind == "trip_waiting_for_square":
            self.moving = False
            floor = self._floor(event.get("from_floor"))
            self.current_floor = floor
            self.target = self._floor(event.get("button"))
            self.reason = "呼梯已完成，等待驶入电梯"

        if self.manual_floor is not None:
            return f"当前楼层估计：{self.manual_floor} 楼（手动测试值）"

        if self.current_floor is not None and not self.moving:
            return f"当前楼层估计：{self.current_floor} 楼（{self.reason}）"
        if self.moving:
            if isinstance(self.target, int):
                destination = f"{self.target} 楼"
            elif self.target == "up":
                destination = "上行呼梯"
            elif self.target == "down":
                destination = "下行呼梯"
            else:
                destination = "目标未识别"
            prior = (
                f"；上次确认 {self.last_confirmed_floor} 楼"
                if self.last_confirmed_floor is not None else ""
            )
            return f"当前楼层估计：未知（正在前往{destination}{prior}）"
        if self.arrived and self.target in {"up", "down"}:
            return f"当前楼层估计：未知（{self.reason}）"
        prior = (
            f"；上次确认 {self.last_confirmed_floor} 楼"
            if self.last_confirmed_floor is not None else ""
        )
        return f"当前楼层估计：未知（{self.reason}{prior}）"


class AssistantEventPresenter:
    """Small stateful view model; it never owns or dispatches robot commands."""

    def __init__(self, assistant_name: str = "优小妹"):
        self.assistant_name = str(assistant_name)
        self.task_active = False
        self.failure_reported = False
        self.press_result_uncertain = False
        self.uncertainty_reported = False

    def consume(self, event: dict) -> list[ChatMessage]:
        kind = str(event.get("event") or "")
        phase = str(event.get("phase") or "")
        value = event.get("value") if isinstance(event.get("value"), dict) else {}
        messages: list[ChatMessage] = []

        if kind == "transcript" and str(event.get("text") or "").strip():
            messages.append(ChatMessage("用户", str(event["text"]).strip()))
        elif kind == "manual_command" and str(event.get("text") or "").strip():
            messages.append(ChatMessage("用户", str(event["text"]).strip()))
        elif kind == "assistant_wake":
            messages.append(ChatMessage(self.assistant_name, "在的"))
        elif kind == "assistant_wake_cancelled":
            messages.append(ChatMessage(self.assistant_name, "好的"))
        elif kind == "assistant_wake_prompt":
            messages.append(ChatMessage(self.assistant_name, "请给我任务"))
        elif kind == "assistant_wake_timeout":
            messages.append(ChatMessage(self.assistant_name, "任务未下达"))
        elif kind == "voice_state":
            state = "已开启" if event.get("enabled") else "已关闭"
            messages.append(ChatMessage(self.assistant_name, f"语音呼叫{state}"))
        elif kind == "motion_started":
            self.task_active = True
            self.failure_reported = False
            self.press_result_uncertain = False
            self.uncertainty_reported = False
            messages.append(ChatMessage(self.assistant_name, "正在执行"))
        elif kind == "mission_progress":
            if press_result_is_uncertain(event):
                self.press_result_uncertain = True
                if not self.uncertainty_reported:
                    messages.append(ChatMessage(self.assistant_name, PRESS_RESULT_UNCERTAIN_MESSAGE))
                    self.uncertainty_reported = True
            records = event.get("mission_status_events") or []
            terminal = records[-1] if records else {}
            hard_failure = (terminal.get("evidence") or {}).get("process_exit_code") == 1
            if (terminal.get("phase") == "failed" and not self.failure_reported
                    and (hard_failure or not self.press_result_uncertain)):
                messages.append(ChatMessage(self.assistant_name,
                                            arm_failure_message(terminal.get('evidence'))))
                self.failure_reported = True
        elif kind == "trip_waiting_for_square":
            self.task_active = False
            self.failure_reported = False
            messages.append(ChatMessage(
                self.assistant_name,
                f"呼梯并返回已完成。请驾驶进入电梯后按 □，继续按{event.get('button')}楼。",
            ))
        elif kind == "dry_run":
            messages.extend((
                ChatMessage(self.assistant_name, "正在执行（预览模式）"),
                ChatMessage(self.assistant_name, "预览完成，未发送机器人指令"),
            ))
        elif kind == "motion_completed":
            self.task_active = False
            detail = completion_message(event, uncertainty_seen=self.press_result_uncertain)
            if detail == PRESS_RESULT_UNCERTAIN_MESSAGE:
                self.press_result_uncertain = True
                if not self.uncertainty_reported:
                    messages.append(ChatMessage(self.assistant_name, detail))
                    self.uncertainty_reported = True
            elif int(event.get("return_code", 1)) == 0:
                messages.append(ChatMessage(self.assistant_name, detail))
            elif not self.failure_reported:
                messages.append(ChatMessage(self.assistant_name, detail))
                self.failure_reported = True
        elif kind == "decision" and event.get("status") == "rejected":
            messages.append(ChatMessage(
                self.assistant_name,
                str(event.get("message") or "未识别到可执行任务，请再说一次"),
            ))
        elif kind in ("manual_command_rejected", "control_rejected"):
            messages.append(ChatMessage(
                self.assistant_name,
                str(event.get("message") or "操作未执行"),
            ))
        elif kind == "stop":
            self.task_active = False
            messages.append(ChatMessage(self.assistant_name, "任务已停止"))
        elif kind == "abort_return_started":
            self.task_active = False
            messages.append(ChatMessage(self.assistant_name, "正在中止任务并返回原位"))
        elif kind == "decision" and event.get("status") == "cancelled":
            messages.append(ChatMessage(self.assistant_name, "任务已取消"))

        if phase == "flow:retry_required":
            messages.append(ChatMessage(self.assistant_name, "尝试再次按压"))
        elif phase == "visual_safety_paused":
            value = event.get("value") or {}
            messages.append(ChatMessage(self.assistant_name, safety_pause_message(value)))
        elif phase == "visual_safety_resumed":
            messages.append(ChatMessage(self.assistant_name, "安全条件恢复，继续任务"))
        elif phase == "visual_safety_failed":
            self.task_active = False
            if not self.failure_reported:
                messages.append(ChatMessage(self.assistant_name, "任务失败"))
                self.failure_reported = True
        return messages
