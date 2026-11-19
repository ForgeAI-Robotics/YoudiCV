from __future__ import annotations

import json
import queue
import threading
from typing import TYPE_CHECKING

from piper_handeye.voice_command import (
    VoiceCommand,
    VoiceIntent,
    assistant_name_variants,
    normalize_transcript,
    parse_voice_command,
)

if TYPE_CHECKING:
    from run_voice_command_gateway import CommandDispatcher


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


def dialogue_relevant_transcript(
    primary_text: str,
    verifier_text: str = "",
    wake_word: str | None = None,
) -> bool:
    """Only expose wake words and supported control intents in the chat UI."""
    names = () if not wake_word else assistant_name_variants(wake_word)
    for text in (primary_text, verifier_text):
        compact = normalize_transcript(text).replace(" ", "")
        if compact == "没事了":
            return True
        if any(name in compact for name in names):
            return True
        if parse_voice_command(text).intent is not VoiceIntent.UNKNOWN:
            return True
    return False


class JsonControlInput:
    """Read newline-delimited GUI controls without blocking an ASR loop."""

    def __init__(self, stream):
        self.stream = stream
        self.messages: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read(self) -> None:
        for line in self.stream:
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("control message must be an object")
                self.messages.put(value)
            except Exception as exc:
                self.messages.put({
                    "command": "invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                })


def handle_control_message(
    message: dict,
    dispatcher: "CommandDispatcher",
    *,
    voice_enabled: bool,
) -> tuple[bool, list[dict]]:
    """Apply one GUI control and return the new voice state plus events."""
    command = str(message.get("command") or "")
    if command == "set_voice_enabled":
        raw_enabled = message.get("enabled")
        if not isinstance(raw_enabled, bool):
            return voice_enabled, [{
                "event": "control_rejected",
                "message": "语音开关状态无效",
            }]
        return raw_enabled, [{"event": "voice_state", "enabled": raw_enabled}]
    if command == "set_current_floor":
        try:
            dispatcher.set_current_floor(message.get("floor"))
        except (TypeError, ValueError) as exc:
            return voice_enabled, [{
                "event": "control_rejected", "message": str(exc),
            }]
        return voice_enabled, [{
            "event": "current_floor_set", "floor": dispatcher.current_floor,
        }]
    if command == "resume_trip":
        try:
            return voice_enabled, [dispatcher.resume_trip()]
        except Exception as exc:
            return voice_enabled, [{
                "event": "trip_resume_rejected",
                "message": f"无法继续乘梯任务：{type(exc).__name__}: {exc}",
            }]
    if command == "manual_floor":
        raw_floor = message.get("floor")
        if isinstance(raw_floor, bool) or not isinstance(raw_floor, int):
            return voice_enabled, [{
                "event": "manual_command_rejected",
                "message": "楼层必须是 1–16 的整数",
            }]
        if not 1 <= raw_floor <= 16:
            return voice_enabled, [{
                "event": "manual_command_rejected",
                "message": "楼层必须在 1–16 之间",
            }]
        supplied_text = str(message.get("text") or "").strip()
        text = supplied_text[:120] if supplied_text else f"送外卖到{raw_floor}楼"
        events = [{"event": "manual_command", "floor": raw_floor, "text": text}]
        if dispatcher.active is not None:
            events.append({
                "event": "manual_command_rejected",
                "message": "当前任务正在执行，请等待任务结束",
            })
            return voice_enabled, events
        motion = VoiceCommand(
            VoiceIntent.PRESS_BUTTON, text, text, button=raw_floor
        )
        try:
            events.append(dispatcher.dispatch_motion(motion))
        except Exception as exc:
            events.append({
                "event": "manual_command_rejected",
                "message": f"任务启动失败：{type(exc).__name__}: {exc}",
            })
        return voice_enabled, events
    if command == "manual_button":
        target = str(message.get("target") or "").strip().casefold()
        if target not in {"up", "down"}:
            return voice_enabled, [{
                "event": "manual_command_rejected",
                "message": "方向按钮只能是 up 或 down",
            }]
        text = str(message.get("text") or ("按上行按钮" if target == "up" else "按下行按钮"))[:120]
        events = [{"event": "manual_command", "target": target, "text": text}]
        if dispatcher.active is not None:
            events.append({
                "event": "manual_command_rejected",
                "message": "当前任务正在执行，请等待任务结束",
            })
            return voice_enabled, events
        motion = VoiceCommand(VoiceIntent.PRESS_BUTTON, text, text, button=target)
        try:
            events.append(dispatcher.dispatch_motion(motion))
        except Exception as exc:
            events.append({
                "event": "manual_command_rejected",
                "message": f"任务启动失败：{type(exc).__name__}: {exc}",
            })
        return voice_enabled, events
    if command in {"origin_check", "origin_return"}:
        operation = "check" if command == "origin_check" else "return"
        if dispatcher.active is not None:
            return voice_enabled, [{
                "event": "origin_operation_rejected",
                "operation": operation,
                "message": "当前任务正在执行，请等待任务结束",
            }]
        try:
            return voice_enabled, [dispatcher.dispatch_origin_operation(operation)]
        except Exception as exc:
            return voice_enabled, [{
                "event": "origin_operation_rejected",
                "operation": operation,
                "message": f"原位操作启动失败：{type(exc).__name__}: {exc}",
            }]
    if command == "abort_and_return":
        try:
            return voice_enabled, [dispatcher.abort_and_return()]
        except Exception as exc:
            return voice_enabled, [{
                "event": "origin_operation_rejected",
                "operation": "abort_and_return",
                "message": f"中止返程启动失败：{type(exc).__name__}: {exc}",
            }]
    return voice_enabled, [{
        "event": "control_rejected",
        "message": str(message.get("error") or "未知控制命令"),
    }]
