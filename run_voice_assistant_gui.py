#!/usr/bin/env python3
"""Live chat GUI for the local streaming elevator-task entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont, ttk

from piper_handeye.assistant_gui_events import AssistantEventPresenter, FloorEstimate, safety_pause_message, completion_message
from piper_handeye.voice_command import VoiceIntent, parse_voice_command


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = "alsa_input.usb-CF-IC_DOV_2025-0711-1800-00.analog-stereo"
DEFAULT_MODEL_DIR = (
    ROOT / "third_party/sherpa-onnx-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
)
DEFAULT_VAD_MODEL = ROOT / "third_party/sherpa-onnx-models/silero_vad.onnx"
DEFAULT_HOLD_REPORT = ROOT / "button_fast_chain_runs/20260826_150725/chain_report.yaml"
DEFAULT_BACKEND_PYTHON = Path(
    "/home/qianjue/anaconda3/envs/piper-handeye/bin/python"
)


DEFAULT_UI_DPI = 96.0
PREFERRED_FONT_FAMILIES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Droid Sans Fallback",
    "DejaVu Sans",
)


@dataclass(frozen=True)
class UiFonts:
    family: str
    body: tkfont.Font
    title: tkfont.Font
    mode: tkfont.Font
    chat: tkfont.Font
    chat_bold: tkfont.Font


def configure_crisp_ui(root: tk.Tk) -> UiFonts:
    """Align Tk with Xft DPI and avoid per-glyph Chinese fallback fonts."""
    try:
        dpi = float(os.environ.get("YOUXIAOMEI_UI_DPI", DEFAULT_UI_DPI))
    except ValueError as exc:
        raise ValueError("YOUXIAOMEI_UI_DPI 必须是数字") from exc
    if not 72.0 <= dpi <= 384.0:
        raise ValueError("YOUXIAOMEI_UI_DPI 必须在 72–384 之间")
    root.tk.call("tk", "scaling", dpi / 72.0)

    installed = set(tkfont.families(root))
    family = next(
        (candidate for candidate in PREFERRED_FONT_FAMILIES if candidate in installed),
        None,
    )
    if family is None:
        raise RuntimeError(
            "当前 Python 的 Tk 不支持 Xft 矢量字体；请使用桌面启动器或 /usr/bin/python3"
        )
    for named in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(named, root=root).configure(family=family, size=11)
    tkfont.nametofont("TkFixedFont", root=root).configure(family=family, size=11)

    fonts = UiFonts(
        family=family,
        body=tkfont.Font(root=root, family=family, size=11),
        title=tkfont.Font(root=root, family=family, size=18, weight="bold"),
        mode=tkfont.Font(root=root, family=family, size=11, weight="bold"),
        chat=tkfont.Font(root=root, family=family, size=15),
        chat_bold=tkfont.Font(root=root, family=family, size=15, weight="bold"),
    )
    style = ttk.Style(root)
    style.configure(".", font=fonts.body)
    style.configure("Title.TLabel", font=fonts.title)
    return fonts


class VoiceBackend:
    def __init__(
        self,
        events: queue.Queue,
        argv: list[str],
        event_log: Path | None = None,
    ):
        self.events = events
        self.argv = argv
        self.event_log = event_log
        self.process: subprocess.Popen | None = None
        self.intentional_stop = False
        self.stop_requested = threading.Event()
        self.write_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            self.process = subprocess.Popen(
                self.argv,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            if self.stop_requested.is_set():
                os.kill(self.process.pid, signal.SIGINT)
            assert self.process.stdout is not None
            for line in self.process.stdout:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    if self.event_log is not None:
                        logged = dict(value)
                        logged["gui_received_unix_sec"] = time.time()
                        with self.event_log.open("a", encoding="utf-8") as output:
                            output.write(json.dumps(
                                logged, ensure_ascii=False, separators=(",", ":")
                            ) + "\n")
                    self.events.put(value)
            return_code = self.process.wait()
            self.events.put({
                "event": "backend_exit",
                "return_code": return_code,
                "intentional": self.intentional_stop,
            })
        except Exception as exc:
            self.events.put({
                "event": "backend_error",
                "error": f"{type(exc).__name__}: {exc}",
            })

    def stop(self) -> None:
        self.intentional_stop = True
        self.stop_requested.set()
        process = self.process
        if process is not None and process.poll() is None:
            # SIGINT enters the streaming backend's guarded shutdown, which
            # independently stops any active robot task before it exits.
            os.kill(process.pid, signal.SIGINT)

    def send(self, message: dict) -> bool:
        process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            return False
        try:
            encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            with self.write_lock:
                process.stdin.write(encoded + "\n")
                process.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False


def parse_floor_entry(text: str) -> int:
    stripped = str(text).strip()
    if not stripped:
        raise ValueError("请输入楼层或任务指令")
    if re.search(r"\d\s*[.,，。、/\\-]\s*\d", stripped):
        raise ValueError("一条指令只能包含一个目标楼层")
    if stripped.isdecimal():
        floor = int(stripped)
    else:
        command = parse_voice_command(stripped)
        if command.intent is not VoiceIntent.PRESS_BUTTON or command.button is None:
            raise ValueError("请输入 1–16 楼，例如：送外卖到15楼")
        floor = command.button
    if not 1 <= floor <= 16:
        raise ValueError("楼层必须在 1–16 之间")
    return floor


class VoiceAssistantGui:
    def __init__(
        self, root: tk.Tk, args: argparse.Namespace, *,
        container: tk.Widget | None = None, auto_start: bool = True,
    ):
        self.root = root
        self.container = container or root
        self.embedded = container is not None
        self.args = args
        # Keep the named Font objects alive for the lifetime of every widget.
        # Passing a multi-word family as a raw Tcl font string can fall back to
        # the bitmap "fixed" font even when Noto is installed.
        self.fonts = configure_crisp_ui(root)
        self.events: queue.Queue = queue.Queue()
        self.presenter = AssistantEventPresenter(args.assistant_name)
        self.floor_estimate = FloorEstimate(execute=bool(args.execute))
        self.closing = False
        self.backend_ready = False
        self.task_active = False
        self.trip_waiting_for_square = False
        self.manual_pending = False
        self.origin_busy = False
        self.voice_enabled = False
        self.status = tk.StringVar(
            value="正在启动本地流式语音…" if auto_start else "任务与语音服务尚未启动"
        )
        self.floor_value = tk.StringVar()
        self.direction_status = tk.StringVar(value="未选择上下行按钮")
        self.floor_status = tk.StringVar(
            value=self.floor_estimate.consume({})
        )
        self.test_floor_value = tk.StringVar()
        self.voice_button_text = tk.StringVar(value="开启语音")
        self.start_button_text = tk.StringVar(value="启动任务与语音服务")
        self.pending_voice_state: bool | None = None
        self.on_voice_state = None
        self.origin_status = tk.StringVar(value="尚未检查原位")
        self._build()
        log_dir = ROOT / "voice_gui_runs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = log_dir / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl"
        )
        self.backend: VoiceBackend | None = None
        if auto_start:
            self.start_backend()
        self.add_message(
            args.assistant_name,
            (
                (
                    f"先设定测试起始楼层；按手柄△开启语音，说“{args.wake_word}”后再说“去一楼”。"
                    "呼梯并回位后驾驶进入电梯，再按手柄□继续。"
                )
                if auto_start else "先启动任务与语音服务并设定测试起始楼层；"
                f"按手柄△开启语音，说“{args.wake_word}”后再说目的楼层。"
            ),
        )
        if not self.embedded:
            self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self.poll_events)

    def _backend_argv(self) -> list[str]:
        argv = [
            str(self.args.backend_python),
            str(ROOT / "run_voice_streaming.py"),
            "--source", self.args.source,
            "--model-dir", str(self.args.model_dir),
            "--vad-model", str(self.args.vad_model),
            "--vad-threshold", str(self.args.vad_threshold),
            "--config", str(self.args.config),
            "--hold-report", str(self.args.hold_report),
            "--assistant-name", self.args.assistant_name,
            "--wake-word", self.args.wake_word,
            "--wake-window-sec", str(self.args.wake_window_sec),
            "--control-stdin",
            "--start-voice-disabled",
        ]
        if self.args.execute:
            argv.extend(("--execute", "--acknowledge-physical-motion"))
        return argv

    def _build(self) -> None:
        if not self.embedded:
            self.root.title("优小妹 · 电梯任务助手")
            self.root.geometry("820x740")
            self.root.minsize(620, 480)
        header = ttk.Frame(self.container, padding=14)
        header.pack(fill="x")
        ttk.Label(header, text="优小妹", style="Title.TLabel").pack(side="left")
        mode = "机器人执行模式" if self.args.execute else "安全预览模式"
        mode_color = "#b42318" if self.args.execute else "#176b3a"
        tk.Label(
            header,
            text=mode,
            fg=mode_color,
            font=self.fonts.mode,
        ).pack(
            side="right"
        )
        ttk.Label(self.container, textvariable=self.status, padding=(14, 0, 14, 8)).pack(
            fill="x"
        )
        # Keep both interaction paths above the scrolling transcript so they
        # remain visible at every supported window size.
        workspace = ttk.Frame(self.container, padding=(14, 4, 14, 10))
        workspace.pack(fill="both", expand=True)
        task_panel = ttk.Frame(workspace, width=420)
        task_panel.pack(side="left", fill="y", padx=(0, 10))
        task_panel.pack_propagate(False)
        conversation_panel = ttk.Frame(workspace)
        conversation_panel.pack(side="left", fill="both", expand=True)

        controls = ttk.LabelFrame(task_panel, text="任务控制", padding=10)
        controls.pack(fill="x", pady=(0, 8))
        controls.columnconfigure(1, weight=1)
        self.start_backend_button = ttk.Button(
            controls, textvariable=self.start_button_text,
            command=self.start_backend,
        )
        self.start_backend_button.grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(controls, text="语音控制：").grid(
            row=0, column=1, sticky="w", padx=(10, 8), pady=(0, 8)
        )
        self.voice_button = ttk.Button(
            controls,
            textvariable=self.voice_button_text,
            command=self.toggle_voice,
            state="disabled",
        )
        self.voice_button.grid(row=0, column=2, sticky="w", pady=(0, 8))
        ttk.Label(controls, text="手动指令：").grid(
            row=1, column=0, sticky="w", padx=(0, 8)
        )
        self.floor_entry = ttk.Entry(
            controls, textvariable=self.floor_value, state="disabled"
        )
        self.floor_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8))
        self.floor_entry.bind("<Return>", lambda _event: self.execute_manual_floor())
        self.execute_button = ttk.Button(
            controls, text="执行任务", command=self.execute_manual_floor, state="disabled"
        )
        self.execute_button.grid(row=1, column=2, sticky="e")
        ttk.Label(controls, text="呼梯按钮：").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0)
        )
        direction_frame = ttk.Frame(controls)
        direction_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=(8, 0))
        self.up_button = ttk.Button(
            direction_frame, text="按上行 ↑", command=lambda: self.execute_direction("up"),
            state="disabled",
        )
        self.up_button.pack(side="left", padx=(0, 8))
        self.down_button = ttk.Button(
            direction_frame, text="按下行 ↓", command=lambda: self.execute_direction("down"),
            state="disabled",
        )
        self.down_button.pack(side="left", padx=(0, 12))
        ttk.Label(direction_frame, textvariable=self.direction_status).pack(side="left")
        ttk.Label(controls, text="测试起始楼层：").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0)
        )
        self.test_floor_entry = ttk.Entry(
            controls, textvariable=self.test_floor_value, width=8
        )
        self.test_floor_entry.grid(row=3, column=1, sticky="w", pady=(8, 0))
        self.test_floor_entry.bind(
            "<Return>", lambda _event: self.set_test_floor()
        )
        test_floor_actions = ttk.Frame(controls)
        test_floor_actions.grid(row=3, column=2, sticky="e", pady=(8, 0))
        self.set_test_floor_button = ttk.Button(
            test_floor_actions, text="设定", command=self.set_test_floor
        )
        self.set_test_floor_button.pack(side="left", padx=(0, 4))
        self.clear_test_floor_button = ttk.Button(
            test_floor_actions, text="自动", command=self.clear_test_floor
        )
        self.clear_test_floor_button.pack(side="left")
        ttk.Label(
            controls, textvariable=self.floor_status, anchor="w", wraplength=385,
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        origin = ttk.LabelFrame(task_panel, text="原位控制", padding=10)
        origin.pack(fill="x")
        origin.columnconfigure(0, weight=1)
        origin.columnconfigure(1, weight=1)
        origin.columnconfigure(2, weight=1)
        self.origin_check_button = ttk.Button(
            origin, text="检查原位", command=self.check_origin, state="disabled"
        )
        self.origin_check_button.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.origin_return_button = ttk.Button(
            origin, text="返回原位", command=self.return_origin, state="disabled"
        )
        self.origin_return_button.grid(row=0, column=1, padx=4, sticky="ew")
        self.abort_return_button = ttk.Button(
            origin,
            text="中止任务并返回",
            command=self.abort_and_return,
            state="disabled",
        )
        self.abort_return_button.grid(row=0, column=2, padx=(4, 0), sticky="ew")
        ttk.Label(
            origin, textvariable=self.origin_status, anchor="w", wraplength=390
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        body = ttk.Frame(conversation_panel)
        body.pack(fill="both", expand=True)
        self.chat = tk.Text(
            body,
            wrap="word",
            state="disabled",
            font=self.fonts.chat,
            padx=14, pady=14, spacing1=4, spacing3=10,
        )
        scrollbar = ttk.Scrollbar(body, command=self.chat.yview)
        self.chat.configure(yscrollcommand=scrollbar.set)
        self.chat.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.chat.tag_configure(
            "user",
            foreground="#234f9b",
            font=self.fonts.chat_bold,
        )
        self.chat.tag_configure(
            "assistant",
            foreground="#7a2c91",
            font=self.fonts.chat_bold,
        )
        self.chat.tag_configure(
            "text", foreground="#202124", font=self.fonts.chat
        )
        footer = ttk.Frame(task_panel, padding=(0, 8, 0, 0))
        footer.pack(fill="x", pady=(8, 0))
        ttk.Label(
            footer,
            text="本地 SenseVoiceSmall + Silero VAD；关闭窗口会进入安全停止流程。",
        ).pack(side="left")
        if not self.embedded:
            ttk.Button(footer, text="停止并退出", command=self.close).pack(side="right")

    def start_backend(self) -> None:
        if self._cycle_test_blocked():
            return
        if self.backend is not None:
            return
        self.start_backend_button.configure(state="disabled")
        self.start_button_text.set("正在启动任务与语音服务…")
        self.status.set("正在加载本地流式语音模型…")
        self.backend = VoiceBackend(
            self.events, self._backend_argv(), event_log=self.event_log
        )
        self.backend.start()

    def set_voice_enabled(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        if enabled and self._cycle_test_blocked():
            return False
        if self.backend is None:
            self.pending_voice_state = enabled
            self.start_backend()
            return True
        if not self.backend_ready:
            self.pending_voice_state = enabled
            return True
        if self.voice_enabled == enabled:
            return True
        if not self.backend.send({"command": "set_voice_enabled", "enabled": enabled}):
            return False
        return True

    def add_message(self, speaker: str, text: str) -> None:
        self.chat.configure(state="normal")
        tag = "user" if speaker == "用户" else "assistant"
        self.chat.insert("end", f"{speaker}：", tag)
        self.chat.insert("end", f"{text}\n", "text")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event.get("event")
            self.floor_status.set(self.floor_estimate.consume(event))
            if kind == "loading_model":
                self.status.set("正在加载本地流式语音模型…")
            elif kind == "model_ready":
                self.status.set("语音模型已就绪，正在启动麦克风…")
            elif kind == "ready":
                self.backend_ready = True
                self.voice_enabled = bool(event.get("voice_enabled", False))
                self.start_button_text.set("任务与语音服务已启动")
                self.start_backend_button.configure(state="disabled")
                self.status.set("语音已关闭；设定测试起始楼层后按手柄 △ 开启")
                self._refresh_controls()
                if self.pending_voice_state is not None:
                    pending = self.pending_voice_state
                    self.pending_voice_state = None
                    if pending != self.voice_enabled:
                        self.set_voice_enabled(pending)
            elif kind == "voice_state":
                self.voice_enabled = bool(event.get("enabled"))
                self.status.set("正在聆听" if self.voice_enabled else "语音已关闭")
                if self.on_voice_state is not None:
                    self.on_voice_state(self.voice_enabled)
                self._refresh_controls()
            elif kind == "partial_transcript":
                self.status.set(f"正在识别：{event.get('text', '')}")
            elif kind == "motion_started":
                self.manual_pending = False
                self.task_active = True
                self.trip_waiting_for_square = False
                if event.get("trip_stage") == "call":
                    direction = event.get("call_button")
                    direction_text = "上行" if direction == "up" else "下行"
                    self.status.set(
                        f"当前 {event.get('current_floor')} 楼，目标 "
                        f"{event.get('trip_destination')} 楼，正在按{direction_text}按钮"
                    )
                else:
                    self.status.set("任务执行中")
                self._refresh_controls()
            elif kind == "mission_progress":
                self.task_active = True
                message = str(event.get("message") or "机械臂任务执行中")
                self.status.set(message)
                statuses = event.get("mission_status_events") or []
                if statuses:
                    phase = statuses[-1].get("phase")
                    if phase != getattr(self, "last_mission_phase", None):
                        self.last_mission_phase = phase
                        if phase == "retry_required":
                            self.add_message(self.args.assistant_name, "尝试再次按压")
                        elif phase == "awaiting_operator_inside_elevator":
                            self.add_message(
                                self.args.assistant_name,
                                "呼梯按钮已完成，请驶入电梯后继续",
                            )
                        elif phase == "safety_paused":
                            evidence = statuses[-1].get("evidence") or {}
                            self.add_message(self.args.assistant_name, safety_pause_message(evidence))
                        elif phase == "safety_resumed":
                            self.add_message(self.args.assistant_name, "安全条件恢复，继续任务")
                        elif phase in {
                            "arm_button_task_completed",
                            "target_floor_button_segment_completed",
                        }:
                            flow=(statuses[-1].get('evidence') or {}).get('flow_status')
                            self.add_message(self.args.assistant_name,
                                "外呼已确认按亮，等待熄灭满30秒，按测试规则超时放行并归位；未确认熄灭"
                                if flow=='external_call_off_timeout_after_verified_press' else
                                "按钮原本已亮，已确认熄灭并归位；本次未按压"
                                if flow=='arrived_after_initial_on' else "机械臂按压、亮灭确认及归位已完成")
                self._refresh_controls()
            elif kind == "trip_waiting_for_square":
                self.task_active = False
                self.trip_waiting_for_square = True
                floor = event.get("button")
                self.status.set(
                    f"呼梯并返回已完成；驶入电梯后按 □ 继续按{floor}楼"
                )
                self._refresh_controls()
            elif kind == "trip_resume_rejected":
                self.status.set(str(event.get("message") or "当前没有可继续的乘梯任务"))
            elif kind == "current_floor_set":
                self.status.set(f"测试起始楼层已设为 {event.get('floor')} 楼")
            elif kind in ("origin_check_started", "origin_return_started"):
                self.origin_busy = True
                if kind == "origin_return_started":
                    self.trip_waiting_for_square = False
                self.origin_status.set(
                    "正在检查当前位置…"
                    if kind == "origin_check_started"
                    else "正在返回原位…"
                )
                self._refresh_controls()
            elif kind == "abort_return_started":
                self.manual_pending = False
                self.task_active = False
                self.origin_busy = True
                detail = (
                    f"；安全停止异常：{event['stop_error']}"
                    if event.get("stop_error") else ""
                )
                self.origin_status.set("正在中止任务，等待任务进程安全退出" + detail)
                self._refresh_controls()
            elif kind == "abort_completed_return_started":
                self.task_active = False
                self.trip_waiting_for_square = False
                self.origin_busy = True
                self.origin_status.set("任务已经中止，正在返回原位…")
                self._refresh_controls()
            elif kind == "abort_return_already_returning":
                self.origin_busy = True
                self.origin_status.set("机械臂已经在返回原位")
                self._refresh_controls()
            elif kind == "abort_return_preview":
                self.origin_busy = False
                self.origin_status.set("安全预览模式不会中止或驱动机械臂")
                self._refresh_controls()
            elif kind in ("origin_check_result", "origin_return_result"):
                if event.get("ok"):
                    error = event.get("max_error_rad")
                    error_text = (
                        f"，最大关节误差 {float(error):.4f} rad"
                        if error is not None else ""
                    )
                    self.origin_status.set(
                        ("已位于原位" if event.get("at_origin") else "当前不在原位")
                        + error_text
                    )
                else:
                    self.origin_status.set(
                        f"原位操作失败：{event.get('error') or '未获得机器人状态'}"
                    )
            elif kind == "origin_return_preview":
                self.origin_busy = False
                self.origin_status.set("安全预览模式不会驱动机械臂返程")
                self._refresh_controls()
            elif kind == "origin_operation_rejected":
                self.origin_busy = False
                self.origin_status.set(str(event.get("message") or "原位操作未执行"))
                self._refresh_controls()
            elif kind == "origin_operation_completed":
                self.origin_busy = False
                self.trip_waiting_for_square = False
                if int(event.get("return_code", 1)) != 0:
                    self.origin_status.set(
                        "原位检查失败" if event.get("operation") == "check"
                        else "返回原位失败"
                    )
                self._refresh_controls()
            elif kind == "motion_completed":
                self.manual_pending = False
                self.task_active = False
                self.trip_waiting_for_square = False
                self.status.set(
                    completion_message(event, uncertainty_seen=self.presenter.press_result_uncertain)
                    if int(event.get("return_code", 1)) != 0 else
                    ("正在聆听" if self.voice_enabled else "语音已关闭"))
                self._refresh_controls()
            elif kind in ("dry_run", "manual_command_rejected", "control_rejected"):
                self.manual_pending = False
                self.trip_waiting_for_square = False
                self._refresh_controls()
            elif kind == "backend_error":
                self.backend_ready = False
                self.backend = None
                self.start_backend_button.configure(state="normal")
                self.start_button_text.set("重新启动任务与语音服务")
                self.manual_pending = False
                self.trip_waiting_for_square = False
                self.origin_busy = False
                self._refresh_controls()
                self.status.set(f"语音后端失败：{event.get('error')}")
                if self.presenter.task_active:
                    self.add_message(self.args.assistant_name, "任务失败")
            elif kind == "backend_exit" and not event.get("intentional"):
                self.backend_ready = False
                self.backend = None
                self.start_backend_button.configure(state="normal")
                self.start_button_text.set("重新启动任务与语音服务")
                self.manual_pending = False
                self.origin_busy = False
                self._refresh_controls()
                self.status.set(f"语音后端已退出：{event.get('return_code')}")
                if self.presenter.task_active:
                    self.add_message(self.args.assistant_name, "任务失败")
            for message in self.presenter.consume(event):
                self.add_message(message.speaker, message.text)
        if not self.closing:
            self.root.after(50, self.poll_events)
        elif (
            self.backend is None
            or not self.backend.thread.is_alive()
            or (
                self.backend is not None
                and self.backend.process is not None
                and self.backend.process.poll() is not None
            )
        ):
            self.root.destroy()
        else:
            self.root.after(100, self.poll_events)

    def _refresh_controls(self) -> None:
        cycle_busy = bool(getattr(self, "cycle_test_busy", lambda: False)())
        manual_state = (
            "normal"
            if (
                self.backend_ready and not self.task_active
                and not self.manual_pending and not self.origin_busy
                and not self.trip_waiting_for_square
                and not cycle_busy
            )
            else "disabled"
        )
        voice_state = "normal" if self.backend_ready and not cycle_busy else "disabled"
        if self.backend_ready:
            self.start_backend_button.configure(state="disabled")
        self.floor_entry.configure(state=manual_state)
        self.execute_button.configure(state=manual_state)
        self.up_button.configure(state=manual_state)
        self.down_button.configure(state=manual_state)
        self.test_floor_entry.configure(state=manual_state)
        self.set_test_floor_button.configure(state=manual_state)
        self.clear_test_floor_button.configure(state=manual_state)
        self.voice_button.configure(state=voice_state)
        origin_state = (
            "normal"
            if self.backend_ready and not self.task_active and not self.manual_pending
            and not self.origin_busy and not self.trip_waiting_for_square
            and not cycle_busy
            else "disabled"
        )
        self.origin_check_button.configure(state=origin_state)
        self.origin_return_button.configure(
            state=origin_state if self.args.execute else "disabled"
        )
        abort_state = (
            "normal"
            if self.backend_ready and self.args.execute
            and not self.manual_pending and not self.origin_busy
            else "disabled"
        )
        self.abort_return_button.configure(state=abort_state)
        self.voice_button_text.set("关闭语音" if self.voice_enabled else "开启语音")

    def _cycle_test_blocked(self) -> bool:
        if bool(getattr(self, "cycle_test_busy", lambda: False)()):
            self.status.set("循环测试持有执行权，请使用循环测试页暂停、结束或终止并归位")
            return True
        return False

    def execute_manual_floor(self) -> None:
        if self._cycle_test_blocked():
            return
        typed_text = self.floor_value.get().strip()
        try:
            floor = parse_floor_entry(typed_text)
        except ValueError as exc:
            self.add_message(self.args.assistant_name, str(exc))
            return
        if not self.backend.send({
            "command": "manual_floor", "floor": floor, "text": typed_text
        }):
            self.add_message(self.args.assistant_name, "语音后端尚未就绪")
            return
        self.manual_pending = True
        self._refresh_controls()

    def set_test_floor(self) -> None:
        try:
            value = self.test_floor_value.get().strip()
            self.floor_status.set(self.floor_estimate.set_manual_floor(value))
            floor = int(value)
            if self.backend_ready and self.backend is not None:
                self.backend.send({"command": "set_current_floor", "floor": floor})
        except ValueError as exc:
            self.add_message(self.args.assistant_name, str(exc))

    def clear_test_floor(self) -> None:
        self.test_floor_value.set("")
        self.floor_status.set(self.floor_estimate.set_manual_floor(None))
        if self.backend_ready and self.backend is not None:
            self.backend.send({"command": "set_current_floor", "floor": None})

    def resume_trip_from_controller(self) -> bool:
        if self._cycle_test_blocked():
            return False
        if self.backend is None or not self.backend_ready:
            self.add_message(self.args.assistant_name, "语音任务服务尚未就绪，无法继续乘梯任务")
            return False
        return self.backend.send({"command": "resume_trip"})

    def execute_direction(self, target: str) -> None:
        if self._cycle_test_blocked():
            return
        target = str(target).casefold()
        if target not in {"up", "down"}:
            return
        text = "按上行按钮" if target == "up" else "按下行按钮"
        if not self.backend.send({
            "command": "manual_button", "target": target, "text": text
        }):
            self.add_message(self.args.assistant_name, "语音后端尚未就绪")
            return
        self.direction_status.set(f"已请求{text}")
        self.manual_pending = True
        self._refresh_controls()

    def toggle_voice(self) -> None:
        if not self.set_voice_enabled(not self.voice_enabled):
            self.add_message(self.args.assistant_name, "语音后端尚未就绪")
            return
        self.voice_button.configure(state="disabled")

    def check_origin(self) -> None:
        if self._cycle_test_blocked():
            return
        if not self.backend.send({"command": "origin_check"}):
            self.origin_status.set("语音后端尚未就绪")
            return
        self.origin_busy = True
        self.origin_status.set("正在检查当前位置…")
        self._refresh_controls()

    def return_origin(self) -> None:
        if self._cycle_test_blocked():
            return
        if not self.backend.send({"command": "origin_return"}):
            self.origin_status.set("语音后端尚未就绪")
            return
        self.origin_busy = True
        self.origin_status.set("正在准备返回原位…")
        self._refresh_controls()

    def abort_and_return(self) -> None:
        if bool(getattr(self, "cycle_test_busy", lambda: False)()):
            callback = getattr(self, "cycle_abort_callback", None)
            if callback is not None:
                callback()
            return
        if not self.backend.send({"command": "abort_and_return"}):
            self.origin_status.set("语音后端尚未就绪")
            return
        self.origin_busy = True
        self.origin_status.set("正在请求安全中止并返回原位…")
        self._refresh_controls()

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.status.set("正在安全停止语音与活动任务…")
        if self.backend is None:
            self.root.destroy()
            return
        self.backend.stop()
        self.root.after(50, self.poll_events)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assistant-name", default="优小妹")
    parser.add_argument("--wake-word", default="优小妹")
    parser.add_argument("--wake-window-sec", type=float, default=12.0)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--vad-model", type=Path, default=DEFAULT_VAD_MODEL)
    parser.add_argument("--vad-threshold", type=float, default=0.55)
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--hold-report", type=Path, default=DEFAULT_HOLD_REPORT)
    parser.add_argument(
        "--backend-python",
        type=Path,
        default=(
            DEFAULT_BACKEND_PYTHON
            if DEFAULT_BACKEND_PYTHON.is_file()
            else Path(sys.executable)
        ),
        help="Python interpreter used for streaming ASR and robot task processes",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-physical-motion", action="store_true")
    args = parser.parse_args()
    if args.execute != args.acknowledge_physical_motion:
        raise RuntimeError(
            "physical execution requires both --execute and --acknowledge-physical-motion"
        )
    root = tk.Tk()
    root.title("优小妹 · 电梯任务与机器人控制")
    root.geometry("1240x660")
    root.minsize(1000, 600)
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)
    task_tab = ttk.Frame(notebook)
    cycle_tab = ttk.Frame(notebook)
    controller_tab = ttk.Frame(notebook)
    notebook.add(task_tab, text="任务与语音")
    notebook.add(cycle_tab, text="循环测试")
    notebook.add(controller_tab, text="手柄与相机")
    gui = VoiceAssistantGui(root, args, container=task_tab, auto_start=False)
    from tools.dualsense_input_gui import InputViewer
    controller = InputViewer(root, container=controller_tab, embedded=True)
    from piper_handeye.button_cycle_gui import ButtonCyclePanel

    def can_start_cycle():
        if gui.task_active or gui.manual_pending or gui.origin_busy or gui.trip_waiting_for_square:
            return "任务页仍有任务或归位操作，请先结束"
        if gui.voice_enabled or gui.pending_voice_state:
            return "请先关闭语音，等待界面确认语音已关闭"
        if gui.backend is not None and not gui.backend_ready:
            return "语音后端正在启动或退出，请等待状态稳定"
        if controller.control_enabled or controller.arm_return_process is not None or controller.control_confirm_pending:
            return "请先关闭手柄控制并等待机械臂归位结束"
        if controller.voice_process is not None and controller.voice_process.poll() is None:
            return "手柄语音后端仍运行，请先关闭"
        return None

    cycle = ButtonCyclePanel(root, cycle_tab, args, VoiceBackend, can_start_cycle, gui._refresh_controls)
    gui.cycle_test_busy = lambda: cycle.busy
    gui.cycle_abort_callback = cycle.abort_return
    controller.external_task_guard = lambda: cycle.busy
    controller.voice_toggle_callback = gui.set_voice_enabled
    controller.trip_resume_callback = gui.resume_trip_from_controller
    gui.on_voice_state = lambda enabled: controller.voice_status.set(
        "语音控制：开启" if enabled else "语音控制：关闭"
    )

    def close_combined() -> None:
        if not cycle.close():
            root.after(100, close_combined)
            return
        if not controller.close(destroy_root=False):
            return
        gui.close()

    root.protocol("WM_DELETE_WINDOW", close_combined)
    signal.signal(signal.SIGINT, lambda _signum, _frame: root.after(0, close_combined))
    signal.signal(signal.SIGTERM, lambda _signum, _frame: root.after(0, close_combined))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
