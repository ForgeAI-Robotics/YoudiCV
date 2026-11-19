#!/usr/bin/env python3
"""GUI for receiving Android elevator goals and dispatching arm-only tasks."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import tkinter as tk
from tkinter import messagebox, ttk

from piper_handeye.android_agx_protocol import (
    AcceptedPacketCache,
    AndroidPacket,
    AndroidUdpListener,
)
from piper_handeye.voice_command import VoiceCommand, VoiceIntent
from run_voice_assistant_gui import configure_crisp_ui
from run_voice_command_gateway import CommandDispatcher


ROOT = Path(__file__).resolve().parent
DEFAULT_HOLD_REPORT = ROOT / "button_fast_chain_runs/20260826_150725/chain_report.yaml"
MISSION_MODE_TRIP = "trip"
MISSION_MODE_FLOOR_ONLY = "floor_only"


def dispatch_android_target(
    dispatcher: CommandDispatcher,
    target: int,
    mission_mode: str,
) -> dict:
    """Dispatch an Android floor through the existing production arm pipeline."""
    command = VoiceCommand(
        VoiceIntent.PRESS_BUTTON,
        f"Android 请求前往{target}楼",
        f"Android 请求前往{target}楼",
        button=target,
    )
    if mission_mode == MISSION_MODE_FLOOR_ONLY:
        return dispatcher.dispatch_motion(command)
    if mission_mode == MISSION_MODE_TRIP:
        return dispatcher.start_trip(command)
    raise ValueError(f"未知 Android 任务模式：{mission_mode}")


class AndroidCommandGui:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.fonts = configure_crisp_ui(root)
        self.events: queue.Queue = queue.Queue()
        self.dispatcher = CommandDispatcher(
            execute=bool(args.execute),
            config_path=args.config,
            hold_report=args.hold_report,
        )
        self.listener = AndroidUdpListener(
            self.events,
            bind_ip=args.bind_ip,
            port=args.port,
            allowed_source_ip=args.android_ip,
            journal=args.journal,
        )
        self.arm_enabled = False
        self.enable_pending = False
        self.enable_deadline_ms = 0
        self.pending_target: int | None = None
        self.pending_packet: AndroidPacket | None = None
        self.accepted_target: int | None = None
        self.accepted_packets = AcceptedPacketCache()
        self.last_source: tuple[str, int] | None = None
        self.last_sequence: int | None = None
        self.closed = False

        self.listener_state = tk.StringVar(value="正在启动 UDP 监听…")
        self.execution_state = tk.StringVar(value="机械臂执行已关闭")
        self.command_state = tk.StringVar(value="尚未收到 Android 命令")
        self.task_state = tk.StringVar(value="无任务")
        self.last_mission_phase: str | None = None
        self.current_floor = tk.StringVar(value="")
        self.target_floor = tk.StringVar(value="—")
        self.map_id = tk.StringVar(value="—")
        self.mission_mode = tk.StringVar(value=MISSION_MODE_TRIP)
        self.committed_mission_mode = MISSION_MODE_TRIP
        self._build()
        self.listener.start()
        self.root.after(100, self._poll)

    def _build(self) -> None:
        self.root.title("Android 电梯任务监视与机械臂执行")
        self.root.geometry("960x650")
        self.root.minsize(820, 560)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Android 电梯任务网关", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="仅允许机械臂闭环任务；不发送 active_result，不控制底盘、地图初始化或滑盖。",
        ).pack(anchor="w", pady=(2, 10))

        status = ttk.LabelFrame(outer, text="连接与任务状态", padding=10)
        status.pack(fill="x")
        for label, variable in (
            ("UDP 监听", self.listener_state),
            ("执行权限", self.execution_state),
            ("最新命令", self.command_state),
            ("机械臂任务", self.task_state),
        ):
            row = ttk.Frame(status)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label + "：", width=12).pack(side="left")
            ttk.Label(row, textvariable=variable).pack(side="left", fill="x", expand=True)

        values = ttk.LabelFrame(outer, text="Android 数据与分段乘梯", padding=10)
        values.pack(fill="x", pady=(10, 0))
        row = ttk.Frame(values)
        row.pack(fill="x")
        ttk.Label(row, text="当前楼层：").pack(side="left")
        ttk.Entry(row, textvariable=self.current_floor, width=6).pack(side="left")
        ttk.Button(row, text="保存当前楼层", command=self._set_current_floor).pack(side="left", padx=(6, 18))
        ttk.Label(row, text="Android 目标：").pack(side="left")
        ttk.Label(row, textvariable=self.target_floor, width=5).pack(side="left")
        ttk.Label(row, text="地图：").pack(side="left", padx=(20, 0))
        ttk.Label(row, textvariable=self.map_id, width=8).pack(side="left")

        mode_row = ttk.Frame(values)
        mode_row.pack(fill="x", pady=(9, 0))
        ttk.Label(mode_row, text="任务模式：").pack(side="left")
        ttk.Radiobutton(
            mode_row,
            text="完整分段乘梯（先按上/下，再按目标楼层）",
            value=MISSION_MODE_TRIP,
            variable=self.mission_mode,
            command=self._mission_mode_changed,
        ).pack(side="left")
        ttk.Radiobutton(
            mode_row,
            text="仅按目标楼层",
            value=MISSION_MODE_FLOOR_ONLY,
            variable=self.mission_mode,
            command=self._mission_mode_changed,
        ).pack(side="left", padx=(16, 0))

        controls = ttk.Frame(values)
        controls.pack(fill="x", pady=(10, 0))
        self.enable_button = ttk.Button(
            controls,
            text="启用 Android 机械臂任务（需按两次）",
            command=self._toggle_execution,
        )
        self.enable_button.pack(side="left")
        ttk.Button(
            controls, text="执行当前 Android 目标", command=self._execute_pending
        ).pack(side="left", padx=8)
        ttk.Button(
            controls, text="停止 Android 重发（不执行）",
            command=self._stop_android_repeat,
        ).pack(side="left", padx=8)
        ttk.Button(
            controls, text="已进入电梯，继续按目标楼层", command=self._resume_trip
        ).pack(side="left", padx=8)
        ttk.Button(
            controls, text="中止并返回原位", command=self._abort_return
        ).pack(side="right")

        log_frame = ttk.LabelFrame(outer, text="通信与执行记录", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.log = tk.Text(
            log_frame,
            wrap="word",
            font=self.fonts.body,
            state="disabled",
            background="#ffffff",
        )
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"{datetime.now():%H:%M:%S}  {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _reply_transport_ack(
        self, source: tuple[str, int] | None, *, context: str
    ) -> bool:
        if source is None:
            return False
        try:
            self.listener.reply(source)
        except Exception as exc:
            self._append(
                f"{context}，但 transport ACK 发送失败："
                f"{type(exc).__name__}: {exc}；等待 Android 重发"
            )
            return False
        return True

    def _set_current_floor(self) -> None:
        try:
            floor = int(self.current_floor.get().strip())
            self.dispatcher.set_current_floor(floor)
        except Exception as exc:
            self.task_state.set(str(exc))
            return
        self.task_state.set(f"当前楼层已设为 {floor}")
        self._append(f"当前楼层：{floor}")

    def _toggle_execution(self) -> None:
        if self.arm_enabled:
            if self.dispatcher.active is not None or self.dispatcher.trip_stage is not None:
                self.task_state.set("任务期间不能直接关闭；请先中止并返回原位")
                return
            self.arm_enabled = False
            self.enable_pending = False
            self.execution_state.set("机械臂执行已关闭")
            self.enable_button.configure(text="启用 Android 机械臂任务（需按两次）")
            self._append("机械臂执行权限已关闭")
            return
        now_ms = int(self.root.tk.call("clock", "milliseconds"))
        if not self.enable_pending or now_ms > self.enable_deadline_ms:
            self.enable_pending = True
            self.enable_deadline_ms = now_ms + 5000
            self.execution_state.set("请在 5 秒内再次确认；确认后仅机械臂可执行")
            self.enable_button.configure(text="再次确认启用机械臂任务")
            return
        self.enable_pending = False
        self.arm_enabled = True
        self.execution_state.set("机械臂任务已启用；底盘输出永久禁用")
        self.enable_button.configure(text="关闭 Android 机械臂任务")
        self._append("已双重确认启用 Android 机械臂任务")

    def _mission_mode_changed(self) -> None:
        if self.dispatcher.active is not None or self.dispatcher.trip_stage is not None:
            self.mission_mode.set(self.committed_mission_mode)
            self.task_state.set("当前任务进行中，不能切换任务模式")
            return
        self.committed_mission_mode = self.mission_mode.get()
        if self.mission_mode.get() == MISSION_MODE_FLOOR_ONLY:
            self.task_state.set("已选择仅按目标楼层；无需设置当前楼层")
            self._append("任务模式：仅按目标楼层")
        else:
            self.task_state.set("已选择完整分段乘梯；执行前需设置当前楼层")
            self._append("任务模式：完整分段乘梯")

    def _handle_android_packet(self, packet: AndroidPacket) -> None:
        self.last_source = packet.source
        self.last_sequence = packet.sequence
        self.command_state.set(
            f"seq={packet.sequence} topic={packet.topic} data={packet.raw_data}"
        )
        self._append(
            f"收到 {packet.source[0]}:{packet.source[1]} seq={packet.sequence} "
            f"{packet.topic} {packet.raw_data}"
        )
        if packet.topic != "bt_control" or not isinstance(packet.data, dict):
            self._append("未知主题或数据格式：只记录，不回包")
            return
        command = packet.data.get("command")
        value = str(packet.data.get("value", ""))
        if command == 10:
            replay_status = self.accepted_packets.classify(packet)
            if replay_status == "replay":
                if self._reply_transport_ack(
                    packet.source, context="已接受数据包重发，不重复执行"
                ):
                    self._append("已接受数据包重发：仅重发 transport ACK，不重复执行")
                return
            if replay_status == "conflict":
                self.task_state.set("拒绝协议冲突：相同序列号携带了不同任务内容")
                self._append(self.task_state.get())
                return
            try:
                floor = int(value)
                if not 1 <= floor <= 16:
                    raise ValueError
            except ValueError:
                self.task_state.set(f"拒绝无效目标楼层：{value}")
                return
            self.pending_target = floor
            self.pending_packet = packet
            self.target_floor.set(str(floor))
            if (
                self.accepted_target == floor
                and (self.dispatcher.active is not None or self.dispatcher.trip_stage is not None)
            ):
                self.accepted_packets.mark_accepted(packet)
                if self._reply_transport_ack(
                    packet.source, context="重复目标已去重，不重复执行"
                ):
                    self._append("重复目标已去重，并重发 transport ACK")
            elif self.arm_enabled and (
                self.mission_mode.get() == MISSION_MODE_FLOOR_ONLY
                or self.dispatcher.current_floor is not None
            ):
                self._execute_pending(source=packet.source, packet=packet)
            else:
                if not self.arm_enabled:
                    self.task_state.set("目标已保存；等待双重确认启用机械臂")
                else:
                    self.task_state.set("目标已保存；完整分段乘梯需先设置当前楼层")
            return
        if command == 1 and value == "arrive":
            self.task_state.set("收到 arrive；含义未确认，只记录，不继续任务")
            return
        if command == 5:
            self.map_id.set(value or "—")
            self.task_state.set(f"收到地图 {value}；只记录，不初始化底盘")
            return
        self.task_state.set(f"命令 {command} 未列入机械臂白名单；不回包")

    def _execute_pending(
        self,
        source: tuple[str, int] | None = None,
        packet: AndroidPacket | None = None,
    ) -> None:
        if not self.arm_enabled:
            self.task_state.set("请先双重确认启用机械臂任务")
            return
        if self.pending_target is None:
            self.task_state.set("尚未收到有效 Android 目标楼层")
            return
        mission_mode = self.mission_mode.get()
        if mission_mode == MISSION_MODE_TRIP and self.dispatcher.current_floor is None:
            self.task_state.set("请先设置当前楼层")
            return
        target = self.pending_target
        try:
            result = dispatch_android_target(self.dispatcher, target, mission_mode)
        except Exception as exc:
            self.task_state.set(f"任务启动失败：{type(exc).__name__}: {exc}")
            self._append(self.task_state.get())
            return
        self.accepted_target = target
        accepted_packet = packet or self.pending_packet
        if accepted_packet is not None:
            self.accepted_packets.mark_accepted(accepted_packet)
        self._reply_transport_ack(
            source or self.last_source,
            context="任务已成功调度",
        )
        if mission_mode == MISSION_MODE_FLOOR_ONLY:
            self.task_state.set(f"正在按目标楼层 {target}")
        else:
            self.task_state.set(
                f"正在呼梯：{result.get('call_button')}；完成后等待进入电梯"
            )
        self._append("任务已接受：" + json.dumps(result, ensure_ascii=False, default=str))

    def _resume_trip(self) -> None:
        if self.mission_mode.get() == MISSION_MODE_FLOOR_ONLY:
            self.task_state.set("仅按目标楼层模式没有分段继续步骤")
            return
        try:
            result = self.dispatcher.resume_trip()
            self.task_state.set(f"正在按目标楼层 {self.dispatcher.trip_destination}")
            self._append("继续任务：" + json.dumps(result, ensure_ascii=False, default=str))
        except Exception as exc:
            self.task_state.set(f"无法继续：{type(exc).__name__}: {exc}")

    def _stop_android_repeat(self) -> None:
        if self.last_source is None:
            self.task_state.set("尚未收到 Android 数据包，无法发送中性响应")
            return
        try:
            self.listener.stop_repeating(self.last_source)
            self.task_state.set("已发送中性 null 响应；Android 应停止当前重发")
            self._append(
                f"向 {self.last_source[0]}:{self.last_source[1]} 发送 null；"
                "未接受任务，未调用任何机器人接口"
            )
        except Exception as exc:
            self.task_state.set(f"中性响应失败：{type(exc).__name__}: {exc}")

    def _abort_return(self) -> None:
        try:
            result = self.dispatcher.abort_and_return()
            self.task_state.set("正在中止任务并返回原位")
            self._append("中止返程：" + json.dumps(result, ensure_ascii=False, default=str))
        except Exception as exc:
            self.task_state.set(f"中止返程失败：{type(exc).__name__}: {exc}")

    def _poll(self) -> None:
        if self.closed:
            return
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event.get("event")
            if kind == "listener_ready":
                self.listener_state.set(
                    f"监听 {event['bind']}，仅接受 {event['allowed_source_ip']}"
                )
                self._append(self.listener_state.get())
            elif kind == "android_packet":
                self._handle_android_packet(event["packet"])
            elif kind == "packet_rejected":
                self._append("拒绝数据包：" + str(event.get("reason")))
            elif kind == "listener_error":
                self.listener_state.set("监听失败：" + str(event.get("error")))
                self._append(self.listener_state.get())
        try:
            completed = self.dispatcher.refresh()
        except Exception as exc:
            completed = {"event": "refresh_failed", "error": f"{type(exc).__name__}: {exc}"}
        if completed is not None:
            self._append("任务状态：" + json.dumps(completed, ensure_ascii=False, default=str))
            event = completed.get("event")
            if event == "mission_progress":
                self.task_state.set(str(completed.get("message") or "机械臂任务执行中"))
                status_events = completed.get("mission_status_events") or []
                if status_events and status_events[-1].get("phase") == "safety_paused":
                    evidence = status_events[-1].get("evidence") or {}
                    reason = str(evidence.get("reason") or "").lower()
                    self.task_state.set(
                        "视觉变化待确认，安全暂停"
                        if "rgb" in reason or "scene_change" in reason
                        else "安全检测到障碍，任务暂停"
                    )
            elif event == "trip_waiting_for_square":
                self.task_state.set("呼梯按钮完成；请驶入电梯后点击继续")
            elif event == "motion_completed":
                if completed.get("return_code") == 0:
                    self.task_state.set("目标楼层按钮任务完成")
                else:
                    self.task_state.set(f"机械臂任务失败，退出码 {completed.get('return_code')}")
            elif event == "origin_operation_completed":
                self.task_state.set(f"返原位结束，退出码 {completed.get('return_code')}")
        self.root.after(100, self._poll)

    def _close(self) -> None:
        if self.dispatcher.active is not None:
            if not messagebox.askyesno("任务正在执行", "关闭窗口将安全停止当前机械臂任务，是否继续？"):
                return
            self.dispatcher.stop(reason="android_gateway_gui_closed")
        self.closed = True
        self.listener.stop()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-ip", default="192.168.168.162")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--android-ip", default="192.168.168.110")
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--hold-report", type=Path, default=DEFAULT_HOLD_REPORT)
    parser.add_argument(
        "--journal", type=Path,
        default=ROOT / "android_command_gui_runs/android_commands.jsonl",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = tk.Tk()
    AndroidCommandGui(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
