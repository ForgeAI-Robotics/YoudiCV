"""Compact Tk cycle-test tab; no robot access until an explicit Start/Resume."""
from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import filedialog, ttk

from .button_cycle import CycleSettings, CycleStore, VERIFICATIONS, parse_buttons

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/runs/button_cycle_tests"
POLICY_LABELS = {"pause": "故障后暂停",
                 "continue_safe": "历史策略：普通失败归位后继续",
                 "recover_wait": "无人值守：自动归位并续测"}
# Keep legacy plan values readable without offering overlapping new-group policies.
SELECTABLE_POLICIES = ("recover_wait", "pause")
DEFAULT_FAILURE_POLICY = "recover_wait"


def label_text(value):
    return {"up": "上行", "down": "下行"}.get(str(value), str(value))


def rate(value):
    return "—" if value is None else f"{100*value:.1f}%"


def number(value):
    return "—" if value is None else f"{value:.2f}"


def session_metadata(args):
    paths = [Path(args.config).resolve(), Path(args.hold_report).resolve(),
             ROOT / "run_button_light_closed_loop.py", ROOT / "run_button_batch.py",
             ROOT / "piper_handeye/button_light_hardware.py", ROOT / "piper_handeye/paired_button_light.py",
             ROOT / "piper_handeye/button_light_flow.py", ROOT / "tools/opencv_light_worker.py",
             ROOT / "piper_handeye/button_cycle.py", ROOT / "piper_handeye/button_cycle_runner.py",
             ROOT / "tools/button_cycle_worker.py", ROOT / "run_button_vision_server.py"]
    paths += [ROOT / name for name in (
        'run_online_replan_no_contact.py','piper_handeye/depth_safety.py',
        'piper_handeye/numeric_layout_handoff.py','piper_handeye/numeric_layout_sampling.py',
        'piper_handeye/retry_panel_localization.py',
        'piper_handeye/retry_observation.py',
        'piper_handeye/numeric_layout_localization.py','piper_handeye/numeric_layout_geometry.py',
        'piper_handeye/detection_enhancement.py',
        'piper_handeye/numeric_visibility.py','config/numeric_panel_layout_current.json')]
    paths += [ROOT / name for name in (
        'piper_handeye/numeric_yolo_consistency.py', 'piper_handeye/live_button.py',
        'piper_handeye/button_pose.py',
        'piper_handeye/button_background_registration.py',
        'piper_handeye/off_reference_alignment.py',
        'piper_handeye/cycle_stationarity.py', 'piper_handeye/cycle_recovery.py',
        'piper_handeye/cycle_fault_recovery.py', 'piper_handeye/near_button_depth_pause.py',
        'piper_handeye/observation_pose.py', 'piper_handeye/robot.py',
        'piper_handeye/observation_visibility.py',
        'piper_handeye/stationary_light_freshness.py',
        'piper_handeye/failure_image_archive.py',
        'piper_handeye/arm_report.py', 'piper_handeye/mission_status.py',
        'piper_handeye/exception_reporting.py',
        'piper_handeye/safety_stop.py', 'tools/orangepi_visual_safety_stop_bridge.py',
        'piper_handeye/joint_convergence.py', 'piper_handeye/guarded_press.py',
        'piper_handeye/current_baseline.py',
        'piper_handeye/contact_prevention.py', 'piper_handeye/press_numeric_trace.py',
        'tools/orangepi_guarded_press_bridge.py', 'run_direct_return_to_origin.py',
        'run_voice_command_gateway.py')]
    return dict(verification="程序视觉判定，人工核验另列；不是独立硬件验收",
                config=str(Path(args.config).resolve()), hold_report=str(Path(args.hold_report).resolve()),
                sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


class ButtonCyclePanel:
    def __init__(self, root, container, args, backend_factory, can_start, on_busy_changed):
        self.root, self.container, self.args = root, container, args
        self.backend_factory, self.can_start = backend_factory, can_start
        self.on_busy_changed = on_busy_changed
        self.events = queue.Queue()
        self.backend = None
        self.store = None
        self.rows = {}
        self.state = "idle"
        self.current = None
        self.reserved = False
        self.starting = False
        self.closing = False
        self.auto_resume = False
        self.pending_action = None
        self.group = tk.StringVar(value="内呼")
        self.buttons = tk.StringVar(value="1-5")
        self.rounds = tk.StringVar(value="1")
        self.interval = tk.StringVar(value="0")
        self.policy = tk.StringVar(value=POLICY_LABELS[DEFAULT_FAILURE_POLICY])
        self.status = tk.StringVar(value="未启动。底盘固定在对应面板前；内呼与外呼分组测试。")
        self.summary = tk.StringVar(value="尚无测试结果；首次/三次内按亮率均为程序判定，人工实物核验另列。")
        self.planned = tk.StringVar(value="预计5个按钮任务；内呼持续亮10秒后归位跳过，不记作已熄灭")
        self.directory = tk.StringVar(value="记录保存在工程 artifacts/runs/button_cycle_tests 内")
        self.verification = tk.StringVar(value="未核验")
        self._build()
        self.root.after(100, self.poll)

    @property
    def busy(self):
        return self.starting or self.reserved

    def _build(self):
        area = ttk.Frame(self.container, padding=8)
        area.pack(fill="both", expand=True)
        settings = ttk.LabelFrame(area, text="分组及顺序（每个按钮都调用正式按压闭环）", padding=6)
        settings.pack(fill="x")
        group = ttk.Combobox(settings, textvariable=self.group, values=("内呼", "外呼"), state="readonly", width=6)
        group.grid(row=0, column=0, padx=3)
        group.bind("<<ComboboxSelected>>", self.change_group)
        ttk.Label(settings, text="按钮顺序").grid(row=0, column=1)
        order = ttk.Entry(settings, textvariable=self.buttons, width=29)
        order.grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Label(settings, text="轮数").grid(row=0, column=3)
        rounds = ttk.Entry(settings, textvariable=self.rounds, width=7)
        rounds.grid(row=0, column=4, padx=4)
        ttk.Label(settings, text="任务间隔/秒").grid(row=0, column=5)
        interval = ttk.Entry(settings, textvariable=self.interval, width=6)
        interval.grid(row=0, column=6, padx=4)
        policy = ttk.Combobox(settings, textvariable=self.policy,
                values=tuple(POLICY_LABELS[key] for key in SELECTABLE_POLICIES), state="readonly", width=25)
        policy.grid(row=0, column=7, padx=4)
        settings.columnconfigure(2, weight=1)
        self.setting_widgets = (group, order, rounds, interval, policy)
        ttk.Label(settings, textvariable=self.planned).grid(row=1, column=0, columnspan=8, sticky="w", pady=(5, 0))
        for variable in (self.buttons, self.rounds, self.interval):
            variable.trace_add("write", lambda *_: self.update_plan())
        bar = ttk.Frame(area)
        bar.pack(fill="x", pady=6)
        specs = (("新建并开始", self.start), ("本项完成后暂停", self.pause),
                 ("继续剩余任务", self.resume), ("立即终止并归位", self.abort_return),
                 ("结束本组", self.finish), ("加载历史", self.load_history),
                 ("导出CSV", self.export))
        self.controls = []
        for text, action in specs:
            button = ttk.Button(bar, text=text, command=action)
            button.pack(side="left", padx=(0, 5))
            self.controls.append(button)
        ttk.Label(area, textvariable=self.status, wraplength=1170).pack(fill="x")
        ttk.Label(area, textvariable=self.summary, wraplength=1170).pack(fill="x", pady=(4, 6))
        self.progress = ttk.Progressbar(area, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 5))
        tables = ttk.Notebook(area)
        tables.pack(fill="both", expand=True)
        results, totals = ttk.Frame(tables), ttk.Frame(tables)
        tables.add(results, text="逐项记录（最近2000条；CSV保留全部）")
        tables.add(totals, text="按按钮汇总")
        columns = ("round", "button", "result", "attempts", "on_time", "wall", "manual", "reason")
        self.tree = ttk.Treeview(results, columns=columns, show="headings", height=11, selectmode="browse")
        for key, text, width in zip(columns,
                ("轮次", "按钮", "程序判定", "按压次数", "至按亮/秒", "完整/秒", "人工核验", "原因"),
                (50, 60, 175, 70, 90, 80, 90, 330)):
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, minwidth=40, stretch=key in {"result", "reason"})
        scroll = ttk.Scrollbar(results, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.totals = ttk.Treeview(totals, columns=("button", "tested", "first", "retry", "fail", "rate", "full", "manual"), show="headings", height=10)
        for key, text in zip(self.totals["columns"], ("按钮", "已测任务", "首次按亮", "重试按亮", "流程失败/不确定", "程序按亮率", "完整流程率", "人工核验数")):
            self.totals.heading(key, text=text)
            self.totals.column(key, width=120, minwidth=65)
        self.totals.pack(fill="both", expand=True)
        review = ttk.Frame(area)
        review.pack(fill="x", pady=(6, 0))
        ttk.Label(review, text="选中记录，填写实物核验：").pack(side="left")
        ttk.Combobox(review, textvariable=self.verification, values=VERIFICATIONS, state="readonly", width=10).pack(side="left")
        ttk.Button(review, text="保存核验", command=self.review).pack(side="left", padx=5)
        ttk.Label(review, text="无人值守包含普通失败及故障恢复后的归位续测；稳定0.5秒，归位失败每10秒复检。人工停止不续测。", wraplength=660).pack(side="left", padx=5)
        ttk.Label(area, textvariable=self.directory, wraplength=1170).pack(fill="x", pady=(5, 0))
        self.refresh_controls()

    def change_group(self, _event=None):
        self.buttons.set("1-5" if self.group.get() == "内呼" else "上行,下行")

    def settings(self, *, for_new_group=False):
        policy = next(key for key, label in POLICY_LABELS.items() if label == self.policy.get())
        if for_new_group and policy not in SELECTABLE_POLICIES:
            policy = DEFAULT_FAILURE_POLICY
        return CycleSettings(self.group.get(), parse_buttons(self.buttons.get(), self.group.get()),
                             int(self.rounds.get()), float(self.interval.get()),
                             policy)

    def update_plan(self):
        try:
            settings = self.settings()
            timeout_policy = "；内呼持续亮10秒后归位跳过，不记熄灭" if settings.group == "内呼" else ""
            self.planned.set(f"预计{settings.total}个按钮任务：每轮{len(settings.buttons)}项 × {settings.rounds}轮；实际按压次数另计" + timeout_policy)
        except ValueError as exc:
            self.planned.set(str(exc))

    def _allowed(self):
        if self.closing:
            raise RuntimeError("窗口正在关闭")
        if not self.args.execute:
            raise RuntimeError("当前为预览版，请使用桌面执行版控制台")
        reason = self.can_start()
        if reason:
            raise RuntimeError(reason)

    def _replace_backend(self, action):
        if self.backend is not None and self.backend.thread.is_alive():
            self.pending_action = action
            self.backend.stop()
            self.status.set("正在关闭旧的空闲测试后端…")
        else:
            self.backend = None
            action()

    def start(self):
        try:
            if self.busy:
                raise RuntimeError("请先暂停并结束当前组，或终止并归位")
            self._allowed()
            settings = self.settings(for_new_group=True)
            metadata = session_metadata(self.args)
            metadata["numeric_cycle_on_timeout_sec"] = 10. if settings.group == "内呼" else None
            def create():
                directory = RUNS / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                self.store = CycleStore.create(directory, settings, metadata)
                self._show_store()
                self.launch(auto_resume=True)
            self._replace_backend(create)
        except Exception as exc:
            self.status.set(str(exc))

    def launch(self, *, auto_resume, initial_command=None):
        self._allowed()
        self.auto_resume = auto_resume
        self.initial_worker_command = initial_command
        self.starting = True
        argv = [str(self.args.backend_python), str(ROOT / "tools/button_cycle_worker.py"),
                "--run-dir", str(self.store.directory), "--config", str(self.args.config),
                "--hold-report", str(self.args.hold_report), "--execute", "--acknowledge-physical-motion"]
        self.backend = self.backend_factory(self.events, argv, event_log=self.store.directory / "backend_events.jsonl")
        self.backend.start()
        self.status.set("正在启动循环测试后端；尚未发送按压动作")
        self.refresh_controls()

    def send(self, command, **values):
        if self.backend is None or not self.backend.send(dict(command=command,
                issued_monotonic_sec=time.monotonic(), **values)):
            self.status.set("测试后端未就绪，请重新加载本组记录后继续")
            return False
        return True

    def pause(self):
        self.send("pause")

    def resume(self):
        try:
            self._allowed()
            pending = self.store is not None and (self.store.directory/"pending_recovery.json").exists()
            if self.store is None or self.store.cursor >= self.store.settings.total and not pending:
                raise RuntimeError("没有剩余任务")
            if self.settings() != self.store.settings:
                raise RuntimeError("继续须使用本组原设置；修改按钮顺序或轮数请新建测试组")
            if self.backend is None or not self.backend.thread.is_alive():
                self.launch(auto_resume=True)
            else:
                self.starting = True
                if not self.send("resume"):
                    self.starting = False
                self.refresh_controls()
        except Exception as exc:
            self.starting = False
            self.status.set(str(exc))

    def abort_return(self):
        if self.backend is None or not self.backend.thread.is_alive():
            if self.store is None:
                self.status.set("尚无测试组，请使用任务页机械臂归位")
                return
            try:
                self.launch(auto_resume=False, initial_command="abort_return")
            except Exception as exc:
                self.starting = False
                self.status.set("归位后端启动失败：" + str(exc))
            return
        self.send("abort_return")

    def finish(self):
        self.send("finish")

    def load_history(self):
        if self.busy:
            self.status.set("请先结束当前组或终止并归位，再加载历史")
            return
        path = filedialog.askopenfilename(parent=self.root, title="选择测试组的plan.json",
                    initialdir=RUNS, filetypes=(("测试计划", "plan.json"),))
        if not path:
            return
        def load():
            try:
                self.store = CycleStore(Path(path).parent)
                self.state = "paused"
                self._show_store()
                self.status.set("历史记录已加载，只读查看；明确点击继续才会重新检查并启动剩余任务")
                self.refresh_controls()
            except Exception as exc:
                self.status.set("加载失败：" + str(exc))
        self._replace_backend(load)

    def _show_store(self):
        self.rows.clear()
        self.tree.delete(*self.tree.get_children())
        for row in self.store.rows[-2000:]:
            self.show_row(row)
        settings = self.store.settings
        self.group.set(settings.group)
        self.buttons.set(",".join(label_text(label) for label in settings.buttons))
        self.rounds.set(str(settings.rounds))
        self.interval.set(str(settings.interval_sec))
        self.policy.set(POLICY_LABELS[settings.failure_policy])
        self.directory.set("记录目录：" + str(self.store.directory))
        from .button_cycle import summarize
        self.show_summary(dict(total=settings.total, overall=summarize(self.store.rows),
                  per_button={label: summarize([r for r in self.store.rows if r["button"] == label])
                              for label in dict.fromkeys(settings.buttons)}))

    def show_row(self, row):
        key = row["execution_id"]
        self.rows[key] = row
        values = (row["round"], label_text(row["button"]), row["result"],
                  "未知" if row.get("press_attempts") is None else row["press_attempts"],
                  number(row.get("time_to_on_sec")), number(row.get("wall_sec")),
                  row.get("manual_verification", "未核验"), row.get("reason", ""))
        if self.tree.exists(key):
            self.tree.item(key, values=values)
        else:
            self.tree.insert("", "end", iid=key, values=values)
        children = self.tree.get_children()
        if len(children) > 2000:
            old = children[0]
            self.tree.delete(old)
            self.rows.pop(old, None)
        self.tree.see(key)

    def show_summary(self, summary):
        overall = summary["overall"]
        self.progress.configure(maximum=summary["total"], value=overall["completed"])
        self.summary.set(
            f"已测 {overall['completed']}/{summary['total']}项｜首次按亮 {overall['first_success']}｜"
            f"重试按亮 {overall['retry_success']}｜流程失败/不确定 {overall['failed_or_uncertain']}｜"
            f"程序按亮率 {rate(overall['lit_success_rate'])}｜完整流程率 {rate(overall['full_cycle_success_rate'])}\n"
            f"已知按压命令 {overall['actual_press_commands']}次，另有{overall['unknown_press_counts']}项次数未知｜"
            f"持续亮超时跳过 {overall.get('on_timeout_skips',0)}项（其中初始已亮 {overall.get('initial_on_timeout_skips',0)}项）｜"
            f"采用OpenCV后备 {overall.get('opencv_fallback_items',0)}项｜"
            f"初始已亮等待 {overall['already_on_waits']}项、中止 {overall['interrupted']}项均未计入成功率｜"
            f"人工已核验 {overall['manually_verified']}项｜平均至按亮 {number(overall['mean_time_to_on_sec'])}秒，完整 {number(overall['mean_wall_sec'])}秒"
        )
        self.totals.delete(*self.totals.get_children())
        for button, values in summary["per_button"].items():
            self.totals.insert("", "end", values=(label_text(button), values["completed"],
                values["first_success"], values["retry_success"], values["failed_or_uncertain"],
                rate(values["lit_success_rate"]), rate(values["full_cycle_success_rate"]), values["manually_verified"]))

    def review(self):
        selected = self.tree.selection()
        if not selected:
            self.status.set("请先选中一条记录")
            return
        try:
            if self.backend is not None and self.backend.thread.is_alive():
                self.send("review", execution_id=selected[0], value=self.verification.get())
            elif self.store is not None:
                self.store = CycleStore(self.store.directory)
                self.store.review(selected[0], self.verification.get())
                self._show_store()
        except Exception as exc:
            self.status.set("核验保存失败：" + str(exc))

    def export(self):
        if self.store is not None:
            self.status.set("CSV已随结果自动保存，Windows Excel可直接打开：" + str(self.store.directory / "results.csv"))
        else:
            self.status.set("请先新建测试或加载历史记录")

    def refresh_controls(self):
        busy = self.busy
        for widget in self.setting_widgets:
            widget.configure(state="disabled" if busy else "readonly" if isinstance(widget, ttk.Combobox) else "normal")
        start, pause, resume, abort, finish, load, export = self.controls
        start.configure(state="normal" if not busy and self.args.execute and not self.closing else "disabled")
        pause.configure(state="normal" if self.state in {"running", "checking", "waiting_next", "recovery_checking", "fault_wait", "fault_stabilizing", "fault_planning", "fault_returning"} else "disabled")
        resume.configure(state="normal" if self.store is not None and self.state in {"paused", "ended"}
                         and (self.store.cursor < self.store.settings.total or (self.store.directory/"pending_recovery.json").exists()) and not self.closing else "disabled")
        abort.configure(state="normal" if not self.closing and (self.backend is not None and self.backend.thread.is_alive() or self.store is not None and self.args.execute) else "disabled")
        finish.configure(state="normal" if self.reserved and self.state == "paused" else "disabled")
        load.configure(state="disabled" if busy or self.closing else "normal")
        export.configure(state="normal" if self.store is not None else "disabled")
        self.on_busy_changed()

    def poll(self):
        if self.backend is not None and self.backend.thread.is_alive() and not self.closing:
            self.backend.send(dict(command="heartbeat", issued_monotonic_sec=time.monotonic()))
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event.get("event")
            if event.get("channel") == "button_cycle_worker":
                if kind == "cycle_ready":
                    for row in event.get("rows", [])[-2000:]:
                        self.show_row(row)
                    pending = getattr(self, "initial_worker_command", None)
                    if pending and not self.closing:
                        self.initial_worker_command = None
                        self.send(pending)
                    if self.auto_resume and not self.closing:
                        self.auto_resume = False
                        self.send("resume")
                elif kind == "cycle_state":
                    self.state, self.reserved = event["state"], event["reserved"]
                    if self.state != "paused" or self.reserved:
                        self.starting = False
                    if self.store is not None:
                        self.store.cursor = event["summary"]["next_index"]
                    self.show_summary(event["summary"])
                    current = event.get("current")
                    self.current = current
                    prefix = (f"第{current['round']}轮，按钮{label_text(current['button'])}：" if current else "")
                    self.status.set(prefix + event["reason"])
                elif kind == "cycle_progress":
                    current = getattr(self, "current", None)
                    prefix = f"第{current['round']}轮，按钮{label_text(current['button'])}：" if current else ""
                    self.status.set(prefix + str(event.get("message") or "正式按压闭环执行中"))
                elif kind == "cycle_result":
                    self.show_row(event["row"])
                elif kind == "cycle_error":
                    self.starting = False
                    self.status.set("循环测试：" + event["message"])
            elif kind in {"backend_exit", "backend_error"}:
                self.backend = None
                self.reserved = self.starting = False
                if self.state not in {"completed", "ended", "closed"}:
                    self.state = "paused"
                    self.status.set("测试后端已退出，未自动恢复；记录已保留，继续前重新核验")
                action, self.pending_action = self.pending_action, None
                if action is not None and not self.closing:
                    try:
                        action()
                    except Exception as exc:
                        self.status.set(str(exc))
            self.refresh_controls()
        if not self.closing:
            self.root.after(150, self.poll)

    def close(self):
        already_closing = self.closing
        self.closing = True
        self.pending_action = None
        if self.backend is not None and self.backend.thread.is_alive():
            if not already_closing:
                self.backend.stop()
            self.status.set("正在安全停止循环任务；等待后端退出后关闭窗口")
            return False
        return True
