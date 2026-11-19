#!/usr/bin/env python3
"""DualSense input tester with an explicit two-click chassis control gate."""

from __future__ import annotations

import glob
import base64
import io
import json
import os
import queue
import select
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import replace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import font as tkfont, ttk
from typing import Any
import yaml

from evdev import InputDevice, ecodes, list_devices

from piper_handeye.dualsense_evdev import DualSenseEvdevAdapter
from piper_handeye.arm_gamepad_udp import ArmGamepadUdpClient
from piper_handeye.gamepad_teleop import (
    ChassisMotionLimiter, DualModeGamepad, GamepadSample, TeleopMode,
)
from piper_handeye.ros1_topic_monitor import (
    Ros1TopicMonitor,
    decode_lr_speed,
    decode_odometry,
    decode_twist_stamped,
)
from piper_handeye.retention import prune_old_entries
from piper_handeye.rospy_twist import RospyTwistStampedPublisher
from piper_handeye.smoother_policy import VelocitySmootherRuntimeLease
from piper_handeye.vision_rpc import VisionRpcClient


CHASSIS_MASTER_URI = "http://192.168.168.100:11311"
JETSON_CHASSIS_IP = "192.168.168.130"
CHASSIS_COMMAND_TOPIC = "/gamepad/plt_cmd_vel"
CONTROL_CONFIRM_TIMEOUT_SEC = 10.0
# The Jetson-side subscriber can complete its TCPROS handshake several
# seconds after the lease service responds, especially after arm/chassis
# switching.  This is still a bounded safety gate; no motion is enabled
# unless the actual subscriber connection is observed.
CHASSIS_SUBSCRIBER_HANDSHAKE_TIMEOUT_SEC = 10.0
ACTION_CONFIRM_TIMEOUT_SEC = 1.5
CONTROL_CHORD_TRIGGER_THRESHOLD = 0.80
ORIGIN_RETURN_HOLD_SEC = 3.0
JOURNAL_PERIOD_SEC = 0.25  # Analogue diagnostic snapshots only; edge events stay immediate.
CONTROL_UPDATE_MS = 20
MAX_HIDRAW_REPORTS_PER_TICK = 8
MAX_AUX_EVENTS_PER_TICK = 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARM_BACKEND_PYTHON = Path("/home/qianjue/anaconda3/envs/piper-handeye/bin/python")
DEFAULT_HOLD_REPORT = (
    PROJECT_ROOT / "button_fast_chain_runs/20260826_150725/chain_report.yaml"
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


def configure_crisp_ui(root: tk.Tk) -> UiFonts:
    """Use the same DPI and vector-font setup as the voice-control GUI."""
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
        raise RuntimeError("当前 Tk 找不到可用的中文矢量字体")
    for named in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(named, root=root).configure(family=family, size=11)
    tkfont.nametofont("TkFixedFont", root=root).configure(family=family, size=11)

    fonts = UiFonts(
        family=family,
        body=tkfont.Font(root=root, family=family, size=11),
        title=tkfont.Font(root=root, family=family, size=18, weight="bold"),
    )
    style = ttk.Style(root)
    style.configure(".", font=fonts.body)
    style.configure("Title.TLabel", font=fonts.title)
    return fonts


BUTTONS = (
    (ecodes.BTN_WEST, "□ Square"),
    (ecodes.BTN_NORTH, "△ Triangle"),
    (ecodes.BTN_EAST, "○ Circle"),
    (ecodes.BTN_SOUTH, "× Cross"),
    (ecodes.BTN_TL, "L1"),
    (ecodes.BTN_TR, "R1"),
    (ecodes.BTN_TL2, "L2 按键"),
    (ecodes.BTN_TR2, "R2 按键"),
    (ecodes.BTN_THUMBL, "L3 左摇杆按压"),
    (ecodes.BTN_THUMBR, "R3 右摇杆按压"),
    (ecodes.BTN_SELECT, "Create"),
    (ecodes.BTN_START, "Options"),
    (ecodes.BTN_MODE, "PS"),
)

COMMON_BUTTON_ROLES = {
    ecodes.BTN_THUMBL: "与 R3 同按切换模式",
    ecodes.BTN_THUMBR: "与 L3 同按切换模式",
}
CHASSIS_BUTTON_ROLES = {
    ecodes.BTN_START: "未分配",
    ecodes.BTN_SOUTH: "未分配",
    ecodes.BTN_EAST: "未分配",
    ecodes.BTN_NORTH: "开启语音控制",
    ecodes.BTN_WEST: "继续分段乘梯任务",
}
ARM_BUTTON_ROLES = {
    ecodes.BTN_START: "未分配",
    ecodes.BTN_SOUTH: "夹爪开关",
    ecodes.BTN_EAST: "双击返原位",
    ecodes.BTN_NORTH: "开启/关闭语音控制",
    ecodes.BTN_TL: "J4 逆时针",
    ecodes.BTN_TR: "J4 顺时针",
}

GAME_AXES = (
    (ecodes.ABS_X, "左摇杆 X", -1.0, 1.0),
    (ecodes.ABS_Y, "左摇杆 Y", -1.0, 1.0),
    (ecodes.ABS_RX, "右摇杆 X", -1.0, 1.0),
    (ecodes.ABS_RY, "右摇杆 Y", -1.0, 1.0),
    (ecodes.ABS_Z, "L2 模拟量", 0.0, 1.0),
    (ecodes.ABS_RZ, "R2 模拟量", 0.0, 1.0),
    (ecodes.ABS_HAT0X, "方向键 X", -1.0, 1.0),
    (ecodes.ABS_HAT0Y, "方向键 Y", -1.0, 1.0),
)

MOTION_AXES = (
    (ecodes.ABS_X, "加速度 X"),
    (ecodes.ABS_Y, "加速度 Y"),
    (ecodes.ABS_Z, "加速度 Z"),
    (ecodes.ABS_RX, "陀螺仪 X"),
    (ecodes.ABS_RY, "陀螺仪 Y"),
    (ecodes.ABS_RZ, "陀螺仪 Z"),
)


class InputViewer:
    def __init__(
        self, root: tk.Tk, *, container: tk.Widget | None = None,
        embedded: bool = False,
    ):
        self.root = root
        self.container = container or root
        self.embedded = bool(embedded)
        self.closed = False
        self.voice_toggle_callback = None
        self.trip_resume_callback = None
        self.fonts = configure_crisp_ui(root)
        if not self.embedded:
            self.root.title("优小妹 · 机器人控制")
            self.root.geometry("1100x820")
            self.root.minsize(900, 680)
        self.adapter: DualSenseEvdevAdapter | None = None
        self.aux_devices: dict[str, InputDevice] = {}
        self.retry_ticks = 0
        self.status = tk.StringVar(value="等待 DualSense…")
        self.progress = tk.StringVar(value="按键测试：0 / 15")
        self.last_event = tk.StringVar(value="尚无输入")
        self.input_status = tk.StringVar(value="等待手柄按键输入")
        self.previous_button_state: dict[int, bool] = {}
        self.preview_queue: queue.Queue[tuple[bytes | None, str]] = queue.Queue(maxsize=1)
        self.preview_photo = None
        self.preview_source = None
        self.preview_render_size = (0, 0)
        self.preview_running = True
        self.previous_dpad = (0.0, 0.0)
        self.preview_status = tk.StringVar(value="摄像头连接中…")
        vision_config = yaml.safe_load(
            (PROJECT_ROOT / "piper_button_pose.yaml").read_text(encoding="utf-8")
        )["vision_service"]
        self.preview_client = VisionRpcClient(
            str(vision_config["host"]), int(vision_config["port"]),
            str(vision_config["token"]), timeout=0.7,
        )
        self.button_vars = {code: tk.StringVar() for code, _ in BUTTONS}
        self.button_labels: dict[int, ttk.Label] = {}
        self.tested_buttons: set[int | str] = set()
        self.axis_vars: dict[int, tk.DoubleVar] = {}
        self.axis_text: dict[int, tk.StringVar] = {}
        self.motion_text = {
            code: tk.StringVar(value="0") for code, _ in MOTION_AXES
        }
        self.touch_click = tk.StringVar(value="触摸板按压：未按下")
        self.touch_contact = tk.StringVar(value="触摸：无")
        self.touch_position = tk.StringVar(value="坐标：—")
        self.touch_pressed = False
        self.touching = False
        self.touch_x: int | None = None
        self.touch_y: int | None = None
        self.mic_pressed = False
        self.mic_button = tk.StringVar(value="麦克风静音键：未按下")
        self.hidraw_fd: int | None = None
        self.latest_sample: GamepadSample | None = None
        self.control_mapper = DualModeGamepad()
        self.chassis_limiter = ChassisMotionLimiter()
        self.arm_mapper = DualModeGamepad(initial_mode=TeleopMode.ARM)
        self.control_publisher: RospyTwistStampedPublisher | None = None
        self.smoother_policy: VelocitySmootherRuntimeLease | None = None
        self.arm_client: ArmGamepadUdpClient | None = None
        self.last_arm_bridge_status: dict[str, Any] = {}
        self.control_kind: str | None = None
        self.control_enabled = False
        self.control_confirm_pending = False
        self.control_confirm_deadline = 0.0
        self.control_status = tk.StringVar(value="控制已关闭")
        self.control_command = tk.StringVar(value="线速度 0.000 m/s　转向 0.000 rad/s")
        self.ui_mode = "chassis"
        self.last_mode_chord_pressed = False
        self.mode_chord_control_seen = False
        self.last_control_chord_pressed = False
        self.control_chord_seen = False
        self.control_chord_first_at: float | None = None
        self.control_chord_enable_armed = False
        self.control_chord_enable_deadline = 0.0
        self.last_circle_pressed = False
        self.mode_switch_first_at: float | None = None
        self.circle_pressed_at: float | None = None
        self.circle_first_click_at: float | None = None
        self.circle_hold_fired = False
        self.arm_return_process: subprocess.Popen | None = None
        self.arm_return_log = None
        self.return_target_mode = "chassis"
        self.return_reason = ""
        self.return_resume_control = False
        self.voice_process: subprocess.Popen | None = None
        self.voice_log = None
        self.voice_status = tk.StringVar(value="语音控制：关闭")
        self.last_triangle_pressed = False
        self.last_square_pressed = False
        config = yaml.safe_load((PROJECT_ROOT / "gamepad_teleop.yaml").read_text())
        self.arm_output_config = dict(config["reserved_ports"]["arm_output"])
        self.mode_status = tk.StringVar(value="当前模式：底盘模式")
        self.mode_assignment = tk.StringVar(
            value="L3+R3 双击：切换模式并先返原位　"
            "左摇杆：转向　R2：前进/加速　L2：倒车"
        )
        self.alignment_status = tk.StringVar(
            value="左摇杆：回中　右摇杆：回中　L2：0%　R2：0%"
        )
        run_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.journal_dir = PROJECT_ROOT / "gamepad_control_runs" / run_stamp
        self.journal_dir.mkdir(parents=True, exist_ok=False)
        prune_old_entries(self.journal_dir.parent)
        self.journal_path = self.journal_dir / "control_journal.jsonl"
        self.journal_file = self.journal_path.open("a", encoding="utf-8", buffering=1)
        self.journal_display = tk.StringVar(value="运行日志：已记录")
        self.last_journal_at = 0.0
        self.last_published_linear = 0.0
        self.last_published_angular = 0.0
        self.last_publish_at: float | None = None
        self.last_publish_gap_sec = 0.0
        self.smoothed_monitor = Ros1TopicMonitor(
            master_uri=CHASSIS_MASTER_URI,
            topic="/plt_cmd_vel",
            message_type="geometry_msgs/TwistStamped",
            decoder=decode_twist_stamped,
            preferred_publisher_node="/gamepad_smoother_runtime_manager",
        )
        self.odom_monitor = Ros1TopicMonitor(
            master_uri=CHASSIS_MASTER_URI,
            topic="/odom",
            message_type="nav_msgs/Odometry",
            decoder=decode_odometry,
        )
        self.motor_speed_monitor = Ros1TopicMonitor(
            master_uri=CHASSIS_MASTER_URI,
            topic="/RosAria/lr_speed",
            message_type="custom_msgs_srvs/LRSpeed",
            decoder=decode_lr_speed,
            preferred_publisher_node="/chassis_server_node",
        )
        self._journal("session_start", gui_pid=os.getpid())
        self._build()
        if not self.embedded:
            self.root.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self._preview_worker, daemon=True).start()
        self.root.after(100, self._show_preview)
        self.root.after(20, self.update)

    def _build(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Released.TLabel", padding=7, anchor="center")
        style.configure(
            "Pressed.TLabel", padding=7, anchor="center",
            foreground="#ffffff", background="#167d32",
        )
        style.configure(
            "Tested.TLabel", padding=7, anchor="center", foreground="#167d32",
        )
        style.configure(
            "Mode.TLabel", padding=6, anchor="center",
            foreground="#ffffff", background="#176b3a",
            font=(self.fonts.family, 13, "bold"),
        )

        workspace = ttk.Frame(self.container, padding=(10, 8, 10, 10))
        workspace.pack(fill="both", expand=True)
        sidebar = ttk.Frame(workspace, width=360)
        sidebar.pack(side="left", fill="y", padx=(0, 10))
        sidebar.pack_propagate(False)
        camera_column = ttk.Frame(workspace)
        camera_column.pack(side="left", fill="both", expand=True)

        ttk.Label(sidebar, text="手柄与底盘 / 机械臂控制", style="Title.TLabel").pack(
            anchor="w", pady=(0, 5)
        )
        ttk.Label(sidebar, textvariable=self.status, wraplength=340, justify="left").pack(
            fill="x", pady=(0, 7)
        )

        mode = ttk.LabelFrame(sidebar, text="模式与按键分配", padding=6)
        mode.pack(fill="x", pady=(0, 7))
        ttk.Label(
            mode, textvariable=self.mode_status, style="Mode.TLabel", width=22
        ).pack(fill="x", padx=2, pady=(0, 5))
        ttk.Label(
            mode, textvariable=self.mode_assignment, wraplength=330, justify="left"
        ).pack(fill="x", padx=2, pady=(0, 2))

        control = ttk.LabelFrame(sidebar, text="手柄控制输出", padding=7)
        control.pack(fill="x", pady=(0, 7))
        self.control_button = ttk.Button(
            control,
            text="开启手柄控制（需按两次）",
            command=self._toggle_control,
        )
        self.control_button.pack(fill="x", pady=(0, 5))
        ttk.Label(control, textvariable=self.control_status, wraplength=330).pack(
            anchor="w", pady=2
        )
        ttk.Label(control, textvariable=self.control_command, wraplength=330).pack(
            anchor="w", pady=2
        )
        ttk.Label(control, textvariable=self.voice_status).pack(anchor="w", pady=2)
        ttk.Label(
            sidebar, textvariable=self.journal_display, foreground="#555555",
            wraplength=340, justify="left",
        ).pack(fill="x", pady=(0, 7))

        feedback = ttk.LabelFrame(sidebar, text="手柄输入", padding=7)
        feedback.pack(fill="x")
        ttk.Label(feedback, textvariable=self.input_status, foreground="#167d32").pack(
            anchor="w", pady=(0, 3)
        )
        ttk.Label(
            feedback, textvariable=self.last_event, wraplength=330, justify="left"
        ).pack(
            anchor="w"
        )

        camera = ttk.LabelFrame(camera_column, text="实时摄像头画面", padding=5)
        camera.pack(fill="both", expand=True)
        ttk.Label(camera, textvariable=self.preview_status, anchor="w").pack(
            fill="x", padx=4, pady=(0, 4)
        )
        self.preview_label = ttk.Label(camera, text="正在连接摄像头…", anchor="center")
        self.preview_label.pack(fill="both", expand=True, padx=2, pady=2)
        self.preview_label.bind("<Configure>", self._render_preview)

    def _preview_worker(self) -> None:
        while self.preview_running:
            try:
                response = self.preview_client.request("preview")
                payload = base64.b64decode(response["jpeg_base64"])
                item = (payload, f"摄像头在线 · 帧 {response['frame_sequence']}")
            except Exception as exc:
                item = (None, f"摄像头连接中：{type(exc).__name__}: {exc}")
            try:
                self.preview_queue.put_nowait(item)
            except queue.Full:
                try:
                    self.preview_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.preview_queue.put_nowait(item)
                except queue.Full:
                    pass
            time.sleep(0.20)

    def _show_preview(self) -> None:
        try:
            payload, status = self.preview_queue.get_nowait()
        except queue.Empty:
            payload = None
            status = None
        if status is not None:
            self.preview_status.set(status)
            if payload:
                try:
                    from PIL import Image, ImageTk
                    self.preview_source = Image.open(io.BytesIO(payload)).convert("RGB")
                    self.preview_render_size = (0, 0)
                    self._render_preview()
                except Exception as exc:
                    self.preview_label.configure(
                        text=f"预览图像显示失败：{type(exc).__name__}: {exc}", image=""
                    )
            else:
                self.preview_label.configure(text="等待摄像头画面…", image="")
        if self.preview_running:
            self.root.after(100, self._show_preview)

    def _render_preview(self, _event: Any = None) -> None:
        if self.preview_source is None:
            return
        width = max(1, self.preview_label.winfo_width() - 4)
        height = max(1, self.preview_label.winfo_height() - 4)
        source_width, source_height = self.preview_source.size
        scale = min(width / source_width, height / source_height)
        size = (
            max(1, int(source_width * scale)),
            max(1, int(source_height * scale)),
        )
        if size == self.preview_render_size:
            return
        from PIL import Image, ImageTk
        frame = self.preview_source.resize(size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(frame)
        self.preview_render_size = size
        self.preview_label.configure(image=self.preview_photo, text="")

    @staticmethod
    def _find_aux_devices(unique_id: str) -> dict[str, InputDevice]:
        devices: dict[str, InputDevice] = {}
        for path in list_devices():
            candidate = InputDevice(path)
            name = str(candidate.name or "")
            if str(candidate.uniq or "") != unique_id:
                candidate.close()
                continue
            if name.endswith("Touchpad"):
                devices["touchpad"] = candidate
            elif name.endswith("Motion Sensors"):
                devices["motion"] = candidate
            else:
                candidate.close()
        return devices

    def _connect(self) -> None:
        try:
            self.adapter = DualSenseEvdevAdapter("auto")
            identity = self.adapter.identity
            self.aux_devices = self._find_aux_devices(str(identity["uniq"]))
            self.hidraw_fd = self._open_hidraw()
            self.previous_button_state = {
                int(code): bool(value)
                for code, value in self.adapter.state.buttons.items()
            }
            aux = "、".join(sorted(self.aux_devices)) or "无附加节点"
            raw = "、raw mic" if self.hidraw_fd is not None else "、raw mic 不可用"
            self.status.set(
                f"已连接：{identity['name']}  {identity['path']}；附加：{aux}{raw}"
            )
            self._journal("controller_connected", identity=identity, auxiliary=aux)
        except FileNotFoundError:
            self.status.set("等待 DualSense；请检查蓝牙或 USB 连接")
        except PermissionError:
            self.status.set("无权读取 /dev/input；请重新登录以应用 input 组权限")
        except Exception as exc:
            self.status.set(f"输入不可用：{type(exc).__name__}: {exc}")

    @staticmethod
    def _open_hidraw() -> int | None:
        """Open Sony DualSense hidraw to expose the mic-mute button.

        hid-playstation consumes this button internally and deliberately does
        not publish it as an evdev key, so it must be inspected in the raw
        input report. The file is opened read-only and non-blocking.
        """
        for uevent in glob.glob("/sys/class/hidraw/hidraw*/device/uevent"):
            try:
                metadata = open(uevent, encoding="utf-8").read().upper()
            except OSError:
                continue
            if "DRIVER=PLAYSTATION" not in metadata:
                continue
            if "0000054C:00000CE6" not in metadata and "0000054C:00000DF2" not in metadata:
                continue
            device = "/dev/" + uevent.split("/")[-3]
            try:
                return os.open(device, os.O_RDONLY | os.O_NONBLOCK)
            except PermissionError:
                return None
            except OSError:
                continue
        return None

    def _read_hidraw_mic(self) -> None:
        if self.hidraw_fd is None:
            return
        # Bluetooth DualSense reports arrive continuously. Never try to drain
        # the entire queue in the Tk thread or chassis publishing can miss its
        # 0.2 s watchdog deadline.
        for _ in range(MAX_HIDRAW_REPORTS_PER_TICK):
            try:
                report = os.read(self.hidraw_fd, 128)
            except BlockingIOError:
                break
            if not report:
                break
            pressed = self._mic_pressed_from_report(report)
            if pressed is None:
                continue
            self.mic_pressed = pressed
            if pressed:
                self.tested_buttons.add("mic")
            tested = "（已测试）" if "mic" in self.tested_buttons else ""
            self.mic_button.set(
                f"麦克风静音键：{'按下' if pressed else '未按下'}{tested}"
            )

    @staticmethod
    def _mic_pressed_from_report(report: bytes) -> bool | None:
        # USB report 0x01: common payload begins at byte 1.
        # Bluetooth full report 0x31: common payload begins at byte 2.
        if report and report[0] == 0x31 and len(report) >= 12:
            return bool(report[11] & 0x04)
        if report and report[0] == 0x01 and len(report) >= 11:
            return bool(report[10] & 0x04)
        return None

    def _read_aux(self) -> None:
        if not self.aux_devices:
            return
        devices = list(self.aux_devices.values())
        readable, _, _ = select.select([item.fd for item in devices], [], [], 0.0)
        by_fd = {item.fd: (name, item) for name, item in self.aux_devices.items()}
        for fd in readable:
            name, device = by_fd[fd]
            try:
                events = device.read()
            except (BlockingIOError, OSError):
                continue
            for index, event in enumerate(events):
                if index >= MAX_AUX_EVENTS_PER_TICK:
                    break
                if name == "motion" and event.type == ecodes.EV_ABS:
                    variable = self.motion_text.get(event.code)
                    if variable is not None:
                        variable.set(str(event.value))
                elif name == "touchpad":
                    self._update_touch(event)

    def _update_touch(self, event: Any) -> None:
        if event.type == ecodes.EV_KEY:
            if event.code == ecodes.BTN_LEFT:
                self.touch_pressed = bool(event.value)
                if self.touch_pressed:
                    self.tested_buttons.add("touchpad")
                    self.input_status.set("输入正常")
                    self.last_event.set("已收到：触摸板按压")
            elif event.code == ecodes.BTN_TOUCH:
                self.touching = bool(event.value)
        elif event.type == ecodes.EV_ABS:
            if event.code in (ecodes.ABS_X, ecodes.ABS_MT_POSITION_X):
                self.touch_x = int(event.value)
            elif event.code in (ecodes.ABS_Y, ecodes.ABS_MT_POSITION_Y):
                self.touch_y = int(event.value)
        tested = "（已测试）" if "touchpad" in self.tested_buttons else ""
        self.touch_click.set(
            f"触摸板按压：{'按下' if self.touch_pressed else '未按下'}{tested}"
        )
        self.touch_contact.set(f"触摸：{'有手指' if self.touching else '无'}")
        if self.touch_x is not None and self.touch_y is not None:
            self.touch_position.set(f"坐标：({self.touch_x}, {self.touch_y})")

    def _refresh_buttons(self) -> None:
        if self.adapter is None:
            return
        mode_roles = (
            ARM_BUTTON_ROLES
            if self.ui_mode in {"arm", "arm_returning"}
            else CHASSIS_BUTTON_ROLES
        )
        for code, label in BUTTONS:
            pressed = bool(self.adapter.state.buttons.get(code, False))
            if pressed:
                self.tested_buttons.add(code)
            tested = code in self.tested_buttons
            marker = "●" if pressed else ("✓" if tested else "○")
            role = COMMON_BUTTON_ROLES.get(code, mode_roles.get(code))
            role_text = f" — {role}" if role else ""
            self.button_vars[code].set(f"{marker} {label}{role_text}")
            self.button_labels[code].configure(
                style="Pressed.TLabel" if pressed else (
                    "Tested.TLabel" if tested else "Released.TLabel"
                )
            )
        tested_count = len(self.tested_buttons.intersection({c for c, _ in BUTTONS}))
        tested_count += int("touchpad" in self.tested_buttons)
        tested_count += int("mic" in self.tested_buttons)
        self.progress.set(f"按键测试：{tested_count} / {len(BUTTONS) + 2}")

    def _refresh_axes(self, sample: Any) -> None:
        values = {
            ecodes.ABS_X: sample.left_x,
            ecodes.ABS_Y: sample.left_y,
            ecodes.ABS_RX: sample.right_x,
            ecodes.ABS_RY: sample.right_y,
            ecodes.ABS_Z: sample.left_trigger,
            ecodes.ABS_RZ: sample.right_trigger,
            ecodes.ABS_HAT0X: sample.dpad_x,
            ecodes.ABS_HAT0Y: sample.dpad_y,
        }
        for code, value in values.items():
            low = 0.0 if code in (ecodes.ABS_Z, ecodes.ABS_RZ) else -1.0
            self.axis_vars[code].set(float(value) - low)
            self.axis_text[code].set(f"{float(value):+.3f}")
        self.alignment_status.set(
            f"左摇杆：{self._stick_alignment(sample.left_x, sample.left_y)}　"
            f"右摇杆：{self._stick_alignment(sample.right_x, sample.right_y)}　"
            f"L2：{sample.left_trigger * 100:.0f}%　"
            f"R2：{sample.right_trigger * 100:.0f}%"
        )

    @staticmethod
    def _stick_alignment(x: float, y: float, deadzone: float = 0.10) -> str:
        horizontal = ""
        vertical = ""
        if x < -deadzone:
            horizontal = "左"
        elif x > deadzone:
            horizontal = "右"
        if y < -deadzone:
            vertical = "上"
        elif y > deadzone:
            vertical = "下"
        return horizontal + vertical if horizontal or vertical else "回中"

    def _set_mode_display(self, mode: str) -> None:
        self.ui_mode = mode
        if mode == "arm":
            self.mode_status.set("当前模式：机械臂模式")
            self.mode_assignment.set(
                "方向键左右：J1 左右　方向键上下：J2 上下　左摇杆上下：J3 上下　"
                "R1/L1：J4 顺/逆时针　右摇杆上下：J5 上下　R2/L2：J6 正/反转　"
                "×：夹爪开关　○双击：仅返原位　○长按3秒：关闭控制并返原位　"
                "△：语音开关　"
                "L3+R3 双击：切换模式并先返原位；碰撞或关节限位前自动停止"
            )
            self.control_button.configure(state="normal")
            self.control_status.set("机械臂控制已关闭；需按两次确认开启")
        elif mode == "chassis":
            self.mode_status.set("当前模式：底盘模式")
            self.mode_assignment.set(
                "L3+R3 双击：切换模式并先返原位　"
                "L3+R3+R1+L2 双击：开启控制　"
                "左摇杆：转向　R2：前进/加速　L2：倒车"
            )
            self.control_button.configure(state="normal")
            self.control_status.set("控制已关闭；需按两次确认开启")
        else:
            self.mode_status.set("当前模式：机械臂返原位中（其它功能已锁定）")
            self.mode_assignment.set("等待返原位完成；期间不接受运动、夹爪或语音命令")
            self.control_button.configure(state="disabled")

    def _start_arm_return(self, target_mode: str, reason: str) -> None:
        if self._external_task_blocked():
            return
        if self.arm_return_process is not None:
            self.control_status.set("机械臂正在返原位，请等待")
            return
        self.return_resume_control = bool(
            reason == "pure_return_keep_control" and self.control_enabled
        )
        if self.control_enabled or self.control_publisher is not None:
            # Keep the ROS publisher alive across a mode switch.  Closing and
            # immediately recreating it leaves a stale TCPROS connection on
            # the Jetson subscriber and can make the safety gate reject the
            # next chassis enable even though the manager is healthy.
            self._disable_control(
                "返原位开始，底盘已停车并关闭",
                retain_publisher=(reason == "mode_switch"),
            )
        if self.arm_client is not None:
            self._disable_control("返原位开始，机械臂伺服已停止")
        self._stop_voice_backend()
        source_mode = self.ui_mode
        self.return_target_mode = target_mode
        self.return_reason = reason
        self._set_mode_display("arm_returning")
        log_path = self.journal_dir / "arm_return.log"
        self.arm_return_log = log_path.open("ab", buffering=0)
        argv = [
            str(ARM_BACKEND_PYTHON),
            str(PROJECT_ROOT / "run_direct_return_to_origin.py"),
            "--config", str(PROJECT_ROOT / "piper_button_pose.yaml"),
            "--hold-report", str(DEFAULT_HOLD_REPORT),
            "--execute", "--acknowledge-clear-workspace",
        ]
        try:
            self.arm_return_process = subprocess.Popen(
                argv,
                cwd=PROJECT_ROOT,
                stdout=self.arm_return_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            self.arm_return_log.close()
            self.arm_return_log = None
            self._set_mode_display(source_mode)
            raise
        self.control_status.set("正在返原位；其它功能已关闭")
        self._journal(
            "arm_return_started", reason=reason, target_mode=target_mode,
            pid=self.arm_return_process.pid, argv=argv,
        )

    def _poll_arm_return(self) -> None:
        process = self.arm_return_process
        if process is None:
            return
        return_code = process.poll()
        if return_code is None:
            return
        self.arm_return_process = None
        if self.arm_return_log is not None:
            self.arm_return_log.close()
            self.arm_return_log = None
        if return_code == 0:
            target = self.return_target_mode
            reason = self.return_reason
            self._set_mode_display(target)
            self._journal(
                "arm_return_completed", target_mode=target, reason=reason
            )
            auto_enable = reason == "mode_switch" or (
                reason == "pure_return_keep_control"
                and self.return_resume_control
            )
            if auto_enable:
                # The released double L3+R3 chord is already an explicit
                # confirmation. Do not require a second, unrelated double click
                # in the GUI after the collision-checked return completes.
                self.control_confirm_pending = True
                self.control_confirm_deadline = (
                    time.monotonic() + CONTROL_CONFIRM_TIMEOUT_SEC
                )
                self._toggle_control()
                if self.control_enabled:
                    self.control_status.set(
                        "返原位完成；机械臂关节直连控制已自动开启"
                        if target == "arm"
                        else "返原位完成；底盘控制已自动开启"
                    )
                    self._journal(
                        "return_control_auto_enabled",
                        target_mode=target,
                        reason=reason,
                    )
            else:
                self.control_status.set(
                    "返原位完成；机械臂控制保持关闭"
                    if target == "arm" else "返原位完成；底盘控制保持关闭"
                )
        else:
            self.control_status.set(
                f"返原位失败（退出码 {return_code}）；其它功能保持锁定，○长按3秒可重试"
            )
            self._journal("arm_return_failed", return_code=return_code)

    def _refresh_mode(self, sample: GamepadSample) -> None:
        """Apply double-click return and three-second close-and-return hold."""
        if self._external_task_blocked():
            if sample is not None:
                # Consume held inputs without arming a deferred action when
                # the cycle releases ownership. A new gesture is required.
                self.last_mode_chord_pressed = bool(sample.left_stick_button and sample.right_stick_button)
                self.mode_chord_control_seen = True
                self.last_control_chord_pressed = bool(sample.l1 and sample.r1
                    and sample.left_trigger >= CONTROL_CHORD_TRIGGER_THRESHOLD
                    and sample.right_trigger >= CONTROL_CHORD_TRIGGER_THRESHOLD)
                self.control_chord_seen = False
                self.control_chord_first_at = self.mode_switch_first_at = None
                self.control_chord_enable_armed = self.control_confirm_pending = False
                self.last_circle_pressed = bool(sample.b)
                self.circle_pressed_at = self.circle_first_click_at = None
                self.circle_hold_fired = bool(sample.b)
            return
        now = time.monotonic()
        chord = bool(sample.left_stick_button and sample.right_stick_button)
        chord_rising = chord and not self.last_mode_chord_pressed
        chord_falling = not chord and self.last_mode_chord_pressed
        if chord_rising:
            self.mode_chord_control_seen = False
        control_chord = bool(
            sample.l1 and sample.r1
            and sample.left_trigger >= CONTROL_CHORD_TRIGGER_THRESHOLD
            and sample.right_trigger >= CONTROL_CHORD_TRIGGER_THRESHOLD
        )
        control_chord_rising = (
            control_chord and not self.last_control_chord_pressed
        )
        control_chord_falling = (
            not control_chord and self.last_control_chord_pressed
        )
        if control_chord_rising:
            self.control_chord_seen = True
        if chord and control_chord:
            self.mode_chord_control_seen = True
        control_chord_click = control_chord_falling and self.control_chord_seen
        mode_chord_click = chord_falling and not self.mode_chord_control_seen
        circle = bool(sample.b)
        circle_rising = circle and not self.last_circle_pressed

        if control_chord_click and self.ui_mode != "arm_returning":
            self.mode_switch_first_at = None
            if self.control_enabled:
                self.control_chord_first_at = None
                self.control_status.set("控制已经开启")
                self._journal("control_chord_ignored", reason="already_enabled")
            elif (
                self.control_chord_first_at is not None
                and now - self.control_chord_first_at <= ACTION_CONFIRM_TIMEOUT_SEC
            ):
                self.control_chord_first_at = None
                self.control_chord_enable_armed = True
                self.control_chord_enable_deadline = now + CONTROL_CONFIRM_TIMEOUT_SEC
                self.control_status.set("启用组合已确认；请松开全部按键和扳机")
                self._journal("control_chord_double_click_confirmed")
            else:
                self.control_chord_first_at = now
                self.control_status.set(
                    "请在 1.5 秒内再次按下并松开 L1+R1+L2+R2 确认开启"
                )
                self._journal("control_chord_confirmation_requested")

        if mode_chord_click and self.ui_mode != "arm_returning":
            if (
                self.mode_switch_first_at is not None
                and now - self.mode_switch_first_at <= ACTION_CONFIRM_TIMEOUT_SEC
            ):
                self.mode_switch_first_at = None
                target = "arm" if self.ui_mode == "chassis" else "chassis"
                try:
                    self._start_arm_return(target, "mode_switch")
                except Exception as exc:
                    self.control_status.set(f"返原位启动失败：{type(exc).__name__}: {exc}")
                    self._journal("arm_return_start_failed", error=str(exc))
            else:
                self.mode_switch_first_at = now
                self.control_status.set("请在 1.5 秒内再次同时按下 L3+R3 确认切换")
                self._journal("mode_switch_confirmation_requested")

        if self.control_chord_enable_armed:
            neutral = (
                sample.left_trigger <= 0.10
                and sample.right_trigger <= 0.10
                and abs(sample.left_x) <= 0.10
                and abs(sample.left_y) <= 0.10
                and abs(sample.right_x) <= 0.10
                and abs(sample.right_y) <= 0.10
                and abs(sample.dpad_x) <= 0.10
                and abs(sample.dpad_y) <= 0.10
                and not sample.l1 and not sample.r1
                and not sample.left_stick_button
                and not sample.right_stick_button
            )
            if now > self.control_chord_enable_deadline:
                self.control_chord_enable_armed = False
                self.control_status.set("启用组合超时；控制保持关闭")
                self._journal("control_chord_enable_expired")
            elif neutral:
                self.control_chord_enable_armed = False
                self.control_confirm_pending = True
                self.control_confirm_deadline = now + CONTROL_CONFIRM_TIMEOUT_SEC
                self._journal("control_chord_neutral_confirmed")
                self._toggle_control()

        if circle_rising:
            self.circle_pressed_at = now
            self.circle_hold_fired = False
            self.control_status.set(
                "松开完成点击；持续按住 ○ 3 秒将关闭控制并返原位"
            )
            self._journal("origin_return_hold_started")
        if (
            circle
            and not self.circle_hold_fired
            and self.circle_pressed_at is not None
        ):
            held = now - self.circle_pressed_at
            if held >= ORIGIN_RETURN_HOLD_SEC:
                self.circle_hold_fired = True
                self.circle_first_click_at = None
                target = (
                    self.return_target_mode
                    if self.ui_mode == "arm_returning" else self.ui_mode
                )
                self._journal("origin_return_hold_confirmed", held_sec=held)
                try:
                    self._start_arm_return(target, "pure_return_close")
                except Exception as exc:
                    self.control_status.set(f"返原位启动失败：{type(exc).__name__}: {exc}")
                    self._journal("arm_return_start_failed", error=str(exc))
        if not circle and self.last_circle_pressed:
            if not self.circle_hold_fired:
                held = (
                    0.0 if self.circle_pressed_at is None
                    else now - self.circle_pressed_at
                )
                if (
                    self.circle_first_click_at is not None
                    and now - self.circle_first_click_at
                    <= ACTION_CONFIRM_TIMEOUT_SEC
                ):
                    self.circle_first_click_at = None
                    target = self.ui_mode
                    self._journal(
                        "origin_return_double_click_confirmed", held_sec=held
                    )
                    try:
                        self._start_arm_return(
                            target, "pure_return_keep_control"
                        )
                    except Exception as exc:
                        self.control_status.set(
                            f"返原位启动失败：{type(exc).__name__}: {exc}"
                        )
                        self._journal(
                            "arm_return_start_failed", error=str(exc)
                        )
                else:
                    self.circle_first_click_at = now
                    self.control_status.set(
                        "请在 1.5 秒内再次短按 ○；长按3秒则关闭控制并返原位"
                    )
                    self._journal("origin_return_first_click", held_sec=held)
            self.circle_pressed_at = None
            self.circle_hold_fired = False

        if (
            not circle
            and self.circle_first_click_at is not None
            and now - self.circle_first_click_at > ACTION_CONFIRM_TIMEOUT_SEC
        ):
            self.circle_first_click_at = None
            if self.control_enabled:
                self.control_status.set(
                    "机械臂关节直连控制已开启（MoveIt 碰撞保护）"
                    if self.control_kind == "arm"
                    else "控制已开启，等待底盘指令"
                )
            else:
                self.control_status.set("控制已关闭")
            self._journal("origin_return_double_click_expired")

        self.last_mode_chord_pressed = chord
        self.last_control_chord_pressed = control_chord
        if control_chord_falling:
            self.control_chord_seen = False
        self.last_circle_pressed = circle

    def _clear_tested(self) -> None:
        self.tested_buttons.clear()
        self.touch_click.set("触摸板按压：未按下")
        self.mic_button.set("麦克风静音键：未按下")

    def _external_task_blocked(self) -> bool:
        if bool(getattr(self, "external_task_guard", lambda: False)()):
            self.control_status.set("循环测试持有执行权，手柄控制/模式切换已锁定")
            return True
        return False

    def _toggle_control(self) -> None:
        if self._external_task_blocked():
            return
        if self.ui_mode == "arm_returning":
            self.control_status.set("机械臂正在返原位，暂不能开启控制")
            self._journal("enable_rejected", reason="arm_returning")
            return
        if self.control_enabled:
            self._disable_control("控制已关闭（已发送停车命令）")
            return
        now = time.monotonic()
        if (
            not self.control_confirm_pending
            or now > self.control_confirm_deadline
        ):
            self.control_confirm_pending = True
            self.control_confirm_deadline = now + CONTROL_CONFIRM_TIMEOUT_SEC
            self.control_status.set("你确认开启测试么？")
            self.control_button.configure(text="再次点击确认开启")
            self._journal("enable_confirmation_requested")
            return
        self.control_confirm_pending = False
        sample = self.latest_sample
        if sample is None or not sample.connected:
            self.control_status.set("开启失败：手柄未连接")
            self.control_button.configure(text="开启手柄控制（需按两次）")
            self._journal("enable_rejected", reason="controller_disconnected")
            return
        chassis_not_neutral = (
            sample.left_trigger > 0.10 or sample.right_trigger > 0.10
            or abs(sample.left_x) > 0.10
        )
        arm_not_neutral = (
            sample.left_trigger > 0.10 or sample.right_trigger > 0.10
            or abs(sample.left_y) > 0.10 or abs(sample.right_y) > 0.10
            or abs(sample.dpad_x) > 0.10 or abs(sample.dpad_y) > 0.10
            or sample.l1 or sample.r1
        )
        if (
            (self.ui_mode == "chassis" and chassis_not_neutral)
            or (self.ui_mode == "arm" and arm_not_neutral)
        ):
            self.control_status.set("开启失败：请松开扳机、方向键并让对应摇杆回中")
            self.control_button.configure(text="开启手柄控制（需按两次）")
            self._journal(
                "enable_rejected", reason="input_not_neutral",
                left_x=sample.left_x, left_trigger=sample.left_trigger,
                right_trigger=sample.right_trigger,
            )
            return
        refreshed = ()
        lease_details = None
        try:
            if self.ui_mode == "chassis":
                if self.control_publisher is None:
                    self.control_publisher = RospyTwistStampedPublisher(
                        master_uri=CHASSIS_MASTER_URI,
                        bind_ip=JETSON_CHASSIS_IP,
                        topic=CHASSIS_COMMAND_TOPIC,
                    )
                self.control_mapper = DualModeGamepad()
                self.chassis_limiter.reset()
                self.control_publisher.publish(0.0, 0.0)
                self.smoother_policy = VelocitySmootherRuntimeLease()
                lease_details = self.smoother_policy.acquire()
                # Keep the command stream explicitly stopped while the new
                # publisher completes its TCPROS handshake after a mode
                # switch.  The runtime lease watchdog also enforces zero.
                self.control_publisher.publish(0.0, 0.0)
                refreshed = self.control_publisher.refresh_subscriber_node(
                    "/gamepad_smoother_runtime_manager",
                    timeout_sec=CHASSIS_SUBSCRIBER_HANDSHAKE_TIMEOUT_SEC,
                )
                if "/gamepad_smoother_runtime_manager" not in refreshed:
                    raise RuntimeError(
                        "底盘安全仲裁器尚未订阅手柄速度话题（"
                        f"等待 {CHASSIS_SUBSCRIBER_HANDSHAKE_TIMEOUT_SEC:.1f} 秒后仍未建立连接）"
                    )
                self.control_kind = "chassis"
            else:
                cfg = self.arm_output_config
                self.arm_client = ArmGamepadUdpClient(
                    str(cfg["host"]), int(cfg["port"]), str(cfg["token"]),
                    timeout_sec=0.30,
                )
                self.arm_client.enable()
                self.arm_mapper = DualModeGamepad(initial_mode=TeleopMode.ARM)
                self.control_kind = "arm"
        except Exception as exc:
            if self.smoother_policy is not None:
                try:
                    self.smoother_policy.restore()
                except Exception as restore_exc:
                    self._journal(
                        "smoother_policy_restore_failed",
                        error_type=type(restore_exc).__name__,
                        error=str(restore_exc),
                    )
                self.smoother_policy = None
            if self.control_publisher is not None:
                self.control_publisher.close()
                self.control_publisher = None
            if self.arm_client is not None:
                try:
                    self.arm_client.close()
                except Exception:
                    pass
                self.arm_client = None
            self.control_kind = None
            self.control_status.set(f"开启失败：{type(exc).__name__}: {exc}")
            self.control_button.configure(text="开启手柄控制（需按两次）")
            self._journal(
                "enable_failed", error_type=type(exc).__name__, error=str(exc)
            )
            return
        self.control_enabled = True
        self.control_status.set(
            "机械臂关节直连控制已开启（MoveIt 碰撞保护）"
            if self.control_kind == "arm" else "控制已开启，等待底盘订阅"
        )
        self.control_button.configure(text="关闭手柄控制（按一次）")
        if lease_details is not None:
            self._journal("smoother_runtime_lease_acquired", manager=lease_details)
        self._journal(
            "control_enabled", control_kind=self.control_kind,
            refreshed_publishers=refreshed,
        )

    def _apply_control(self, sample: GamepadSample) -> None:
        if self._external_task_blocked():
            if sample is not None:
                self.last_triangle_pressed = bool(sample.y)
                self.last_square_pressed = bool(sample.square)
            return
        triangle_rising = bool(sample.y) and not self.last_triangle_pressed
        square_rising = bool(sample.square) and not self.last_square_pressed
        self.last_triangle_pressed = bool(sample.y)
        self.last_square_pressed = bool(sample.square)
        if self.ui_mode == "chassis" and triangle_rising:
            self._set_voice_enabled(True)
            self._journal("chassis_triangle_voice_enable_requested")
        if (
            self.ui_mode == "chassis" and square_rising
            and self.trip_resume_callback is not None
        ):
            try:
                accepted = bool(self.trip_resume_callback())
            except Exception as exc:
                accepted = False
                self._journal("trip_resume_callback_failed", error=str(exc))
            self._journal("trip_resume_requested_by_square", accepted=accepted)
        if (
            self.control_confirm_pending
            and time.monotonic() > self.control_confirm_deadline
        ):
            self.control_confirm_pending = False
            self.control_status.set("确认超时，控制未开启")
            self.control_button.configure(text="开启手柄控制（需按两次）")
        if not self.control_enabled:
            return
        if self.control_kind == "arm":
            self._apply_arm_control(sample)
            return
        if self.control_publisher is None:
            return
        # Mode switching is handled by the GUI's double-confirmed return state
        # machine. Do not also pass those buttons into the chassis mapper.
        chassis_sample = replace(
            sample,
            menu=False,
            left_stick_button=False,
            right_stick_button=False,
            a=False,
            b=False,
            y=False,
        )
        intent = self.control_mapper.update(chassis_sample)
        linear, angular = self.chassis_limiter.update(
            intent.chassis_linear_x_m_s,
            intent.chassis_angular_z_rad_s,
        )
        try:
            if self.smoother_policy is None:
                raise RuntimeError("底盘安全仲裁器租约丢失")
            self.smoother_policy.heartbeat()
            if (
                "/gamepad_smoother_runtime_manager"
                not in self.control_publisher.subscriber_names
            ):
                self.control_publisher.publish(0.0, 0.0)
                self.last_published_linear = 0.0
                self.last_published_angular = 0.0
                self.control_status.set("控制已开启，等待底盘订阅")
                self.control_command.set("线速度 0.000 m/s　转向 0.000 rad/s")
                return
            self.control_publisher.publish(linear, angular)
            publish_at = time.monotonic()
            if self.last_publish_at is not None:
                self.last_publish_gap_sec = publish_at - self.last_publish_at
            self.last_publish_at = publish_at
            self.last_published_linear = linear
            self.last_published_angular = angular
        except Exception as exc:
            self._journal(
                "publish_failed", error_type=type(exc).__name__, error=str(exc)
            )
            self._disable_control(f"控制故障并已停车：{type(exc).__name__}: {exc}")
            return
        self.control_status.set("控制已开启（左摇杆回中即直行）")
        self.control_command.set(
            f"线速度 {linear:+.3f} m/s　转向 {angular:+.3f} rad/s"
        )

    def _apply_arm_control(self, sample: GamepadSample) -> None:
        if self.arm_client is None:
            self._disable_control("机械臂控制连接丢失，输出已关闭")
            return
        arm_sample = replace(
            sample, menu=False, left_stick_button=False,
            right_stick_button=False, b=False,
        )
        intent = self.arm_mapper.update(arm_sample)
        try:
            response = self.arm_client.update_joints(
                velocities_rad_s=intent.arm_joint_velocities_rad_s,
            )
            self.last_arm_bridge_status = dict(response.get("status", {}))
            for action in intent.actions:
                if action == "toggle_clamp":
                    clamp_response = self.arm_client.toggle_clamp()
                    self.last_arm_bridge_status = dict(
                        clamp_response.get("status", {})
                    )
                    self._journal(
                        "clamp_command_sent", action=action,
                        bridge_status=self.last_arm_bridge_status,
                    )
                elif action == "enable_voice_elevator_mode":
                    self._set_voice_enabled(True)
                elif action == "disable_voice_elevator_mode":
                    self._set_voice_enabled(False)
        except Exception as exc:
            self._journal("arm_publish_failed", error=str(exc))
            self._disable_control(f"机械臂控制故障并已停止：{type(exc).__name__}: {exc}")
            return
        blocked = self.last_arm_bridge_status.get("blocked_reason")
        if blocked:
            self.control_status.set(
                "已在最后安全位置停止："
                + str(self.last_arm_bridge_status.get("blocked_detail") or blocked)
            )
        else:
            self.control_status.set("机械臂关节直连控制已开启（MoveIt 碰撞保护）")
        velocities = intent.arm_joint_velocities_rad_s
        active = [
            f"J{index + 1}{value:+.3f}"
            for index, value in enumerate(velocities) if abs(value) > 1e-6
        ]
        self.control_command.set(
            ("关节速度 " + "　".join(active) if active else "关节保持")
            + f"　上限 {intent.arm_joint_speed_rad_s:.2f} rad/s"
        )

    def _set_voice_enabled(self, enabled: bool) -> None:
        if enabled and self._external_task_blocked():
            return
        if self.voice_toggle_callback is not None:
            try:
                accepted = bool(self.voice_toggle_callback(bool(enabled)))
            except Exception as exc:
                accepted = False
                self._journal("voice_control_toggle_failed", error=str(exc))
            self.voice_status.set(
                "语音控制：未就绪" if not accepted
                else "语音控制：启动中" if enabled
                else "语音控制：关闭"
            )
            self._journal("voice_control_toggled", enabled=bool(enabled), accepted=accepted)
            return
        process = self.voice_process
        if process is not None and process.poll() is not None:
            self.voice_process = None
            process = None
        if enabled and process is None:
            log_path = self.journal_dir / "controller_voice.log"
            self.voice_log = log_path.open("ab", buffering=0)
            argv = [
                str(ARM_BACKEND_PYTHON), str(PROJECT_ROOT / "run_voice_streaming.py"),
                "--control-stdin", "--start-voice-disabled",
                "--wake-word", "优小妹",
                "--execute", "--acknowledge-physical-motion",
            ]
            process = subprocess.Popen(
                argv, cwd=PROJECT_ROOT, stdin=subprocess.PIPE,
                stdout=self.voice_log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.voice_process = process
        if process is None or process.stdin is None:
            self.voice_status.set("语音控制：关闭")
            return
        message = json.dumps(
            {"command": "set_voice_enabled", "enabled": bool(enabled)},
            ensure_ascii=False,
        ).encode("utf-8") + b"\n"
        process.stdin.write(message)
        process.stdin.flush()
        self.voice_status.set("语音控制：开启" if enabled else "语音控制：关闭")
        self._journal("voice_control_toggled", enabled=bool(enabled))

    def _stop_voice_backend(self) -> None:
        process, self.voice_process = self.voice_process, None
        self.voice_status.set("语音控制：关闭")
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if self.voice_log is not None:
            self.voice_log.close()
            self.voice_log = None

    def _disable_control(
        self, message: str, *, retain_publisher: bool = False
    ) -> None:
        publisher = self.control_publisher
        if not retain_publisher:
            self.control_publisher = None
        self.control_enabled = False
        self.control_kind = None
        self.control_confirm_pending = False
        self.control_button.configure(text="开启手柄控制（需按两次）")
        self.control_command.set("线速度 0.000 m/s　转向 0.000 rad/s")
        self.last_published_linear = 0.0
        self.last_published_angular = 0.0
        self.last_publish_at = None
        self.last_publish_gap_sec = 0.0
        self.chassis_limiter.reset()
        if publisher is not None:
            for _ in range(5):
                try:
                    publisher.publish(0.0, 0.0)
                except Exception:
                    break
                time.sleep(0.01)
            if not retain_publisher:
                publisher.close()
        arm_client, self.arm_client = self.arm_client, None
        if arm_client is not None:
            try:
                arm_client.close()
            except Exception as exc:
                self._journal("arm_disable_failed", error=str(exc))
        policy, self.smoother_policy = self.smoother_policy, None
        if policy is not None:
            try:
                restored = policy.restore()
                self._journal("smoother_runtime_lease_restored", restored=restored)
            except Exception as exc:
                self._journal(
                    "smoother_policy_restore_failed",
                    error_type=type(exc).__name__, error=str(exc),
                )
                message = (
                    f"{message}；警告：底盘安全仲裁器恢复失败："
                    f"{type(exc).__name__}: {exc}"
                )
        self.control_status.set(message)
        self._journal("control_disabled", reason=message)

    @staticmethod
    def _control_deadzone(value: float, deadzone: float = 0.10) -> float:
        value = float(value)
        magnitude = abs(value)
        if magnitude <= deadzone:
            return 0.0
        scaled = (magnitude - deadzone) / (1.0 - deadzone)
        return scaled if value > 0 else -scaled

    def _journal(self, event: str, **fields: Any) -> None:
        handle = getattr(self, "journal_file", None)
        if handle is None or handle.closed:
            return
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "monotonic_sec": round(time.monotonic(), 6),
            "event": event,
            "mode": getattr(self, "ui_mode", "chassis"),
            "control_enabled": bool(getattr(self, "control_enabled", False)),
            "confirmation_pending": bool(
                getattr(self, "control_confirm_pending", False)
            ),
            **fields,
        }
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _record_sample(self, sample: GamepadSample) -> None:
        now = time.monotonic()
        if now - self.last_journal_at < JOURNAL_PERIOD_SEC:
            return
        self.last_journal_at = now
        forward = self._control_deadzone(sample.right_trigger)
        reverse = self._control_deadzone(sample.left_trigger)
        drive = forward - reverse
        steer = self._control_deadzone(sample.left_x)
        expected_linear = drive * 0.10
        expected_angular = (
            0.0 if abs(drive) <= 1e-12
            else (-steer if drive > 0.0 else steer) * 0.40
        )
        subscribers = (
            self.control_publisher.subscriber_count
            if self.control_publisher is not None else 0
        )
        if self.control_kind == "arm" and self.arm_client is not None:
            subscribers = 1
        subscriber_names = (
            self.control_publisher.subscriber_names
            if self.control_publisher is not None else ()
        )
        smoothed = self.smoothed_monitor.snapshot()
        odometry = self.odom_monitor.snapshot()
        motor_speed = self.motor_speed_monitor.snapshot()
        arm_input_active = any(abs(value) > 0.10 for value in (
            sample.dpad_x, sample.dpad_y, sample.right_x, sample.right_y,
            sample.left_trigger, sample.right_trigger,
        ))
        input_active = arm_input_active if self.ui_mode == "arm" else abs(drive) > 1e-12
        if self.control_kind == "arm" and self.control_enabled:
            comparison = "arm_command_published" if input_active else "arm_idle_or_stop"
        elif not input_active:
            comparison = "idle_or_stop"
        elif self.ui_mode != "chassis":
            comparison = "blocked_arm_mode"
        elif not self.control_enabled:
            comparison = "blocked_control_gate"
        elif subscribers < 1:
            comparison = "blocked_no_ros_subscriber"
        elif (
            abs(expected_linear - self.last_published_linear) > 1e-6
            or abs(expected_angular - self.last_published_angular) > 1e-6
        ):
            comparison = "command_rate_limited"
        else:
            comparison = "command_published"
        self._journal(
            "sample_compare",
            controller={
                "left_x": round(sample.left_x, 5),
                "left_y": round(sample.left_y, 5),
                "right_x": round(sample.right_x, 5),
                "right_y": round(sample.right_y, 5),
                "left_trigger": round(sample.left_trigger, 5),
                "right_trigger": round(sample.right_trigger, 5),
                "dpad_x": round(sample.dpad_x, 5),
                "dpad_y": round(sample.dpad_y, 5),
                "left_stick_button": bool(sample.left_stick_button),
                "right_stick_button": bool(sample.right_stick_button),
                "cross": bool(sample.a),
                "circle": bool(sample.b),
                "triangle": bool(sample.y),
                "square": bool(sample.square),
                "l1": bool(sample.l1),
                "r1": bool(sample.r1),
            },
            expected_twist={
                "linear_x": round(expected_linear, 6),
                "angular_z": round(expected_angular, 6),
            },
            published_twist={
                "linear_x": round(self.last_published_linear, 6),
                "angular_z": round(self.last_published_angular, 6),
            },
            arm_control={
                "connected": self.arm_client is not None,
                "control_kind": self.control_kind,
                "input_active": arm_input_active,
                "bridge_status": self.last_arm_bridge_status,
            },
            ros_subscriber_count=subscribers,
            ros_subscriber_names=list(subscriber_names),
            publish_gap_sec=round(self.last_publish_gap_sec, 6),
            smoothed_twist=smoothed,
            odometry_twist=odometry,
            motor_speed=motor_speed,
            comparison=comparison,
        )

    def update(self) -> None:
        if self.closed:
            return
        self._poll_arm_return()
        if self.adapter is None:
            self.retry_ticks += 1
            if self.retry_ticks >= 50:
                self.retry_ticks = 0
                self._connect()
            self.root.after(CONTROL_UPDATE_MS, self.update)
            return
        sample = self.adapter.read_sample(timeout_sec=0.0)
        self.latest_sample = sample
        if not sample.connected:
            self._disconnect("手柄已断开，等待重新连接…")
            self.root.after(CONTROL_UPDATE_MS, self.update)
            return
        self._track_button_input()
        self._refresh_mode(sample)
        self._apply_control(sample)
        self._record_sample(sample)
        try:
            self._read_aux()
            self._read_hidraw_mic()
        except (EOFError, OSError):
            self._disconnect("手柄附加输入已断开，等待重新连接…")
        self.root.after(CONTROL_UPDATE_MS, self.update)

    def _track_button_input(self) -> None:
        if self.adapter is None:
            return
        states = self.adapter.state.buttons
        labels = dict(BUTTONS)
        for code, pressed in states.items():
            pressed = bool(pressed)
            if pressed and not self.previous_button_state.get(code, False):
                name = labels.get(code, f"按键 {code}")
                self.input_status.set("输入正常")
                self.last_event.set(f"已收到：{name}")
                self._journal("controller_button_input", code=int(code), label=name)
            self.previous_button_state[code] = pressed
        sample = self.latest_sample
        if sample is not None:
            dpad = (float(sample.dpad_x), float(sample.dpad_y))
            if dpad != (0.0, 0.0) and self.previous_dpad == (0.0, 0.0):
                direction = f"方向键 X={dpad[0]:g}, Y={dpad[1]:g}"
                self.input_status.set("输入正常")
                self.last_event.set(f"已收到：{direction}")
                self._journal("controller_dpad_input", x=dpad[0], y=dpad[1])
            self.previous_dpad = dpad

    def _disconnect(self, message: str) -> None:
        self._journal("controller_disconnected", reason=message)
        if self.control_enabled or self.control_publisher is not None:
            self._disable_control("手柄断开，控制已停车并关闭")
        else:
            self.control_confirm_pending = False
            self.control_button.configure(text="开启手柄控制（需按两次）")
        if self.adapter is not None:
            self.adapter.close()
        self.adapter = None
        self.last_mode_chord_pressed = False
        self.mode_chord_control_seen = False
        self.last_control_chord_pressed = False
        self.control_chord_seen = False
        self.control_chord_first_at = None
        self.control_chord_enable_armed = False
        self.control_chord_enable_deadline = 0.0
        self.last_circle_pressed = False
        self.circle_pressed_at = None
        self.circle_first_click_at = None
        self.circle_hold_fired = False
        for device in self.aux_devices.values():
            device.close()
        self.aux_devices.clear()
        if self.hidraw_fd is not None:
            os.close(self.hidraw_fd)
            self.hidraw_fd = None
        self.status.set(message)

    def close(self, *, destroy_root: bool = True) -> bool:
        if self.arm_return_process is not None:
            self.control_status.set("机械臂正在返原位，完成前不能关闭控制界面")
            return False
        if self.closed:
            return True
        self._disconnect("已关闭")
        self.closed = True
        self.preview_running = False
        self._stop_voice_backend()
        self.smoothed_monitor.close()
        self.odom_monitor.close()
        self.motor_speed_monitor.close()
        self._journal("session_end")
        self.journal_file.close()
        if destroy_root and not self.embedded:
            self.root.destroy()
        return True


def main() -> int:
    root = tk.Tk()
    InputViewer(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
