#!/usr/bin/python3
"""中文纸袋检测 GUI；系统 Tk 渲染，独立 Python 子进程推理。"""
import argparse
import base64
import io
import json
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from PIL import Image, ImageTk
from bag_handle_v3.infer_bag import (
    DEFAULT_CONFIDENCE, DEFAULT_KEYPOINT_CONFIDENCE, DEFAULT_WEIGHTS,
)

ROOT = Path(__file__).resolve().parent
BACKEND = Path('/home/qianjue/anaconda3/envs/elevator-yolo26/bin/python')
ORIGIN_BACKEND = Path('/home/qianjue/anaconda3/envs/piper-handeye/bin/python')
STATES = {'valid': '提手点可用', 'invalid': '无有效提手', 'uncertain': '不确定'}


class BagGUI:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.process = None
        self.events = queue.Queue(maxsize=2)
        self.session = 0
        self.frame = None
        self.last_received = 0
        self.last_display = 0
        self.photo = None
        self.closing = False
        self.stopping = False
        self.arm_process = None
        self.arm_events = queue.Queue()
        self.arm_session = 0
        self.arm_terminal = False
        self.close_pending = False
        self.abort_return_pending = False
        self.return_request_id = 0
        self.return_job = None
        self.arm_action = None
        self.arm_last_report = None
        self.cycle_auto_return = False
        self.pending_cycle_report = None
        root.title('纸袋实时检测 v3 · YOLO26')
        root.geometry('1160x680')
        root.minsize(780, 480)
        root.tk.call('tk', 'scaling', 96 / 72)
        root.option_add('*Font', ('Noto Sans CJK SC', 11))
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TButton', padding=7)
        outer = ttk.Frame(root, padding=12)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='纸袋实时检测 v3', font=('Noto Sans CJK SC', 19, 'bold')).pack(anchor='w')
        ttk.Label(outer, text='实时摄像头  /  袋子检测框  /  提手中心像素坐标').pack(anchor='w', pady=(2, 10))
        bar = ttk.Frame(outer)
        bar.pack(fill='x')
        self.start_button = ttk.Button(bar, text='开始检测', command=self.start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(bar, text='停止检测', command=self.stop, state='disabled')
        self.stop_button.pack(side='left', padx=6)
        ttk.Label(bar, text='检测置信度').pack(side='left', padx=(14, 4))
        self.conf = tk.StringVar(value=f'{args.conf:g}')
        ttk.Spinbox(bar, from_=0.05, to=1.0, increment=0.05, width=5, textvariable=self.conf).pack(side='left')
        ttk.Label(bar, text='调整后重新开始生效').pack(side='left', padx=8)
        self.status = tk.StringVar(value='准备就绪')
        ttk.Label(outer, textvariable=self.status).pack(anchor='w', pady=8)
        arm_bar = ttk.Frame(outer)
        arm_bar.pack(fill='x', pady=(0, 5))
        self.save_observation_button = ttk.Button(arm_bar, text='保存当前为观测位', command=lambda: self.start_arm_task('save'))
        self.plan_button = ttk.Button(arm_bar, text='仅规划（不运动）', command=lambda: self.start_arm_task('plan'))
        self.touch_button = ttk.Button(arm_bar, text='开始任务：抓取→原位放置', command=lambda: self.start_arm_task('cycle'))
        self.arm_stop_button = ttk.Button(arm_bar, text='停止机械臂', command=self.stop_arm_task, state='disabled')
        self.abort_return_button = ttk.Button(arm_bar, text='终止任务并返回原点', command=self.abort_and_return)
        self.arm_buttons = [self.save_observation_button, self.plan_button, self.touch_button]
        for button in self.arm_buttons + [self.arm_stop_button, self.abort_return_button]:
            button.pack(side='left', padx=(0, 6))
        if args.source:
            for button in self.arm_buttons:
                button.configure(state='disabled')
            self.abort_return_button.configure(state='disabled')
        observation = ROOT / 'config/bag_observation.json'
        self.arm_status = tk.StringVar(value='纸袋观测位已保存；开爪定位→水平越过7厘米→闭爪垂直抬起5厘米→返回→原路下降放置→归位' if observation.exists() else '请先保存当前姿态为纸袋观测位')
        ttk.Label(outer, textvariable=self.arm_status, wraplength=1080).pack(anchor='w', pady=(0, 6))
        self.target_info = tk.StringVar(value='提手三维位置：等待本次任务定位')
        ttk.Label(outer, textvariable=self.target_info).pack(anchor='w', pady=(0, 4))
        body = ttk.Frame(outer)
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(body, bg='#101925', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.bind('<Configure>', lambda _: self.render())
        side = ttk.Frame(body, padding=(12, 0, 0, 0), width=280)
        side.grid(row=0, column=1, sticky='nsew')
        self.count = tk.StringVar(value='等待画面')
        ttk.Label(side, textvariable=self.count, font=('Noto Sans CJK SC', 12, 'bold')).pack(anchor='w', pady=(0, 8))
        self.details = tk.Text(side, width=27, bg='#f4f6f8', relief='flat', wrap='word', state='disabled')
        self.details.pack(fill='both', expand=True)
        ttk.Label(outer, text='绿色：2D 提手点通过检查    红色：无可用提手点    坐标单位：像素').pack(anchor='w', pady=(10, 0))
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_job = root.after(50, self.poll)
        self.autostart_job = None
        if not args.no_autostart:
            self.autostart_job = root.after(150, self.start)

    def start(self):
        if self.process is not None:
            return
        try:
            conf = float(self.conf.get())
            if not 0 <= conf <= 1:
                raise ValueError()
        except ValueError:
            self.status.set('请输入 0 到 1 之间的置信度')
            return
        self.session += 1
        session = self.session
        env = os.environ.copy()
        lib = str(self.args.backend_python.parent.parent / 'lib')
        env['LD_LIBRARY_PATH'] = lib + ':/usr/local/cuda-12.6/lib64:' + env.get('LD_LIBRARY_PATH', '')
        command = [str(self.args.backend_python), '-u', str(ROOT / 'tools/bag_detection_worker.py'),
                   '--weights', str(self.args.weights), '--conf', str(conf),
                   '--point-conf', str(self.args.point_conf), '--device', self.args.device]
        if self.args.source:
            command += ['--source', str(self.args.source)]
        try:
            logs = ROOT / 'artifacts/diagnostics/bag_detection_gui'
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / 'backend.log').open('a') as log:
                proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        except OSError as exc:
            self.status.set('无法启动检测：' + str(exc))
            return
        self.process = proc
        self.stopping = False
        self.last_display = 0
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status.set('正在启动检测…')

        def reader():
            for line in proc.stdout:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                self.enqueue(session, item)
            proc.stdout.close()
            self.enqueue(session, {'type': 'exit', 'code': proc.wait()})
        threading.Thread(target=reader, daemon=True).start()

    def enqueue(self, session, item):
        while True:
            try:
                self.events.put_nowait((session, item))
                return
            except queue.Full:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    pass

    def start_arm_task(self, action):
        if self.arm_process is not None or self.args.source or self.close_pending or self.abort_return_pending:
            return
        try:
            conf = .65 if action == 'return' else max(.65, float(self.conf.get()))
            if not .65 <= conf <= 1:
                raise ValueError()
        except ValueError:
            self.arm_status.set('动作检测置信度须为0.65到1之间')
            return
        # Every task starts a fresh backend and inherits no prior target/depth.
        self.arm_session += 1
        session = self.arm_session
        command = [str(self.args.backend_python), '-u', str(ROOT / 'tools/bag_grasp_worker.py'),
                   '--action', action, '--gui-heartbeat', '--conf', str(conf)]
        if action == 'return':
            command = [str(ORIGIN_BACKEND), '-u', str(ROOT / 'tools/bag_origin_worker.py'), '--gui-heartbeat']
        if action in ('touch','cycle'):
            command += ['--execute']
        env = os.environ.copy()
        # Match the voice GUI return environment. Conda's libcrypto breaks
        # the system SSH used by the existing return mode handoff.
        if action != 'return':
            env['LD_LIBRARY_PATH'] = str(self.args.backend_python.parent.parent / 'lib') + ':/usr/local/cuda-12.6/lib64:' + env.get('LD_LIBRARY_PATH', '')
        try:
            logdir = ROOT / 'artifacts/diagnostics/bag_detection_gui'
            logdir.mkdir(parents=True, exist_ok=True)
            with (logdir / 'arm_backend.log').open('a') as log:
                proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=log, text=True, bufsize=1, env=env)
        except OSError as exc:
            self.arm_status.set('机械臂任务启动失败：' + str(exc))
            for button in self.arm_buttons:
                button.configure(state='normal')
            self.abort_return_button.configure(state='normal')
            self.arm_stop_button.configure(state='disabled')
            return
        self.arm_process = proc
        self.arm_action = action
        self.cycle_auto_return = action == 'cycle'
        if action != 'return':
            self.pending_cycle_report = None
        self.arm_last_report = None
        self.arm_terminal = False
        if action != 'save':
            self.target_info.set('提手三维位置：等待本次任务的新鲜 RGB-D')
        for button in self.arm_buttons:
            button.configure(state='disabled')
        self.arm_stop_button.configure(state='normal')
        self.abort_return_button.configure(state='disabled' if action == 'return' else 'normal')
        self.arm_status.set({'save':'正在读取当前编码器…', 'plan':'正在进行无运动预检…',
                             'touch':'任务启动：前往观测位并定位提手…',
                             'cycle':'抓放任务启动：前往观测位、打开夹爪并定位…',
                             'return':'正在返回机械臂原点…'}[action])
        if action == 'return':
            self.target_info.set('归位使用项目原有机械臂原点，纸袋观测位保持不变')

        def reader():
            for line in proc.stdout:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                self.arm_events.put((session, item))
            proc.stdout.close()
            self.arm_events.put((session, {'type':'exit', 'code':proc.wait()}))
        threading.Thread(target=reader, daemon=True).start()

    def cancel_pending_return(self):
        self.abort_return_pending = False
        self.return_request_id += 1
        if self.return_job is not None:
            self.root.after_cancel(self.return_job)
            self.return_job = None

    def abort_and_return(self):
        if self.args.source or self.close_pending or self.abort_return_pending or self.arm_action == 'return' and self.arm_process is not None:
            return
        self.abort_return_pending = True
        self.return_request_id += 1
        self.abort_return_button.configure(state='disabled')
        for button in self.arm_buttons:
            button.configure(state='disabled')
        if self.arm_process is not None:
            self.stop_arm_task(cancel_return=False)
            self.arm_status.set('正在终止纸袋任务；等待停止与后台退出后返回原点…')
        else:
            self.arm_stop_button.configure(state='normal')
            self.return_job = self.root.after_idle(self.begin_pending_return, self.return_request_id)

    def begin_pending_return(self, request_id):
        self.return_job = None
        if (request_id != self.return_request_id or not self.abort_return_pending
                or self.close_pending or self.arm_process is not None):
            return
        self.abort_return_pending = False
        self.start_arm_task('return')

    def stop_arm_task(self, *, cancel_return=True):
        if cancel_return:
            self.cycle_auto_return = False
            self.pending_cycle_report = None
            self.cancel_pending_return()
        proc = self.arm_process
        if proc is None or proc.poll() is not None:
            self.arm_status.set('已取消待执行的返回原点')
            self.arm_stop_button.configure(state='disabled')
            self.abort_return_button.configure(state='disabled' if self.args.source else 'normal')
            for button in self.arm_buttons:
                button.configure(state='disabled' if self.args.source else 'normal')
            return
        self.arm_status.set('正在通过独立停止接口中止任务…')
        self.arm_stop_button.configure(state='disabled')
        try:
            proc.stdin.write('{"command":"stop"}\n')
            proc.stdin.flush()
        except (OSError, ValueError):
            proc.terminate()

    def poll_arm(self):
        proc = self.arm_process
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({'command':'heartbeat','monotonic':time.monotonic()})+'\n')
                proc.stdin.flush()
            except (OSError, ValueError):
                self.arm_status.set('机械臂控制通道异常，等待独立停止完成')
        try:
            while True:
                session, item = self.arm_events.get_nowait()
                if session != self.arm_session:
                    continue
                kind = item.get('type')
                if kind == 'exit':
                    if proc is not None and proc.stdin:
                        proc.stdin.close()
                    if (self.arm_action == 'cycle' and self.cycle_auto_return
                            and item['code'] == 0 and self.arm_last_report
                            and self.arm_last_report.get('return_to_origin_on_success') is True
                            and self.arm_last_report.get('cycle_completed') is True
                            and not self.close_pending and not self.arm_last_report.get('stop_error')):
                        self.pending_cycle_report = dict(self.arm_last_report)
                        self.abort_return_pending = True
                        self.return_request_id += 1
                    self.cycle_auto_return = False
                    self.arm_process = None
                    self.arm_action = None
                    self.arm_stop_button.configure(state='disabled')
                    self.abort_return_button.configure(state='disabled' if self.args.source else 'normal')
                    for button in self.arm_buttons:
                        button.configure(state='disabled' if self.args.source else 'normal')
                    if not self.arm_terminal:
                        self.arm_status.set(f'机械臂任务已退出（{item["code"]}），请检查任务日志')
                    if self.abort_return_pending and not self.close_pending:
                        if self.arm_last_report is None or self.arm_last_report.get('stop_error'):
                            self.cancel_pending_return()
                            self.arm_status.set('旧任务停止结果未确认，已取消自动归位')
                        else:
                            self.arm_stop_button.configure(state='normal')
                            self.abort_return_button.configure(state='disabled')
                            for button in self.arm_buttons:
                                button.configure(state='disabled')
                            self.return_job = self.root.after_idle(self.begin_pending_return, self.return_request_id)
                elif kind == 'depth':
                    x,y,z = item['base_xyz_m']
                    self.target_info.set(f'提手深度 {item["depth_m"]*100:.1f}厘米 · 基座坐标 ({x*1000:.1f}, {y*1000:.1f}, {z*1000:.1f})毫米')
                else:
                    self.arm_status.set(item.get('message', '任务状态更新'))
                    if kind in ('complete','error'):
                        self.arm_terminal = True
                        self.arm_last_report = item.get('report')
                        if self.arm_action == 'return' and self.pending_cycle_report:
                            result = dict(self.pending_cycle_report)
                            result['origin_return'] = self.arm_last_report
                            result['status'] = 'grasp_place_completed_at_origin' if kind == 'complete' else 'grasp_place_origin_return_failed'
                            result['returned_to_origin'] = kind == 'complete'
                            try:
                                path = ROOT / 'artifacts/diagnostics/bag_detection_gui/latest_arm_task.json'
                                path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
                            except OSError:
                                self.arm_status.set('归位结果已收到，但任务报告保存失败')
                            self.arm_last_report = result
                            self.pending_cycle_report = None
                            if kind == 'complete':
                                self.arm_status.set('抓取与原位放置动作已执行，机械臂已归位；实际夹持结果未核验')
                    if kind == 'error':
                        self.target_info.set('本次目标不可用于继续运动；重新开始将重新定位')
                        if (self.arm_last_report or {}).get('possibly_holding_bag'):
                            self.arm_status.set(item.get('message','任务已停止')+'；夹爪可能仍夹持纸袋')
        except queue.Empty:
            pass
        if self.close_pending and self.arm_process is None:
            self._destroy()
            return False
        return True

    def stop(self):
        if self.process is None:
            return
        self.process.terminate()
        self.stopping = True
        self.status.set('正在停止检测…')
        self.stop_button.configure(state='disabled')
        proc = self.process
        def reap():
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        threading.Thread(target=reap, daemon=True).start()
        self.frame = None
        self.render()

    def poll(self):
        if self.closing:
            return
        if not self.poll_arm():
            return
        try:
            for _ in range(2):
                session, item = self.events.get_nowait()
                if session != self.session:
                    continue
                kind = item['type']
                if kind == 'frame':
                    if self.stopping:
                        continue
                    self.frame = item
                    self.last_received = time.monotonic()
                    if not self.args.source and time.time() - item['capture_time'] > 2:
                        self.frame = None
                        self.status.set('检测画面已过期，等待新帧')
                    else:
                        now = time.monotonic()
                        fps = 1 / (now - self.last_display) if self.last_display else 0
                        self.last_display = now
                        device = 'GPU' if item['device'] == '0' else 'CPU'
                        self.status.set(f'v3 检测中 · {device} · {fps:.1f} 帧/秒 · 推理 {item["inference_ms"]:.0f} 毫秒 · 帧号 {item["frame_sequence"]}')
                    self.render()
                elif kind in ('error', 'status'):
                    self.status.set(item['message'])
                    if kind == 'error':
                        self.frame = None
                        self.render()
                elif kind == 'exit':
                    self.process = None
                    self.start_button.configure(state='normal')
                    self.stop_button.configure(state='disabled')
                    if item['code'] == 0 and self.args.source:
                        self.status.set('测试图片检测完成')
                    elif item['code'] in (0, -15, -9):
                        self.status.set('检测已停止')
                    else:
                        self.status.set(self.status.get() + '；后端已退出，详见日志')
        except queue.Empty:
            pass
        if self.frame and not self.args.source and time.time() - self.frame['capture_time'] > 2:
            self.frame = None
            self.status.set('画面已过期，等待摄像头或检测恢复')
            self.render()
        self.poll_job = self.root.after(50, self.poll)

    def render(self):
        self.canvas.delete('all')
        w, h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        rows = []
        if self.frame:
            image = Image.open(io.BytesIO(base64.b64decode(self.frame['jpeg_base64'])))
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            resampling = getattr(Image, 'Resampling', Image)
            image = image.resize((max(1, int(iw * scale)), max(1, int(ih * scale))), resampling.LANCZOS)
            ox, oy = (w - image.width) / 2, (h - image.height) / 2
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(ox, oy, image=self.photo, anchor='nw')
            detections = self.frame['detections']
            self.count.set(f'检测到 {len(detections)} 个袋子')
            for obj in detections:
                point = obj['handle_center_xy'] if obj['grasp_ready_2d'] else None
                color = '#39e092' if point else '#ff6565'
                name = STATES[obj['status']]
                x1, y1, x2, y2 = obj['bbox_xyxy']
                box = [ox+x1*scale, oy+y1*scale, ox+x2*scale, oy+y2*scale]
                self.canvas.create_rectangle(*box, outline=color, width=2)
                self.canvas.create_text(box[0]+4, max(oy+2, box[1]-21), text=f'#{obj["object_index"]} {name} {obj["confidence"]:.0%}', fill=color, anchor='nw', font=('Noto Sans CJK SC', 11, 'bold'))
                if point:
                    x, y = ox+point[0]*scale, oy+point[1]*scale
                    self.canvas.create_oval(x-5,y-5,x+5,y+5,outline=color,width=2)
                    self.canvas.create_line(x-9,y,x+9,y,fill=color,width=2)
                    self.canvas.create_line(x,y-9,x,y+9,fill=color,width=2)
                coords = f'({point[0]:.1f}, {point[1]:.1f})' if point else '无可用点'
                rows.append(f'袋子 #{obj["object_index"]} · {name}\n检测置信度：{obj["confidence"]:.1%}\n提手中心：{coords}\n提手置信度：{obj["keypoint_confidence"]:.1%}\n')
            if not rows:
                rows = ['当前画面未检测到袋子']
        else:
            self.count.set('等待画面')
            self.canvas.create_text(w/2,h/2,text='等待实时摄像头画面',fill='#a8b8cc',font=('Noto Sans CJK SC',16))
        self.details.configure(state='normal')
        self.details.delete('1.0','end')
        self.details.insert('1.0','\n'.join(rows))
        self.details.configure(state='disabled')

    def close(self):
        self.cancel_pending_return()
        if self.close_pending:
            return
        if self.arm_process is not None:
            self.close_pending = True
            self.stop_arm_task()
            self.arm_status.set('正在停止机械臂任务，停止完成后关闭窗口…')
            return
        self._destroy()

    def _destroy(self):
        self.stop()
        self.closing = True
        self.root.after_cancel(self.poll_job)
        if self.autostart_job is not None:
            self.root.after_cancel(self.autostart_job)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='纸袋实时检测窗口')
    parser.add_argument('--backend-python', type=Path, default=BACKEND)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--conf', type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument('--point-conf', type=float, default=DEFAULT_KEYPOINT_CONFIDENCE)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--no-autostart', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.point_conf <= 1:
        parser.error('置信度须在 0 到 1 之间')
    root = tk.Tk()
    BagGUI(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
