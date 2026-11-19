#!/usr/bin/env python3
"""纸袋观测位、RGB-D规划及受监督触碰；GUI 经 JSON 行协议调用。"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import struct
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from piper_handeye import bag_grasp as bg
from piper_handeye.bag_task_timing import timed_stage
from piper_handeye.arm_gamepad_udp import ArmGamepadUdpClient
from piper_handeye.robot import PiperUdpRobot
from piper_handeye.kinematics import forward_kinematics
from piper_handeye.safety_stop import SafetyStopUdpClient
from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient, reverse_timed_trajectory
from piper_handeye.guarded_press import GuardedPressUdpClient
from piper_handeye.vision_rpc import VisionRpcClient
from piper_handeye.ros1_topic_monitor import Ros1TopicMonitor


def decode_odom(payload):
    n = struct.unpack_from('<I', payload, 12)[0]
    off = 16+n
    child = struct.unpack_from('<I',payload,off)[0]
    off += 4+child
    pose = struct.unpack_from('<7d',payload,off)
    vel = struct.unpack_from('<6d',payload,off+7*8+36*8)
    return {'pose':list(pose),'velocity':list(vel)}


def chassis_clear(sample):
    if not sample.get('available') or sample.get('error') or sample.get('age_sec',99)>.5:
        raise RuntimeError('底盘速度反馈不可用或已过期')
    v = np.asarray(sample['velocity'])
    if not np.isfinite(v).all() or np.linalg.norm(v[:3])>.005 or np.linalg.norm(v[3:])>.02:
        raise RuntimeError('底盘未停稳，机械臂任务已停止')


class Task:
    def __init__(self,args):
        self.args=args
        self.config=yaml.safe_load((ROOT/'piper_button_pose.yaml').read_text())
        gp=yaml.safe_load((ROOT/'gamepad_teleop.yaml').read_text())
        self.arm_cfg=gp['reserved_ports']['arm_output']
        self.master_uri=gp['reserved_ports']['chassis_output']['ros_master']
        self.K,self.dist,self.tcp_camera,self.cal_sha=bg.calibration(self.config)
        self.cancel=threading.Event();self.closed=threading.Event()
        self.heartbeat=time.monotonic();self.error=None
        self.require_heartbeat=False
        self.moving=False;self.has_sent_motion=False;self.motion_owned=False
        self.last_guard=time.monotonic();self.camera_watch=False
        self.send_lock=threading.RLock()
        self.capture_lock=threading.Lock()
        self.arm_query_gate=threading.RLock()
        self.gripper_active=threading.Event()
        self.gripper_expected=None
        self.motion_gate=threading.RLock()
        self.live=None;self.robot=None;self.odom=None
        self.stop_cfg=self.config['safety_stop']
        self.stopper=SafetyStopUdpClient(self.stop_cfg['host'],self.stop_cfg['port'],timeout_sec=.6)
        self.monitor_thread=None;self.heartbeat_thread=None
        self.report={'status':'initializing','actual_motion_requested':False,'contact_confirmed':False,
                     'gripper_commanded':False,'bag_collision_heuristics_enabled':False,'observation_path':str(bg.OBSERVATION_PATH)}

    def emit(self,kind,message,**values):
        data={'type':kind,'message':message,**values}
        try:
            with self.send_lock:
                print(json.dumps(data,ensure_ascii=False,allow_nan=False),flush=True)
        except BrokenPipeError:
            self.abort('GUI 通信已断开')
            raise RuntimeError('GUI 通信已断开')

    def abort(self,reason):
        with self.motion_gate:
            if not self.cancel.is_set():
                self.error=str(reason)
                self.cancel.set()
        if self.motion_owned and self.moving:
            try:
                result=self.stopper.stop(token=self.stop_cfg['token'],reason='bag_task:'+str(reason),frame_sequence=0)
                self.report['independent_stop']=result
            except Exception as exc:
                self.report['stop_error']=str(exc)

    def check(self):
        if self.cancel.is_set():
            raise RuntimeError(self.error or '任务已中止')
        if self.require_heartbeat and self.args.gui_heartbeat and time.monotonic()-self.heartbeat>1.0:
            self.abort('GUI 心跳超时');raise RuntimeError('GUI 心跳超时')

    def heartbeat_watch(self):
        while not self.closed.wait(.1):
            if self.require_heartbeat and self.args.gui_heartbeat and time.monotonic()-self.heartbeat>1.0:
                self.abort('GUI 心跳超时')

    def controls(self):
        for line in sys.stdin:
            try:
                data=json.loads(line)
                cmd=data.get('command')
            except Exception:continue
            if cmd=='heartbeat':
                stamp=data.get('monotonic')
                if isinstance(stamp,(int,float)) and np.isfinite(stamp) and -.05<=time.monotonic()-stamp<=.5:
                    self.heartbeat=float(stamp)
            elif cmd=='stop':self.abort('用户停止任务')
        if not self.closed.is_set():self.abort('GUI 控制通道已关闭')

    def arm_client(self):
        return ArmGamepadUdpClient(self.arm_cfg['host'],self.arm_cfg['port'],self.arm_cfg['token'],timeout_sec=.4)

    def healthy(self,live,*,idle=False):
        status=live.status()['status']
        age=status.get('joint_feedback_age_sec')
        if age is None or not np.isfinite(age) or not 0<=age<=.25:
            raise RuntimeError('机械臂编码器不新鲜')
        if status.get('active') or status.get('fault_reason'):
            raise RuntimeError('手柄机械臂控制正在占用，或机械臂存在故障')
        if idle and status.get('other_motion_active'):
            raise RuntimeError('另一机械臂运动正在执行')
        q=bg.joints(status['actual_joints_rad'])
        state=self.robot.state(max_age=.25)
        if state.raw.get('hardware_ok')!=1:
            raise RuntimeError('机械臂硬件未就绪')
        return q

    def vision_client(self):
        c=self.config['vision_service']
        return VisionRpcClient(c['host'],c['port'],c['token'],timeout=.7)

    def capture(self,vision,live,*,stationary=False,q_reader=None):
        # snapshot RPC writes/prunes groups of three files. Serialize local
        # callers through decoding so the monitor and locator cannot race pruning.
        with self.capture_lock:
            return self._capture(vision,live,stationary=stationary,q_reader=q_reader)

    def _capture(self,vision,live,*,stationary=False,q_reader=None):
        import cv2
        self.check()
        read_q=q_reader or (lambda:live.live_joints(max_age_sec=.25))
        before=bg.joints(read_q())
        started=time.monotonic()
        response=vision.request('snapshot')
        after=bg.joints(read_q())
        if time.monotonic()-started>.25:
            raise RuntimeError('RGB-D 与编码器配对超时')
        if np.max(np.abs(after-before))>(.003 if stationary else .012):
            raise RuntimeError('RGB-D 采样期间机械臂位姿变化超限')
        bg.fresh(response['capture_time_unix_sec'])
        # snapshot server copies RGB and registered depth from one LiveRgbdFrame.
        rgb=cv2.imread(response['path'])
        raw=cv2.imread(response['depth_mm_path'],-1)
        if rgb is None or raw is None or rgb.shape!=(360,640,3) or raw.shape!=(360,640) or raw.dtype!=np.uint16:
            raise RuntimeError('同帧 RGB-D 读取失败')
        return {'rgb':rgb,'depth':raw.astype(float)/1000,'q':(before+after)/2,
                'sequence':response['frame_sequence'],'captured_unix':response['capture_time_unix_sec']}

    def independent_joints(self):
        """Read the continuously pushed Piper state, independently of 5015 RPC."""
        state=self.robot.state(max_age=.25)
        if state.raw.get('hardware_ok')!=1:
            raise RuntimeError('机械臂独立反馈硬件未就绪')
        return bg.joints(state.joints)

    def monitor_frame(self,live,vision):
        # 5015 serves synchronously: a clamp command blocks its status queue.
        # Never send competing short-timeout RPCs while the clamp owns it.
        if self.arm_query_gate.acquire(blocking=False):
            try:
                self.healthy(live)
                return self.capture(vision,live) if self.camera_watch else None
            finally:self.arm_query_gate.release()
        q=self.independent_joints()
        if self.gripper_active.is_set() and np.max(np.abs(q-self.gripper_expected))>.005:
            raise RuntimeError('夹爪动作期间机械臂发生非预期移动')
        return self.capture(vision,live,q_reader=self.independent_joints) if self.camera_watch else None

    def monitor(self):
        live=self.arm_client();vision=self.vision_client()
        last_seq=-1;last_new=time.monotonic();last_health=0
        try:
            while not self.closed.is_set() and not self.cancel.is_set():
                self.check()
                chassis_clear(self.odom.snapshot())
                if time.monotonic()-last_health>.3:
                    h=self.stopper.health(token=self.stop_cfg['token'])
                    if h.get('latched'):
                        raise RuntimeError('独立停止已锁存；本任务不自动清锁或重启')
                    last_health=time.monotonic()
                f=self.monitor_frame(live,vision)
                if f is not None:
                    if f['sequence']==last_seq:
                        if time.monotonic()-last_new>.4:raise RuntimeError('摄像头没有新深度帧')
                    else:
                        last_seq=f['sequence'];last_new=time.monotonic()
                        if np.mean(np.isfinite(f['depth']) & (f['depth']>.08)) < .10:
                            raise RuntimeError('摄像头深度数据覆盖不足')
                self.last_guard=time.monotonic()
                self.closed.wait(.05)
        except Exception as exc:self.abort('任务监督：'+str(exc))
        finally:live.close()

    def stable_frames(self,*,detect=False,near_contact=False):
        frames=[];seen=set();start=time.monotonic();q0=self.healthy(self.live,idle=True)
        while len(frames)<4:
            self.check()
            if time.monotonic()-start>4:raise RuntimeError('四帧纸袋/深度定位超时')
            f=self.capture(self.vision,self.live,stationary=True)
            if f['sequence'] in seen:
                self.closed.wait(.025);continue
            if np.max(np.abs(f['q']-q0))>.003:raise RuntimeError('静止采样时机械臂发生移动')
            seen.add(f['sequence'])
            if detect:
                with contextlib.redirect_stdout(sys.stderr):
                    data,_=self.predict_frame(f['rgb'],self.model,device=self.device,conf=max(.65,self.args.conf),point_conf=.5)
                bg.fresh(f['captured_unix'])
                f['target']=bg.localize(data['detections'],f['depth'],self.K,self.dist,f['q'],self.tcp_camera,
                                       require_full_box=not near_contact)
                self.emit('depth','已获取提手与同帧深度',**f['target'],sequence=f['sequence'])
            frames.append(f)
            if len(frames)<4:self.closed.wait(.04)
        self.last_frames=frames
        if detect:return frames,bg.stable_target(frames)
        return frames,q0

    def validate_path(self,points,name):
        self.check()
        with timed_stage(self.report,'path_preflight',name=name):
            result=self.timed.execute(points,token=self.config['timed_trajectory']['token'],
                                      dry_run=True,task_id=f'bag-{name}-validate-{time.time_ns()}')
        self.check();return result

    def stopped_at(self,expected):
        q=self.healthy(self.live,idle=True)
        if np.max(np.abs(q-bg.joints(expected)))>.005:raise RuntimeError('机械臂未到预期关节位')
        return q

    def execute_path(self,points,name,*,guarded=None):
        with timed_stage(self.report,'motion',name=name,
                         planned_duration_sec=points[-1].get('time_from_start_sec')):
            return self._execute_path(points,name,guarded=guarded)

    def _execute_path(self,points,name,*,guarded=None):
        self.check()
        self.stopped_at(points[0]['positions'])
        if time.monotonic()-self.last_guard>.5:raise RuntimeError('安全监督未更新，拒绝开始运动')
        chassis_clear(self.odom.snapshot())
        if self.stopper.health(token=self.stop_cfg['token']).get('latched'):
            raise RuntimeError('独立停止处于锁存状态')
        with self.motion_gate:
            self.check()
            self.moving=True;self.has_sent_motion=True
            self.report['actual_motion_requested']=True
        self.emit('progress','正在触碰提手' if guarded else '正在移动：'+name)
        try:
            # Generation and live-pose mode handoff are implemented by the existing bridges.
            if guarded is None:
                result=self.timed.execute(points,token=self.config['timed_trajectory']['token'],task_id=f'bag-{name}-{time.time_ns()}')
            else:
                result=self.guarded.execute(points,token=self.config['guarded_press']['token'],
                    current_guard=self.config['guarded_press']['current_guard'],max_forward_travel_m=.020,
                    contact_normal_base=guarded['normal_base'],contact_panel_origin_base=guarded['target_base_m'],
                    task_id=f'bag-touch-{time.time_ns()}')
            self.check()
        except BaseException as exc:
            self.abort(str(exc));raise
        else:self.moving=False
        self.stopped_at(points[0]['positions'] if guarded else points[-1]['positions'])
        return result

    def execute_fixed_target(self, plan):
        """Use this task's initial base-frame plan throughout; no near-field relocalization."""
        self.execute_path(plan['approach'],'按初始点位到提手前2厘米')
        self.stopped_at(plan['precontact_joints'])
        self.check()
        self.emit('progress','已到接触前位置，沿本次初始定位点执行触碰')
        gc=self.config['guarded_press']
        self.guarded.execute(plan['forward'],token=gc['token'],current_guard=gc['current_guard'],
            max_forward_travel_m=.020,contact_normal_base=plan['normal_base'],
            contact_panel_origin_base=plan['target_base_m'],dry_run=True,
            task_id=f'bag-fixed-target-validate-{time.time_ns()}')
        self.check()
        return self.execute_path(plan['forward'],'按初始点位受力触碰',guarded=plan)

    def set_gripper(self, opening, expected):
        with timed_stage(self.report,'gripper',opening=opening):
            return self._set_gripper(opening,expected)

    def _set_gripper(self, opening, expected):
        # Wait for the independently pushed encoder stream to catch the
        # just-finished arm motion before relying on it during clamp RPCs.
        expected=bg.joints(expected).copy()
        deadline=time.monotonic()+.6
        while True:
            self.check()
            if np.max(np.abs(self.independent_joints()-expected))<=.005:break
            if time.monotonic()>deadline:raise RuntimeError('夹爪动作前独立编码器尚未确认停稳')
            self.closed.wait(.02)
        with self.arm_query_gate:
            self.gripper_expected=expected
            self.gripper_active.set()
            try:return self._set_gripper_owned(opening,expected)
            finally:self.gripper_active.clear()

    def _set_gripper_owned(self, opening, expected):
        """Reuse the teleoperation clamp actions, checking live motor feedback."""
        self.check();self.stopped_at(expected)
        chassis_clear(self.odom.snapshot())
        if time.monotonic()-self.last_guard>.5 or self.stopper.health(token=self.stop_cfg['token']).get('latched'):
            raise RuntimeError('监督未更新或已停止，未执行夹爪动作')
        before=self.live.status()['status']
        position=before.get('gripper_position')
        age=before.get('joint_feedback_age_sec')
        if (position is None or not np.isfinite(position) or age is None
                or not np.isfinite(age) or not 0<=age<=.25):
            raise RuntimeError('夹爪实时反馈不可用')
        state=bool(position>=.35)
        if before.get('clamp_open') is not state:
            raise RuntimeError('夹爪反馈状态不一致')
        self.emit('progress','正在打开夹爪' if opening else '正在闭合夹爪')
        self.report['stage']='打开夹爪' if opening else '闭合夹爪'
        with self.motion_gate:
            self.check();self.moving=True;self.has_sent_motion=True
            self.report['actual_motion_requested']=True
            self.report['gripper_commanded']=True
        try:
            if state==opening:
                after=before;sent=False
            else:
                self.report['gripper_attempt']={'opening':opening,'before_motor_rad':position,
                    'reply_timeout_sec':6.5,'result':'waiting'}
                try:
                    response=(self.live.toggle_clamp(timeout_sec=6.5) if opening
                              else self.live.close_clamp(timeout_sec=6.5))
                except TimeoutError as exc:
                    raise RuntimeError('夹爪动作回执超时（6.5秒），结果未知，未自动重发') from exc
                after=response['status'];sent=True
            self.check()
            final=after.get('gripper_position')
            age=after.get('joint_feedback_age_sec')
            if (final is None or not np.isfinite(final) or age is None
                    or not np.isfinite(age) or not 0<=age<=.25):
                raise RuntimeError('夹爪动作后实时反馈不可用')
            if opening:
                valid=after.get('clamp_open') is True and final>=.35
            else:
                # A handle may prevent full closure; motion feedback cannot prove a grasp.
                valid=final<.35 or (sent and after.get('last_gripper_target')==0 and position-final>=.08)
            if not valid:raise RuntimeError('夹爪开合未得到实时反馈确认')
            self.stopped_at(expected)
            record={'opening':opening,'command_sent':sent,'before_motor_rad':position,'after_motor_rad':final}
            self.report.setdefault('gripper_actions',[]).append(record)
            if sent:self.report['gripper_attempt']['result']='feedback_verified'
        except BaseException as exc:
            attempt=self.report.get('gripper_attempt',{})
            if attempt.get('result')=='waiting':
                attempt.update(result='unconfirmed',error=str(exc))
            self.abort(str(exc));raise
        else:self.moving=False
        return record

    def execute_grasp_cycle(self, plan):
        # These paths are from this task's initial RGB-D localization only.
        def travel(path, name):
            self.report['stage']=name
            # Full-task preflight has passed. execute_path still invokes the
            # bridge's fresh MoveIt check and every live start/stop gate.
            self.check()
            self.execute_path(plan[path],name)
        travel('approach','到初始提手定位点')
        travel('insertion','水平越过提手7厘米')
        self.report['possibly_holding_bag']=True
        self.set_gripper(False,plan['grasp_joints'])
        travel('lift','闭爪后垂直抬起5厘米')
        travel('carry_out_near','抬起后携袋返回中间位')
        travel('carry_out_far','携袋返回观测位')
        self.report['stage']='returned_to_observation_with_closed_gripper'
        self.emit('progress','已闭爪回到观测位，沿本次保存路径返回原放置点')
        travel('carry_back_far','沿保存路径返回中间位')
        travel('carry_back_near','返回原放置点上方5厘米')
        travel('lower','垂直下降5厘米放回原位')
        self.set_gripper(True,plan['grasp_joints'])
        self.report['possibly_holding_bag']=False
        travel('withdraw_insertion','开爪退出原放置点')
        travel('withdraw_approach','放置后返回观测位')
        self.check()
        self.report.update(status='grasp_place_completed_pending_origin',
                           cycle_completed=True,grasp_confirmed=False,return_to_origin_on_success=True)
        self.emit('complete','抓取与原位放置动作已执行，等待返回机械臂原点；未核验实际夹持结果',report=self.report)

    def run(self):
        lock=Path(os.environ.get('YOUXIAOMEI_ROBOT_MOTION_LOCK','/tmp/youxiaomei_robot_motion.lock')).open('a+')
        try:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('另一机器人任务正在执行，纸袋任务未启动')
            self.motion_owned=True
            robot_cfg=dict(self.config['robot']);robot_cfg['state_timeout_sec']=.5
            self.robot=PiperUdpRobot(robot_cfg);self.robot.open()
            self.live=self.arm_client();q=self.healthy(self.live,idle=True)
            if self.args.action=='save':
                for _ in range(3):
                    self.check();time.sleep(.06)
                    if np.max(np.abs(self.healthy(self.live,idle=True)-q))>.003:
                        raise RuntimeError('机械臂未静止，不能保存观测位')
                data=bg.save_observation(q,self.cal_sha)
                self.report.update(status='observation_saved',observation=data)
                self.emit('complete','已将当前编码器姿态保存为纸袋观测位',observation=data,report=self.report)
                return
            self.vision=self.vision_client()
            tc=self.config['timed_trajectory'];gc=self.config['guarded_press']
            self.timed=TimedTrajectoryUdpClient(tc['host'],tc['port'],timeout_sec=20)
            self.guarded=GuardedPressUdpClient(gc['host'],gc['port'],timeout_sec=12)
            th=self.timed.health(token=tc['token'])
            if th.get('busy'):
                raise RuntimeError('轨迹接口正在执行其他任务')
            gh=self.guarded.health(token=gc['token'])
            if gh.get('busy'):
                raise RuntimeError('受力接口正在执行其他任务')
            if not th.get('safety_handoff_v2') or not gh.get('safety_handoff_v2'):
                raise RuntimeError('机器人接口尚未加载本次停止交接保护，拒绝执行')
            h=self.stopper.health(token=self.stop_cfg['token'])
            if h.get('latched'):raise RuntimeError('独立停止处于锁存状态，本任务不自动清锁')
            self.emit('progress','正在加载并预热 v3 纸袋模型（未请求运动）')
            with contextlib.redirect_stdout(sys.stderr):
                import torch
                from ultralytics import YOLO
                from bag_handle_v3.infer_bag import predict_frame,DEFAULT_WEIGHTS
                torch.set_num_threads(2)
                self.model=YOLO(str(DEFAULT_WEIGHTS))
                self.predict_frame=predict_frame
                self.device='0' if torch.cuda.is_available() else 'cpu'
                self.model.predict(np.zeros((360,640,3),dtype=np.uint8),imgsz=640,
                                   conf=max(.65,self.args.conf),device=self.device,verbose=False)
            self.check()
            if self.args.gui_heartbeat:
                deadline=time.monotonic()+2
                while time.monotonic()-self.heartbeat>.5:
                    self.check()
                    if time.monotonic()>deadline:
                        raise RuntimeError('模型已加载，但GUI未提供新鲜控制心跳')
                    self.closed.wait(.05)
                self.require_heartbeat=True
            self.odom=Ros1TopicMonitor(master_uri=self.master_uri,topic='/odom',message_type='nav_msgs/Odometry',decoder=decode_odom)
            start=time.monotonic();stable_since=None
            while True:
                self.check()
                try:chassis_clear(self.odom.snapshot())
                except RuntimeError:stable_since=None
                else:
                    if stable_since is None:stable_since=time.monotonic()
                    if time.monotonic()-stable_since>=.5:break
                if time.monotonic()-start>4:raise RuntimeError('底盘尚未连续停稳或反馈不可用')
                self.closed.wait(.05)
            self.monitor_thread=threading.Thread(target=self.monitor,daemon=True);self.monitor_thread.start()
            frames,q=self.stable_frames();self.camera_watch=True
            observation=bg.load_observation(self.cal_sha)
            if np.max(np.abs(q-observation))>.005:
                self.emit('progress','正在预检到纸袋观测位的路径')
                points,audit=bg.trajectory(q,observation,speed=.05)
                self.report['observation_validation']=self.validate_path(points,'observation')
                if not self.args.execute:
                    self.report['status']='observation_path_validated'
                    self.emit('complete','观测位路径已通过预检；实际到位后才能定位纸袋',report=self.report);return
                self.execute_path(points,'到观测位')
                self.stopped_at(observation)
            is_cycle=self.args.action in ('cycle','plan')
            if is_cycle and self.args.execute:
                self.set_gripper(True,observation)
            self.emit('progress','已在观测位，正在核验四帧提手与深度')
            frames,target=self.stable_frames(detect=True)
            q=self.stopped_at(observation);self.camera_watch=True
            target=np.array(target,dtype=float,copy=True)
            target.setflags(write=False)
            self.report['target_base_m']=target.tolist()
            self.report['target_policy']='initial_observation_fixed_base_target'
            self.report['initial_localization']={
                'frame_sequences':[f['sequence'] for f in frames],
                'capture_times_unix_sec':[f['captured_unix'] for f in frames],
                'observation_joints_rad':q.tolist(),
                'target_base_m':target.tolist(),
            }
            self.emit('progress','初始提手三维坐标已固定，抓取与放置沿用本次路径' if is_cycle else '初始提手三维坐标已固定，后续接近和触碰沿用此点位')
            self.report['stage']='规划抓放路径' if is_cycle else '规划触碰路径'
            with timed_stage(self.report,'planning',cycle=is_cycle):
                plan=bg.grasp_cycle_plan(q,target) if is_cycle else bg.contact_plan(q,target,self.tcp_camera)
            self.report['plan']=plan
            if is_cycle:
                self.emit('progress','正在预检抓取、携袋返回与原路放置路径')
                self.report['cycle_path_validations']={name:self.validate_path(plan[name],name)
                    for name in bg.CYCLE_PATH_NAMES}
                self.check()
                if not self.args.execute:
                    self.report['status']='grasp_place_plan_validated'
                    self.emit('complete','水平前伸7厘米、垂直抬起5厘米及原路放置路径预检通过；未执行开合或运动',report=self.report)
                    return
                self.execute_grasp_cycle(plan)
                return
            self.emit('progress','正在调用机械臂原有接口预检接近、触碰及回撤路径')
            self.report['approach_validation']=self.validate_path(plan['approach'],'approach')
            reverse=reverse_timed_trajectory(plan['forward'])
            self.report['contact_retreat_validation']=self.validate_path(reverse,'contact-retreat')
            self.report['guarded_validation']=self.guarded.execute(plan['forward'],token=gc['token'],
                current_guard=gc['current_guard'],max_forward_travel_m=.020,contact_normal_base=plan['normal_base'],
                contact_panel_origin_base=plan['target_base_m'],dry_run=True,task_id=f'bag-guard-validate-{time.time_ns()}')
            self.check()
            if not self.args.execute:
                self.report['status']='contact_plan_validated'
                self.emit('complete','接近、触碰与回撤路径预检通过；未执行运动',report=self.report);return
            result=self.execute_fixed_target(plan)
            self.report['contact_result']=result
            confirmed=result.get('feedback_contact_confirmed') is True
            self.report['contact_confirmed']=confirmed
            self.report['status']='contact_retracted' if confirmed else 'contact_unconfirmed_retracted'
            self.emit('complete','已检测受力接触并回撤到接触前位置' if confirmed else '已到触碰行程上限并回撤，接触未确认',report=self.report)
        finally:
            # Keep task ownership until outstanding work is stopped and all monitors are joined.
            if self.moving:self.abort(self.error or '任务退出')
            self.closed.set()
            if self.monitor_thread:self.monitor_thread.join(timeout=2)
            if self.odom:self.odom.close()
            if self.live:self.live.close()
            if self.robot:self.robot.close()
            self.motion_owned=False
            lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['save','plan','touch','cycle'],required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--gui-heartbeat',action='store_true')
    p.add_argument('--conf',type=float,default=.65)
    args=p.parse_args()
    if args.execute and args.action not in ('touch','cycle'):p.error('只有触碰或抓放任务可以执行运动')
    if not np.isfinite(args.conf) or not .65<=args.conf<=1:p.error('动作检测置信度须在0.65到1之间')
    task=Task(args)
    def interrupted(*_):task.abort('用户中止任务')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    if args.gui_heartbeat:
        threading.Thread(target=task.controls,daemon=True).start()
        task.heartbeat_thread=threading.Thread(target=task.heartbeat_watch,daemon=True);task.heartbeat_thread.start()
    try:
        task.run()
        return 0
    except Exception as exc:
        task.report['status']='failed';task.report['error']=str(exc)
        task.emit('error','任务停止：'+str(exc),report=task.report)
        return 2
    finally:
        task.closed.set()
        # Persist only one current report; no extra diagnostic directory per launch.
        path=ROOT/'artifacts/diagnostics/bag_detection_gui/latest_arm_task.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(task.report,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    raise SystemExit(main())
