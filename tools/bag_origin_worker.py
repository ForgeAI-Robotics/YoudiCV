#!/usr/bin/env python3
"""纸袋GUI归位适配器：复用生产CommandDispatcher，不另写归位轨迹。"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from run_voice_command_gateway import CommandDispatcher, DEFAULT_HOLD_REPORT
from tools.navigation_arm_worker import check_origin
from tools.bag_grasp_worker import chassis_clear, decode_odom
from piper_handeye.arm_gamepad_udp import ArmGamepadUdpClient
from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient
from piper_handeye.guarded_press import GuardedPressUdpClient
from piper_handeye.ros1_topic_monitor import Ros1TopicMonitor
from piper_handeye.robot import PiperUdpRobot
from run_direct_return_to_origin import ensure_origin_clamp_closed, GRIPPER_CLOSED_THRESHOLD_RAD
from piper_handeye.bag_task_timing import timed_stage


class OriginTask:
    def __init__(self, emit, *, gui_heartbeat=True, dispatcher=None):
        self.emit=emit;self.gui_heartbeat=gui_heartbeat
        self.cancel=threading.Event();self.last_heartbeat=time.monotonic()
        self.reason='用户停止归位'
        self.report={'status':'initializing','timings':[]}
        self.dispatcher=dispatcher or CommandDispatcher(execute=True,
            config_path=ROOT/'piper_button_pose.yaml',hold_report=DEFAULT_HOLD_REPORT,
            origin_return_bridge_handoff=True)

    def stop(self,reason):
        self.reason=reason;self.cancel.set()

    def check(self):
        if self.cancel.is_set():raise RuntimeError(self.reason)
        if self.gui_heartbeat and time.monotonic()-self.last_heartbeat>1:
            self.stop('GUI心跳超时');raise RuntimeError(self.reason)

    def safe_stop(self):
        # Never send a global stop when this adapter owns no child motion.
        while self.dispatcher.active is not None:
            result=self.dispatcher.stop(reason='bag_gui_origin_cancelled')
            if result.get('event')=='stop_pending':
                self.emit('progress','归位仍在停止中，保持控制权并等待旧进程退出')
                time.sleep(.1)
            else:break

    def close_clamp_before_return(self, arm_config, robot, live, odom, expected):
        """Finish and verify clamp recovery before starting the arm return child."""
        self.emit('progress','先将夹爪归位：闭合并等待实时反馈确认…')
        outcome={}
        def close():
            try:outcome['result']=ensure_origin_clamp_closed(arm_config,clamp_timeout_sec=6.5)
            except Exception as exc:outcome['error']=exc
        def supervise():
            self.check();chassis_clear(odom.snapshot())
            state=robot.state(max_age=.25)
            q=np.asarray(state.joints,dtype=float)
            if (state.raw.get('hardware_ok')!=1 or q.shape!=(6,) or not np.isfinite(q).all()
                    or np.max(np.abs(q-expected))>.005):
                raise RuntimeError('夹爪归位期间机械臂硬件反馈或静止核验失败')
        supervise()
        thread=threading.Thread(target=close,daemon=True)
        thread.start()
        try:
            while thread.is_alive():
                # 5015 is synchronously occupied by the clamp. Use the separate
                # pushed state stream, so status polling cannot time out it.
                supervise()
                self.cancel.wait(.05)
        finally:
            # Keep the motion lock until the bounded clamp RPC has returned,
            # including after stop/close. No arm command is sent from this thread.
            thread.join()
        supervise()
        if 'error' in outcome:
            raise RuntimeError('夹爪归位未确认，未启动机械臂返程：'+str(outcome['error'])) from outcome['error']
        result=outcome['result'];status=result['status'];position=status.get('gripper_position')
        if (status.get('clamp_open') is not False or position is None
                or not np.isfinite(position) or position>=GRIPPER_CLOSED_THRESHOLD_RAD):
            raise RuntimeError('夹爪尚未闭合，未启动机械臂返程')
        # Once the RPC finishes, restore fresh 5015 checks before handing over.
        status=live.status()['status'];age=status.get('joint_feedback_age_sec')
        position=status.get('gripper_position')
        q=np.asarray(status.get('actual_joints_rad'),dtype=float)
        if (age is None or not np.isfinite(age) or not 0<=age<=.25
                or status.get('clamp_open') is not False or position is None
                or not np.isfinite(position) or position>=GRIPPER_CLOSED_THRESHOLD_RAD
                or status.get('active') or status.get('other_motion_active') or status.get('fault_reason')
                or q.shape!=(6,) or not np.isfinite(q).all() or np.max(np.abs(q-expected))>.005):
            raise RuntimeError('夹爪归位后的实时状态核验失败，未启动机械臂返程')
        self.check()
        return result

    def wait_for_standstill(self, live, timed, guarded, odom, tc, gc):
        deadline=time.monotonic()+15;stable_since=None;anchor=None
        while True:
            self.check()
            if time.monotonic()>deadline:raise RuntimeError('旧运动尚未停止或底盘反馈未就绪，未启动归位')
            status=live.status()['status']
            if status.get('active') or status.get('fault_reason'):
                raise RuntimeError('手柄机械臂控制占用或硬件故障，未启动归位')
            # Position and age from the same reply; do not issue a second status RPC.
            q=np.asarray(status.get('actual_joints_rad'),dtype=float)
            age=status.get('joint_feedback_age_sec')
            if (q.shape!=(6,) or not np.isfinite(q).all() or age is None
                    or not np.isfinite(age) or not 0<=age<=.25):
                raise RuntimeError('归位前编码器无效或陈旧')
            busy=timed.health(token=tc['token']).get('busy') or guarded.health(token=gc['token']).get('busy')
            try:chassis_clear(odom.snapshot())
            except RuntimeError:clear=False
            else:clear=True
            if busy or status.get('other_motion_active') or not clear:
                stable_since=None;anchor=None
            elif anchor is None or np.max(np.abs(q-anchor))>.003:
                anchor=q;stable_since=time.monotonic()
            elif time.monotonic()-stable_since>=.5:return q
            self.cancel.wait(.05)

    def return_and_wait(self, odom):
        self.dispatcher.dispatch_origin_operation('return')
        deadline=time.monotonic()+90
        while self.dispatcher.active is not None:
            self.check();chassis_clear(odom.snapshot())
            if time.monotonic()>deadline:raise RuntimeError('归位超时')
            result=self.dispatcher.refresh()
            if result is not None and result.get('event')=='origin_operation_completed':
                if result.get('return_code')!=0:raise RuntimeError('既有机械臂归位程序执行失败')
                break
            self.cancel.wait(.05)

    def run(self, *, origin_reader=check_origin):
        cfg=yaml.safe_load((ROOT/'piper_button_pose.yaml').read_text())
        gp=yaml.safe_load((ROOT/'gamepad_teleop.yaml').read_text())
        ac=gp['reserved_ports']['arm_output'];tc=cfg['timed_trajectory'];gc=cfg['guarded_press']
        live=None;odom=None;robot=None
        try:
            self.check()
            self.dispatcher._acquire_motion_lock()
            live=ArmGamepadUdpClient(ac['host'],ac['port'],ac['token'],timeout_sec=.5)
            timed=TimedTrajectoryUdpClient(tc['host'],tc['port'],timeout_sec=.7)
            guarded=GuardedPressUdpClient(gc['host'],gc['port'],timeout_sec=.7)
            odom=Ros1TopicMonitor(master_uri=gp['reserved_ports']['chassis_output']['ros_master'],
                topic='/odom',message_type='nav_msgs/Odometry',decoder=decode_odom)
            self.emit('progress','等待旧运动接口空闲、机械臂与底盘停稳…')
            with timed_stage(self.report,'origin_standstill_confirmation'):
                q=self.wait_for_standstill(live,timed,guarded,odom,tc,gc)
            self.check()
            robot_config=dict(cfg['robot']);robot_config['state_timeout_sec']=.5
            robot=PiperUdpRobot(robot_config);robot.open()
            with timed_stage(self.report,'origin_clamp'):
                clamp=self.close_clamp_before_return(ac,robot,live,odom,q)
            robot.close();robot=None
            self.check();chassis_clear(odom.snapshot())
            self.emit('progress','夹爪已归位闭合，正在调用现有机械臂归位流程')
            with timed_stage(self.report,'origin_program'):
                self.return_and_wait(odom)
            self.check();chassis_clear(odom.snapshot())
            with timed_stage(self.report,'origin_verification'):
                verified=origin_reader(self.dispatcher)
            self.check()
            if verified.get('ok') is not True or verified.get('at_origin') is not True or verified.get('hardware_ok')!=1:
                raise RuntimeError('归位后的实时编码器核验未通过')
            self.report.update(status='returned_to_origin',returned_to_origin=True,
                               pre_return_clamp=clamp,origin_verification=verified)
            self.emit('complete','机械臂已返回原点，实时编码器核验通过',report=self.report)
        finally:
            self.safe_stop()
            self.dispatcher._release_motion_lock()
            if odom:odom.close()
            if live:live.close()
            if robot:robot.close()


def isolate_control_stdin():
    """Keep the GUI pipe private; SSH children must not consume heartbeats."""
    control=os.fdopen(os.dup(sys.stdin.fileno()),'r')
    with open(os.devnull,'rb') as null:
        os.dup2(null.fileno(),sys.stdin.fileno())
    return control


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui-heartbeat',action='store_true')
    args=parser.parse_args()
    # Production children print YAML. Isolate their stdout from this JSON protocol.
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    control=isolate_control_stdin() if args.gui_heartbeat else None
    def emit(kind,message,**values):
        protocol.write(json.dumps({'type':kind,'message':message,**values},ensure_ascii=False)+'\n')
        protocol.flush()
    task=OriginTask(emit,gui_heartbeat=args.gui_heartbeat)
    def read():
        for line in control:
            try:data=json.loads(line)
            except ValueError:continue
            if data.get('command')=='stop':task.stop('用户停止归位')
            if data.get('command')=='heartbeat':
                stamp=data.get('monotonic')
                if isinstance(stamp,(int,float)) and np.isfinite(stamp) and -.05<=time.monotonic()-stamp<=.5:
                    task.last_heartbeat=stamp
        task.stop('GUI控制通道关闭')
    if args.gui_heartbeat:threading.Thread(target=read,daemon=True).start()
    signal.signal(signal.SIGTERM,lambda *_:task.stop('用户停止归位'))
    signal.signal(signal.SIGINT,lambda *_:task.stop('用户停止归位'))
    try:task.run();return 0
    except Exception as exc:
        task.report.update(status='return_failed',error=str(exc))
        emit('error','归位停止：'+str(exc),report=task.report)
        return 2
    finally:protocol.close()


if __name__=='__main__':raise SystemExit(main())
