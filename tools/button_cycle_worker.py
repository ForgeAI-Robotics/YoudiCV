#!/usr/bin/env python3
"""Explicitly started cycle tests around the unchanged production dispatcher."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from run_voice_command_gateway import CommandDispatcher
from piper_handeye.arm_gamepad_udp import ArmGamepadUdpClient
from piper_handeye.button_cycle import CycleStore
from piper_handeye.button_cycle_runner import CycleRunner
from piper_handeye.button_cycle import append_json
from piper_handeye.cycle_stationarity import StationarityGuard, ChassisNotStationary
from piper_handeye.cycle_recovery import require_unlatched_generation
from piper_handeye.cycle_fault_recovery import EmergencyFeedbackError, emergency_stop_reason, require_emergency_released, recovery_stop_identity, require_fresh_clear_visual
from piper_handeye.ros1_topic_monitor import Ros1TopicMonitor
from piper_handeye.safety_stop import SafetyStopUdpClient
from piper_handeye.vision_rpc import VisionRpcClient


class SessionDispatcher(CommandDispatcher):
    """Keep the existing global arm lock across task gaps and pauses.

    Production CommandDispatcher behavior is unchanged for all other callers.
    No second dispatcher or motion implementation is used inside a cycle.
    """
    def __init__(self, **kwargs):
        # Cycle return uses the existing verified trajectory bridge mode handoff.
        kwargs.setdefault("origin_return_bridge_handoff", True)
        super().__init__(**kwargs)
        self.session_reserved = False

    def reserve_session(self):
        self._acquire_motion_lock()
        self.session_reserved = True

    def _release_motion_lock(self):
        if not self.session_reserved:
            super()._release_motion_lock()

    def release_session(self):
        if self.active is not None:
            raise RuntimeError("机械臂进程尚未退出，不能释放测试执行权")
        self.session_reserved = False
        super()._release_motion_lock()


def decode_full_odometry(payload):
    frame = struct.unpack_from("<I", payload, 12)[0]
    offset = 16 + frame
    child = struct.unpack_from("<I", payload, offset)[0]
    offset += 4 + child + 7*8 + 36*8
    return {"velocity": list(struct.unpack_from("<6d", payload, offset))}


def decode_cycle_odometry(payload):
    value = decode_full_odometry(payload)
    sequence, secs, nsecs = struct.unpack_from('<3I', payload, 0)
    frame = struct.unpack_from('<I', payload, 12)[0]
    child_offset = 16 + frame
    child = struct.unpack_from('<I', payload, child_offset)[0]
    x, y, z, qx, qy, qz, qw = struct.unpack_from('<7d', payload, child_offset+4+child)
    norm = math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)
    if not math.isfinite(norm) or abs(norm-1.) > .01 or nsecs >= 1_000_000_000:
        raise ValueError('invalid odometry orientation or timestamp')
    yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return dict(value, sequence=sequence, source_stamp_sec=secs+nsecs*1e-9, pose_xy_yaw=[x,y,yaw])


def require_stationary(sample):
    age = sample.get("age_sec")
    velocity = sample.get("velocity")
    if (sample.get("available") is not True or sample.get("error")
            or type(age) not in (int, float) or not math.isfinite(age) or not 0 <= age <= .5
            or not isinstance(velocity, (list, tuple)) or len(velocity) != 6
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in velocity)):
        raise RuntimeError("底盘速度反馈不可用或已过期，测试保持停止")
    if math.sqrt(sum(v*v for v in velocity[:3])) > .005 or math.sqrt(sum(v*v for v in velocity[3:])) > .02:
        raise RuntimeError("底盘未停稳，测试保持停止")


def control_timestamp(packet, now, max_age):
    issued = packet.get("issued_monotonic_sec")
    if type(issued) not in (int, float) or not math.isfinite(issued) or not -.1 <= now-issued <= max_age:
        raise ValueError("启动/继续请求或GUI心跳已过期，请明确重新操作")
    return issued



def control_shutdown_reason(*, interrupted, disconnected, reader_error, heartbeat_age):
    if interrupted:
        return "收到退出信号，取消后续测试"
    if disconnected:
        return ("GUI控制通道读取异常（" + reader_error + "），取消后续测试"
                if reader_error else "GUI控制通道已关闭，取消后续测试")
    if heartbeat_age > 3.:
        return "GUI心跳超时（%.3f秒，允许3秒），取消后续测试" % heartbeat_age
    return None


def check_recorded_files(metadata):
    for path, expected in metadata.get("sha256", {}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("代码或配置已变化，请新建一组测试：" + Path(path).name)


class Readiness:
    def __init__(self, dispatcher, store, odom):
        self.dispatcher, self.store, self.odom = dispatcher, store, odom
        metadata = store.plan["metadata"]
        if (metadata.get("config") != str(dispatcher.config_path)
                or metadata.get("hold_report") != str(dispatcher.hold_report)):
            raise RuntimeError("历史组的配置或原位文件与当前界面不同，请新建测试组")
        config = yaml.safe_load(dispatcher.config_path.read_text())
        gamepad = yaml.safe_load((ROOT / "gamepad_teleop.yaml").read_text())
        hold = yaml.safe_load(dispatcher.hold_report.read_text())
        self.origin = hold["stage_1_states"][0]["joints_rad"]
        cfg = gamepad["reserved_ports"]["arm_output"]
        self.arm = ArmGamepadUdpClient(cfg["host"], cfg["port"], cfg["token"], timeout_sec=.5)
        cfg = config["vision_service"]
        self.vision = VisionRpcClient(cfg["host"], cfg["port"], cfg["token"], timeout=.5)
        self.stop_cfg = config["safety_stop"]
        self.stopper = SafetyStopUdpClient(self.stop_cfg["host"], self.stop_cfg["port"], timeout_sec=.5)
        self.config = config
        self.emergency = None
        self.requires_emergency_feedback = store.settings.failure_policy == 'recover_wait'
        self._last_emergency_signature = None
        self._planning_pool = None
        self.cancel_epoch = [0]
        self._recovery_feedback = None
        self._recovery_feedback_stop = None
        self._recovery_feedback_thread = None
        self._recovery_feedback_error = None
        self._recovery_communication_links = {}
        self._recovery_communication_epoch = 0
        self._recovery_communication_observer_id = str(time.time_ns())

    def _communication_request(self, name, request):
        """Track observed transport loss/recovery, not elapsed time or BUSY."""
        links = getattr(self, "_recovery_communication_links", None)
        if links is None:
            links = self._recovery_communication_links = {}
        if not getattr(self, "_recovery_communication_observer_id", None):
            self._recovery_communication_observer_id = str(time.time_ns())
        previous = links.get(name)
        try:
            value = request()
        except Exception as exc:
            transport_error = (isinstance(exc, (OSError, TimeoutError))
                or isinstance(exc.__cause__, (OSError, TimeoutError))
                or (name.startswith("vision")
                    and str(exc) == "vision service closed without a response"))
            if not transport_error:
                raise
            if previous is not False:
                self._recovery_communication_epoch = getattr(self, "_recovery_communication_epoch", 0)+1
            links[name] = False
            raise
        if previous is False:
            self._recovery_communication_epoch = getattr(self, "_recovery_communication_epoch", 0)+1
        links[name] = True
        return value

    def __call__(self):
        return self._probe(require_origin=True)

    def _probe(self, *, require_origin):
        if self.requires_emergency_feedback:
            self.emergency_guard()
        require_stationary(self.odom.snapshot())
        check_recorded_files(self.store.plan["metadata"])
        status = self.arm.status()["status"]
        if (status.get("active") is not False or status.get("other_motion_active") is not False
                or status.get("gripper_command_active") is not False or status.get("fault_reason")):
            raise RuntimeError("机械臂手动控制/其他动作尚未关闭，或桥报告故障")
        joints, age = status.get("actual_joints_rad"), status.get("joint_feedback_age_sec")
        if (not isinstance(joints, list) or len(joints) != 6 or len(self.origin) != 6
                or any(type(q) not in (int, float) or not math.isfinite(q) for q in [*joints, *self.origin])
                or type(age) not in (int, float) or not math.isfinite(age) or not 0 <= age <= .25):
            raise RuntimeError("机械臂编码器反馈不可用或已过期")
        if require_origin and max(abs(a-b) for a, b in zip(joints, self.origin)) > .03:
            raise RuntimeError("机械臂未归位，请先使用原机械臂归位功能")
        health = self.vision.request("health")
        if health.get("status") != "ready" or not health.get("camera_sequence"):
            raise RuntimeError("相机或常驻视觉服务未就绪")
        stopped = self.stopper.health(token=self.stop_cfg["token"])
        if stopped.get("status") != "ok" or stopped.get("latched") is not False:
            raise RuntimeError("独立停止未就绪或仍锁存，请先通过原归位流程处理")
        require_stationary(self.odom.snapshot())
        error = max(abs(a-b) for a, b in zip(joints, self.origin))
        return dict(at_origin=error <= .03, max_error_rad=error)

    def recovery_probe(self, expected_generation):
        self._probe(require_origin=False)
        from piper_handeye.guarded_press import GuardedPressUdpClient
        from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient
        for name, cls in (('guarded_press', GuardedPressUdpClient), ('timed_trajectory', TimedTrajectoryUdpClient)):
            cfg = self.config[name]
            health = cls(cfg['host'], int(cfg['port']), timeout_sec=.5).health(token=cfg['token'])
            if health.get('status') != 'ok' or health.get('busy') is not False:
                raise RuntimeError('旧运动桥尚未空闲，禁止自动归位')
        return require_unlatched_generation(self.stopper.health(token=self.stop_cfg['token']), expected_generation)

    def _return_arm_snapshot(self, *, require_origin=False):
        require_stationary(self.odom.snapshot())
        status = self._communication_request('arm', self.arm.status)['status']
        joints, age = status.get('actual_joints_rad'), status.get('joint_feedback_age_sec')
        if (status.get('active') is not False or status.get('other_motion_active') is not False
                or status.get('gripper_command_active') is not False or status.get('fault_reason')
                or not isinstance(joints,list) or len(joints)!=6
                or any(type(q) not in (int,float) or not math.isfinite(q) for q in joints)
                or type(age) not in (int,float) or not math.isfinite(age) or not 0<=age<=.25):
            raise RuntimeError('机械臂未空闲或编码器反馈异常，等待恢复')
        error=max(abs(a-b) for a,b in zip(joints,self.origin))
        if require_origin and error>.03:
            raise RuntimeError('机械臂实际关节尚未归位')
        return dict(at_origin=error<=.03,max_error_rad=error,joints=list(joints))

    def verify_return(self):
        # Return truth does not depend on YOLO, lamp rules, or a historical group's fingerprint.
        return self._return_arm_snapshot(require_origin=True)

    def emergency_guard(self):
        sample = self.emergency.snapshot() if self.emergency else {}
        failure = None
        try:
            require_emergency_released(sample)
        except EmergencyFeedbackError as exc:
            failure = exc
        code = 'released' if failure is None else failure.code
        signature = (code, sample.get('engaged'), str(sample.get('error')))
        if signature != getattr(self, '_last_emergency_signature', None):
            self._last_emergency_signature = signature
            directory = getattr(self.store, 'directory', None)
            if directory is not None:
                append_json(Path(directory)/'emergency_events.jsonl', dict(
                    event='emergency_feedback_changed', topic='/emergency',
                    reason_code=code, sample=sample,
                    monotonic_sec=time.monotonic(), time_unix_sec=time.time()))
        if failure is not None:
            raise failure

    def auto_recovery_probe(self):
        check_recorded_files(self.store.plan['metadata'])
        self.emergency_guard()
        arm=self._return_arm_snapshot()
        from piper_handeye.guarded_press import GuardedPressUdpClient
        from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient
        for name,cls in [('guarded_press',GuardedPressUdpClient),('timed_trajectory',TimedTrajectoryUdpClient)]:
            c=self.config[name]
            h=self._communication_request(name, lambda:cls(c['host'],int(c['port']),timeout_sec=.5).health(token=c['token']))
            if h.get('status')!='ok' or h.get('busy') is not False:
                raise RuntimeError('运动桥尚未空闲或通信不可用，等待恢复')
            if (name=='timed_trajectory' and getattr(self.dispatcher,'origin_return_bridge_handoff',False) is True
                    and h.get('safety_handoff_v2') is not True):
                raise RuntimeError('轨迹桥模式交接能力未确认，禁止自动归位')
        h=self._communication_request('stop', lambda:self.stopper.health(token=self.stop_cfg['token']));identity=recovery_stop_identity(h)
        health=self._communication_request('vision', lambda:self.vision.request('health'))
        if health.get('status')!='ready' or not health.get('camera_sequence'):
            raise RuntimeError('相机或视觉服务未恢复')
        if identity['latched'] and identity['reason'] not in {'cycle_physical_estop','cycle_feedback_unavailable'}:
            self._ensure_recovery_feedback()
            if self._recovery_feedback_error:
                raise RuntimeError(self._recovery_feedback_error)
            # Do not erase an obstacle reference or recapture the scene to manufacture clearance.
            safety=self._communication_request('vision_safety',lambda:self.vision.request('safety'))['visual_safety']
            require_fresh_clear_visual(safety,time.monotonic())
        return dict(**arm,stop_identity=identity,camera_sequence=int(health['camera_sequence']),
                    communication_epoch=getattr(self,'_recovery_communication_epoch',0),
                    communication_observer_id=getattr(self,'_recovery_communication_observer_id',None))

    def _ensure_recovery_feedback(self):
        if self._recovery_feedback_thread is not None:return
        from piper_handeye.robot import PiperUdpRobot
        from run_online_replan_no_contact import load_handeye,resolve_path
        handeye,_,_=load_handeye(resolve_path(self.config['calibration']['handeye_yaml'],self.dispatcher.config_path))
        robot=PiperUdpRobot(self.config['robot'])
        robot.open()  # Subscribe only; no enable/mode/trajectory command.
        self._recovery_feedback=robot
        self._recovery_feedback_stop=threading.Event()
        stop=self._recovery_feedback_stop
        def publish():
            while not stop.is_set():
                try:
                    sample=robot.state(max_age=.25)
                    if sample.raw.get('hardware_ok')!=1:raise RuntimeError('机械臂硬件反馈未恢复')
                    self.vision.request('safety_camera_pose',timestamp_monotonic=float(sample.timestamp),
                        base_to_camera=(sample.base_to_gripper@handeye).tolist())
                    self._recovery_feedback_error=None
                except Exception as exc:self._recovery_feedback_error=str(exc)
                stop.wait(.05)
        self._recovery_feedback_thread=threading.Thread(target=publish,daemon=True,name='cycle-recovery-feedback')
        self._recovery_feedback_thread.start()

    def end_recovery(self):
        if self._recovery_feedback_stop is not None:self._recovery_feedback_stop.set()
        if self._recovery_feedback_thread is not None:self._recovery_feedback_thread.join(timeout=2.)
        if self._recovery_feedback is not None:self._recovery_feedback.close()
        self._recovery_feedback=None;self._recovery_feedback_thread=None
        self._recovery_feedback_stop=None;self._recovery_feedback_error=None

    def begin_return_preflight(self, probe):
        if self._planning_pool is None:
            self._planning_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cycle-return-preflight')
        probe={**probe,"control_cancel_epoch":self.cancel_epoch[0]}
        return self._planning_pool.submit(self._preflight_return_path,probe)

    def _preflight_return_path(self, probe):
        # Pure planning + actual MoveIt dry_run. Never enable, clear, grasp or move.
        import numpy as np
        from run_direct_return_to_origin import (build_transportable_origin_return_trajectory,
                                                select_collision_checked_origin_return)
        from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient
        start=np.asarray(probe['joints'],float);origin=np.asarray(self.origin,float)
        c=self.config['timed_trajectory'];token=c['token']
        prepared=build_transportable_origin_return_trajectory(start,origin,requested_speed_m_s=.20,token=token)
        bridge=TimedTrajectoryUdpClient(c['host'],int(c['port']),timeout_sec=20.)
        _,audit,_,validation=select_collision_checked_origin_return(start,origin,prepared,
            lambda points:bridge.execute(points,token=token,task_id=f'cycle-recovery-validate-{time.time_ns()}',dry_run=True),token=token)
        return dict(start_joints=list(probe['joints']),stop_identity=probe['stop_identity'],
                    duration_sec=audit.duration_sec,validation=validation,
                    control_cancel_epoch=probe['control_cancel_epoch'])

    def authorize_auto_return(self, proof):
        if proof.get("control_cancel_epoch")!=self.cancel_epoch[0]:
            raise RuntimeError("用户已取消待执行的恢复")
        current=self.auto_recovery_probe()
        if current['stop_identity']!=proof['stop_identity']:
            raise RuntimeError('规划期间停止状态变化，必须重新复检')
        if max(abs(a-b) for a,b in zip(current['joints'],proof['start_joints']))>.003:
            raise RuntimeError('规划期间关节位置变化，必须重新规划')
        identity=current['stop_identity'];generation=identity['generation']
        if proof.get('control_cancel_epoch')!=self.cancel_epoch[0]:
            raise RuntimeError('用户已取消待执行的恢复')
        if identity['latched']:
            ack=self.stopper.reset_latch_if_generation(token=self.stop_cfg['token'],
                generation=generation,reason=identity['reason'])
            if ack.get('latched') is not False or ack.get('generation')!=generation:
                raise RuntimeError('按代号解除停止未获得确认')
        require_unlatched_generation(self.stopper.health(token=self.stop_cfg['token']),generation)
        self.emergency_guard()
        if proof.get("control_cancel_epoch")!=self.cancel_epoch[0]:
            raise RuntimeError("用户已取消待执行的恢复")
        self.end_recovery()
        return generation

    def close(self):
        # This client was never enabled: close sends no disable/motion command.
        self.end_recovery()
        self.arm.close()
        if getattr(self,"_planning_pool",None) is not None:
            self._planning_pool.shutdown(wait=False,cancel_futures=True)


def stop_before_exit(dispatcher, readiness):
    if dispatcher.active is None:
        return
    try:
        dispatcher.stop(reason="cycle_worker_finalizing")
    except Exception:
        # Use already-loaded stop settings if a file/log failure prevented the
        # production stop method from reaching its independent stop request.
        dispatcher.return_after_abort = False
        try:
            readiness.stopper.stop(token=readiness.stop_cfg["token"],
                reason="cycle_worker_stop_fallback", frame_sequence=0)
        except Exception as exc:
            print("cycle independent stop fallback failed: " + str(exc), file=sys.stderr, flush=True)
        finally:
            process = dispatcher.active
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    while dispatcher.active is not None:
        try:
            dispatcher.refresh()
        except Exception:
            # Broken reporting must not release the lock while the producer
            # lives. Once it exits, no return or next task is launched here.
            if dispatcher.active is not None and dispatcher.active.poll() is not None:
                dispatcher.active = None
        time.sleep(.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hold-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-physical-motion", action="store_true")
    args = parser.parse_args()
    if not args.execute or not args.acknowledge_physical_motion:
        raise RuntimeError("循环实测须从执行版GUI明确启动")
    os.environ["YOUXIAOMEI_MISSION_STATUS_OUTBOX"] = str(args.run_dir / "mission_events.jsonl")
    os.environ['YOUXIAOMEI_CYCLE_PARENT_RECOVERY'] = '1'
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def emit(value):
        protocol.write(json.dumps(dict(channel="button_cycle_worker", **value), ensure_ascii=False,
                                  allow_nan=False) + "\n")
        protocol.flush()

    controls = queue.Queue(maxsize=128)
    heartbeat_last = [time.monotonic()]
    control_cancel_epoch = [0]
    disconnected = threading.Event()
    reader_error = [None]

    def read_controls():
        try:
            while True:
                line = sys.stdin.readline(16385)
                if not line:
                    break
                if len(line) > 16384:
                    raise ValueError("控制消息过长")
                packet = json.loads(line)
                if not isinstance(packet, dict):
                    raise ValueError("控制消息必须为对象")
                if packet.get("command") in {"pause","abort_return","close","finish"}:
                    control_cancel_epoch[0]+=1
                if packet.get("command")=="heartbeat":
                    heartbeat_last[0]=max(heartbeat_last[0],control_timestamp(packet,time.monotonic(),3.))
                    continue
                try:
                    controls.put(packet, timeout=.2)
                except queue.Full:
                    if packet.get("command") != "heartbeat":
                        raise RuntimeError("控制消息积压")
        except Exception as exc:
            # Log the type only; malformed control content may be sensitive.
            reader_error[0] = type(exc).__name__
        finally:
            disconnected.set()

    interrupted = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: interrupted.set())
    signal.signal(signal.SIGTERM, lambda *_: interrupted.set())
    threading.Thread(target=read_controls, daemon=True).start()
    dispatcher = SessionDispatcher(execute=True, config_path=args.config, hold_report=args.hold_report)
    store = CycleStore(args.run_dir)
    os.environ["YOUXIAOMEI_CYCLE_AUTO_RECOVERY"]="1" if store.settings.failure_policy=="recover_wait" else "0"
    # Scope the demonstration timeout to numeric buttons in explicit cycle tests.
    os.environ["YOUXIAOMEI_CYCLE_NUMERIC_ON_TIMEOUT"] = "1" if store.settings.group == "内呼" else "0"
    gp = yaml.safe_load((ROOT / "gamepad_teleop.yaml").read_text())
    master = gp["reserved_ports"]["chassis_output"]["ros_master"]
    odom = Ros1TopicMonitor(master_uri=master, topic="/odom", message_type="nav_msgs/Odometry", decoder=decode_cycle_odometry)
    readiness = Readiness(dispatcher, store, odom)
    readiness.cancel_epoch = control_cancel_epoch
    emergency = None
    if store.settings.failure_policy=="recover_wait":
        def decode_emergency(payload):
            if len(payload)!=1 or payload[0] not in (0,1):raise ValueError("invalid emergency Bool")
            return {"engaged":bool(payload[0])}
        emergency=Ros1TopicMonitor(master_uri=master,topic="/emergency",
            message_type="std_msgs/Bool",decoder=decode_emergency)
        readiness.emergency=emergency
    runner = CycleRunner(store, dispatcher, readiness, emit)
    last_heartbeat = time.monotonic()
    closing = False
    guard = None
    guard_process = None
    transient_logged = False
    try:
        runner.publish()
        emit(dict(event="cycle_ready", directory=str(store.directory), rows=store.rows,
                  settings=store.plan["settings"]))
        while True:
            link_age = time.monotonic()-heartbeat_last[0]
            shutdown_reason = control_shutdown_reason(
                interrupted=interrupted.is_set(), disconnected=disconnected.is_set(),
                reader_error=reader_error[0], heartbeat_age=link_age)
            if not closing and shutdown_reason:
                closing = True
                append_json(store.directory / "control_events.jsonl", dict(
                    event="control_shutdown", reason=shutdown_reason,
                    interrupted=interrupted.is_set(), disconnected=disconnected.is_set(),
                    reader_error=reader_error[0], heartbeat_age_sec=link_age,
                    state=runner.state, monotonic_sec=time.monotonic()))
                runner.stop(closing=True, reason=shutdown_reason)
            while not closing:
                try:
                    packet = controls.get_nowait()
                except queue.Empty:
                    break
                command = packet.get("command")
                try:
                    if command == "heartbeat":
                        last_heartbeat = max(last_heartbeat, control_timestamp(packet, time.monotonic(), 3.))
                    elif command == "resume":
                        control_timestamp(packet, time.monotonic(), 1.5)
                        runner.resume()
                    elif command == "pause":
                        runner.pause()
                    elif command == "finish":
                        runner.finish()
                    elif command == "abort_return":
                        append_json(store.directory / "control_events.jsonl", dict(
                            event="abort_return_received", state=runner.state,
                            monotonic_sec=time.monotonic()))
                        runner.stop(return_origin=True)
                    elif command == "review":
                        row = store.review(packet.get("execution_id"), packet.get("value"))
                        emit(dict(event="cycle_result", row=row))
                        runner.publish()
                    elif command == "close":
                        closing = True
                        runner.stop(closing=True, reason="窗口关闭，取消后续测试")
                    else:
                        raise ValueError("未知循环测试命令")
                except Exception as exc:
                    if command in {"abort_return", "close"}:
                        runner.stop(closing=True, reason="终止/归位请求异常，取消后续测试：" + str(exc))
                        closing = True
                    emit(dict(event="cycle_error", message=str(exc)))
            if dispatcher.active is not None and not runner.aborting:
                sample = {}
                try:
                    if store.settings.failure_policy=="recover_wait":readiness.emergency_guard()
                    sample = odom.snapshot()
                    if guard_process is not dispatcher.active:
                        guard, guard_process = StationarityGuard(), dispatcher.active
                        transient_logged = False
                    if runner.state == 'recovering':
                        require_stationary(sample)
                    audit = guard.observe(sample, time.monotonic())
                    if audit['decision'] == 'transient_cleared' or (audit['decision'] == 'transient_pending' and not transient_logged):
                        append_json(store.directory / 'stationarity_events.jsonl', dict(task_id=getattr(dispatcher, '_active_task_id', None), **audit))
                        transient_logged = audit['decision'] == 'transient_pending'
                except Exception as exc:
                    append_json(store.directory / 'stationarity_events.jsonl', dict(task_id=getattr(dispatcher, '_active_task_id', None),
                        **getattr(exc, 'audit', dict(decision='stop', sample=sample, reason=str(exc)))))
                    if store.settings.failure_policy=="recover_wait" and not runner.pause_requested:
                        cause=emergency_stop_reason(exc)
                        runner.interrupt_for_recovery(str(exc),stop_reason=cause)
                    else:
                        runner.stop(closing=False, reason=str(exc))
            runner.tick()
            if closing and dispatcher.active is None:
                break
            time.sleep(.02)
        return 0
    except BaseException as exc:
        try:
            runner.stop(closing=True, reason="测试后端异常：" + str(exc))
        finally:
            emit(dict(event="cycle_error", message=str(exc)))
        raise
    finally:
        # Do not release ownership or leave a child moving on a backend error.
        stop_before_exit(dispatcher, readiness)
        dispatcher.release_session()
        readiness.close()
        odom.close()
        if emergency is not None:emergency.close()
        protocol.close()


if __name__ == "__main__":
    raise SystemExit(main())
