#!/usr/bin/env python3
"""JSON-only navigation adapter for the existing production arm dispatcher."""
import argparse
import collections
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_voice_command_gateway import CommandDispatcher, DEFAULT_HOLD_REPORT
from piper_handeye.voice_command import VoiceCommand, VoiceIntent
from piper_handeye.external_call_mission import SUCCESS_FLOWS, direction_for
from piper_handeye.mission_status import read_events
from piper_handeye.arm_report import plain_data


def arm_result(records, return_code):
    terminal=next((r for r in reversed(records) if r.get('producer')=='arm_pressing'
                   and r.get('raw_event')=='task_terminal'),{})
    evidence=terminal.get('evidence') or {}
    origin=evidence.get('origin_verification') or {}
    phases={r.get('phase') for r in records if r.get('producer')=='arm_pressing'}
    result = dict(return_code=return_code,on_confirmed='button_on_confirmed' in phases,
                off_confirmed='floor_arrival_observed' in phases,
                off_wait_timed_out='external_call_timeout_release' in phases,
                flow_status=evidence.get('flow_status'),
                returned_to_origin=evidence.get('returned_to_origin') is True,
                origin_verified=origin.get('verified') is True and origin.get('hardware_ok')==1)
    for key in ('origin_verification', 'press_execution_started', 'press_attempts',
                'failure_stage', 'failure_code', 'failure_type', 'failure_reason', 'reporting_errors',
                'external_call_off_wait_timeout'):
        if key in evidence and evidence[key] is not None:
            result[key] = evidence[key]
    return plain_data(result)


def check_origin(dispatcher):
    if dispatcher.active is not None:raise RuntimeError('机械臂任务尚未退出')
    dispatcher._acquire_motion_lock()
    try:
        p=subprocess.run([sys.executable,str(ROOT/'run_origin_position_check.py'),
                          '--config',str(dispatcher.config_path),
                          '--hold-report',str(dispatcher.hold_report)],
                         cwd=ROOT,capture_output=True,text=True,timeout=12)
        rows=[]
        for line in p.stdout.splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
        result=next((r for r in reversed(rows) if r.get('event')=='origin_check_result'),None)
        if p.returncode or result is None:raise RuntimeError('原位检查失败：'+(p.stderr or p.stdout)[-700:])
        return result
    finally:dispatcher._release_motion_lock()


class Worker:
    def __init__(self, dispatcher, emit, origin_reader=check_origin):
        self.dispatcher=dispatcher;self.emit=emit;self.origin_reader=origin_reader
        self.press=None;self.records=[];self.completed=collections.OrderedDict()

    def reply(self, request, event, **values):
        message=dict(channel='navigation_arm_worker',request_id=request['request_id'],
                     mission_id=request['mission_id'],operation=request['operation'],event=event,**values)
        if event in {'completed','error'}:
            self.completed[request['request_id']]=message
            while len(self.completed)>256:self.completed.popitem(last=False)
        self.emit(message)

    def handle(self, request):
        for key in ('request_id','mission_id','operation'):
            if not isinstance(request.get(key),str) or not request[key]:raise ValueError('请求缺少'+key)
        if request['request_id'] in self.completed:
            self.emit(self.completed[request['request_id']]);return
        if self.press and request['request_id']==self.press['request_id']:return
        op=request['operation']
        try:
            if op=='stop':
                result=self.dispatcher.stop(reason='navigation_mission_stopped') if self.dispatcher.active is not None else {'event':'idle'}
                if result.get('event')=='stop_pending':
                    raise RuntimeError(str(result))
                if self.press:
                    self.reply(self.press,'error',error='任务已停止');self.press=None;self.records=[]
                if result.get('error'):raise RuntimeError(str(result))
                self.reply(request,'completed',result=result)
            elif op=='check_origin':
                self.reply(request,'completed',result=self.origin_reader(self.dispatcher))
            elif op=='return_origin':
                if self.press or self.dispatcher.active is not None:raise RuntimeError('机械臂已有任务')
                started=self.dispatcher.dispatch_origin_operation('return')
                self.press=request;self.records=[];self.reply(request,'started',result=started)
            elif op=='press':
                if self.press or self.dispatcher.active is not None:raise RuntimeError('机械臂已有任务')
                expected=direction_for(request.get('current_floor'),request.get('target_floor'))
                if request.get('direction')!=expected:raise ValueError('上下行方向与楼层不一致')
                self.dispatcher.set_current_floor(request['current_floor'])
                cmd=VoiceCommand(VoiceIntent.PRESS_BUTTON,expected,expected,button=expected)
                started=self.dispatcher.dispatch_motion(cmd)
                self.press=request;self.records=[]
                self.reply(request,'started',result=started)
            elif op=='press_floor':
                if self.press or self.dispatcher.active is not None:raise RuntimeError('机械臂已有任务')
                target=request.get('target_floor')
                if type(target) is not int or not 1<=target<=16:
                    raise ValueError('目标楼层必须为1–16的整数')
                self.dispatcher.set_current_floor(request.get('current_floor'))
                cmd=VoiceCommand(VoiceIntent.PRESS_BUTTON,f'按下{target}楼按钮',f'按下{target}楼按钮',button=target)
                started=self.dispatcher.dispatch_motion(cmd)
                self.press=request;self.records=[]
                self.reply(request,'started',result=started)
            else:raise ValueError('不支持的机械臂操作')
        except Exception as exc:self.reply(request,'error',error=type(exc).__name__+': '+str(exc))

    def refresh(self):
        if not self.press:return
        result=self.dispatcher.refresh()
        if result is None:return
        records=result.get('mission_status_events') or []
        self.records.extend(records)
        if result.get('event')=='origin_operation_completed':
            self.reply(self.press,'completed',result={'return_code':result.get('return_code')})
            self.press=None;self.records=[]
        elif result.get('event')=='motion_completed':
            # The child's terminal events may be appended between the
            # dispatcher's initial outbox read and its process-exit check.
            # Collect only this task's remaining evidence before replying.
            path=getattr(self.dispatcher,'_status_outbox',None)
            offset=getattr(self.dispatcher,'_status_cursor',None)
            task_id=result.get('task_id')
            if isinstance(path,Path) and type(offset) is int and isinstance(task_id,str):
                tail,self.dispatcher._status_cursor=read_events(path,offset)
                self.records.extend(r for r in tail if r.get('task_id')==task_id)
            self.reply(self.press,'completed',result=arm_result(self.records,result.get('return_code')))
            self.press=None;self.records=[]
        else:self.reply(self.press,'progress',result=result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=True)
    os.environ['YOUXIAOMEI_MISSION_STATUS_OUTBOX']=str(args.run_dir/'arm_mission_events.jsonl')
    # Child motion programs inherit stderr for logs; only this duplicate fd
    # carries JSON replies. YAML/stdout from the production pipeline cannot
    # masquerade as worker protocol packets.
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    def emit(m):protocol.write(json.dumps(m,ensure_ascii=False)+'\n');protocol.flush()
    dispatcher=CommandDispatcher(execute=True,config_path=ROOT/'piper_button_pose.yaml',hold_report=DEFAULT_HOLD_REPORT)
    worker=Worker(dispatcher,emit);requests=queue.Queue();closing=threading.Event()
    def read():
        while True:
            line=sys.stdin.readline(65537)
            if not line:requests.put(None);return
            if len(line)>65536:requests.put(None);return
            try:requests.put(json.loads(line))
            except ValueError:requests.put(None);return
    threading.Thread(target=read,daemon=True).start()
    signal.signal(signal.SIGTERM,lambda *_:closing.set());signal.signal(signal.SIGINT,lambda *_:closing.set())
    try:
        while not closing.is_set():
            try:r=requests.get(timeout=.1)
            except queue.Empty:r={}
            if r is None:break
            if r:
                try:worker.handle(r)
                except Exception as exc:
                    print('worker protocol error:',exc,file=sys.stderr);break
            worker.refresh()
    finally:
        if dispatcher.active is not None:dispatcher.stop(reason='navigation_arm_worker_closed')
        protocol.close()


if __name__=='__main__':main()
