"""Cycle scheduling around the existing dispatcher; all hardware is injected."""
from __future__ import annotations

from datetime import datetime
import time
import uuid

from .button_cycle import TaskEvidence, append_json, atomic_json
from .voice_command import VoiceCommand, VoiceIntent
from .cycle_fault_recovery import (RECOVERY_STABLE_SEC,RETURN_REASSESS_SEC,FAULT_RECHECK_SEC,RECOVERY_POLL_SEC)


class CycleRunner:
    def __init__(self, store, dispatcher, readiness, emit, *, clock=time.monotonic):
        self.store, self.dispatcher = store, dispatcher
        self.readiness, self.emit, self.clock = readiness, emit, clock
        self.state = "paused"
        self.reason = "待明确点击开始或继续；不会自动运动"
        self.pause_requested = False
        self.current = None
        self.evidence = None
        self.next_at = 0.
        self.check_until = 0.
        self.stable_since = None
        self.next_check = 0.
        self.closing = False
        self.aborting = False
        self.return_requested = False
        self.recovery = None
        self.auto_recovery = None
        self.recovery_future = None
        self.restore_interrupted()

    def restore_interrupted(self):
        path = self.store.directory / "active.json"
        if path.exists():
            import json
            active = json.loads(path.read_text())
            if not any(row["execution_id"] == active["execution_id"] for row in self.store.rows):
                evidence = TaskEvidence(active.get("task_id"))
                self.store.append({**active, **evidence.result(None, started=0., finished=0., aborted=True),
                                   "reason": "上次执行中断，结果未核验；不会自动重发旧动作",
                                   "wall_sec": None,
                                   "finished_at": datetime.now().astimezone().isoformat()})
            path.unlink()

    def publish(self, state=None, reason=None):
        if state is not None:
            self.state = state
        if reason is not None:
            self.reason = reason
        summary = self.store.save(self.state, self.reason)
        self.emit(dict(event="cycle_state", state=self.state, reason=self.reason,
                       reserved=self.dispatcher.session_reserved, summary=summary,
                       current=self.current, pause_requested=self.pause_requested))

    def resume(self):
        if self.state not in {"paused", "ended", "completed"} or self.dispatcher.active is not None:
            raise RuntimeError("当前不能继续测试")
        pending_recovery = self.store.settings.failure_policy=="recover_wait" and (self.store.directory/"pending_recovery.json").exists()
        if self.store.cursor >= self.store.settings.total and not pending_recovery:
            self.publish("completed", "全部计划任务已结束")
            return
        self.dispatcher.reserve_session()
        self.pause_requested = False
        self.closing = self.aborting = self.return_requested = False
        self.recovery = None
        self.stable_since = None
        self.next_check = 0.
        self.check_until = self.clock() + 8.
        pending=self.store.directory/"pending_recovery.json"
        if self.store.settings.failure_policy=="recover_wait" and pending.exists():
            import json
            value=json.loads(pending.read_text())
            if value.get("next_index")!=self.store.cursor:
                raise RuntimeError("恢复断点与当前计划不匹配")
            self.auto_recovery=value
            self.next_check=self.clock()
            self.publish("fault_wait","已加载故障恢复断点；正在重新核验，不重发旧动作")
        else:
            self.publish("checking", "检查原位、相机、独立停止和底盘停稳状态")

    def pause(self):
        if self.state in {"fault_wait","fault_stabilizing","fault_planning","fault_stopping"}:
            self._cancel_auto_recovery()
            self.pause_requested=True
            self.publish("paused","已暂停自动恢复；进度和失败记录保留")
            return
        if self.state=="fault_returning":
            self.pause_requested=True
            self.reason="当前归位完成后暂停，不继续剩余任务"
            return
        if self.state == 'recovery_checking':
            self.recovery = None
            self.publish('paused', '已取消待执行的故障归位')
            return
        if self.state == "running":
            self.pause_requested = True
            self.publish(reason="当前按钮任务结束并归位后暂停")
        elif self.state in {"checking", "waiting_next"}:
            self.publish("paused", "已暂停，未启动下一个按钮")

    def finish(self):
        if self.dispatcher.active is not None:
            raise RuntimeError("当前按钮仍在执行，请先暂停或使用立即终止并归位")
        self.dispatcher.release_session()
        self.recovery = None
        self._cancel_auto_recovery()
        self.publish("ended", "本组已结束，剩余任务保留，可明确点击继续")

    def _append_current(self, values):
        if self.current is None:
            return
        row = {**self.current, **values, "finished_at": datetime.now().astimezone().isoformat()}
        self.store.append(row)
        (self.store.directory / "active.json").unlink(missing_ok=True)
        self.current = self.evidence = None
        self.emit(dict(event="cycle_result", row=row))

    def _abort_row(self):
        if self.current is not None:
            # stop() can drain the final dispatcher event internally. Read only
            # this task's durable status tail after its process has exited.
            from .mission_status import read_events
            rows, _ = read_events(self.dispatcher._status_outbox, 0)
            self.evidence.add(dict(mission_status_events=rows))
            values = self.evidence.result(None, started=self.current["started_monotonic_sec"],
                                          finished=self.clock(), aborted=True)
            self._append_current(values)

    def stop(self, *, return_origin=False, closing=False, reason="用户终止测试"):
        self._cancel_auto_recovery()
        self.recovery = None
        self.closing = self.closing or closing
        self.aborting = True
        self.pause_requested = True
        self.return_requested = return_origin and not self.closing
        self.reason = reason
        if self.return_requested:
            self.dispatcher.reserve_session()
            event = self.dispatcher.abort_and_return()
            self.state = "returning"
            if event.get("stop_error"):
                self.reason += "；独立停止异常：" + str(event["stop_error"])
        else:
            event = self.dispatcher.stop(reason="cycle_test_stopped") if self.dispatcher.active is not None else {}
            self.state = "stopping"
        if self.dispatcher.active is None:
            self._abort_row()
            self.dispatcher.release_session()
            self.state = "closed" if self.closing else "ended"
        self.publish(reason=self.reason)

    def _completed(self, event):
        if not self.aborting and self.state=="fault_stopping":
            if self.evidence is not None:self.evidence.add(event)
            if self.dispatcher.active is None:self._finish_fault_stop()
            return
        if not self.aborting and self.state=="fault_returning":
            if event.get("event")!="origin_operation_completed" or event.get("operation")!="return":
                raise RuntimeError("恢复归位收到不匹配的结束消息")
            try:
                if event.get("return_code")!=0:raise RuntimeError("正式归位流程失败")
                getattr(self.readiness,"verify_return",self.readiness)()
                self._auto_recovery_complete()
            except Exception as exc:
                self._auto_wait(str(exc),delay=RETURN_REASSESS_SEC)
            return
        if not self.aborting and self.state == 'recovering':
            if event.get('event') != 'origin_operation_completed' or event.get('operation') != 'return':
                raise RuntimeError('故障归位返回了不匹配的完成消息')
            self.recovery = None
            try:
                if event.get('return_code') != 0:
                    raise RuntimeError('正式归位流程失败')
                self.readiness()
            except Exception as exc:
                self.publish('paused', '故障归位未通过核验，保持暂停：' + str(exc))
            else:
                self.publish('paused', '故障后已核验归位；原失败结果保留，需明确点击继续')
            return
        if self.aborting:
            if event.get("event") == "abort_completed_return_started":
                self._abort_row()
                self.publish("returning", "旧任务已退出，正在按正式流程归位")
                return
            if event.get("event") == "origin_operation_completed":
                self._abort_row()
                ok = event.get("return_code") == 0
                try:
                    if not ok:
                        raise RuntimeError("归位程序执行失败")
                    getattr(self.readiness,"verify_return",self.readiness)()
                except Exception as exc:
                    self.dispatcher.release_session()
                    self.publish("ended", "终止完成，但归位未通过实时核验：" + str(exc))
                    return
                self.dispatcher.release_session()
                self.publish("closed" if self.closing else "ended", "已终止并核验归位；未自动继续")
                return
            if self.dispatcher.active is None:
                self._abort_row()
                self.dispatcher.release_session()
                self.publish("closed" if self.closing else "ended", self.reason)
            return
        if self.current is None or event.get("task_id") != self.current["task_id"]:
            raise RuntimeError("收到不匹配的任务结束消息，暂停测试")
        self.evidence.add(event)
        result = self.evidence.result(event.get("return_code"),
                    started=self.current["started_monotonic_sec"], finished=self.clock())
        candidate = self.evidence.recovery_candidate
        eligible = (candidate is not None and not self.evidence.recovery_veto
                    and self.evidence.terminal is not None
                    and not self.evidence.terminal.get('reporting_errors')
                    and self.evidence.terminal.get('failure_type') != 'TerminationRequested'
                    and type(event.get('return_code')) is int and event['return_code'] > 0
                    and not result['returned_to_origin'] and not self.closing
                    and not self.pause_requested and hasattr(self.readiness, 'recovery_probe'))
        failed_task_id = self.current['task_id']
        self._append_current(result)
        if (self.store.settings.failure_policy=="recover_wait"
                and not (result["full_cycle_success"] or result["initial_on_wait"] or result.get("on_timeout_skipped"))
                and not self.closing and not self.pause_requested):
            self._start_auto_recovery(failed_task_id,result["reason"])
            return
        # A missing/failed return or malformed terminal never authorizes a new
        # motion, even if the child returned zero or once reported ON.
        ordinary = result["safe_to_continue"] and self.store.settings.failure_policy == "continue_safe"
        if not (result["full_cycle_success"] or result["initial_on_wait"] or ordinary or result.get("on_timeout_skipped")):
            if eligible:
                self.recovery = dict(candidate, task_id=failed_task_id)
                self.stable_since, self.next_check = None, 0.
                self.check_until = self.clock()+4.
                self.publish('recovery_checking', '旧任务已退出；核验停稳、反馈和停止代号后受控归位')
            else:
                self.publish("paused", "已记录结果并暂停：" + result["result"] + "；" + result["reason"])
        elif self.store.cursor >= self.store.settings.total:
            self.dispatcher.release_session()
            self.publish("completed", "全部计划任务已结束；程序判定与人工实物核验分别统计")
        elif self.pause_requested:
            self.publish("paused", "当前按钮任务已结束并归位，已暂停")
        else:
            self.next_at = self.clock() + self.store.settings.interval_sec
            self.publish("waiting_next", "初始已亮按钮已等灭，重新测试同一项" if result["initial_on_wait"]
                         else "本项已记录，准备下一个按钮")

    def _cancel_auto_recovery(self):
        getattr(self.readiness,"end_recovery",lambda:None)()
        future=getattr(self,'recovery_future',None)
        if future is not None:future.cancel()
        self.recovery_future=None
        self.auto_recovery=None
        self.stable_since=None

    def _persist_auto_recovery(self):
        if self.auto_recovery is not None:
            atomic_json(self.store.directory/'pending_recovery.json',self.auto_recovery)

    def _start_auto_recovery(self, task_id, reason):
        self.auto_recovery=dict(failed_task_id=task_id,next_index=self.store.cursor,
                               reason=reason,return_attempts=0)
        self._persist_auto_recovery()
        append_json(self.store.directory/'events.jsonl',dict(event='fault_wait_started',**self.auto_recovery))
        self._auto_wait(reason,delay=0.)

    def _auto_wait(self, reason, *, delay=FAULT_RECHECK_SEC):
        self.stable_since=None
        self.recovery_future=None
        self.next_check=self.clock()+delay
        if self.auto_recovery is not None:
            self.auto_recovery['last_reason']=reason
            self._persist_auto_recovery()
        append_json(self.store.directory/'recovery_events.jsonl',dict(
            event='recovery_wait',reason=reason,next_evaluation_after_sec=delay,
            monotonic_sec=self.clock(),next_index=self.store.cursor))
        self.publish('fault_wait',f'等待故障恢复：{reason}；{delay:g}秒后复检，未发送运动')

    def interrupt_for_recovery(self, reason, *, stop_reason):
        if self.closing or self.aborting or self.pause_requested:
            self.stop(reason=reason);return
        self._cancel_auto_recovery()
        self.fault_interrupt_reason=reason
        self.fault_interrupt_task=self.current['task_id'] if self.current else None
        result=self.dispatcher.stop(reason=stop_reason) if self.dispatcher.active is not None else {}
        if self.evidence is not None and isinstance(result.get('completion'),dict):
            self.evidence.add(result['completion'])
        self.publish('fault_stopping','故障已请求停止，等待旧任务完全退出：'+reason)
        if self.dispatcher.active is None:self._finish_fault_stop()

    def _finish_fault_stop(self):
        if self.current is not None:
            from .mission_status import read_events
            rows,_=read_events(self.dispatcher._status_outbox,0)
            self.evidence.add(dict(mission_status_events=rows))
            values=self.evidence.result(-15,started=self.current['started_monotonic_sec'],finished=self.clock())
            values.update(result='失败',counted=True,full_cycle_success=False,safe_to_continue=False,
                          reason=self.fault_interrupt_reason)
            self._append_current(values)
        self._start_auto_recovery(self.fault_interrupt_task,self.fault_interrupt_reason)

    def _auto_recovery_complete(self):
        getattr(self.readiness,"end_recovery",lambda:None)()
        append_json(self.store.directory/'recovery_events.jsonl',dict(
            event='recovery_verified',monotonic_sec=self.clock(),next_index=self.store.cursor))
        self.auto_recovery=None;self.recovery_future=None;self.stable_since=None
        (self.store.directory/'pending_recovery.json').unlink(missing_ok=True)
        if self.pause_requested or self.closing:
            self.publish('paused','已核验归位，按用户请求暂停，不自动继续')
        elif self.store.cursor>=self.store.settings.total:
            self.dispatcher.release_session()
            self.publish('completed','全部计划项目已记录，故障后也已核验归位；失败记录保留')
        else:
            self.next_at=self.clock()+self.store.settings.interval_sec
            self.publish('waiting_next','已核验恢复和归位，继续下一未执行项目；原失败结果保留')

    def _tick_auto_recovery(self):
        if self.dispatcher.active is not None or self.auto_recovery is None:return
        now=self.clock()
        if self.state=='fault_planning':
            if self.recovery_future is None or not self.recovery_future.done():return
            try:
                proof=self.recovery_future.result()
                generation=self.readiness.authorize_auto_return(proof)
                self.auto_recovery['return_attempts']+=1
                self._persist_auto_recovery()
                started=self.dispatcher.dispatch_origin_operation('return',required_unlatched_generation=generation)
                if started.get('event')!='origin_return_started':raise RuntimeError('正式归位未确认启动')
                append_json(self.store.directory/'recovery_events.jsonl',dict(
                    event='auto_return_started',generation=generation,attempt=self.auto_recovery['return_attempts'],
                    monotonic_sec=self.clock()))
                self.recovery_future=None
                self.publish('fault_returning','按当前实测关节重新规划归位，完成后复检并继续')
            except Exception as exc:
                if self.dispatcher.active is not None:
                    self.interrupt_for_recovery(str(exc),stop_reason='cycle_feedback_unavailable')
                else:self._auto_wait(str(exc),delay=RETURN_REASSESS_SEC)
            return
        if now<self.next_check:return
        self.next_check=now+RECOVERY_POLL_SEC
        if self.stable_since is not None and now<self.stable_since+RECOVERY_STABLE_SEC:
            self.next_check=min(self.next_check,self.stable_since+RECOVERY_STABLE_SEC)
        try:
            probe=self.readiness.auto_recovery_probe()
            signature=probe['stop_identity']
            if self.auto_recovery.get('return_attempts',0)>=3:
                blocked=self.auto_recovery.get('retry_blocked_condition')
                observer_id=probe.get('communication_observer_id')
                if blocked is not None and blocked.get('communication_observer_id')!=observer_id:
                    # A restarted observer is not evidence of network recovery.
                    # Rebind its baseline without granting a new motion attempt.
                    blocked['communication_observer_id']=observer_id
                    blocked['communication_epoch']=probe.get('communication_epoch',0)
                    self._persist_auto_recovery()
                if blocked is None:
                    self.auto_recovery['retry_blocked_condition']=dict(
                        identity=dict(signature),joints=list(probe['joints']),
                        communication_epoch=probe.get('communication_epoch',0),
                        communication_observer_id=observer_id)
                    self._persist_auto_recovery()
                elif (blocked['identity']!=signature or
                      blocked.get('communication_epoch',0)!=probe.get('communication_epoch',0) or
                      max(abs(a-b) for a,b in zip(blocked['joints'],probe['joints']))>.003):
                    append_json(self.store.directory/'recovery_events.jsonl',dict(
                        event='recovery_condition_changed', previous_condition=blocked,
                        stop_identity=dict(signature), communication_epoch=probe.get('communication_epoch',0),
                        monotonic_sec=self.clock(), next_index=self.store.cursor))
                    self.auto_recovery['return_attempts']=0
                    self.auto_recovery.pop('retry_blocked_condition',None)
                    self._persist_auto_recovery()
                    self.stable_since=None
                if self.auto_recovery['return_attempts']>=3:
                    self._auto_wait('同条件下归位已失败3次；仅复检，等待通信恢复、停止状态或实际姿态发生有效变化',delay=RETURN_REASSESS_SEC)
                    return
            if self.stable_since is None or getattr(self,'recovery_signature',None)!=signature:
                self.stable_since=self.clock();self.recovery_signature=dict(signature)
                self.recovery_initial_joints=list(probe['joints'])
                self.recovery_last_camera=probe['camera_sequence']
                self.publish('fault_stabilizing','恢复条件已满足，核验持续稳定0.5秒')
                return
            if max(abs(a-b) for a,b in zip(probe['joints'],self.recovery_initial_joints))>.003:
                raise RuntimeError('等待恢复期间关节仍在变化')
            if self.clock()-self.stable_since<RECOVERY_STABLE_SEC:return
            if probe['camera_sequence']<=self.recovery_last_camera:
                raise RuntimeError('未取得独立的新相机帧')
            if probe['at_origin'] and not signature['latched']:
                self.readiness()
                self._auto_recovery_complete();return
            self.recovery_future=self.readiness.begin_return_preflight(probe)
            self.publish('fault_planning','纯规划并校验完整归位路径，尚未清锁或发运动')
        except Exception as exc:
            self._auto_wait(str(exc),delay=FAULT_RECHECK_SEC)

    def tick(self):
        event = self.dispatcher.refresh() if self.dispatcher.active is not None else None
        if event:
            if self.evidence is not None:
                self.evidence.add(event)
            if event.get("event") == "mission_progress":
                self.emit(dict(event="cycle_progress", message=event.get("message"),
                               task_id=event.get("task_id")))
            else:
                self._completed(event)
        if self.aborting:
            if self.dispatcher.active is None and self.state in {"stopping", "returning"}:
                self._abort_row()
                self.dispatcher.release_session()
                self.publish("closed" if self.closing else "ended", self.reason)
            return
        now = self.clock()
        if self.state in {"fault_wait","fault_stabilizing","fault_planning","fault_stopping"}:
            self._tick_auto_recovery()
            return
        if self.state == 'recovery_checking' and now >= self.next_check:
            self.next_check = now+.1
            try:
                if self.dispatcher.active is not None or self.recovery is None:
                    raise RuntimeError('旧任务尚未退出或归位已取消')
                generation = self.recovery['safety_generation']
                self.readiness.recovery_probe(generation)
                checked = self.clock()
                if self.stable_since is None:
                    self.stable_since = checked
                if checked < self.check_until and checked-self.stable_since >= .5:
                    started = self.dispatcher.dispatch_origin_operation('return', required_unlatched_generation=generation)
                    if started.get('event') != 'origin_return_started':
                        raise RuntimeError('正式归位入口未确认启动')
                    append_json(self.store.directory / 'events.jsonl', dict(event='controlled_recovery_started', **self.recovery))
                    self.publish('recovering', '正在按正式流程归位；不会清除停止锁或自动重按')
            except Exception as exc:
                # A failed gate is not permission to retry after somebody
                # clears a stop. Remain paused until explicit user action.
                self.recovery = None
                if self.dispatcher.active is not None:
                    self.stop(reason='故障归位交接失败：' + str(exc))
                else:
                    self.publish('paused', '故障归位检查未通过，保持暂停：' + str(exc))
            if self.clock() >= self.check_until and self.state == 'recovery_checking':
                self.recovery = None
                self.publish('paused', '故障归位检查超时，保持暂停')
        if self.state == "checking" and now >= self.next_check:
            self.next_check = now + .15
            try:
                self.readiness()
                if self.stable_since is None:
                    self.stable_since = now
                if now - self.stable_since >= .5:
                    self.next_at = now
                    self.publish("waiting_next", "检查通过，准备启动本项")
            except Exception as exc:
                self.stable_since = None
                self.reason = str(exc)
            if now >= self.check_until and self.state == "checking":
                if self.store.settings.failure_policy=="recover_wait" and not self.pause_requested:
                    self._start_auto_recovery(None,"启动检查未通过："+self.reason)
                else:self.publish("paused", "启动检查未通过：" + self.reason)
        if self.state == "waiting_next" and now >= self.next_at:
            try:
                self.readiness()
            except Exception as exc:
                if self.store.settings.failure_policy=="recover_wait" and not self.pause_requested:
                    self._start_auto_recovery(None,"下一项启动检查未通过："+str(exc))
                else:self.publish("paused", "下一项启动检查未通过：" + str(exc))
                return
            item = self.store.settings.item(self.store.cursor)
            button = int(item["button"]) if item["button"].isdecimal() else item["button"]
            command = VoiceCommand(VoiceIntent.PRESS_BUTTON, str(button), str(button), button=button)
            started = self.clock()
            try:
                dispatched = self.dispatcher.dispatch_motion(command)
                if dispatched.get("event") != "motion_started" or not dispatched.get("task_id"):
                    raise RuntimeError("正式执行入口没有确认任务已启动")
                self.current = dict(**item, execution_id=str(uuid.uuid4()), task_id=dispatched["task_id"],
                                    started_monotonic_sec=started,
                                    started_at=datetime.now().astimezone().isoformat())
                self.evidence = TaskEvidence(self.current["task_id"])
                atomic_json(self.store.directory / "active.json", self.current)
                append_json(self.store.directory / "events.jsonl", dict(event="task_started", **self.current))
                self.publish("running", "执行正式按压闭环")
            except Exception:
                # Failure after launch must stop that child; never just mark the
                # GUI paused while an untracked arm task is still moving.
                if self.dispatcher.active is not None:
                    self.stop(closing=False, reason="任务启动或记录失败，已请求安全停止")
                else:
                    self.publish("paused", "任务未能启动，请检查执行日志")
                raise
