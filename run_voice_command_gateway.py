#!/usr/bin/env python3
"""Convert ASR transcripts into confirmed, allow-listed Piper commands."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from piper_handeye.safety_stop import SafetyStopUdpClient
from piper_handeye.mission_status import (
    DEFAULT_OUTBOX as DEFAULT_MISSION_STATUS_OUTBOX,
    append_status as append_mission_status,
    new_task_id,
    read_events as read_mission_status_events,
)
from piper_handeye.mission_status import describe_phase_zh
from piper_handeye.voice_command import (
    DecisionStatus,
    VoiceCommand,
    VoiceCommandSession,
    VoiceIntent,
    parse_voice_command,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_HOLD_REPORT = ROOT / "button_fast_chain_runs/20260826_150725/chain_report.yaml"


class CommandDispatcher:
    STOP_WAIT_TIMEOUT_SEC = 45.0

    def __init__(self, *, execute: bool, config_path: Path, hold_report: Path,
                 origin_return_bridge_handoff: bool = False):
        self.execute = bool(execute)
        self.origin_return_bridge_handoff = bool(origin_return_bridge_handoff)
        self.config_path = config_path.resolve()
        self.hold_report = hold_report.resolve()
        self.active: subprocess.Popen | None = None
        self.active_command: VoiceCommand | None = None
        self.active_operation: str | None = None
        self.return_after_abort = False
        self.current_floor: int | None = None
        self.trip_destination: int | None = None
        self.trip_stage: str | None = None
        self.mission_id: str | None = None
        self._active_task_id: str | None = None
        self._active_segment_id: str | None = None
        self._status_outbox = Path(os.environ.get(
            "YOUXIAOMEI_MISSION_STATUS_OUTBOX", str(DEFAULT_MISSION_STATUS_OUTBOX)
        ))
        self._status_cursor = 0
        self._arm_terminal_record = None
        self._motion_lock_stream = None
        self._motion_lock_path = Path(
            os.environ.get(
                "YOUXIAOMEI_ROBOT_MOTION_LOCK",
                "/tmp/youxiaomei_robot_motion.lock",
            )
        )

    def _acquire_motion_lock(self) -> None:
        if not self.execute or self._motion_lock_stream is not None:
            return
        stream = self._motion_lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise RuntimeError("另一控制界面正在占用机械臂任务执行权")
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        self._motion_lock_stream = stream

    def _release_motion_lock(self) -> None:
        stream = self._motion_lock_stream
        if stream is None:
            return
        self._motion_lock_stream = None
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def _motion_argv(self, command: VoiceCommand) -> list[str]:
        if command.intent is VoiceIntent.PRESS_BUTTON:
            return [
                sys.executable,
                str(ROOT / "run_button_batch.py"),
                str(command.button),
                "--config", str(self.config_path),
                "--hold-report", str(self.hold_report),
            ]
        if command.intent is VoiceIntent.RETURN_ORIGIN:
            return [
                sys.executable,
                str(ROOT / "run_clearance_return_to_origin.py"),
                "--hold-report",
                str(self.hold_report),
                "--config",
                str(self.config_path),
                "--execute",
                "--acknowledge-clear-workspace",
            ]
        raise ValueError(f"not a motion command: {command.intent.value}")

    def refresh(self) -> dict | None:
        if self.active is None:
            return None
        new_records, self._status_cursor = read_mission_status_events(
            self._status_outbox, self._status_cursor
        )
        task_records = [
            value for value in new_records
            if self._active_task_id is not None
            and value.get("task_id") == self._active_task_id
        ]
        return_code = self.active.poll()
        for value in task_records:
            if value.get('producer') == 'arm_pressing' and value.get('raw_event') == 'task_terminal':
                self._arm_terminal_record = value
        if return_code is None:
            if not task_records:
                return None
            return {
                "event": "mission_progress",
                "task_id": self._active_task_id,
                "mission_id": self.mission_id,
                "mission_status_events": task_records,
                "message": describe_phase_zh(task_records[-1]),
            }
        # The child can append its terminal evidence between the first read
        # and poll(). Drain the completed tail without waiting or resubmitting
        # any motion, and keep other tasks' records out of this completion.
        tail, self._status_cursor = read_mission_status_events(
            self._status_outbox, self._status_cursor)
        task_records.extend(value for value in tail
                            if self._active_task_id is not None
                            and value.get("task_id") == self._active_task_id)
        operation = self.active_operation
        for value in task_records:
            if value.get('producer') == 'arm_pressing' and value.get('raw_event') == 'task_terminal':
                self._arm_terminal_record = value
        trip_stage = self.trip_stage
        trip_destination = self.trip_destination
        button = self.active_command.button if self.active_command else None
        task_id = self._active_task_id
        segment_id = self._active_segment_id
        if self.return_after_abort:
            aborted_return_code = return_code
            self.active = None
            self.active_command = None
            self.active_operation = None
            self._active_task_id = None
            self._active_segment_id = None
            self.return_after_abort = False
            started = self._start_origin_operation("return")
            started.update({
                "event": "abort_completed_return_started",
                "aborted_process_return_code": aborted_return_code,
            })
            return started
        completed = {
            "event": (
                "origin_operation_completed" if operation else "motion_completed"
            ),
            "operation": operation,
            "intent": self.active_command.intent.value if self.active_command else None,
            "button": self.active_command.button if self.active_command else None,
            "trip_stage": trip_stage,
            "return_code": return_code,
            "task_id": task_id,
            "mission_id": self.mission_id,
            "segment_id": segment_id,
            "current_floor": self.current_floor,
            "destination_floor": trip_destination if trip_stage else (
                button if isinstance(button, int) else None
            ),
            "mission_status_events": task_records,
        }
        terminal = getattr(self, '_arm_terminal_record', None)
        if isinstance(terminal, dict) and terminal.get('task_id') == task_id:
            # The terminal row may have been delivered in an earlier progress poll.
            completed['arm_report'] = terminal.get('evidence') or {}
        append_mission_status(self._status_outbox, {
            "task_id": task_id,
            "mission_id": self.mission_id or task_id,
            "segment_id": segment_id,
            "producer": "dispatcher",
            "phase": (
                {"call": "call_button_segment_completed",
                 "floor": "target_floor_button_segment_completed"}.get(trip_stage)
                or ("arm_button_task_completed" if return_code == 0 else "failed")
            ) if return_code == 0 else "failed",
            "outcome": "completed" if return_code == 0 else "failed",
            "reason_code": None if return_code == 0 else "ARM_PROCESS_NONZERO_EXIT",
            "evidence": {"return_code": return_code, "button": button},
            "current_floor": self.current_floor,
            "destination_floor": trip_destination if trip_stage else (
                button if isinstance(button, int) else None
            ),
            "raw_event": "dispatcher_process_exited",
        })
        self.active = None
        self.active_command = None
        self.active_operation = None
        self._active_task_id = None
        self._active_segment_id = None
        if trip_stage == "call" and return_code == 0:
            self.trip_stage = "waiting_square"
            wait_record = append_mission_status(self._status_outbox, {
                "task_id": task_id,
                "mission_id": self.mission_id or task_id,
                "segment_id": "operator_transition",
                "target": trip_destination,
                "current_floor": self.current_floor,
                "destination_floor": trip_destination,
                "producer": "dispatcher",
                "phase": "awaiting_operator_inside_elevator",
                "outcome": "waiting",
                "reason_code": None,
                "evidence": {"call_button_segment_return_code": return_code},
                "raw_event": "trip_waiting_for_operator",
            })
            return {
                "event": "trip_waiting_for_square",
                "from_floor": self.current_floor,
                "button": trip_destination,
                "return_code": return_code,
                "task_id": task_id,
                "mission_id": self.mission_id,
                "mission_status_events": [*task_records, wait_record],
                "message": describe_phase_zh(wait_record),
            }
        if trip_stage == "call":
            self.trip_stage = None
            self.trip_destination = None
            self.mission_id = None
        if trip_stage == "floor":
            self.trip_stage = None
            self.trip_destination = None
            if return_code == 0 and isinstance(button, int):
                self.current_floor = button
            self.mission_id = None
        elif trip_stage is None:
            self.mission_id = None
        self._release_motion_lock()
        return completed

    def dispatch_motion(self, command: VoiceCommand) -> dict:
        self.refresh()
        if self.active is not None:
            raise RuntimeError("已有机械臂任务正在执行")
        argv = self._motion_argv(command)
        if not self.execute:
            return {"event": "dry_run", "argv": argv}
        # A dedicated process group lets emergency stop terminate the fixed
        # command and any descendants after the independent hardware stop has
        # been requested.
        self._acquire_motion_lock()
        task_id = new_task_id()
        mission_id = self.mission_id or task_id
        if self.trip_stage not in {"call", "floor"}:
            mission_id = task_id
        segment_id = (
            "call" if self.trip_stage == "call"
            else "target_floor" if self.trip_stage == "floor"
            else "arm_task"
        )
        try:
            self._status_cursor = self._status_outbox.stat().st_size
        except FileNotFoundError:
            self._status_cursor = 0
        env = os.environ.copy()
        destination_floor = (
            self.trip_destination
            if self.trip_stage in {"call", "floor"}
            else int(command.button)
            if isinstance(command.button, int)
            else None
        )
        env.update({
            "YOUXIAOMEI_TASK_ID": task_id,
            "YOUXIAOMEI_MISSION_ID": mission_id,
            "YOUXIAOMEI_SEGMENT_ID": segment_id,
            "YOUXIAOMEI_MISSION_STATUS_OUTBOX": str(self._status_outbox),
            "YOUXIAOMEI_DESTINATION_FLOOR": (
                "" if destination_floor is None else str(destination_floor)
            ),
            "YOUXIAOMEI_CURRENT_FLOOR": (
                "" if self.current_floor is None else str(self.current_floor)
            ),
        })
        try:
            append_mission_status(self._status_outbox, {
                "task_id": task_id,
                "mission_id": mission_id,
                "segment_id": segment_id,
                "target": command.button,
                "producer": "dispatcher",
                "phase": "accepted",
                "outcome": "accepted",
                "reason_code": None,
                "evidence": {"intent": command.intent.value},
                "current_floor": self.current_floor,
                "destination_floor": destination_floor,
                "raw_event": "task_accepted",
            })
            self._status_cursor = self._status_outbox.stat().st_size
            self.active = subprocess.Popen(
                argv, cwd=ROOT, start_new_session=True, env=env,
                stdin=subprocess.DEVNULL
            )
        except Exception:
            append_mission_status(self._status_outbox, {
                "task_id": task_id,
                "mission_id": mission_id,
                "segment_id": segment_id,
                "producer": "dispatcher",
                "phase": "failed",
                "outcome": "failed",
                "reason_code": "ARM_PROCESS_START_FAILED",
                "raw_event": "task_start_failed",
            })
            self._release_motion_lock()
            raise
        self.active_command = command
        self.active_operation = None
        self._active_task_id = task_id
        self._active_segment_id = segment_id
        self.mission_id = mission_id
        append_mission_status(self._status_outbox, {
            "task_id": task_id, "mission_id": mission_id,
            "segment_id": segment_id, "target": command.button,
            "producer": "dispatcher", "phase": "arm_process_started",
            "outcome": "active", "reason_code": None,
            "evidence": {"pid": self.active.pid, "intent": command.intent.value},
            "current_floor": self.current_floor,
            "destination_floor": destination_floor,
            "raw_event": "arm_process_started",
        })
        return {"event": "motion_started", "pid": self.active.pid, "argv": argv,
                "task_id": task_id, "mission_id": mission_id,
                "segment_id": segment_id,
                "current_floor": self.current_floor,
                "destination_floor": destination_floor}

    def set_current_floor(self, value) -> None:
        if value in (None, ""):
            self.current_floor = None
            return
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16:
            raise ValueError("当前楼层必须是 1–16 的整数")
        self.current_floor = value

    def start_trip(self, command: VoiceCommand) -> dict:
        """Call the lift, wait for the operator to drive inside, then press target."""
        target = command.button
        if not isinstance(target, int) or not 1 <= target <= 16:
            raise ValueError("分段乘梯任务只支持数字目标楼层")
        if self.current_floor is None:
            raise RuntimeError("请先在 GUI 设定当前测试楼层，再开始乘梯任务")
        if self.current_floor == target:
            raise RuntimeError("目标楼层与当前楼层相同，无需呼梯")
        if self.active is not None or self.trip_stage is not None:
            raise RuntimeError("已有机械臂任务或分段乘梯任务正在执行")
        direction = "up" if target > self.current_floor else "down"
        if not self.execute:
            return {
                "event": "dry_run",
                "trip_destination": target,
                "current_floor": self.current_floor,
                "call_button": direction,
            }
        self.trip_destination = target
        self.mission_id = new_task_id()
        self.trip_stage = "call"
        call = VoiceCommand(
            VoiceIntent.PRESS_BUTTON,
            f"呼叫电梯{direction}", f"呼叫电梯{direction}", button=direction,
        )
        try:
            started = self.dispatch_motion(call)
        except Exception:
            self.trip_destination = None
            self.trip_stage = None
            self.mission_id = None
            raise
        started.update({
            "trip_destination": target,
            "trip_stage": "call",
            "current_floor": self.current_floor,
            "call_button": direction,
        })
        return started

    def resume_trip(self) -> dict:
        if self.trip_stage != "waiting_square" or self.trip_destination is None:
            raise RuntimeError("当前没有等待 □ 键继续的乘梯任务")
        target = self.trip_destination
        command = VoiceCommand(
            VoiceIntent.PRESS_BUTTON, f"按下{target}楼按钮", f"按下{target}楼按钮",
            button=target,
        )
        self.trip_stage = "floor"
        try:
            started = self.dispatch_motion(command)
        except Exception:
            self.trip_stage = "waiting_square"
            raise
        started.update({"trip_stage": "floor", "trip_destination": target})
        return started

    def dispatch_origin_operation(self, operation: str, *, required_unlatched_generation=None) -> dict:
        """Serialize GUI origin controls with every other robot operation."""
        self.refresh()
        if self.active is not None:
            raise RuntimeError("已有机械臂任务正在执行")
        if self.trip_stage is not None:
            raise RuntimeError("分段乘梯任务待继续；请按 □ 继续或使用中止返原位")
        if required_unlatched_generation is None:
            return self._start_origin_operation(operation)
        return self._start_origin_operation(operation, required_unlatched_generation=required_unlatched_generation)

    def _start_origin_operation(self, operation: str, *, required_unlatched_generation=None) -> dict:
        if operation == "check":
            argv = [
                sys.executable,
                str(ROOT / "run_origin_position_check.py"),
                "--config", str(self.config_path),
                "--hold-report", str(self.hold_report),
            ]
        elif operation == "return":
            argv = [
                sys.executable,
                str(ROOT / "run_direct_return_to_origin.py"),
                "--config", str(self.config_path),
                "--hold-report", str(self.hold_report),
                "--execute",
                "--acknowledge-clear-workspace",
            ]
            if self.origin_return_bridge_handoff:
                argv.append("--bridge-mode-handoff")
            if required_unlatched_generation is not None:
                if type(required_unlatched_generation) is not int or required_unlatched_generation < 0:
                    raise ValueError('invalid recovery safety generation')
                argv += ['--require-unlatched-generation', str(required_unlatched_generation)]
            if not self.execute:
                return {"event": "origin_return_preview"}
        else:
            raise ValueError(f"unknown origin operation: {operation}")
        self._acquire_motion_lock()
        try:
            # Motion children must not share the GUI command pipe. In particular,
            # a nested SSH client can consume it or toggle its file status flags.
            self.active = subprocess.Popen(
                argv, cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL
            )
        except Exception:
            self._release_motion_lock()
            raise
        self.active_command = None
        self.active_operation = operation
        return {
            "event": f"origin_{operation}_started",
            "operation": operation,
            "pid": self.active.pid,
        }

    def abort_and_return(self) -> dict:
        """Stop the active process, wait for its exit, then return to origin."""
        self.refresh()
        self.trip_destination = None
        self.trip_stage = None
        self.mission_id = None
        if not self.execute:
            return {"event": "abort_return_preview"}
        if self.active is None:
            return self._start_origin_operation("return")
        if self.active_operation == "return":
            return {"event": "abort_return_already_returning"}

        process = self.active
        task_id = self._active_task_id
        mission_id = self.mission_id or task_id
        segment_id = self._active_segment_id
        response = None
        error = None
        # A read-only encoder check has no hardware motion to stop. Every other
        # active operation is stopped independently before its process receives
        # SIGTERM. Retaining ``self.active`` prevents the return from starting
        # until that process and all its cleanup have exited.
        if self.active_operation != "check":
            config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            stop_cfg = config["safety_stop"]
            try:
                response = SafetyStopUdpClient(
                    stop_cfg["host"], int(stop_cfg["port"]), timeout_sec=2.0
                ).stop(
                    token=str(stop_cfg["token"]),
                    reason="gui_abort_and_return",
                    frame_sequence=0,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if task_id is not None:
                append_mission_status(self._status_outbox, {
                    "task_id": task_id, "mission_id": mission_id,
                    "segment_id": segment_id, "producer": "dispatcher",
                    "phase": "abort_requested", "outcome": "active",
                    "reason_code": "OPERATOR_ABORT_AND_RETURN",
                    "raw_event": "abort_return_requested",
                })
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        self.return_after_abort = True
        result = {
            "event": "abort_return_started",
            "pid": process.pid,
            "response": response,
        }
        if error is not None:
            result["stop_error"] = error
        return result

    def stop(self, reason: str = "voice_emergency_stop") -> dict:
        process = self.active
        # A plain stop cancels a previously requested abort-and-return chain.
        # Otherwise refresh() would launch a new return process immediately
        # after the stopped child exits.
        self.return_after_abort = False
        task_id = self._active_task_id
        mission_id = self.mission_id or task_id
        segment_id = self._active_segment_id
        if task_id is not None:
            append_mission_status(self._status_outbox, {
                "task_id": task_id, "mission_id": mission_id,
                "segment_id": segment_id, "producer": "dispatcher",
                "phase": "abort_requested", "outcome": "active",
                "reason_code": str(reason).upper(),
                "raw_event": "task_abort_requested",
            })
        response = None
        error = None
        if self.execute:
            config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            stop_cfg = config["safety_stop"]
            try:
                response = SafetyStopUdpClient(
                    stop_cfg["host"], int(stop_cfg["port"]), timeout_sec=2.0
                ).stop(
                    token=str(stop_cfg["token"]),
                    reason=reason,
                    frame_sequence=0,
                )
            except Exception as exc:  # Still terminate the local command producer.
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if process is not None and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        completion = None
        if process is not None:
            try:
                process.wait(timeout=self.STOP_WAIT_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                # Keep both the process reference and the file lock.  Releasing
                # either here would allow a new motion while the old command is
                # still cleaning up or owns a bridge trajectory.
                result = {
                    "event": "stop_pending",
                    "dry_run": not self.execute,
                    "response": response,
                    "pid": process.pid,
                    "message": "旧任务仍在安全停止中，控制权保持锁定",
                }
                if error is not None:
                    result["error"] = error
                return result
            completion = self.refresh()
        else:
            self.active_command = None
            self.active_operation = None
            self._active_task_id = None
            self._active_segment_id = None
            self.return_after_abort = False
            self._release_motion_lock()
        self.trip_destination = None
        self.trip_stage = None
        self.mission_id = None
        result = {
            "event": "stop",
            "dry_run": not self.execute,
            "response": response,
            "completion": completion,
        }
        if error is not None:
            result["error"] = error
        return result


def parse_transcript_line(line: str) -> tuple[str, float]:
    stripped = line.strip()
    if not stripped:
        return "", 1.0
    if stripped.startswith("{"):
        value = json.loads(stripped)
        if not isinstance(value, dict) or "text" not in value:
            raise ValueError("JSON transcript requires a text field")
        return str(value["text"]), float(value.get("confidence", 1.0))
    return stripped, 1.0


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", help="transcript; repeat for a dialogue")
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--hold-report", type=Path, default=DEFAULT_HOLD_REPORT)
    parser.add_argument("--confirmation-timeout-sec", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-physical-motion", action="store_true")
    args = parser.parse_args()
    if args.execute != args.acknowledge_physical_motion:
        raise RuntimeError(
            "physical execution requires both --execute and "
            "--acknowledge-physical-motion"
        )

    session = VoiceCommandSession(
        confirmation_timeout_sec=args.confirmation_timeout_sec
    )
    dispatcher = CommandDispatcher(
        execute=args.execute,
        config_path=args.config,
        hold_report=args.hold_report,
    )
    lines = args.text if args.text is not None else sys.stdin
    emit({"event": "ready", "mode": "execute" if args.execute else "dry_run"})
    try:
        for line in lines:
            completed = dispatcher.refresh()
            if completed is not None:
                emit(completed)
            try:
                text, confidence = parse_transcript_line(line)
                incoming = parse_voice_command(text)
                if (
                    dispatcher.active is not None
                    and incoming.intent not in (VoiceIntent.STOP, VoiceIntent.CANCEL)
                ):
                    emit(
                        {
                            "event": "decision",
                            "status": DecisionStatus.REJECTED.value,
                            "message": "机械臂任务正在执行；当前只接受停止或取消",
                            "intent": incoming.intent.value,
                            "button": incoming.button,
                        }
                    )
                    continue
                decision = session.submit(text, confidence=confidence)
                output = {
                    "event": "decision",
                    "status": decision.status.value,
                    "message": decision.message,
                    "intent": decision.command.intent.value if decision.command else None,
                    "button": decision.command.button if decision.command else None,
                }
                emit(output)
                if decision.status is DecisionStatus.STOP:
                    emit(dispatcher.stop())
                elif decision.status is DecisionStatus.CANCELLED and dispatcher.active:
                    emit(dispatcher.stop(reason="voice_cancel_active_motion"))
                elif decision.status is DecisionStatus.EXECUTE and decision.command:
                    emit(dispatcher.dispatch_motion(decision.command))
            except Exception as exc:
                emit({"event": "error", "error": f"{type(exc).__name__}: {exc}"})

        # A finite --text dialogue is primarily for tests and scripts. If it
        # launched real motion, retain ownership until that command completes.
        if args.text is not None and dispatcher.active is not None:
            return_code = dispatcher.active.wait()
            emit({"event": "motion_completed", "return_code": return_code})
            return return_code
        return 0
    except (KeyboardInterrupt, EOFError):
        if dispatcher.active is not None:
            emit(dispatcher.stop(reason="voice_gateway_interrupted"))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
