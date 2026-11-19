"""Sequential test bookkeeping. No camera, robot, ROS or motion imports."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from .button_light_flow import (NUMERIC_CYCLE_ON_TIMEOUT_FLOW,
    NUMERIC_CYCLE_INITIAL_ON_TIMEOUT_FLOW, NUMERIC_CYCLE_TIMEOUT_FLOWS)
import json
import math
import os
from pathlib import Path
import re
import time


VERIFICATIONS = ("未核验", "已按亮", "未按亮")
SAFE_FAILURE_FLOWS = frozenset({
    "button_not_lit_after_max_attempts", "initial_observation_uncertain",
    "post_press_confirmation_uncertain", "post_press_confirmation_disagreement",
})


def parse_buttons(text: str, group: str) -> tuple[str, ...]:
    if group not in {"内呼", "外呼"}:
        raise ValueError("请选择内呼或外呼")
    tokens = re.split(r"[,，、;；\s]+", str(text).strip())
    labels = []
    aliases = {"上": "up", "上行": "up", "下": "down", "下行": "down"}
    for token in tokens:
        token = aliases.get(token, token.casefold())
        span = re.fullmatch(r"(\d{1,2})\s*[-~～]\s*(\d{1,2})", token)
        if span:
            a, b = map(int, span.groups())
            if not 1 <= a <= 16 or not 1 <= b <= 16:
                raise ValueError("数字按钮必须在1–16之间")
            labels.extend(str(i) for i in range(a, b + (1 if b >= a else -1), 1 if b >= a else -1))
        elif token:
            labels.append(str(int(token)) if token.isdecimal() else token)
    allowed = {str(i) for i in range(1, 17)} if group == "内呼" else {"up", "down"}
    if not labels or len(labels) > 64 or any(label not in allowed for label in labels):
        raise ValueError("内呼填写1–16，如1-5；外呼填写上行,下行。每轮最多64项")
    return tuple(labels)


@dataclass(frozen=True)
class CycleSettings:
    group: str
    buttons: tuple[str, ...]
    rounds: int
    interval_sec: float = 0.
    failure_policy: str = "pause"

    def __post_init__(self):
        if type(self.rounds) is not int or not 1 <= self.rounds <= 10000:
            raise ValueError("轮数须为1–10000的整数")
        if not isinstance(self.buttons, (tuple, list)) or not self.buttons:
            raise ValueError("按钮顺序不能为空")
        if tuple(parse_buttons(",".join(self.buttons), self.group)) != tuple(self.buttons):
            raise ValueError("按钮顺序无效")
        if type(self.interval_sec) not in (int, float) or not math.isfinite(self.interval_sec) or not 0 <= self.interval_sec <= 60:
            raise ValueError("间隔须为0–60秒")
        if self.failure_policy not in {"pause", "continue_safe", "recover_wait"} or self.total > 100000:
            raise ValueError("失败策略无效或总任务数超过100000")

    @property
    def total(self):
        return len(self.buttons) * self.rounds

    def item(self, index):
        if not 0 <= index < self.total:
            raise IndexError(index)
        return dict(index=index, round=index // len(self.buttons) + 1,
                    button=self.buttons[index % len(self.buttons)])


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path, value):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        # Recover a torn final append only when explicitly writing/resuming;
        # merely viewing history never changes the original evidence file.
        with path.open("r+b") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            stream.seek(max(0, end-65536))
            offset = stream.tell()
            tail = stream.read()
            if not tail.endswith(b"\n"):
                boundary = tail.rfind(b"\n") + 1
                fragment = tail[boundary:]
                try:
                    json.loads(fragment)
                except ValueError:
                    if offset and boundary == 0:
                        raise ValueError("记录末行过长，拒绝自动修复")
                    evidence = path.with_name(path.name + ".partial_" + str(time.time_ns()))
                    evidence.write_bytes(fragment)
                    stream.truncate(offset + boundary)
                else:
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            # A power loss may leave only the final append incomplete. Never
            # infer a successful task from that fragment or ignore middle errors.
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("测试记录损坏，请保留文件并检查")
    return rows


def summarize(rows):
    measured = [r for r in rows if r.get("counted") is True]
    first = sum(r.get("on_confirmed") is True and r.get("press_attempts") == 1 for r in measured)
    lit = sum(r.get("on_confirmed") is True for r in measured)
    complete = sum(r.get("full_cycle_success") is True for r in measured)
    known = [r["press_attempts"] for r in rows if type(r.get("press_attempts")) is int]
    wall = [r["wall_sec"] for r in measured if type(r.get("wall_sec")) in (int, float)]
    to_on = [r["time_to_on_sec"] for r in measured if type(r.get("time_to_on_sec")) in (int, float)]
    return dict(completed=len(measured), first_success=first, retry_success=lit-first,
                lit_success=lit, full_cycle_success=complete,
                failed_or_uncertain=len(measured)-complete,
                uncertain=sum(r.get("result") == "亮灯不确定" for r in measured),
                opencv_fallback_items=sum(r.get('opencv_fallback_used') is True for r in measured),
                on_timeout_skips=sum(r.get("on_timeout_skipped") is True for r in measured),
                initial_on_timeout_skips=sum(r.get("on_timeout_skipped") is True and r.get("press_attempts") == 0 for r in measured),
                no_press_failures=sum(r.get("press_attempts") == 0 for r in measured),
                actual_press_commands=sum(known),
                unknown_press_counts=sum(r.get("press_attempts") is None for r in rows),
                already_on_waits=sum(r.get("result") == "初始已亮，未计数" for r in rows),
                interrupted=sum(r.get("result") == "中止/中断，未计数" for r in rows),
                first_success_rate=first/len(measured) if measured else None,
                lit_success_rate=lit/len(measured) if measured else None,
                full_cycle_success_rate=complete/len(measured) if measured else None,
                mean_wall_sec=sum(wall)/len(wall) if wall else None,
                mean_time_to_on_sec=sum(to_on)/len(to_on) if to_on else None,
                manually_verified=sum(r.get("manual_verification") != "未核验" for r in measured),
                manual_on=sum(r.get("manual_verification") == "已按亮" for r in measured))


CSV_COLUMNS = (
    ("execution_id", "执行编号"), ("round", "轮次"), ("button", "按钮"),
    ("result", "程序判定"), ("counted", "计入已测任务"),
    ("press_attempts", "实际按压命令次数"), ("on_confirmed", "程序确认按亮"),
    ("off_confirmed", "视觉确认熄灭"), ("off_wait_timed_out", "等灭超时"),
    ("on_timeout_skipped", "内呼持续亮超时跳过"),
    ('opencv_fallback_used', '采用OpenCV后备'), ('on_confirmation_source', 'ON采用来源'),
    ("full_cycle_success", "完整流程成功"), ("time_to_on_sec", "启动至按亮秒"),
    ("wall_sec", "完整任务秒"), ("manual_verification", "人工实物核验"),
    ("reason", "原因"), ("task_id", "正式任务编号"), ("flow_status", "流程状态"),
    ("started_at", "开始时间"), ("finished_at", "结束时间"),
)


class CycleStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.plan = json.loads((self.directory / "plan.json").read_text())
        if self.plan.get("schema_version") != 1:
            raise ValueError("测试记录版本不支持，不能自动继续")
        self.settings = CycleSettings(**{**self.plan["settings"], "buttons": tuple(self.plan["settings"]["buttons"])})
        self.rows = read_rows(self.directory / "results.jsonl")
        self._csv_version = None
        self._review_version = 0
        if len({r["execution_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("存在重复执行编号，不能自动继续")
        for review in read_rows(self.directory / "reviews.jsonl"):
            self._review_version += 1
            for row in self.rows:
                if row["execution_id"] == review["execution_id"]:
                    row["manual_verification"] = review["value"]
        self.cursor = 0
        for row in self.rows:
            if (type(row.get("index")) is not int or row["index"] != self.cursor
                    or type(row.get("counted")) is not bool or self.cursor >= self.settings.total):
                raise ValueError("测试顺序记录不连续，不能自动继续")
            item = self.settings.item(self.cursor)
            if row.get("button") != item["button"] or row.get("round") != item["round"]:
                raise ValueError("记录的按钮或轮次与计划不匹配，不能自动继续")
            if row.get("counted"):
                self.cursor += 1

    @classmethod
    def create(cls, directory, settings, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "plan.json", dict(schema_version=1, settings=asdict(settings),
                    created_at=datetime.now().astimezone().isoformat(), metadata=metadata))
        store = cls(directory)
        store.save("paused", "等待明确启动；尚未发送动作")
        return store

    def append(self, row):
        if any(r["execution_id"] == row["execution_id"] for r in self.rows):
            raise ValueError("执行结果不能重复计数")
        if row["index"] != self.cursor:
            raise ValueError("结果与当前测试项不匹配")
        append_json(self.directory / "results.jsonl", row)
        self.rows.append(dict(row))
        if row["counted"]:
            self.cursor += 1
        self.save("paused")

    def review(self, execution_id, value):
        if value not in VERIFICATIONS:
            raise ValueError("核验值无效")
        row = next((r for r in self.rows if r["execution_id"] == execution_id), None)
        if row is None:
            raise ValueError("未找到该执行记录")
        append_json(self.directory / "reviews.jsonl", dict(execution_id=execution_id,
                    value=value, reviewed_at=datetime.now().astimezone().isoformat()))
        row["manual_verification"] = value
        self._review_version += 1
        self.save("paused")
        return row

    def save(self, state, reason=""):
        summary = dict(state=state, reason=reason, next_index=self.cursor, total=self.settings.total,
                       overall=summarize(self.rows),
                       per_button={label: summarize([r for r in self.rows if r["button"] == label])
                                   for label in dict.fromkeys(self.settings.buttons)})
        atomic_json(self.directory / "summary.json", summary)
        version = (len(self.rows), self._review_version)
        if self._csv_version == version and (self.directory / "results.csv").exists():
            return summary
        path = self.directory / "results.csv"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([label for _, label in CSV_COLUMNS])
            writer.writerows([[row.get(key) for key, _ in CSV_COLUMNS] for row in self.rows])
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        self._csv_version = version
        return summary


class TaskEvidence:
    """Keep only scoped milestones; indefinite light waiting stays bounded."""
    def __init__(self, task_id):
        self.task_id = task_id
        self.terminal = None
        self.on_at = None
        self.on_confirmed = False
        self.off_confirmed = False
        self.initial_on = False
        self.timeout = False
        self.numeric_timeout_audit = None
        self.hard_failure = False
        self.recovery_candidate = None
        self.recovery_veto = False
        self.opencv_fallback_used = False
        self.on_confirmation_source = None

    def add(self, event):
        for row in event.get("mission_status_events") or []:
            if row.get('task_id') == self.task_id and row.get('phase') in {'abort_requested', 'aborted', 'safety_paused', 'motion_segment_cleanup_failed'}:
                self.recovery_veto = True
            if row.get("task_id") != self.task_id or row.get("producer") != "arm_pressing":
                continue
            raw, phase = row.get("raw_event"), row.get("phase")
            if raw == 'post_press_opencv_adopted':
                self.opencv_fallback_used = True
            if raw == 'press_verified_on':
                self.on_confirmation_source = (row.get('evidence') or {}).get('confirmation_source')
            if raw == 'guarded_press_failed' or (raw == 'motion_segment_failed' and (row.get('evidence') or {}).get('segment') == 'guarded_press'):
                from .cycle_recovery import recoverable_retract_failure
                self.recovery_candidate = recoverable_retract_failure((row.get('evidence') or {}).get('bridge_failure'))
            if raw in {"press_verified_on", "post_press_on_captured_early"}:
                self.on_confirmed = True
                if self.on_at is None:
                    value = row.get("emitted_monotonic_sec")
                    if type(value) in (int, float) and math.isfinite(value):
                        self.on_at = value
            if raw == "button_became_off":
                self.off_confirmed = True
            if raw == "button_already_on":
                self.initial_on = True
            if raw == "external_call_off_wait_timeout":
                self.timeout = True
            if raw == "numeric_cycle_on_wait_timeout":
                self.timeout = True
                self.numeric_timeout_audit = row.get("evidence") or {}
            if phase in {"motion_segment_failed", "motion_segment_cleanup_failed", "press_failed", "origin_return_failed"}:
                self.hard_failure = True
            if raw == "task_terminal":
                self.terminal = row.get("evidence") or {}
        # Dispatcher retains the terminal even if it arrived in an earlier poll.
        if event.get("task_id") == self.task_id and isinstance(event.get("arm_report"), dict):
            self.terminal = event["arm_report"]

    def result(self, return_code, *, started, finished, aborted=False):
        evidence = self.terminal or {}
        opencv_used = bool(self.opencv_fallback_used or evidence.get('opencv_fallback_used'))
        on_source = evidence.get('on_confirmation_source') or self.on_confirmation_source
        attempts = evidence.get("press_attempts")
        executed = evidence.get("press_execution_started")
        attempts = attempts if type(attempts) is int and attempts >= 0 and executed is True else 0 if executed is False else None
        origin = evidence.get("origin_verification") or {}
        at_origin = (evidence.get("returned_to_origin") is True and origin.get("verified") is True
                     and origin.get("hardware_ok") == 1)
        flow = evidence.get("flow_status")
        on = self.on_confirmed and executed is True and type(attempts) is int and attempts > 0
        full = (return_code == 0 and on and at_origin and self.off_confirmed
                and flow == "arrived_after_verified_press" and not evidence.get("reporting_errors"))
        already = return_code == 0 and flow == "arrived_after_initial_on" and executed is False and attempts == 0 and at_origin and self.off_confirmed
        audit = evidence.get("numeric_cycle_on_wait_timeout") or self.numeric_timeout_audit or {}
        elapsed = audit.get("elapsed_continuous_on_sec")
        numeric_timeout_valid = (
            flow in NUMERIC_CYCLE_TIMEOUT_FLOWS and audit.get("target_label") in {str(i) for i in range(1,17)}
            and audit.get("release_policy") == "cycle_numeric_on_timeout_skip"
            and audit.get("last_observed_state") == "ON" and audit.get("off_confirmed") is False
            and type(audit.get("timeout_sec")) in (int,float) and audit["timeout_sec"] == 10.
            and type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed >= 10.
            and ((flow == NUMERIC_CYCLE_ON_TIMEOUT_FLOW and audit.get("after_verified_press") is True and on)
                 or (flow == NUMERIC_CYCLE_INITIAL_ON_TIMEOUT_FLOW and audit.get("after_verified_press") is False
                     and self.initial_on and executed is False and attempts == 0)))
        timeout_skip = (numeric_timeout_valid and return_code == 0 and at_origin
                        and not self.hard_failure and not self.recovery_veto
                        and not evidence.get("reporting_errors") and not aborted)
        reason = str(evidence.get("failure_reason") or evidence.get("failure_code") or flow or "缺少正式终态证据")
        if aborted:
            result = "中止/中断，未计数"
        elif already:
            result = "初始已亮，未计数"
        elif full:
            result = "首次成功" if attempts == 1 else "重试成功"
            reason = ("按OpenCV采用ON，视觉熄灭及归位均已确认"
                      if str(on_source or '').startswith('opencv_')
                      else "按亮、视觉熄灭及归位均已确认")
        elif timeout_skip:
            result = "按亮确认，持续亮超时跳过" if on else "初始已亮超时，未按压"
            reason = "连续观察亮灯10秒未灭，疑似内呼机异常；已核验归位并跳过，未确认熄灭"
        elif on:
            result = "按亮确认，等灭超时" if self.timeout and return_code == 0 and at_origin else "按亮确认，流程失败"
            if self.timeout and return_code == 0 and at_origin:
                reason = "按亮后等灭超时放行，未取得视觉熄灭证据"
        elif flow in {"post_press_confirmation_uncertain", "post_press_confirmation_disagreement"}:
            result = "亮灯不确定"
        else:
            result = "未执行按压" if attempts == 0 else "失败"
        soft = (return_code == 2 and flow in SAFE_FAILURE_FLOWS and at_origin
                and not self.hard_failure and not evidence.get("reporting_errors"))
        time_to_on = (self.on_at - started if on and self.on_at is not None
                      and started <= self.on_at <= finished else None)
        return dict(result=result, counted=not (aborted or already), press_attempts=attempts,
                    on_confirmed=on, off_confirmed=self.off_confirmed,
                    off_wait_timed_out=self.timeout or numeric_timeout_valid,
                    on_timeout_skipped=timeout_skip, full_cycle_success=full,
                    opencv_fallback_used=opencv_used, on_confirmation_source=on_source,
                    time_to_on_sec=time_to_on, wall_sec=max(0., finished-started),
                    flow_status=flow, return_code=return_code, reason=reason,
                    returned_to_origin=at_origin, safe_to_continue=soft or timeout_skip,
                    initial_on_wait=already, manual_verification="未核验")
