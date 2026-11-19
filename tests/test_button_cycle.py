"""No-motion sequence, evidence and persistence tests for large test groups."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from piper_handeye.button_cycle import CycleSettings, CycleStore, TaskEvidence, parse_buttons, summarize
from piper_handeye.button_cycle_runner import CycleRunner


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now
    def advance(self, value=1.): self.now += value


class Dispatcher:
    def __init__(self, tmp_path):
        self.active = None
        self.session_reserved = False
        self._status_outbox = tmp_path / "mission.jsonl"
        self.events = []
        self.commands = []
        self.returning = False

    def reserve_session(self): self.session_reserved = True
    def release_session(self):
        assert self.active is None
        self.session_reserved = False
    def dispatch_motion(self, command):
        assert self.session_reserved and self.active is None
        self.commands.append(command.button)
        self.active = object()
        return dict(event="motion_started", task_id=f"task{len(self.commands)}")
    def refresh(self):
        if not self.events: return None
        event = self.events.pop(0)
        if event["event"] in {"motion_completed", "origin_operation_completed"}: self.active = None
        return event
    def stop(self, **kwargs):
        self.active = None
        return dict(event="stopped")
    def abort_and_return(self):
        if self.active is not None:
            self.events.append(dict(event="abort_completed_return_started"))
        self.active = object()
        self.returning = True
        return dict(event="abort_return_started")


def terminal(tid="task1", *, attempts=1, flow="arrived_after_verified_press", rc=0,
             origin=True, on=True, off=True, executed=True, timeout=False):
    evidence = dict(flow_status=flow, press_attempts=attempts, press_execution_started=executed,
                    returned_to_origin=origin, origin_verification=dict(verified=origin, hardware_ok=1),
                    process_exit_code=rc)
    milestones = []
    for raw, phase, enabled in (("press_verified_on", "button_on_confirmed", on),
                                ("button_became_off", "floor_arrival_observed", off),
                                ("external_call_off_wait_timeout", "external_call_timeout_release", timeout)):
        if enabled:
            milestones.append(dict(task_id=tid, producer="arm_pressing", raw_event=raw, phase=phase,
                                   emitted_monotonic_sec=101.))
    milestones.append(dict(task_id=tid, producer="arm_pressing", raw_event="task_terminal", evidence=evidence))
    return dict(event="motion_completed", task_id=tid, return_code=rc, mission_status_events=milestones, arm_report=evidence)


def setup(tmp_path, settings=None, readiness=None):
    store = CycleStore.create(tmp_path / "run", settings or CycleSettings("内呼", ("1", "2"), 1), {})
    dispatcher, clock, emitted = Dispatcher(tmp_path), Clock(), []
    runner = CycleRunner(store, dispatcher, readiness or (lambda: dict(at_origin=True)), emitted.append, clock=clock)
    return runner, dispatcher, store, clock, emitted


def begin(runner, clock):
    runner.resume()
    runner.tick()
    clock.advance(.6)
    runner.tick()


def finish_task(runner, dispatcher, clock, **kwargs):
    clock.advance(.1)
    event = terminal(f"task{len(dispatcher.commands)}", **kwargs)
    for row in event["mission_status_events"]:
        row["emitted_monotonic_sec"] = clock()
    dispatcher.events.append(event)
    runner.tick()


def test_ranges_order_and_group_boundaries():
    assert parse_buttons("1-5", "内呼") == ("1", "2", "3", "4", "5")
    assert parse_buttons("3-1, 5，2", "内呼") == ("3", "2", "1", "5", "2")
    assert parse_buttons("上行,下行,up", "外呼") == ("up", "down", "up")
    for value, group in (("", "内呼"), ("0", "内呼"), ("17", "内呼"), ("open", "外呼"),
                         ("1,up", "内呼"), ("1-5", "外呼"), ("1.5", "内呼")):
        with pytest.raises(ValueError): parse_buttons(value, group)


@pytest.mark.parametrize("values", [dict(rounds=0), dict(rounds=True), dict(rounds=10001),
    dict(interval_sec=-1), dict(interval_sec=float('nan')), dict(interval_sec=float('inf')),
    dict(failure_policy="always_continue"), dict(buttons=tuple(["1"]*65)),
    dict(rounds=10000, buttons=tuple(str(i) for i in range(1, 17)))])
def test_settings_reject_unbounded_or_unsafe_input(values):
    with pytest.raises(ValueError):
        CycleSettings(**{**dict(group="内呼", buttons=("1",), rounds=1), **values})


@pytest.mark.parametrize("settings,expected", [
    (CycleSettings("内呼", ("1", "2", "3", "4", "5"), 2), [1,2,3,4,5]*2),
    (CycleSettings("外呼", ("up", "down"), 2), ["up", "down"]*2),
])
def test_complete_sequence_uses_dispatcher_once_per_item(tmp_path, settings, expected):
    r, d, store, clock, _ = setup(tmp_path, settings)
    begin(r, clock)
    for _ in expected:
        assert d.active is not None
        finish_task(r, d, clock)
    assert d.commands == expected
    assert r.state == "completed" and not d.session_reserved
    assert store.cursor == settings.total
    summary = summarize(store.rows)
    assert summary["full_cycle_success"] == len(expected)
    assert summary["first_success_rate"] == 1.
    assert CycleStore(store.directory).cursor == settings.total


def test_pause_finishes_current_does_not_launch_next_until_explicit_resume(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    begin(r, clock)
    r.pause()
    assert r.pause_requested and d.active is not None
    finish_task(r, d, clock)
    assert r.state == "paused" and d.session_reserved and d.commands == [1]
    for _ in range(10): clock.advance(); r.tick()
    assert d.commands == [1]
    begin(r, clock)
    assert d.commands == [1, 2]


def test_initial_on_wait_is_not_success_and_retests_same_item(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    begin(r, clock)
    finish_task(r, d, clock, flow="arrived_after_initial_on", attempts=0, executed=False, on=False)
    assert d.commands == [1, 1] and store.cursor == 0
    assert summarize(store.rows)["completed"] == 0
    assert summarize(store.rows)["already_on_waits"] == 1
    finish_task(r, d, clock)
    assert d.commands == [1, 1, 2] and store.cursor == 1


def test_default_failure_pauses_even_with_good_origin(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    begin(r, clock)
    finish_task(r, d, clock, rc=2, flow="button_not_lit_after_max_attempts", attempts=3, on=False, off=False)
    assert r.state == "paused" and d.commands == [1]
    assert store.cursor == 1 and summarize(store.rows)["lit_success"] == 0
    r.finish()
    assert not d.session_reserved and r.state == "ended"


@pytest.mark.parametrize("rc,flow,origin,expected", [
    (2, "button_not_lit_after_max_attempts", True, [1, 2]),
    (2, "post_press_confirmation_uncertain", True, [1, 2]),
    (2, "post_press_confirmation_uncertain", False, [1]),
    (1, "post_press_confirmation_uncertain", True, [1]),
    (2, "unknown", True, [1]),
])
def test_continue_only_known_ordinary_failure_after_verified_return(tmp_path, rc, flow, origin, expected):
    r, d, store, clock, _ = setup(tmp_path, CycleSettings("内呼", ("1", "2"), 1, failure_policy="continue_safe"))
    begin(r, clock)
    finish_task(r, d, clock, rc=rc, flow=flow, origin=origin, on=False, off=False)
    assert d.commands == expected


def test_stale_or_unready_data_never_dispatches(tmp_path):
    r, d, _, clock, _ = setup(tmp_path, readiness=lambda: (_ for _ in ()).throw(RuntimeError("feedback stale")))
    r.resume()
    for _ in range(11): r.tick(); clock.advance()
    assert r.state == "paused" and not d.commands and d.session_reserved
    r.finish()


def test_interval_is_between_completed_tasks_and_pause_cancels_due_item(tmp_path):
    r, d, _, clock, _ = setup(tmp_path, CycleSettings("内呼", ("1", "2"), 1, interval_sec=2.))
    begin(r, clock)
    finish_task(r, d, clock)
    assert d.commands == [1] and r.state == "waiting_next"
    r.pause(); clock.advance(10); r.tick()
    assert d.commands == [1]


def test_zero_exit_without_terminal_evidence_pauses_and_does_not_fake_success(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    begin(r, clock)
    d.events.append(dict(event="motion_completed", task_id="task1", return_code=0))
    r.tick()
    assert r.state == "paused" and d.commands == [1]
    assert store.rows[0]["press_attempts"] is None and not store.rows[0]["on_confirmed"]


def test_abort_stops_queue_and_waits_for_origin_result(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    begin(r, clock)
    r.stop(return_origin=True)
    assert d.session_reserved and d.returning
    r.tick()
    assert r.state == "returning" and store.cursor == 0
    d.events.append(dict(event="origin_operation_completed", return_code=0))
    r.tick()
    assert r.state == "ended" and not d.session_reserved and d.commands == [1]
    assert summarize(store.rows)["interrupted"] == 1


def test_close_stops_without_automatically_returning_or_dispatching(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    begin(r, clock)
    r.stop(closing=True)
    assert r.state == "closed" and not d.session_reserved and not d.returning
    assert d.commands == [1] and store.cursor == 0


def test_restore_interrupted_execution_is_paused_and_uncounted(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    begin(r, clock)
    restored = CycleStore(store.directory)
    new_d = Dispatcher(tmp_path)
    new_r = CycleRunner(restored, new_d, lambda: True, lambda _: None, clock=clock)
    assert new_r.state == "paused" and not new_d.commands and not new_d.session_reserved
    assert restored.cursor == 0 and summarize(restored.rows)["interrupted"] == 1
    assert restored.rows[0]["press_attempts"] is None
    assert restored.rows[0]["wall_sec"] is None


def test_duplicate_completion_cannot_increment_twice(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    begin(r, clock)
    r.pause(); finish_task(r, d, clock)
    with pytest.raises(ValueError, match="重复"):
        store.append(store.rows[0])
    assert store.cursor == 1


def test_manual_review_does_not_change_program_outcome_and_exports_excel_utf8(tmp_path):
    r, d, store, clock, _ = setup(tmp_path, CycleSettings("内呼", ("1",), 1))
    begin(r, clock); finish_task(r, d, clock)
    row = store.rows[0]
    store.review(row["execution_id"], "未按亮")
    reloaded = CycleStore(store.directory)
    assert reloaded.rows[0]["result"] == "首次成功"
    assert reloaded.rows[0]["manual_verification"] == "未按亮"
    assert summarize(reloaded.rows)["manual_on"] == 0
    assert (store.directory / "results.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_torn_final_append_is_preserved_then_repaired_on_explicit_write(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    begin(r, clock); r.pause(); finish_task(r, d, clock)
    path = store.directory / "results.jsonl"
    with path.open("ab") as stream: stream.write(b'{"broken":"\xe4')
    before = path.read_bytes()
    reloaded = CycleStore(store.directory)
    assert path.read_bytes() == before and reloaded.cursor == 1
    row = {**store.rows[0], "execution_id": "second", "index": 1, "button": "2"}
    reloaded.append(row)
    assert CycleStore(store.directory).cursor == 2
    assert next(store.directory.glob("results.jsonl.partial_*")).read_bytes() == b'{"broken":"\xe4'


def test_foreign_on_and_initial_on_are_not_proof_of_this_press():
    evidence = TaskEvidence("mine")
    evidence.add(terminal("other"))
    assert evidence.terminal is None and not evidence.on_confirmed
    event = terminal("mine", on=False)
    event["mission_status_events"].insert(0, dict(task_id="mine", producer="arm_pressing",
        raw_event="button_already_on", phase="button_on_confirmed"))
    evidence.add(event)
    assert not evidence.result(0, started=100., finished=102.)["on_confirmed"]


def test_timeout_is_distinct_from_visual_off_and_full_cycle_success():
    evidence = TaskEvidence("task1")
    evidence.add(terminal(flow="external_call_off_timeout_after_verified_press", off=False, timeout=True))
    row = evidence.result(0, started=100., finished=140.)
    assert row["on_confirmed"] and not row["off_confirmed"] and not row["full_cycle_success"]
    assert row["result"] == "按亮确认，等灭超时"


def test_new_group_already_has_exportable_empty_csv_and_no_completed_tasks(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    assert (store.directory / "results.csv").is_file()
    assert json.loads((store.directory / "summary.json").read_text())["overall"]["completed"] == 0
    assert not d.commands and not d.session_reserved


def test_modified_history_button_cannot_resume_as_a_different_task(tmp_path):
    r, d, store, clock, _ = setup(tmp_path)
    begin(r, clock); r.pause(); finish_task(r, d, clock)
    path = store.directory / "results.jsonl"
    row = json.loads(path.read_text())
    row["button"] = "8"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="不匹配"):
        CycleStore(store.directory)


def test_one_thousand_virtual_tasks_preserve_order_and_counts(tmp_path):
    settings = CycleSettings("内呼", ("1", "2", "3", "4", "5"), 200)
    r, d, store, clock, _ = setup(tmp_path, settings)
    begin(r, clock)
    for _ in range(1000):
        finish_task(r, d, clock)
    assert d.commands == [1, 2, 3, 4, 5] * 200
    assert store.cursor == 1000 and r.state == "completed"
    summary = summarize(CycleStore(store.directory).rows)
    assert summary["completed"] == summary["first_success"] == summary["actual_press_commands"] == 1000
    assert summary["mean_time_to_on_sec"] == pytest.approx(.1)
