"""GUI protocol/lifecycle without a display, robot or subprocess."""
from pathlib import Path
import queue
from types import SimpleNamespace
from unittest.mock import Mock

from piper_handeye.button_cycle_gui import ButtonCyclePanel


def panel():
    p = ButtonCyclePanel.__new__(ButtonCyclePanel)
    p.root = Mock()
    p.events = queue.Queue()
    p.backend = Mock()
    p.backend.thread.is_alive.return_value = True
    p.store = SimpleNamespace(cursor=0)
    p.starting = True
    p.reserved = False
    p.closing = False
    p.auto_resume = True
    p.pending_action = None
    p.status = Mock()
    p.state = "paused"
    p.show_summary = Mock()
    p.show_row = Mock()
    p.refresh_controls = Mock()
    return p


def test_ready_only_resumes_after_explicit_start_flag():
    p = panel()
    p.events.put(dict(channel="button_cycle_worker", event="cycle_ready", rows=[]))
    p.poll()
    commands = [call.args[0]["command"] for call in p.backend.send.call_args_list]
    assert commands == ["heartbeat", "resume"]
    assert p.starting  # holds GUI guards until worker confirms reservation
    p.backend.send.reset_mock()
    p.events.put(dict(channel="button_cycle_worker", event="cycle_ready", rows=[]))
    p.poll()
    assert [call.args[0]["command"] for call in p.backend.send.call_args_list] == ["heartbeat"]


def test_history_ready_does_not_start_motion():
    p = panel()
    p.auto_resume = False
    p.events.put(dict(channel="button_cycle_worker", event="cycle_ready", rows=[]))
    p.poll()
    assert [call.args[0]["command"] for call in p.backend.send.call_args_list] == ["heartbeat"]


def test_foreign_protocol_cannot_change_cycle_state_or_rows():
    p = panel()
    p.events.put(dict(channel="other", event="cycle_state", state="completed", reserved=False))
    p.events.put(dict(channel="other", event="cycle_result", row={}))
    p.poll()
    assert p.state == "paused"
    p.show_row.assert_not_called()
    p.show_summary.assert_not_called()


def test_progress_does_not_clear_reservation_or_start_next():
    p = panel()
    p.reserved = True
    p.state = "running"
    p.events.put(dict(channel="button_cycle_worker", event="cycle_progress", message="等待熄灭"))
    p.poll()
    assert p.busy and p.state == "running"
    assert [call.args[0]["command"] for call in p.backend.send.call_args_list] == ["heartbeat"]


def test_close_cancels_pending_start_and_signals_only_once():
    p = panel()
    pending = Mock()
    p.pending_action = pending
    assert not p.close()
    assert not p.close()
    assert p.pending_action is None
    p.backend.stop.assert_called_once()
    p.backend.send.assert_not_called()
    pending.assert_not_called()
    p.backend.thread.is_alive.return_value = False
    assert p.close()


def test_unexpected_exit_never_resumes_automatically():
    p = panel()
    backend = p.backend
    p.events.put(dict(event="backend_exit", return_code=1, intentional=False))
    p.poll()
    assert p.backend is None and not p.busy and p.state == "paused"
    assert [call.args[0]["command"] for call in backend.send.call_args_list] == ["heartbeat"]



def initialized_panel(monkeypatch):
    """Real Tcl settings, without a window, worker or robot connection."""
    import tkinter as tk
    interp = tk.Tcl()
    string_var = tk.StringVar
    monkeypatch.setattr(tk, "StringVar", lambda **kw: string_var(master=interp, **kw))
    monkeypatch.setattr(ButtonCyclePanel, "_build", Mock())
    root = Mock()
    p = ButtonCyclePanel(root, Mock(), SimpleNamespace(execute=True), Mock(), lambda: None, Mock())
    return p


def test_new_gui_defaults_to_one_combined_unattended_policy_without_starting(monkeypatch):
    from piper_handeye.button_cycle_gui import SELECTABLE_POLICIES
    p = initialized_panel(monkeypatch)
    assert SELECTABLE_POLICIES == ("recover_wait", "pause")
    assert p.settings().failure_policy == "recover_wait"
    assert p.backend is None and p.store is None and not p.busy
    p.backend_factory.assert_not_called()


def test_history_preserves_legacy_policy_and_only_new_group_uses_new_default(monkeypatch, tmp_path):
    from piper_handeye.button_cycle import CycleSettings, CycleStore
    from piper_handeye.button_cycle_gui import POLICY_LABELS
    p = initialized_panel(monkeypatch)
    original = CycleSettings("内呼", ("1", "2"), 1, failure_policy="continue_safe")
    p.store = CycleStore.create(tmp_path / "old_group", original, {})
    p.tree = Mock()
    p.tree.get_children.return_value = []
    p.show_summary = Mock()
    p._show_store()
    assert p.policy.get() == POLICY_LABELS["continue_safe"]
    assert p.settings() == original
    assert p.settings(for_new_group=True).failure_policy == "recover_wait"
    assert CycleStore(p.store.directory).settings == original
    assert p.backend is None
    p.backend_factory.assert_not_called()


def test_explicit_pause_remains_available_for_new_group(monkeypatch):
    from piper_handeye.button_cycle_gui import POLICY_LABELS
    p = initialized_panel(monkeypatch)
    p.policy.set(POLICY_LABELS["pause"])
    assert p.settings(for_new_group=True).failure_policy == "pause"


def test_new_group_start_passes_combined_policy_to_saved_plan(monkeypatch, tmp_path):
    import piper_handeye.button_cycle_gui as gui
    p = initialized_panel(monkeypatch)
    monkeypatch.setattr(gui, "RUNS", tmp_path)
    monkeypatch.setattr(gui, "session_metadata", lambda args: {})
    p.launch = Mock()
    p._show_store = Mock()
    p.start()
    assert p.store.settings.failure_policy == "recover_wait"
    assert gui.CycleStore(p.store.directory).settings.failure_policy == "recover_wait"
    p.launch.assert_called_once_with(auto_resume=True)
