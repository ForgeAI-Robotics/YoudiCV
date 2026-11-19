import ast
from pathlib import Path

import pytest

from piper_handeye.trajectory_splice import PreparedSpliceUdpClient


def test_abort_active_sends_authenticated_discard_request(monkeypatch):
    client = PreparedSpliceUdpClient("robot", 5011)
    captured = {}

    def request(payload, timeout_sec=None):
        captured.update(payload)
        captured["timeout_sec"] = timeout_sec
        return {"status": "ok", "execution_status": "aborted"}

    monkeypatch.setattr(client, "_request", request)
    result = client.abort_active(token="secret", active_id="trajectory-1")
    assert result["execution_status"] == "aborted"
    assert captured["command"] == "abort_active"
    assert captured["active_id"] == "trajectory-1"
    assert captured["token"] == "secret"
    assert captured["timeout_sec"] == pytest.approx(5.0)


def test_recovery_discards_prepared_motion_before_unlocking_safety_latch():
    source = Path(__file__).resolve().parents[1] / "run_button_recovery_to_origin.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    abort = next(node for node in calls if node.func.attr == "abort_active")
    unlock = next(node for node in calls if node.func.attr == "reset_latch")
    assert abort.lineno < unlock.lineno


def test_online_near_panel_motion_requires_coverage_degraded_policy(monkeypatch):
    import run_online_replan_no_contact as runner

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_online_replan_no_contact.py",
            "--hold-report", "unused.yaml",
            "--target-distance-m", "0.01",
            "--receding-schedule",
            "--visual-safety",
            "--execute",
            "--acknowledge-clear-workspace",
        ],
    )
    with pytest.raises(ValueError, match="final-depth-degraded-commit"):
        runner.main()
