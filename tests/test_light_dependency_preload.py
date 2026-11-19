"""Startup ordering and failure isolation, without robot or vision connections."""
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

import run_button_light_closed_loop as entry
import piper_handeye.pipeline_timing as timing


@pytest.fixture
def isolated_entry(tmp_path, monkeypatch):
    config = Path(entry.__file__).with_name("piper_button_pose.yaml")
    hold = tmp_path / "hold.yaml"
    hold.write_text(yaml.safe_dump({
        "stage_1_states": [{"joints_rad": [0.] * 6}],
        "stage_1": {"selected_approach_roll_deg": 0.},
    }))
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setenv("YOUXIAOMEI_MISSION_STATUS_OUTBOX", str(tmp_path / "status.jsonl"))
    monkeypatch.setenv("YOUXIAOMEI_TASK_ID", "isolated-preload-test")
    monkeypatch.setenv("YOUXIAOMEI_TIMING_FILE", "")
    monkeypatch.setattr(timing, "_recorder", None)
    monkeypatch.setattr(entry, "load_handeye", Mock(return_value=(np.eye(4), np.eye(3), [])))
    vision = Mock()
    # The live config now enables layout handoff and neighbour consistency.
    # Model the matching read-only health response so these startup tests
    # reach their intended import/feedback boundary, without bypassing checks.
    settings=yaml.safe_load(config.read_text())["numeric_layout_localization"]
    support=dict(version=1,enabled=True,profile_sha256=settings['profile_sha256'])
    vision.request.return_value = {"status": "ok", "barrier_sequence": 0,
        "numeric_layout_task_handoff": dict(support),
        "numeric_yolo_consistency": dict(support)}
    monkeypatch.setattr(entry, "VisionRpcClient", Mock(return_value=vision))
    stop = Mock()
    stop.health.return_value = {"status": "ok", "latched": False}
    monkeypatch.setattr(entry, "SafetyStopUdpClient", Mock(return_value=stop))
    robot_factory = Mock(side_effect=AssertionError("unexpected robot adapter"))
    monkeypatch.setattr(entry, "PiperUdpRobot", robot_factory)
    pool_factory = Mock(side_effect=AssertionError("unexpected worker startup"))
    monkeypatch.setattr(entry, "ThreadPoolExecutor", pool_factory)
    pointing = Mock(side_effect=AssertionError("unexpected pointing subprocess"))
    monkeypatch.setattr(entry.subprocess, "run", pointing)
    supervisor = Mock(side_effect=AssertionError("unexpected safety reset/start"))
    monkeypatch.setattr(entry, "VisualSafetySupervisor", supervisor)
    monkeypatch.setattr(entry.sys, "argv", [
        "run_button_light_closed_loop.py", "4", "--config", str(config),
        "--hold-report", str(hold), "--execute", "--acknowledge-clear-workspace",
    ])
    # main installs a task-local signal handler; don't alter pytest's handler.
    monkeypatch.setattr(entry.signal, "signal", Mock())
    return tmp_path, vision, stop, robot_factory, pool_factory, pointing, supervisor


def report_at(root):
    return yaml.safe_load(next(root.glob("button_light_closed_loop_runs/*/closed_loop_report.yaml")).read_text())


def test_failed_preload_is_reported_before_any_motion_boundary(isolated_entry, monkeypatch):
    root, vision, stop, robot_factory, pool_factory, pointing, supervisor = isolated_entry
    real_import = entry.importlib.import_module

    def missing_cv2(name):
        if name == "cv2":
            raise ImportError("isolated missing OpenCV")
        return real_import(name)

    monkeypatch.setattr(entry.importlib, "import_module", missing_cv2)
    with pytest.raises(ImportError, match="isolated missing OpenCV"):
        entry.main()
    report = report_at(root)
    assert report["phase"] == "light_processing_preload_failed"
    assert report["status"] == "failed"
    assert report["light_processing_preload"]["status"] == "failed"
    assert report["light_processing_preload"]["modules"][0]["status"] == "error"
    assert report["pipeline_timing"]["written"]
    assert [call.args[0] for call in vision.request.call_args_list] == ["health"]
    for boundary in (robot_factory, pool_factory, pointing, supervisor, stop.reset_latch):
        boundary.assert_not_called()


def test_real_entry_preloads_before_worker_and_feedback_start(isolated_entry, monkeypatch):
    root, vision, stop, _, _, pointing, supervisor = isolated_entry
    order = []
    original_preload = entry.preload_light_processing_dependencies

    def preload(audit):
        dependencies = original_preload(audit)
        order.append("preloaded")
        return dependencies

    monkeypatch.setattr(entry, "preload_light_processing_dependencies", preload)
    monkeypatch.setattr(entry, "wait_for_stable_visibility_after_target_switch",
                        Mock(return_value=({}, True)))
    monkeypatch.setattr(entry, "complete_visible_target", Mock(return_value={"bbox_xyxy": [20, 20, 80, 80]}))
    worker = Mock()
    future = Future()
    future.set_result(worker)
    pool = Mock()
    pool.submit.return_value = future

    def pool_factory(**kwargs):
        assert order == ["preloaded"]
        order.append("worker_start")
        return pool

    monkeypatch.setattr(entry, "ThreadPoolExecutor", pool_factory)
    robot = Mock()
    robot._pose_timing = None
    robot.state.side_effect = RuntimeError("no live state in isolated test")

    def open_feedback():
        assert order == ["preloaded", "worker_start"]
        order.append("feedback_start")
        raise RuntimeError("isolated feedback boundary")

    robot.open.side_effect = open_feedback
    monkeypatch.setattr(entry, "PiperUdpRobot", Mock(return_value=robot))
    with pytest.raises(RuntimeError, match="isolated feedback boundary"):
        entry.main()
    report = report_at(root)
    audit = report["light_processing_preload"]
    assert audit["status"] == "ready"
    assert len(audit["modules"]) == 4
    assert all(row["status"] == "ok" for row in audit["modules"])
    assert audit["elapsed_sec"] >= 0
    journal = report["journal"]
    completed = next(row for row in journal if row["kind"] == "light_processing_preload_completed")
    pointing_started = next(row for row in journal if row["kind"] == "pointing_phase_started")
    assert completed["sequence"] < pointing_started["sequence"]
    assert order == ["preloaded", "worker_start", "feedback_start"]
    worker.close.assert_called_once()
    robot.close.assert_called_once()
    pointing.assert_not_called()
    supervisor.assert_not_called()
    stop.reset_latch.assert_not_called()


def test_check_mode_remains_read_only_without_preload(isolated_entry, monkeypatch):
    root, vision, stop, robot_factory, pool_factory, pointing, supervisor = isolated_entry
    args = [arg for arg in entry.sys.argv if arg not in ("--execute", "--acknowledge-clear-workspace")]
    monkeypatch.setattr(entry.sys, "argv", [*args, "--check"])
    preload = Mock(side_effect=AssertionError("check mode must not run preparation"))
    monkeypatch.setattr(entry, "preload_light_processing_dependencies", preload)
    assert entry.main() == 0
    for boundary in (preload, robot_factory, pool_factory, pointing, supervisor, stop.reset_latch):
        boundary.assert_not_called()
    assert not (root / "button_light_closed_loop_runs").exists()


def test_repeated_preload_reuses_only_code_not_task_state(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "_recorder", None)
    monkeypatch.setenv("YOUXIAOMEI_TIMING_FILE", "")
    first, second = {}, {}
    first_classes = entry.preload_light_processing_dependencies(first)
    second_classes = entry.preload_light_processing_dependencies(second)
    assert first_classes == second_classes
    assert all(row["already_loaded"] for row in second["modules"])
    paired_class = first_classes[0]
    one, two = paired_class(tmp_path / "one"), paired_class(tmp_path / "two")
    assert one is not two
    assert one.references is not two.references
    assert not one.references and not two.references
