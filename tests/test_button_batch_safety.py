"""Check physical command arguments without starting any subprocess."""
from types import SimpleNamespace
from unittest.mock import patch

import run_button_batch


def test_batch_uses_the_light_verified_force_feedback_workflow():
    with patch("sys.argv", ["run_button_batch.py", "15"]), patch(
        "run_button_batch.subprocess.run", return_value=SimpleNamespace(returncode=0)
    ) as launch:
        assert run_button_batch.main() == 0
    argv = launch.call_args.args[0]
    assert argv[1].endswith("run_button_light_closed_loop.py")
    assert "run_online_replan_no_contact.py" not in argv[1]
    assert argv[-1] == "15"
    assert argv[argv.index("--max-press-attempts") + 1] == "3"
    assert "--pause-depth-near-button" in argv


def test_batch_propagates_config_and_hold_report():
    with patch("sys.argv", [
        "run_button_batch.py", "15", "--config", "/tmp/custom.yaml",
        "--hold-report", "/tmp/hold.yaml",
    ]), patch(
        "run_button_batch.subprocess.run", return_value=SimpleNamespace(returncode=0)
    ) as launch:
        assert run_button_batch.main() == 0
    argv = launch.call_args.args[0]
    assert "--pause-depth-near-button" in argv
    assert argv[1].endswith("run_button_light_closed_loop.py")
    assert argv[argv.index("--config") + 1] == "/tmp/custom.yaml"
    assert argv[argv.index("--hold-report") + 1] == "/tmp/hold.yaml"
