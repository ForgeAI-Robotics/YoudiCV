import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np


@pytest.mark.parametrize("unmonitored_return", [False, True])
@pytest.mark.parametrize("degraded", [False, True])
def test_press_policy_is_kept_until_release(unmonitored_return, degraded):
    source = Path(__file__).resolve().parents[1] / "run_online_replan_no_contact.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == "execute_blended_press_return")
    callbacks = []
    waits = []
    transitions = []

    class Event:
        def wait(self, timeout):
            waits.append(timeout)
            return False

        def set(self):
            pass

        def is_set(self):
            return False

    class Thread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            callbacks.append(self.target)

        def join(self, **_kwargs):
            pass

    def execute(*_args, **_kwargs):
        assert transitions == []  # Submission must retain the press policy.
        assert len(callbacks) == 1
        callbacks[0]()  # Simulate reaching the scheduled release boundary.
        assert waits == [pytest.approx(3.12)]
        assert transitions == ["return"]
        return {"status": "complete"}

    target = np.eye(4)
    target[0, 3] = 0.50
    poses = []
    for x in (0.49, 0.44):
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append(SimpleNamespace(base_to_gripper=pose))
    namespace = {
        "args": SimpleNamespace(return_without_visual_safety=unmonitored_return,
                                final_depth_degraded_commit=degraded,
                                pause_depth_near_button=False,
                                final_depth_degraded_distance_m=0.05),
        "degraded_commit_active": SimpleNamespace(is_set=lambda: degraded),
        "degraded_commit_lock": nullcontext(), "degraded_commit_target": target,
        "robot": SimpleNamespace(state=Mock(side_effect=poses)),
        "threading": SimpleNamespace(Event=Event, Thread=Thread),
        "standard": SimpleNamespace(execute=execute), "token": "test",
        "transition_to_return_safety_policy": lambda: transitions.append("return"),
        "wait_for_depth_recovery": (
            lambda _finished: transitions.append("return")
        ),
        "execute_return": Mock(side_effect=AssertionError("premature return policy")),
        "report": {},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    result = namespace[function.name]([], SimpleNamespace(prepress_pass_time_sec=3.10), "test")
    assert result["status"] == "complete"
    assert namespace["report"]["continuous_press_return_safety_handoff"]["visual_safety_kept_through_prepress"]


@pytest.mark.parametrize("stopped,error", [(True, RuntimeError()), (False, TimeoutError())])
def test_ambiguous_or_safety_stopped_motion_cannot_launch_recovery(stopped, error):
    source = Path(__file__).resolve().parents[1] / "run_online_replan_no_contact.py"
    tree = ast.parse(source.read_text())
    handler = next(n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)
                   and any(isinstance(x, ast.Constant) and x.value == "visual_safety_stopped"
                           for x in ast.walk(n)))
    assert not any(isinstance(n, ast.Attribute) and n.attr == "reset_latch"
                   for n in ast.walk(handler))
    recovery = next(n for n in handler.body if isinstance(n, ast.If)
                    and any(isinstance(x, ast.Name) and x.id == "robot_open"
                            for x in ast.walk(n.test)))
    assert not eval(compile(ast.Expression(recovery.test), str(source), "eval"), {
        "robot_open": True, "motion_started": True,
        "report": {"returned_to_origin": False}, "safety_stopped": stopped,
        "exc": error,
    })
