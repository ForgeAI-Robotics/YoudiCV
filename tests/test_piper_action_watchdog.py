from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from piper_action_watchdog import wait_for_action_result


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now


class FakeAction:
    def __init__(self, clock, results, on_wait=None):
        self.clock, self.results, self.on_wait = clock, iter(results), on_wait
        self.waits = []
        self.cancel_goal_calls = 0
        self.cancel_all_calls = 0
    def wait_for_result(self, duration):
        self.waits.append(float(duration))
        self.clock.now += float(duration)
        if self.on_wait:
            self.on_wait()
        return next(self.results, False)
    def cancel_goal(self): self.cancel_goal_calls += 1
    def cancel_all_goals(self): self.cancel_all_calls += 1


def test_watchdog_returns_after_terminal_action_result():
    clock = FakeClock()
    action = FakeAction(clock, [False, True])
    wait_for_action_result(
        action, safety_generation=lambda: 4, initial_generation=4,
        duration_sec=1.0, duration_factory=float, monotonic=clock,
    )
    assert action.waits == [0.05, 0.05]
    assert action.cancel_goal_calls == action.cancel_all_calls == 0


def test_watchdog_cancels_immediately_when_safety_generation_changes():
    clock, generation = FakeClock(), [7]
    action = FakeAction(clock, [False], on_wait=lambda: generation.__setitem__(0, 8))
    with pytest.raises(RuntimeError, match="visual safety generation 8"):
        wait_for_action_result(
            action, safety_generation=lambda: generation[0], initial_generation=7,
            duration_sec=10.0, duration_factory=float, monotonic=clock,
        )
    assert action.cancel_all_calls == 1
    assert action.cancel_goal_calls == 0


def test_watchdog_uses_wall_clock_deadline_when_action_never_completes():
    clock = FakeClock()
    action = FakeAction(clock, [])
    with pytest.raises(RuntimeError, match="timed out"):
        wait_for_action_result(
            action, safety_generation=lambda: 1, initial_generation=1,
            duration_sec=0.10, grace_sec=0.0, duration_factory=float,
            monotonic=clock,
        )
    assert action.cancel_goal_calls == 1


def test_stop_during_mode_or_baseline_prevents_send():
    from tools.piper_action_watchdog import send_goal_with_safety
    from unittest.mock import MagicMock
    c=MagicMock()
    with pytest.raises(RuntimeError):
        send_goal_with_safety(c,object(),safety_generation=lambda:2,
                              safety_latched=lambda:False,initial_generation=1)
    c.send_goal.assert_not_called()


def test_stop_between_cancel_and_send_is_cancelled_again():
    from tools.piper_action_watchdog import send_goal_with_safety
    from unittest.mock import MagicMock
    state={'generation':1}
    c=MagicMock()
    c.send_goal.side_effect=lambda _:state.update(generation=2)
    with pytest.raises(RuntimeError):
        send_goal_with_safety(c,object(),safety_generation=lambda:state['generation'],
                              safety_latched=lambda:False,initial_generation=1)
    c.cancel_all_goals.assert_called_once()


def test_latched_stop_blocks_retract_even_without_changed_generation():
    from tools.piper_action_watchdog import send_goal_with_safety
    from unittest.mock import MagicMock
    c=MagicMock()
    with pytest.raises(RuntimeError):
        send_goal_with_safety(c,object(),safety_generation=lambda:1,
                              safety_latched=lambda:True,initial_generation=1)
    c.send_goal.assert_not_called()


def test_normal_goal_sent_exactly_once():
    from tools.piper_action_watchdog import send_goal_with_safety
    from unittest.mock import MagicMock
    c=MagicMock();goal=object()
    send_goal_with_safety(c,goal,safety_generation=lambda:1,
                          safety_latched=lambda:False,initial_generation=1)
    c.send_goal.assert_called_once_with(goal)
    c.cancel_all_goals.assert_not_called()


@pytest.mark.parametrize('filename,classname',[
    ('orangepi_timed_trajectory_bridge.py','Bridge'),
    ('orangepi_guarded_press_bridge.py','GuardedBridge'),
])
def test_actual_bridge_execute_rejects_stop_during_handoff(filename,classname):
    import ast
    import numpy as np
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from tools.piper_action_watchdog import require_safety_generation,send_goal_with_safety
    source=Path(__file__).resolve().parents[1]/'tools'/filename
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='execute' for m in n.body))
    method_name = '_execute' if filename.startswith('orangepi_guarded') else 'execute'
    method=next(m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name==method_name)
    state={'generation':1,'latched':False}
    fake_rospy=SimpleNamespace(get_param=lambda key,default:state['generation'] if key.endswith('generation') else state['latched'])
    ns={'np':np,'rospy':fake_rospy,'require_safety_generation':require_safety_generation,
        'send_goal_with_safety':send_goal_with_safety}
    module=ast.Module(body=[method],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
    bridge=MagicMock();bridge.current_joints.return_value=np.zeros(6)
    bridge.activate_actuation_mode.side_effect=lambda _:state.update(generation=2,latched=True)
    points=[{'positions':[0]*6}]
    with pytest.raises(RuntimeError,match='handoff'):
        if filename.startswith('orangepi_guarded'):
            ns['_execute'](bridge,points,None,[1,0,0],'task',None,None,None,[])
        else:ns['execute'](bridge,points)
    bridge.client.send_goal.assert_not_called()
    bridge.baseline.assert_not_called()
