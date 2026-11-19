"""Virtual CAN/encoder collection and actual bridge integration, no robot IO."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from piper_handeye.current_baseline import collect_stable_current_baseline, baseline_audit_summary
from piper_handeye.guarded_press import MotorCurrentSample, RobustCurrentBaseline
from piper_handeye.joint_convergence import JointConvergenceError
from tools.piper_action_watchdog import require_safety_generation


class Feed:
    def __init__(self, q=None, current=None, period=.005):
        self.t = 100.
        self.q = q or (lambda t: np.zeros(6))
        self.current = current or (lambda t: np.full(6, .1))
        self.period = period
        self.snapshot_override = None
        self.safety = lambda: None

    def clock(self):
        return self.t

    def snapshot(self):
        if self.snapshot_override is not None:
            return self.snapshot_override()
        tick = int((self.t-100.)/.05)
        received = 100.+tick*.05
        return tick, received, self.q(received-100.).tolist()

    def wait_next(self, sequence, timeout_sec):
        self.t += min(self.period, timeout_sec)
        return sequence+1, MotorCurrentSample(self.t, self.current(self.t-100.))

    def collect(self, **kwargs):
        audit = kwargs.pop('audit', {})
        baseline = collect_stable_current_baseline(
            self.wait_next, self.snapshot, self.safety, clock=self.clock, audit=audit,
            expected_start=np.zeros(6), **kwargs)
        return baseline, audit


def moving_then_still(until):
    def joints(t):
        result = np.zeros(6)
        result[2] = .006*np.sin(2*np.pi*t/.20) if t < until else 0.
        return result
    return joints


def test_normal_stable_baseline_returns_early_and_keeps_original_fit():
    f = Feed()
    baseline, audit = f.collect()
    assert .30 <= audit['elapsed_sec'] < .45
    np.testing.assert_allclose(baseline.median_a, .1)
    np.testing.assert_allclose(baseline.sigma_a, .01)
    assert baseline.sample_count >= 40
    assert audit['outcome'] == 'stable' and not audit['discarded_windows']


@pytest.mark.parametrize('until', [1., 2.5, 4.5])
def test_motion_is_discarded_until_stable_within_one_five_second_budget(until):
    f = Feed(q=moving_then_still(until),
             current=lambda t: np.full(6, 1.5 if t < until else .12))
    baseline, audit = f.collect()
    assert until+.3 <= audit['elapsed_sec'] < 5.
    np.testing.assert_allclose(baseline.median_a, .12)
    assert audit['discarded_windows']
    assert any(row.get('max_joint_index_1based') == 3 for row in audit['discarded_windows'])
    assert max(audit['joint_spread_rad']) <= .003+1e-12


def test_continuous_motion_ends_at_five_seconds_without_fitting_bad_data():
    f = Feed(q=moving_then_still(10.))
    with pytest.raises(TimeoutError, match='within 5.0s') as caught:
        f.collect()
    audit = caught.value.baseline_acquisition_audit
    assert 5. <= f.t-100. < 5.01
    assert audit['outcome'] == 'failed' and len(audit['discarded_windows']) > 10


def test_minimum_forty_distinct_current_samples_is_not_removed():
    f = Feed(period=.02)
    baseline, audit = f.collect()
    assert baseline.sample_count >= 40 and audit['elapsed_sec'] >= .8-1e-8


def test_short_current_timeout_discards_old_window_and_recovers():
    f = Feed()
    original = f.wait_next
    def wait(sequence, timeout_sec):
        if .15 < f.t-100. < .23:
            f.t += timeout_sec
            raise TimeoutError('no fresh complete motor-current sample')
        return original(sequence, timeout_sec)
    f.wait_next = wait
    _, audit = f.collect()
    assert audit['current_wait_timeouts'] > 0
    assert audit['elapsed_sec'] >= .5


def test_duplicate_current_packets_cannot_supply_a_fresh_baseline():
    f = Feed()
    def wait(sequence, timeout_sec):
        f.t += min(.005, timeout_sec)
        return sequence+1, MotorCurrentSample(100., np.zeros(6))
    f.wait_next = wait
    with pytest.raises(TimeoutError):
        f.collect()
    assert f.t-100. >= 5.


@pytest.mark.parametrize('snapshot', [
    lambda: None,
    lambda: (0, 99.8, [0.]*6),
    lambda: (0, 100., [float('nan')]*6),
    lambda: (0, 100., [0.]*5),
])
def test_missing_stale_or_invalid_encoders_still_fail_immediately(snapshot):
    f = Feed()
    f.snapshot_override = snapshot
    with pytest.raises(RuntimeError, match='feedback'):
        f.collect()
    assert f.t == 100.


def test_start_pose_cannot_drift_out_of_original_point_zero_three_limit():
    f = Feed(q=lambda t: np.full(6, .031))
    with pytest.raises(RuntimeError, match='start changed'):
        f.collect()


def test_stop_during_collection_propagates_unchanged_with_audit():
    f = Feed()
    original = RuntimeError('new stop generation')
    def check():
        if f.t >= 100.15:
            raise original
    f.safety = check
    with pytest.raises(RuntimeError) as caught:
        f.collect()
    assert caught.value is original
    assert caught.value.baseline_acquisition_audit['elapsed_sec'] < .2


def test_receiver_failure_is_not_treated_as_recoverable_motion():
    f = Feed()
    f.wait_next = Mock(side_effect=RuntimeError('CAN receiver failed'))
    with pytest.raises(RuntimeError, match='CAN receiver failed'):
        f.collect()


def method(name, namespace):
    source = Path('tools/orangepi_guarded_press_bridge.py')
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'GuardedBridge')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


def test_actual_bridge_baseline_uses_cached_encoders_and_checks_stop():
    bridge = Mock()
    bridge.current.sequence = 31
    captured = {}
    def collect(*args, **kwargs):
        captured.update(kwargs)
        args[2]()
        return 'reference'
    state = dict(generation=7, latched=False)
    ros = SimpleNamespace(is_shutdown=lambda: False,
                          get_param=lambda key, default: state['generation'] if key.endswith('generation') else state['latched'])
    fn = method('baseline', dict(rospy=ros, require_safety_generation=require_safety_generation,
                                 collect_stable_current_baseline=collect))
    assert fn(bridge, safety_generation=7, expected_start=np.zeros(6), audit={}) == 'reference'
    assert captured['timeout_sec'] == 5. and captured['stable_sec'] == .3
    bridge.current_joints.assert_not_called()
    state.update(generation=8, latched=True)
    with pytest.raises(RuntimeError, match='handoff'):
        fn(bridge, safety_generation=7, expected_start=np.zeros(6), audit={})


def test_actual_execute_never_sends_forward_goal_when_baseline_fails():
    bridge = Mock()
    bridge.current_joints.return_value = np.zeros(6)
    bridge.baseline.side_effect = TimeoutError('no stable current baseline within 5.0s')
    ros = SimpleNamespace(get_param=lambda key, default: 7 if key.endswith('generation') else False)
    fn = method('_execute', dict(np=np, rospy=ros, require_safety_generation=require_safety_generation))
    with pytest.raises(TimeoutError, match='5.0s'):
        fn(bridge, [dict(positions=[0.]*6)], None, [1,0,0], 'task', None, None, None, [])
    bridge.client.send_goal.assert_not_called()
    bridge.goal.assert_not_called()
    assert bridge.baseline.call_args.kwargs['safety_generation'] == 7


def test_actual_wrapper_saves_full_failure_audit_and_keeps_udp_summary_bounded(tmp_path):
    f = Feed(q=moving_then_still(10.))
    try:
        f.collect()
    except TimeoutError as error:
        failed = error
    full = failed.baseline_acquisition_audit
    short = baseline_audit_summary(full)
    assert short['joint_samples_count'] > 90
    assert short['discarded_window_count'] > 10
    assert 'joint_samples' not in short and len(short['discarded_windows_tail']) == 8
    assert len(json.dumps(short).encode()) < 12000
    bridge = Mock()
    bridge._execute.side_effect = failed
    fn = method('execute', dict(os=os, json=json, rospy=SimpleNamespace(logerr=Mock()),
        baseline_audit_summary=baseline_audit_summary, GUARDED_PRESS_TRACE_DIRECTORY=str(tmp_path)))
    with pytest.raises(TimeoutError):
        fn(bridge, [], None, None, 'baseline-failure')
    saved = json.loads((tmp_path/'baseline-failure.jsonl.retract.json').read_text())
    assert saved['baseline_acquisition'] == full
    assert saved['full_feedback_trace_samples'] == 0


def test_actual_wrapper_saves_full_success_audit_and_returns_summary(tmp_path):
    _, full = Feed(q=moving_then_still(1.)).collect()
    bridge = Mock()
    bridge._execute.return_value = dict(status='contact_retracted', baseline_acquisition=full)
    fn = method('execute', dict(os=os, json=json, rospy=SimpleNamespace(logerr=Mock()),
        baseline_audit_summary=baseline_audit_summary, GUARDED_PRESS_TRACE_DIRECTORY=str(tmp_path)))
    result = fn(bridge, [], None, None, 'baseline-success')
    saved = json.loads((tmp_path/'baseline-success.jsonl.retract.json').read_text())
    assert saved['baseline_acquisition'] == full
    assert 'joint_samples' not in result['baseline_acquisition']
    assert result['baseline_acquisition']['outcome'] == 'stable'


@pytest.mark.parametrize('error,is_timeout', [
    (TimeoutError('no stable current baseline within 5.0s'), True),
    (RuntimeError('trajectory cancelled during safety handoff'), False),
])
def test_actual_failure_reply_preserves_stop_identity_and_baseline_diagnostics(error, is_timeout):
    error.baseline_acquisition_audit = dict(outcome='failed', joint_samples=[], discarded_windows=[])
    bridge = Mock()
    bridge.validate_request.side_effect = error
    fn = method('_process', dict(JointConvergenceError=JointConvergenceError,
                                 baseline_audit_summary=baseline_audit_summary))
    fn(bridge, ('127.0.0.1', 1), 'task', dict(command='execute'))
    call = bridge.response.call_args
    assert call.args[2] == 'failed' and str(error) in call.kwargs['error']
    assert call.kwargs['baseline_acquisition']['outcome'] == 'failed'
    assert (call.kwargs.get('failure_code') == 'PRESS_CURRENT_BASELINE_NOT_READY') is is_timeout
    bridge.busy.release.assert_called_once()
