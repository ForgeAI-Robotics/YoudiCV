"""No hardware: first approach must stop at the exact photo/retract endpoint."""
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

from piper_handeye.button_light_hardware import DryValidatedSegmentExecutor
from tests.test_button_light_hardware import (
    FakeBridge, FakeGuardedBridge, FakeRobot, _plan, _guarded_config,
)


def test_observation_scope_checks_real_pose_and_all_remaining_paths():
    plan = _plan(); robot = FakeRobot(); robot.target = plan.observation_joints.copy()
    bridge = FakeBridge(); guard = FakeGuardedBridge()
    executor = DryValidatedSegmentExecutor(bridge, robot, 'test', guarded_bridge=guard,
        guarded_press_config=_guarded_config())
    executor.load_plan(plan, already_at_observation=True)
    assert executor._require_plan() is plan
    assert 'approach_to_observation' not in executor.validations
    assert 'observation_to_prepress' in executor.validations
    assert len(bridge.calls) == 4 and len(guard.calls) == 1
    assert guard.calls[0][1]['retract_target_joints'] == plan.observation_joints.tolist()
    assert all(options['dry_run'] for _, options in bridge.calls + guard.calls)
    executor.validations.pop('observation_to_prepress')
    with pytest.raises(RuntimeError, match='all dry validations'): executor._require_plan()


@pytest.mark.parametrize('fault', ['pose', 'hardware', 'collision', 'ambiguous', 'nan'])
def test_failed_observation_load_invalidates_previous_plan(fault):
    robot = FakeRobot(); bridge = FakeBridge(); plan = _plan()
    executor = DryValidatedSegmentExecutor(bridge, robot, 'test')
    executor.load_plan(plan)
    robot.target = plan.observation_joints.copy()
    if fault == 'pose': robot.target[4] += .009
    if fault == 'nan': robot.target[4] = np.nan
    if fault == 'hardware':
        robot.state = lambda **kw: NS(joints=robot.target, raw={'hardware_ok': 0})
    if fault == 'collision':
        bridge.execute = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('collision'))
    with pytest.raises((RuntimeError, ValueError)):
        executor.load_plan(plan, already_at_observation=True, already_at_prepress=fault == 'ambiguous')
    with pytest.raises(RuntimeError, match='all dry validations'): executor._require_plan()


@pytest.mark.parametrize('up_policy', [True, False])
def test_direct_photo_stop_keeps_original_lease_without_retreat(monkeypatch, up_policy):
    from tests.test_up_safety_handoff import setup
    from tests.test_up_stationary_light_reference import executor_fixture
    safety, vision, monitor, guard, metrics, calls, state = setup(monkeypatch)
    safety.stationary_reference_handoff = up_policy
    executor, trace, _ = executor_fixture()
    executor.motion_safety = safety
    executor.begin_press_attempt = safety.begin_outbound_cycle
    original_start = safety.near_pause.started
    def capture():
        assert safety._reference_hold
        for _ in range(45): assert not vision.request('safety')['visual_safety']['blocked']
        trace.append('capture'); return {}
    def press():
        safety.before_segment('observation_to_prepress')
        assert safety.near_pause.started == original_start
        trace.append('press'); return []
    executor.press_from_observation = press
    executor.press_with_stationary_reference(capture, already_at_observation=True,
                                             keep_approach_lease=True)
    assert trace == ['validate', 'capture', 'evidence', 'press']
    assert guard.arm_calls == 1 and not safety._reference_hold
    assert not any(c == 'safety_set_depth_degraded_commit' and not k['enabled'] for c, k in calls)


@pytest.mark.parametrize('direct,wrong_endpoint,sample_fault', [
    (True, False, None), (False, False, None), (True, True, None), (True, .0053, None),
    *[(True, False, fault) for fault in
      ('duplicate', 'stale', 'position_jump', 'weak', 'bad_plane', 'missing_pose')],
])
def test_real_receding_orchestrator_uses_last_solved_endpoint(monkeypatch, direct, wrong_endpoint, sample_fault):
    import run_button_light_closed_loop as m
    target = np.eye(4)
    target[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    target[0, 3] = .5
    estimate = NS(base_to_button=target, stable=True, sequence_range=(1, 1))
    robot = FakeRobot()
    def fk(q):
        pose = np.eye(4); pose[0, 3] = q[0]; return pose
    robot.state = lambda **kw: NS(joints=robot.target.copy(), raw={'hardware_ok': 1},
                                  base_to_gripper=fk(robot.target), timestamp=time.monotonic())
    monkeypatch.setattr(m, 'forward_kinematics', fk)
    monkeypatch.setattr(m, 'plan_receding_approach', lambda *a, **k: NS(dynamic_guard_distance_m=.1, speed_cap_m_s=.18))
    solves = []; photo_solves = []; planner_options = []
    class Planner:
        def __init__(self, **kwargs):
            planner_options.append(kwargs)
            self.guard = kwargs['stand_off_m'] != 0
        def solve(self, frame, seed):
            q = np.zeros(6); q[0] = .4 if self.guard else .49 + .001 * len(solves)
            solves.append(q); return NS(joints=q, solve_sec=.001)
    monkeypatch.setattr(m, 'FixedRollFastIkPlanner', Planner)
    def photo(prepress, *args):
        q = prepress.copy(); q[0] -= .04; photo_solves.append(q)
        return q, {'panel_distance_m': .05}
    monkeypatch.setattr(m, 'solve_camera_centered_observation_joints', photo)
    def points(q, stamp=1):
        return [{'positions': q.tolist(), 'velocities': [0]*6, 'accelerations': [0]*6,
                 'time_from_start_sec': float(stamp)}]
    monkeypatch.setattr(m, 'generate_timed_joint_trajectory', lambda start, end, **kw: (points(end), NS(duration_sec=1)))
    monkeypatch.setattr(m, 'select_future_handoff', lambda *a, **kw: NS(active_index=0, positions=np.zeros(6)))
    ends = []
    audit = NS(duration_sec=1, cartesian_chord_m=.05, cartesian_max_overshoot_m=0,
               cartesian_max_backtrack_step_m=0, cartesian_max_cross_track_m=0, joint_max_overshoot_rad=0)
    def splice(active, endpoint, **kw):
        ends.append(endpoint.copy())
        return NS(active_index=1, active_time_sec=1, positions=np.zeros(6)), points(endpoint), audit, []
    monkeypatch.setattr(m, 'generate_first_feasible_splice', splice)
    class History:
        def __init__(self, **kw): pass
        def add(self, *args): pass
        def transform_at(self, *a, **kw):
            if sample_fault == 'missing_pose':
                raise RuntimeError('no time-matched robot pose')
            return np.eye(4)
    monkeypatch.setattr(m, 'RobotTransformHistory', History)
    monkeypatch.setattr(m, 'constrain_button_ray_to_reference_plane',
                        lambda camera, measured, reference: (measured, 0.))
    real_window = m.make_button_window
    windows = []
    def make_window(*args, **kwargs):
        window = real_window(*args, **kwargs)
        windows.append(window)
        return window
    monkeypatch.setattr(m, 'make_button_window', make_window)
    prepared = []; callbacks = []
    class Bridge:
        def __init__(self, *a, **kw): pass
        def prepare(self, p, **kw): prepared.append(p); return {}
        def start_active(self, **kw): return {}
        def schedule_replacement(self, **kw):
            robot.target = np.array(prepared[-1][-1]['positions'])
            if wrong_endpoint: robot.target[4] += .009 if wrong_endpoint is True else wrong_endpoint
            return {}
    monkeypatch.setattr(m, 'PreparedSpliceUdpClient', Bridge)
    monkeypatch.setattr(m, 'wait_terminal', lambda *a: ({'execution_status': 'replacement_complete'}, []))
    sample = dict(sequence=1, capture_time_unix_sec=time.time()-.02,
                  camera_to_button=target.tolist(), confidence=.95, plane_rms_m=.001)
    second_target = target.copy()
    if sample_fault == 'position_jump': second_target[1, 3] += .02
    second = dict(sample, sequence=1 if sample_fault == 'duplicate' else 2,
                  capture_time_unix_sec=time.time()-.01,
                  camera_to_button=second_target.tolist(),
                  confidence=.69 if sample_fault == 'weak' else .95,
                  plane_rms_m=.02 if sample_fault == 'bad_plane' else .001)
    if sample_fault == 'stale':
        sample['capture_time_unix_sec'] -= 1
        second['capture_time_unix_sec'] -= 1
    vision = NS(request=lambda *a, **kw: {'samples': [sample, second]})
    safety = NS(before_segment=lambda n: None, after_segment=lambda n: None, _assert_clear=lambda: None)
    def execute():
        return m.execute_receding_prepress(vision=vision, robot=robot, safety=safety,
            config={'timed_trajectory': {'host': 'no-network'}, 'detector': {},
                    'button_geometry': {}}, handeye=np.eye(4),
            target_label='down' if direct else '8', roll_deg=0, initial_target=estimate,
            token='test', stop_at_reference=direct,
            replacement_ready=lambda e, p, **kw: callbacks.append((p, kw)))
    if sample_fault:
        with pytest.raises(RuntimeError, match='moving RGB-D target was not stable'):
            execute()
        assert not ends and not callbacks  # No near-panel replacement/press plan.
        assert len(planner_options) == 1  # Guard IK only.
        assert windows[0].size == 2
        return
    if wrong_endpoint is True:
        with pytest.raises(RuntimeError, match='observation pose confirmation failed'): execute()
        return
    _, solved, result = execute()
    assert windows[0].size == 2
    assert windows[0].max_input_age_sec == .35
    assert windows[0].max_latest_age_sec == .35
    assert windows[0].max_span_sec == 1.05
    assert windows[0].max_position_rms_m == .0035
    assert windows[0].max_position_jump_m == .007
    assert windows[0].max_normal_rms_deg == 1.5
    assert result['window_sequence_range'] == [1, 2]
    assert len(ends) == 2  # Re-solved from the actual future handoff.
    assert planner_options[0]['downstream_max_segment_joint_deltas_rad'] == (
        None, m.PRESS_SEGMENT_MAX_JOINT_DELTA_RAD)
    assert all(options['downstream_max_segment_joint_deltas_rad'] == (
        m.PRESS_SEGMENT_MAX_JOINT_DELTA_RAD,) for options in planner_options[1:])
    assert not np.array_equal(ends[0], ends[1])
    if wrong_endpoint:
        assert np.max(np.abs(ends[-1]-robot.target))==pytest.approx(.0053)
        assert result['observation_pose_confirmation']['outcome']=='stable'
    else:
        assert np.array_equal(ends[-1], robot.target)
    assert result['endpoint'] == ('reference' if direct else 'prepress')
    if direct:
        assert len(photo_solves) == 2
        assert np.array_equal(callbacks[0][1]['reference_joints'], ends[-1])
        assert not np.array_equal(robot.target, solved.joints)
    else:
        assert not photo_solves and callbacks[0][1] == {}
        assert np.array_equal(robot.target, solved.joints)


@pytest.mark.parametrize('label', ['up', 'down', *map(str, range(1, 17))])
@pytest.mark.parametrize('attempt', [1, 2, 3])
def test_production_first_attempt_and_retry_dispatch_are_distinct(label, attempt):
    import ast
    from pathlib import Path
    from unittest.mock import Mock
    import run_button_light_closed_loop as m
    tree = ast.parse(Path(m.__file__).read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'press_attempt')
    reference = np.ones(6); solved = NS(joints=np.zeros(6)); estimate = object(); plan = object()
    executor = Mock(); executor.press_with_stationary_reference.return_value = 'fixed'
    executor.press_from_prepress.return_value = 'numeric'
    executor.press_from_observation.return_value = 'retry'
    builds = []
    def build(refreshed, audit, **kw):
        builds.append(kw); return plan
    def receding(**kw):
        assert attempt == 1
        assert kw['stop_at_reference'] is True
        if kw['stop_at_reference']:
            kw['replacement_ready'](estimate, solved, reference_joints=reference, reference_alignment={'test': True})
        else: kw['replacement_ready'](estimate, solved)
        return estimate, solved, {}
    observer = NS(capture_stationary_off_reference=lambda: {})
    namespace = dict(vars(m), args=NS(target_label=label), failure_images=Mock(), checkpoint=lambda *a: None,
        executor=executor, observer=observer, latest_target={'estimate': estimate},
        build_plan=build, execute_receding_prepress=receding, vision=None, robot=None,
        safety=None, config={}, handeye=np.eye(4), roll_deg=0,
        trajectory_cfg={'token': 'test'})
    exec(compile(ast.Module(body=[function], type_ignores=[]), m.__file__, 'exec'), namespace)
    namespace['press_attempt'](attempt)
    namespace['failure_images'].begin_attempt.assert_called_once_with(attempt)
    if attempt == 1:
        fixed = True
        executor.load_plan.assert_called_once_with(plan, already_at_prepress=not fixed, already_at_observation=fixed)
        assert np.array_equal(builds[0]['current_override'], reference if fixed else solved.joints)
        if fixed:
            assert np.array_equal(builds[0]['reference_joints'], reference)
            executor.press_with_stationary_reference.assert_called_once_with(
                observer.capture_stationary_off_reference, already_at_observation=True, keep_approach_lease=True)
        else: executor.press_from_prepress.assert_called_once()
    else:
        assert not builds
        executor.load_plan.assert_not_called()
        executor.press_with_stationary_reference.assert_called_once_with(
            observer.capture_stationary_off_reference, already_at_observation=True)
        executor.press_from_observation.assert_not_called()
