"""Full elevator route orchestration with no network or hardware effects."""
from unittest.mock import MagicMock, patch
import math

import pytest

from piper_handeye.android_agx_protocol import AndroidPacket
from piper_handeye.external_call_mission import ExternalCallMission
from tests.test_external_call_mission import ARM, GOAL, ORIGIN, sample
from tests.test_navigation_link_gui import InlineThread, gui, packet, tick, worker_reply
from tools.navigation_arm_worker import Worker
from tools.navigation_task_gui import FULL_ROUTE, CALL_ONLY_ROUTE
from tests.test_android_command_gui import FakeVariable


FLOOR_GOAL = (-1.829298, 2.820949, .708453, .705758)


def full_mission():
    m = ExternalCallMission(include_floor_panel=True)
    mid = m.begin(16, 8, 0)
    assert m.origin_checked(mid, ORIGIN) == 'ready_navigation'
    m.sent('liftout', 10)
    return m, mid


def finish_call(m, mid):
    m.phase = 'arm_call'
    assert m.arm_finished(mid, ARM) == 'check_origin'
    assert m.origin_checked(mid, ORIGIN) == 'navigate_floor'
    assert m.phase == 'preparing_floor'
    m.sent('liftin', 50)


def test_each_navigation_leg_requires_new_arrival_and_continuous_stillness():
    m, mid = full_mission()
    m.arrive(20)
    m.stable_since = m.last_sample = 20
    finish_call(m, mid)
    assert m.arrive_received is None and m.stable_since is None and m.last_sample is None
    assert m.expected_navigation_goal == 'liftin'
    for t in (51, 51.2, 51.4, 51.6):
        assert m.observe(sample(t, FLOOR_GOAL), FLOOR_GOAL, t, True) is None
    m.arrive(52)
    for t in (52, 52.2, 52.4):
        assert m.observe(sample(t, FLOOR_GOAL), FLOOR_GOAL, t, True) is None
    assert m.observe(sample(52.6, FLOOR_GOAL), FLOOR_GOAL, 52.6, True) == 'pause_for_arm'
    assert m.chassis_paused(53, False) is None
    assert m.chassis_paused(53, True) == 'press_floor'
    assert m.arm_finished(mid, ARM) == 'check_origin'
    assert m.phase == 'checking_floor_origin'
    assert m.origin_checked(mid, ORIGIN) == 'navigate_back'
    m.sent('back', 54)
    assert m.charged() and m.phase == 'completed'


@pytest.mark.parametrize('field', list(ARM))
@pytest.mark.parametrize('phase', ['arm_call', 'arm_floor'])
def test_each_arm_segment_requires_all_evidence_before_navigation(field, phase):
    m, mid = full_mission()
    m.phase = phase
    evidence = dict(ARM)
    evidence.pop(field)
    assert m.arm_finished(mid, evidence) == 'failed'
    assert m.phase == 'failed'


@pytest.mark.parametrize('position_error,velocity,fresh,resumed', [
    (.131, (0, 0, 0), True, True),
    (0, (.02, 0, 0), True, True),
    (0, (0, 0, 0), False, True),
    (0, (0, 0, 0), True, False),
])
def test_inner_handoff_retains_thirteen_cm_gate_and_fresh_paused_navigation(
    position_error, velocity, fresh, resumed,
):
    m, mid = full_mission()
    finish_call(m, mid)
    m.arrive(51)
    pose = (FLOOR_GOAL[0] + position_error, *FLOOR_GOAL[1:])
    for t in (51, 51.2, 51.4, 51.6):
        observation = sample(t if fresh else t - 3, pose, velocity)
        assert m.observe(observation, FLOOR_GOAL, t, resumed) is None
    assert m.phase == 'navigating_floor'
    assert m.position_tolerance_for('liftout') == .12
    assert m.position_tolerance_for('liftin') == .13


@pytest.mark.parametrize('position_error', [.08584, .11999, .12758, .12999])
def test_inner_test_gate_allows_parking_only_after_arrival_stillness_and_pause(position_error):
    m, mid = full_mission()
    finish_call(m, mid)
    pose = (FLOOR_GOAL[0] + position_error, *FLOOR_GOAL[1:])
    assert m.observe(sample(50.8, pose), FLOOR_GOAL, 50.8, True) is None
    m.arrive(51)
    for t in (51, 51.2, 51.4):
        assert m.observe(sample(t, pose), FLOOR_GOAL, t, True) is None
    assert m.observe(sample(51.6, pose), FLOOR_GOAL, 51.6, True) == 'pause_for_arm'
    assert m.chassis_paused(52, False) is None
    assert m.chassis_paused(52, True) == 'press_floor'


RECORDED_PARKED_LIFTIN = (-1.7325968742370605, 2.904165744781494,
                          .7212854275378756, .692637951617946)


def test_recorded_stationary_stop_passes_new_inner_gate_but_not_original_gate():
    distance = math.hypot(RECORDED_PARKED_LIFTIN[0]-FLOOR_GOAL[0],
                          RECORDED_PARKED_LIFTIN[1]-FLOOR_GOAL[1])
    assert distance == pytest.approx(.1275779539569755)
    old, mid = full_mission()
    old.floor_position_tolerance_m = .12
    finish_call(old, mid)
    old.arrive(51)
    for t in (51, 51.2, 51.4, 51.6):
        assert old.observe(sample(t, RECORDED_PARKED_LIFTIN), FLOOR_GOAL, t, True) is None

    new, mid = full_mission()
    finish_call(new, mid)
    new.arrive(51)
    for t in (51, 51.2, 51.4):
        assert new.observe(sample(t, RECORDED_PARKED_LIFTIN), FLOOR_GOAL, t, True) is None
    assert new.observe(sample(51.6, RECORDED_PARKED_LIFTIN), FLOOR_GOAL, 51.6, True) == 'pause_for_arm'
    assert new.chassis_paused(52, False) is None
    assert new.chassis_paused(52, True) == 'press_floor'


def test_inner_gate_never_latches_a_passing_position_while_still_moving():
    m, mid = full_mission()
    finish_call(m, mid)
    m.arrive(51)
    inside = (FLOOR_GOAL[0] + .10, *FLOOR_GOAL[1:])
    outside = (FLOOR_GOAL[0] + .13001, *FLOOR_GOAL[1:])
    for t in (51, 51.2, 51.4):
        assert m.observe(sample(t, inside, (.03, 0, 0)), FLOOR_GOAL, t, True) is None
    for t in (51.6, 51.8, 52, 52.2):
        assert m.observe(sample(t, outside), FLOOR_GOAL, t, True) is None
    assert m.phase == 'navigating_floor'
    assert m.stable_since is None


def test_inner_relaxed_distance_does_not_relax_yaw():
    m, mid = full_mission()
    finish_call(m, mid)
    m.arrive(51)
    yaw = 2*math.atan2(FLOOR_GOAL[2], FLOOR_GOAL[3]) + .101
    pose = (FLOOR_GOAL[0] + .08584, FLOOR_GOAL[1], math.sin(yaw/2), math.cos(yaw/2))
    for t in (51, 51.2, 51.4, 51.6):
        assert m.observe(sample(t, pose), FLOOR_GOAL, t, True) is None
    assert m.phase == 'navigating_floor'


def test_inner_navigation_timeout_and_infinite_button_wait_are_distinct():
    m, mid = full_mission()
    finish_call(m, mid)
    assert m.observe(sample(231, FLOOR_GOAL), FLOOR_GOAL, 231, True) == 'failed'
    m, mid = full_mission()
    m.phase = 'arm_floor'
    assert m.observe(sample(100000, FLOOR_GOAL), FLOOR_GOAL, 100000, True) is None
    assert m.phase == 'arm_floor' and not m.charged()


@pytest.mark.parametrize('target', [1, 8, 16])
def test_floor_worker_calls_existing_numeric_pipeline_and_deduplicates(target):
    d = MagicMock()
    d.active = None
    d.dispatch_motion.return_value = {'event': 'motion_started'}
    messages = []
    w = Worker(d, messages.append)
    request = dict(operation='press_floor', request_id='floor-1', mission_id='m',
                   current_floor=3, target_floor=target)
    w.handle(request)
    assert messages[-1]['event'] == 'started'
    assert d.dispatch_motion.call_args.args[0].button == target
    d.start_trip.assert_not_called()
    w.handle(request)
    assert d.dispatch_motion.call_count == 1


@pytest.mark.parametrize('target', [None, True, '8', 0, 17])
def test_invalid_floor_never_dispatches(target):
    d = MagicMock()
    d.active = None
    messages = []
    w = Worker(d, messages.append)
    w.handle(dict(operation='press_floor', request_id='floor-1', mission_id='m',
                  current_floor=3, target_floor=target))
    assert messages[-1]['event'] == 'error'
    d.dispatch_motion.assert_not_called()


def arrive_and_press(g, when, sequence, revision):
    goal = list(g.state.goal_pose)
    tick(g, when, goal_revision=revision, goal=goal, pose=goal)
    tick(g, when + .1, pose=goal)
    arrival = packet(sequence, 1, 'arrive')
    with patch('tools.navigation_task_gui.time.monotonic', return_value=when + 1):
        g.packet(arrival)
    for dt in (1, 1.2, 1.4, 1.6, 1.7, 1.9, 2.1, 2.3):
        tick(g, when + dt, pose=goal)
    return arrival


def gui_at_floor_navigation(current=16, target=8):
    g = gui()
    g.mission.include_floor_panel = True
    g.pending_current_floor = current
    with patch('tools.navigation_task_gui.time.monotonic', return_value=0):
        g.packet(packet(1, 10, str(target)))
    worker_reply(g, 0, ORIGIN)
    tick(g, 10)
    old_arrival = arrive_and_press(g, 11, 2, 1)
    assert g.mission.phase == 'arm_call'
    assert g._arm_client.requests[1]['direction'] == ('up' if target > current else 'down')
    for t in (50, 50.2):
        tick(g, t)
    assert [x[1] for x in g.sent] == ['liftout']
    worker_reply(g, 1, ARM)
    with patch('tools.navigation_task_gui.time.monotonic', return_value=51):
        worker_reply(g, 2, ORIGIN)
    assert g.mission.phase == 'preparing_floor'
    assert not g.resume_confirmed and not g.state.arrived
    tick(g, 51.1)
    assert [x[1] for x in g.sent] == ['liftout', 'liftin']
    assert g.mission.target_floor == target
    return g, old_arrival


def test_gui_recorded_stop_handoff_keeps_numeric_target_and_pause_confirmation():
    g, _ = gui_at_floor_navigation(3, 4)
    tick(g, 52, goal_revision=2, goal=list(g.state.goal_pose), pose=RECORDED_PARKED_LIFTIN)
    tick(g, 52.1, pose=RECORDED_PARKED_LIFTIN)
    with patch('tools.navigation_task_gui.time.monotonic', return_value=53):
        g.packet(packet(3, 1, 'arrive'))
    for dt in (0, .2, .4):
        tick(g, 53+dt, pose=RECORDED_PARKED_LIFTIN)
        assert g.mission.phase == 'navigating_floor'
        assert len(g._arm_client.requests) == 3
    for dt in (.6, .7, .9, 1.1, 1.3):
        tick(g, 53+dt, pose=RECORDED_PARKED_LIFTIN)
    assert g.mission.phase == 'arm_floor'
    assert g.arm_pause_confirmed
    assert g._arm_client.requests[3]['operation'] == 'press_floor'
    assert g._arm_client.requests[3]['target_floor'] == 4
    assert g.mission.position_tolerance_for('liftout') == .12


@pytest.mark.parametrize('current,target', [(1, 4), (16, 4), (1, 11), (16, 11)])
def test_gui_full_route_reuses_original_target_and_verifies_charge(current, target):
    g, old_arrival = gui_at_floor_navigation(current, target)
    g.packet(old_arrival)
    assert g.mission.arrive_received is None  # old outer arrival is not inner arrival
    assert not g.resume_confirmed
    arrive_and_press(g, 52, 3, 2)
    assert g.mission.phase == 'arm_floor'
    request = g._arm_client.requests[3]
    assert request['operation'] == 'press_floor' and request['target_floor'] == target
    worker_reply(g, 1, ARM)  # late outer success must not complete the floor task
    assert g.mission.phase == 'arm_floor' and len(g._arm_client.requests) == 4
    for t in (150, 150.2):
        tick(g, t, pose=list(FLOOR_GOAL))
    assert [x[1] for x in g.sent] == ['liftout', 'liftin']
    worker_reply(g, 3, ARM)
    assert g.mission.phase == 'checking_floor_origin'
    with patch('tools.navigation_task_gui.time.monotonic', return_value=151):
        worker_reply(g, 4, ORIGIN)
    tick(g, 151.1)
    assert [x[1] for x in g.sent] == ['liftout', 'liftin', 'back']
    assert g.mission.phase == 'charging' and not g.resume_confirmed
    g.packet(AndroidPacket(('192.168.168.110', 5002), g.dispatch_sequence,
                           'active_result', {'result': True}, ''))
    evidence = dict(charge_request_revision=1, charge_service_revision=1,
                    charge_request_time=1151.15, charge_service_time=1151.16,
                    charge_request_age=.1, charge_service_age=.1,
                    charge_work_age=0, real_charging=True, charge_signal=True)
    tick(g, 151.2, **evidence)
    tick(g, 151.3, **evidence)
    tick(g, 152.4, **evidence)
    assert g.mission.phase == 'completed'
    assert g.mission.current_floor == current  # mock scene does not change physical floor


@pytest.mark.parametrize('target', [4, 11])
def test_gui_recorded_liftin_pose_dispatches_original_manual_target(target):
    g, _ = gui_at_floor_navigation(current=16, target=target)
    parked = [-1.7725229263305657, 2.8853304386138894,
              .7128714877579979, .7012946898000145]
    tick(g, 52, goal_revision=2, goal=list(FLOOR_GOAL), pose=parked)
    tick(g, 52.1, pose=parked)
    with patch('tools.navigation_task_gui.time.monotonic', return_value=53):
        g.packet(packet(3, 1, 'arrive'))
    for t in (53, 53.2, 53.4, 53.6, 53.7, 53.9, 54.1, 54.3):
        tick(g, t, pose=parked)
    assert g.mission.phase == 'arm_floor'
    request = g._arm_client.requests[3]
    assert request['operation'] == 'press_floor'
    assert request['target_floor'] == target
    handoff = [r for r in g.log_rows if r['event'] == '内呼到位交接'][-1]
    assert handoff['position_tolerance_m'] == .13


def test_wrong_inner_plan_cannot_release_pause_or_start_floor_task():
    g, _ = gui_at_floor_navigation()
    tick(g, 52, goal_revision=2, goal=list(GOAL), pose=list(FLOOR_GOAL))
    with patch('tools.navigation_task_gui.time.monotonic', return_value=53):
        g.packet(packet(3, 1, 'arrive'))
    for t in (53, 53.2, 53.4, 53.6, 54):
        tick(g, t, pose=list(FLOOR_GOAL))
    assert not g.resume_confirmed
    assert g.mission.phase == 'navigating_floor'
    assert not any(r['operation'] == 'press_floor' for r in g._arm_client.requests)


def test_stop_during_floor_press_cancels_back_even_with_late_success():
    g, _ = gui_at_floor_navigation()
    arrive_and_press(g, 52, 3, 2)
    assert g.mission.phase == 'arm_floor'
    with patch('tools.navigation_task_gui.threading.Thread', InlineThread):
        g.stop()
    worker_reply(g, 3, ARM)
    tick(g, 60, pose=list(FLOOR_GOAL))
    assert g.mission.phase == 'aborted'
    assert not any(cmd == 1 and value == 'back' for cmd, value, seq in g.sent)


def test_failed_inner_origin_check_does_not_queue_back():
    g, _ = gui_at_floor_navigation()
    arrive_and_press(g, 52, 3, 2)
    worker_reply(g, 3, ARM)
    with patch('tools.navigation_task_gui.threading.Thread', InlineThread):
        worker_reply(g, 4, dict(ORIGIN, at_origin=False))
    tick(g, 60, pose=list(FLOOR_GOAL))
    assert g.mission.phase == 'failed'
    assert not any(cmd == 1 and value == 'back' for cmd, value, seq in g.sent)


@pytest.mark.parametrize('route,full', [(FULL_ROUTE, True), (CALL_ONLY_ROUTE, False)])
def test_gui_freezes_route_choice_when_execution_is_enabled(route, full):
    g = gui()
    g.state.enabled = False
    g.saved_current_floor = 16
    g.current_floor_input = FakeVariable('16')
    g.link_mode = FakeVariable(True)
    g.route_mode = FakeVariable(route)
    g.selected_point = FakeVariable('liftout')
    g.fresh = lambda: True
    g.preflight = lambda generation: g.events.put(('enabled', generation))
    with patch('tools.navigation_task_gui.time.monotonic', return_value=100), \
         patch('tools.navigation_task_gui.threading.Thread', InlineThread):
        g.enable()
        assert not g.state.enabled
        g.enable()
    tick(g, 100.1)
    assert g.state.enabled and g.mission.include_floor_panel is full
    g.route_mode.set(CALL_ONLY_ROUTE if full else FULL_ROUTE)
    with patch('tools.navigation_task_gui.time.monotonic', return_value=101):
        g.packet(packet(1, 10, '8'))
    assert g.mission.include_floor_panel is full
