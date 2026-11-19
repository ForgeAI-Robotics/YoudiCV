from piper_handeye.navigation_gui_state import NavigationState

def test_default_monitor_only():
    s=NavigationState();assert s.accept('8',1)=='monitor';assert not s.due(99)

def test_three_second_gate_and_semantic_retries():
    s=NavigationState();s.enable();assert s.accept('8',1)=='accepted'
    assert s.accept('8',2)=='duplicate';assert s.accept('9',2)=='busy'
    assert s.deadline==4;assert not s.due(3.99);assert s.due(4);assert not s.due(5)

def test_stop_cancels_at_dispatch_boundary():
    s=NavigationState();s.enable();s.accept('8',1);s.stop()
    assert not s.due(4);assert s.accept('8',5)=='monitor'

def test_explicit_reenable_resets_stale_task():
    s=NavigationState();s.enable();s.accept('8',1);s.due(4);s.stop();s.enable()
    assert not s.due(99);assert s.accept('9',100)=='accepted';assert s.deadline==103


def test_feedback_ignores_cross_machine_wall_clock_offset():
    from piper_handeye.navigation_gui_state import telemetry_fresh
    sample=dict(pose=[0,0,0,1],velocity=[0,0,0],pose_time=10,velocity_time=10,
                pose_age=.02,velocity_age=.03,received_monotonic=100)
    assert telemetry_fresh(sample,100.2)
    assert not telemetry_fresh(sample,101.1)
    sample['velocity_age']=2
    assert not telemetry_fresh(sample,100.2)
    sample['velocity_age']=.03;sample.pop('pose')
    assert not telemetry_fresh(sample,100.2)


def test_resume_requires_new_matching_plan_and_stop_dominates():
    from piper_handeye.navigation_gui_state import resume_allowed
    s=NavigationState();s.enable();s.accept('8',0);s.due(10)
    d=dict(pose=[0,0,0,1],velocity=[0,0,0],pose_age=.01,velocity_age=.02,
           received_monotonic=10,goal_revision=2,goal=[-.525,3.050,0,1])
    assert resume_allowed(s,d,1,10.1)
    assert not resume_allowed(s,d,2,10.1)
    assert not resume_allowed(s,d,1,11.1)
    d['goal']=[1,2,0,1];assert not resume_allowed(s,d,1,10.1)
    d['goal']=[-.525,3.050,0,1];s.stop()
    assert not resume_allowed(s,d,1,10.1)


def test_selected_liftin_plan_and_heading_are_required():
    from piper_handeye.navigation_gui_state import resume_allowed
    goal=(-1.829298,2.820949,.708453,.705758)
    s=NavigationState();s.enable('liftin',goal);s.accept('manual',0)
    assert not s.due(2.99);assert s.due(3)
    d=dict(pose=[0,0,0,1],velocity=[0,0,0],pose_age=.01,velocity_age=.02,
           received_monotonic=10,goal_revision=2,goal=list(goal))
    assert s.goal_name=='liftin' and resume_allowed(s,d,1,10.1)
    d['goal']=[-.525359,3.050205,-.019123,.999817]
    assert not resume_allowed(s,d,1,10.1)
    d['goal']=[goal[0],goal[1],0,1]
    assert not resume_allowed(s,d,1,10.1)
    d['goal']=list(goal);s.stop();assert not resume_allowed(s,d,1,10.1)


def test_reenable_freezes_new_goal_and_cancels_previous_deadline():
    s=NavigationState();s.enable();s.accept('8',0);s.stop()
    s.enable('liftin',(-1.829298,2.820949,.708453,.705758))
    assert s.floor is None and s.deadline is None and not s.due(100)
    assert s.accept('manual',100)=='accepted' and s.deadline==103


def test_unknown_goal_without_coordinates_is_rejected():
    import pytest
    s=NavigationState()
    with pytest.raises(ValueError):s.enable('unknown')
    assert not s.enabled


def test_back_requires_correlated_ack_and_new_charge_service_evidence():
    from piper_handeye.navigation_gui_state import charge_resume_allowed
    s=NavigationState();s.enable('back',(6.9,1.3,0,1));s.accept('manual',0);s.due(10)
    d=dict(pose=[0,0,0,1],velocity=[0,0,0],pose_age=.01,velocity_age=.02,
           received_monotonic=10,charge_request_revision=3,charge_service_revision=5,
           charge_request_time=10.05,charge_service_time=10.06,charge_request_age=.1,charge_service_age=.1,goal=[-1,2,0,1])
    gate=lambda ack=True:charge_resume_allowed(s,d,2,4,ack,10.1,10)
    assert gate()  # ordinary back coordinate does not gate a charging mission
    assert not gate(False)
    d['charge_error_time']=10.05;assert not gate()
    d.pop('charge_error_time')
    d['charge_request_time']=9;assert not gate()
    d['charge_request_time']=10.05
    d['charge_request_revision']=2;assert not gate()
    d['charge_request_revision']=3;d['charge_service_revision']=4;assert not gate()
    d['charge_service_revision']=5;d['charge_request_age']=12;assert not gate()
    d['charge_request_age']=.1;d['velocity_age']=2;assert not gate()
    d['velocity_age']=.02;s.stop();assert not gate()


def test_charge_gate_never_bypasses_normal_navigation():
    from piper_handeye.navigation_gui_state import charge_resume_allowed
    s=NavigationState();s.enable();s.accept('8',0);s.due(10)
    d=dict(pose=[0,0,0,1],velocity=[0,0,0],pose_age=0,velocity_age=0,
           received_monotonic=10,charge_request_revision=1,charge_service_revision=1,
           charge_request_age=0,charge_service_age=0)
    assert not charge_resume_allowed(s,d,0,0,True,10,10)
