import pytest
from piper_handeye.external_call_mission import ExternalCallMission, direction_for

GOAL=(-.528,3.01,-.019123,.999817)
ORIGIN=dict(ok=True,at_origin=True,hardware_ok=1,joint_state_source='orange_pi_ros_joint_states')
ARM=dict(return_code=0,on_confirmed=True,off_confirmed=True,returned_to_origin=True,
         origin_verified=True,flow_status='arrived_after_verified_press')


def sample(now,pose=GOAL,velocity=(0,0,0)):
    return dict(pose=list(pose),velocity=list(velocity),pose_age=0,velocity_age=0,received_monotonic=now)


def start():
    m=ExternalCallMission();mid=m.begin(16,8,0);assert m.origin_checked(mid,ORIGIN)=='ready_navigation'
    m.sent('liftout',10);return m,mid


def test_direction_validation_and_frozen_floor_after_call():
    assert direction_for(16,8)=='down' and direction_for(1,16)=='up'
    for floors in [(8,8),(0,8),(True,8),(1,'8')]:
        with pytest.raises(ValueError):direction_for(*floors)
    m,mid=start();m.arrive(11)
    for t in (11,11.2,11.4):assert m.observe(sample(t),GOAL,t,True) is None
    assert m.observe(sample(11.6),GOAL,11.6,True)=='pause_for_arm'
    assert m.chassis_paused(12,False) is None
    assert m.chassis_paused(12,True)=='press'
    assert m.arm_finished(mid,ARM)=='check_origin'
    assert m.origin_checked(mid,ORIGIN)=='navigate_back'
    m.sent('back',13);assert m.charged() and not m.charged()
    assert m.phase=='completed' and m.current_floor==16


def test_false_arrival_moving_stale_and_observation_gaps_do_not_press():
    m,_=start()
    assert m.observe(sample(11),GOAL,11,True) is None  # no vendor arrival
    m.arrive(12)
    for p,v in [((0,0,0,1),(0,0,0)),(GOAL,(.1,0,0)),((GOAL[0],GOAL[1],0,0),(0,0,0))]:
        assert m.observe(sample(12,p,v),GOAL,12,True) is None
    assert m.observe(sample(10),GOAL,12,True) is None
    assert m.observe(sample(12),GOAL,12,False) is None
    assert m.observe(sample(13),GOAL,13,True) is None
    assert m.observe(sample(14),GOAL,14,True) is None  # continuity broken
    assert m.observe(sample(14.2),GOAL,14.2,True) is None


def test_repeated_arrive_does_not_reset_position_failure_deadline():
    m,_=start();m.arrive(11);m.arrive(39)
    assert m.observe(sample(42,pose=(0,0,0,1)),GOAL,42,True)=='failed'
    assert m.phase=='failed'


def test_temporary_12cm_gate_accepts_recorded_parking_after_stable_stop():
    m,_=start();m.arrive(11)
    parked=(-.492626965,2.917235374,-.011300113,.999936152)
    for t in (11,11.2,11.4):
        assert m.observe(sample(t,parked),GOAL,t,True) is None
    assert m.observe(sample(11.6,parked),GOAL,11.6,True)=='pause_for_arm'
    assert m.chassis_paused(12,False) is None
    assert m.chassis_paused(12,True)=='press'


@pytest.mark.parametrize('pose,velocity',[
    ((GOAL[0]+.121,GOAL[1],GOAL[2],GOAL[3]),(0,0,0)),
    ((GOAL[0]+.099,GOAL[1],.1,.994987),(0,0,0)),
    ((GOAL[0]+.099,GOAL[1],GOAL[2],GOAL[3]),(.02,0,0)),
])
def test_relaxed_gate_still_rejects_distance_yaw_or_motion(pose,velocity):
    m,_=start();m.arrive(11)
    for t in (11,11.2,11.4,11.6,12):
        assert m.observe(sample(t,pose,velocity),GOAL,t,True) is None
    assert m.phase=='navigating_call'


@pytest.mark.parametrize('missing',list(ARM))
def test_no_dock_navigation_without_complete_arm_evidence(missing):
    m,mid=start();m.phase='arm_call';bad=dict(ARM);bad.pop(missing)
    assert m.arm_finished(mid,bad)=='failed';assert m.phase=='failed'


def test_uncertain_origin_and_stale_callbacks_never_advance():
    m=ExternalCallMission();mid=m.begin(16,8,0)
    assert m.origin_checked('old',ORIGIN) is None
    bad=dict(ORIGIN,at_origin=False,hardware_ok=0);assert m.origin_checked(mid,bad)=='failed'
    m.begin(16,8,1);m.abort();assert m.origin_checked(m.mission_id,ORIGIN) is None
    assert m.arm_finished(m.mission_id,ARM) is None and not m.charged()


def test_button_wait_has_no_elapsed_time_completion():
    m,_=start();m.phase='arm_call'
    assert m.observe(sample(100000),GOAL,100000,True) is None
    assert m.phase=='arm_call' and not m.charged()


def test_start_origin_recovery_requires_pause_and_fresh_check_only_once():
    m=ExternalCallMission();mid=m.begin(16,8,0)
    bad=dict(ORIGIN,at_origin=False)
    assert m.origin_checked(mid,bad,1)=='pause_for_origin'
    assert m.chassis_paused(2,False) is None
    assert m.chassis_paused(3,True)=='return_origin'
    assert m.start_return_finished(mid,{'return_code':0})=='check_origin'
    assert m.origin_checked(mid,ORIGIN,4)=='ready_navigation'
    m.abort();m.begin(16,8,10);mid=m.mission_id
    m.origin_checked(mid,bad,11);m.chassis_paused(12,True);m.start_return_finished(mid,{'return_code':0})
    assert m.origin_checked(mid,bad,13)=='failed'  # no endless recovery
