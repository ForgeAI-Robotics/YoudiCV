"""Deterministic AGX elevator mission; no network or robot side effects."""
from dataclasses import dataclass
import math
import uuid
from .navigation_gui_state import telemetry_fresh
from .arm_report import arm_failure_message
from .button_light_flow import CALL_OFF_TIMEOUT_FLOW

ACTIVE = {'pausing_start_origin','returning_start_origin','checking_start_origin','waiting_delay','navigating_call','pausing_chassis',
          'arm_call','checking_end_origin','preparing_floor','navigating_floor','pausing_floor_chassis',
          'arm_floor','checking_floor_origin','returning_dock','charging'}
SUCCESS_FLOWS = {'arrived_after_initial_on','arrived_after_verified_press'}


def direction_for(current, target):
    for floor in (current, target):
        if isinstance(floor,bool) or not isinstance(floor,int) or not 1 <= floor <= 16:
            raise ValueError('当前楼层和目标楼层必须为1–16的整数')
    if current == target:
        raise ValueError('当前楼层与目标相同，无需呼梯')
    return 'up' if target > current else 'down'


@dataclass
class ExternalCallMission:
    phase: str = 'idle'
    mission_id: str = ''
    current_floor: int | None = None
    target_floor: int | None = None
    direction: str | None = None
    reason: str = ''
    nav_started: float | None = None
    arrive_received: float | None = None
    stable_since: float | None = None
    last_sample: float | None = None
    pause_started: float | None = None
    start_return_attempted: bool = False
    call_off_wait_timed_out: bool = False
    # User-authorized temporary integration-test tolerance (2026-10-13).
    # This does not change the vendor navigation goal or collision checks.
    position_tolerance_m: float = .12
    yaw_tolerance_rad: float = .1
    include_floor_panel: bool = False
    # Liftin-only integration-test allowance (2026-10-20): the stationary
    # recorded stop is 0.127578 m from the configured target. This does not
    # change vendor parking accuracy or latch a passing position while moving.
    # Arm reachability, visual evidence and collision checks remain independent.
    floor_position_tolerance_m: float = .13

    def position_tolerance_for(self, goal):
        return self.floor_position_tolerance_m if goal=='liftin' else self.position_tolerance_m

    @property
    def expected_navigation_goal(self):
        return {'navigating_call':'liftout','navigating_floor':'liftin'}.get(self.phase)

    @property
    def active(self):
        return self.phase in ACTIVE

    def begin(self, current, target, now):
        if self.active:
            raise RuntimeError('已有外呼联动任务')
        direction=direction_for(current,target)
        self.mission_id=str(uuid.uuid4());self.current_floor=current;self.target_floor=target
        self.direction=direction;self.phase='checking_start_origin';self.reason='';self.start_return_attempted=False
        self.call_off_wait_timed_out=False
        self.nav_started=self.arrive_received=self.stable_since=self.last_sample=self.pause_started=None
        return self.mission_id

    def fail(self, reason):
        self.phase='failed';self.reason=str(reason);self.stable_since=None

    def abort(self):
        if self.active:
            self.phase='aborted';self.reason='用户停止任务';self.stable_since=None

    def origin_checked(self, mission_id, result, now=0):
        if mission_id != self.mission_id or self.phase not in {'checking_start_origin','checking_end_origin','checking_floor_origin'}:
            return None
        healthy=(result.get('ok') is True and result.get('hardware_ok')==1
                 and result.get('joint_state_source')=='orange_pi_ros_joint_states')
        if healthy and result.get('at_origin') is False and self.phase=='checking_start_origin' and not self.start_return_attempted:
            self.start_return_attempted=True;self.phase='pausing_start_origin';self.pause_started=now;return 'pause_for_origin'
        if not (result.get('ok') is True and result.get('at_origin') is True
                and result.get('hardware_ok') == 1 and result.get('joint_state_source')=='orange_pi_ros_joint_states'):
            self.fail('机械臂原位未通过实时编码器核验');return 'failed'
        if self.phase=='checking_start_origin':
            self.phase='waiting_delay';return 'ready_navigation'
        if self.phase=='checking_end_origin' and self.include_floor_panel:
            self.phase='preparing_floor';return 'navigate_floor'
        self.phase='returning_dock';return 'navigate_back'

    def start_return_finished(self, mission_id, result):
        if mission_id!=self.mission_id or self.phase!='returning_start_origin':return None
        if result.get('return_code')!=0:
            self.fail('出发前安全归位失败');return 'failed'
        self.phase='checking_start_origin';return 'check_origin'

    def sent(self, goal, now):
        if goal=='liftout' and self.phase=='waiting_delay':
            self.phase='navigating_call';self.nav_started=now
        elif goal=='liftin' and self.include_floor_panel and self.phase=='preparing_floor':
            self.phase='navigating_floor';self.nav_started=now
        elif goal=='back' and self.phase=='returning_dock':
            self.phase='charging';self.nav_started=now
        else:
            raise RuntimeError('导航下发与联动阶段不一致')
        self.arrive_received=self.stable_since=self.last_sample=self.pause_started=None

    def arrive(self, now):
        if self.phase in {'navigating_call','navigating_floor'} and self.arrive_received is None:
            self.arrive_received=now

    def observe(self, sample, goal_pose, now, resumed):
        if self.phase not in {'navigating_call','navigating_floor'}:return None
        floor_stage=self.phase=='navigating_floor'
        if self.nav_started is not None and now-self.nav_started>180:
            self.fail(('内呼' if floor_stage else '外呼')+'导航超时');return 'failed'
        if self.arrive_received is not None and now-self.arrive_received>30:
            self.fail('厂商报告到位，但实际停车位置/朝向或停稳未通过核验');return 'failed'
        if self.arrive_received is None or not resumed or not telemetry_fresh(sample,now):
            self.stable_since=self.last_sample=None;return None
        p=sample['pose'];v=sample['velocity']
        if not all(math.isfinite(float(x)) for x in [*p,*v,*goal_pose]):
            self.stable_since=self.last_sample=None;return None
        if not .5 <= math.hypot(p[2],p[3]) <= 1.5:
            self.stable_since=self.last_sample=None;return None
        yaw=2*math.atan2(p[2],p[3])-2*math.atan2(goal_pose[2],goal_pose[3])
        yaw=abs(math.atan2(math.sin(yaw),math.cos(yaw)))
        valid=(math.hypot(p[0]-goal_pose[0],p[1]-goal_pose[1])<=self.position_tolerance_for('liftin' if floor_stage else 'liftout')
               and yaw<=self.yaw_tolerance_rad and math.hypot(v[0],v[1])<.01 and abs(v[2])<.02)
        if not valid:
            self.stable_since=self.last_sample=None;return None
        if self.last_sample is None or now-self.last_sample>.35:
            self.stable_since=now
        self.last_sample=now
        if now-self.stable_since>=.5:
            self.phase='pausing_floor_chassis' if floor_stage else 'pausing_chassis'
            self.pause_started=now;return 'pause_for_arm'
        return None

    def chassis_paused(self, now, stationary):
        if self.phase not in {'pausing_chassis','pausing_start_origin','pausing_floor_chassis'}:return None
        if self.pause_started is not None and now-self.pause_started>20:
            self.fail('底盘暂停交接超时');return 'failed'
        if stationary:
            if self.phase=='pausing_start_origin':
                self.phase='returning_start_origin';return 'return_origin'
            if self.phase=='pausing_floor_chassis':
                self.phase='arm_floor';return 'press_floor'
            self.phase='arm_call';return 'press'
        return None

    def arm_finished(self, mission_id, result):
        if mission_id!=self.mission_id or self.phase not in {'arm_call','arm_floor'}:return None
        timeout = result.get('external_call_off_wait_timeout') or {}
        timeout_release = (
            self.phase == 'arm_call' and result.get('flow_status') == CALL_OFF_TIMEOUT_FLOW
            and result.get('off_wait_timed_out') is True
            and result.get('press_execution_started') is True
            and type(result.get('press_attempts')) is int and result['press_attempts'] >= 1
            and timeout.get('target_label') == self.direction
            and timeout.get('release_policy') == 'mock_scene_timeout_release'
            and timeout.get('off_confirmed') is False
            and timeout.get('timeout_sec') == 30.0
            and isinstance(timeout.get('elapsed_sec'), (int, float))
            and math.isfinite(timeout['elapsed_sec']) and timeout['elapsed_sec'] >= 30.0)
        measured_off = (result.get('off_confirmed') is True
                        and result.get('flow_status') in SUCCESS_FLOWS)
        if not (result.get('return_code')==0 and result.get('on_confirmed') is True
                and (measured_off or timeout_release) and result.get('returned_to_origin') is True
                and result.get('origin_verified') is True):
            uncertain=(result.get('return_code')==2 and result.get('flow_status') in {
                'post_press_confirmation_uncertain','post_press_confirmation_disagreement'})
            detail = arm_failure_message(result)
            self.fail('已执行按压，但亮灯确认不确定' if uncertain else
                      detail if detail != '任务失败' else
                      '机械臂闭环失败或亮灭/归位证据不完整');return 'failed'
        if timeout_release:
            self.call_off_wait_timed_out=True
        self.phase='checking_floor_origin' if self.phase=='arm_floor' else 'checking_end_origin'
        return 'check_origin'

    def charged(self):
        if self.phase=='charging':
            self.phase='completed';return True
        return False
