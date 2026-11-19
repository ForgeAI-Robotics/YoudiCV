#!/usr/bin/env python3
"""Simulate the full elevator state sequence; no sockets, ROS or robot commands."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piper_handeye.external_call_mission import ExternalCallMission


def simulate(current, target):
    mission = ExternalCallMission(include_floor_panel=True)
    rows = []
    def record(event, now):
        rows.append(dict(simulation=True, event=event, time_sec=now, phase=mission.phase))
    origin = dict(ok=True, at_origin=True, hardware_ok=1,
                  joint_state_source='orange_pi_ros_joint_states')
    arm = dict(return_code=0, on_confirmed=True, off_confirmed=True,
               returned_to_origin=True, origin_verified=True,
               flow_status='arrived_after_verified_press')
    mid = mission.begin(current, target, 0)
    record('接收Android目标并保存上下行方向', 0)
    assert mission.origin_checked(mid, origin) == 'ready_navigation'
    record('实时原位检查通过', 1)

    def navigate(goal, pose, started, arrived):
        mission.sent(goal, started)
        record('模拟下发'+goal, started)
        assert mission.arrive_received is None
        # Reaching the coordinate without a fresh arrival notification is insufficient.
        fresh = dict(pose=list(pose), velocity=[0, 0, 0], pose_age=0, velocity_age=0)
        assert mission.observe(dict(fresh, received_monotonic=arrived-.1), pose, arrived-.1, True) is None
        mission.arrive(arrived)
        for dt in (0, .2, .4):
            assert mission.observe(dict(fresh, received_monotonic=arrived+dt), pose, arrived+dt, True) is None
        assert mission.observe(dict(fresh, received_monotonic=arrived+.6), pose, arrived+.6, True) == 'pause_for_arm'
        record('到位位置/朝向及连续停稳通过', arrived+.6)

    navigate('liftout', (-.528, 3.01, -.019123, .999817), 10, 20)
    assert mission.chassis_paused(21, False) is None
    assert mission.chassis_paused(21, True) == 'press'
    record('模拟暂停确认后执行外呼'+mission.direction, 21)
    assert mission.observe({}, (), 100000, True) is None
    record('按钮亮起或不确定期间保持等待，不推进导航', 30)
    assert mission.arm_finished(mid, arm) == 'check_origin'
    record('模拟外呼熄灭及闭环归位完成', 40)
    assert mission.origin_checked(mid, origin) == 'navigate_floor'
    record('外呼后实际原位复核通过', 41)

    navigate('liftin', (-1.829298, 2.820949, .708453, .705758), 41.1, 50)
    assert mission.chassis_paused(51, True) == 'press_floor'
    record('模拟暂停确认后调用数字按钮'+str(target), 51)
    assert mission.target_floor == target
    assert mission.arm_finished(mid, arm) == 'check_origin'
    record('模拟目标楼层按钮亮灭及归位完成', 70)
    assert mission.origin_checked(mid, origin) == 'navigate_back'
    record('内呼后实际原位复核通过', 71)
    mission.sent('back', 71.1)
    record('模拟进入厂商自动回桩流程', 71.1)
    assert mission.charged()
    record('模拟GUI提供真实充电及停稳通过事件', 100)
    assert mission.phase == 'completed' and mission.current_floor == current
    return dict(simulation=True, current_floor=current, target_floor=target,
                direction=mission.direction, events=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'artifacts/diagnostics/20261015_full_elevator_route/simulation.json')
    args = parser.parse_args()
    result = dict(hardware_motion=False, scope='State machine only; physical feedback is simulated',
                  scenarios=[simulate(1, 8), simulate(16, 8)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(str(args.output))


if __name__ == '__main__':
    main()
