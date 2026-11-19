#!/usr/bin/env python3
"""Save a motion-free simulation of both up/down mission state sequences."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from piper_handeye.external_call_mission import ExternalCallMission


def simulate(current,target):
    m=ExternalCallMission();rows=[]
    def record(event,now):rows.append(dict(simulation=True,event=event,time_sec=now,phase=m.phase))
    mid=m.begin(current,target,0);record('目标已接收',0)
    origin=dict(ok=True,at_origin=True,hardware_ok=1,joint_state_source='orange_pi_ros_joint_states')
    m.origin_checked(mid,origin);record('原位检查通过',1)
    m.sent('liftout',10);record('等待10秒后下发liftout',10)
    m.arrive(20)
    pose=[-.528,3.01,-.019,.999]
    for t in [20,20.2,20.4,20.6]:
        m.observe(dict(pose=pose,velocity=[0,0,0],pose_age=0,velocity_age=0,received_monotonic=t),pose,t,True)
    assert m.phase=='pausing_chassis';record('实际位姿及持续停稳通过',20.6)
    m.chassis_paused(21,True);record('三个暂停接口确认后按'+m.direction,21)
    record('亮灯与不确定时持续等待，无回桩下发',30)
    result=dict(return_code=0,on_confirmed=True,off_confirmed=True,returned_to_origin=True,origin_verified=True,flow_status='arrived_after_verified_press')
    m.arm_finished(mid,result);record('熄灭及闭环归位证据完整',40)
    m.origin_checked(mid,origin);record('实际编码器归位复核通过',41)
    m.sent('back',41.1);record('下发back并恢复三个暂停接口',41.1)
    m.charged();record('实际充电确认，任务完成',70)
    assert m.current_floor==current and m.phase=='completed'
    return dict(simulation=True,current_floor=current,target_floor=target,direction=m.direction,events=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/diagnostics/20261013_external_call_link/simulation.json');a=p.parse_args()
    data={'hardware_motion':False,'scenarios':[simulate(16,8),simulate(1,8)]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(str(a.output))

if __name__=='__main__':main()
