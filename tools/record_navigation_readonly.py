#!/usr/bin/python3
"""Record vendor ROS telemetry over SSH without publishers or service calls."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piper_handeye.navigation_ssh import connect_navigation_host
REMOTE = r'''
source /opt/ros/melodic/setup.bash
source /home/ubuntu/ud_ws/install/setup.bash
python -u - <<'RECORDER'
import rospy,roslib.message,json,time,threading
rospy.init_node('agx_navigation_readonly_record',anonymous=True,disable_signals=True)
mutex=threading.Lock();last={}
def plain(m):
 if hasattr(m,'__slots__'):return {k:plain(getattr(m,k)) for k in m.__slots__}
 if isinstance(m,(list,tuple)):return [plain(x) for x in m]
 if isinstance(m,(str,int,float,bool)) or m is None:return m
 return str(m)
def emit(data):
 with mutex:print(json.dumps(data,separators=(',',':')))
def callback(wire,topic):
 now=time.time()
 important=topic in ('/Local/Goal_reached','/SWJ/ApkReceiveResult','/UD/elevator/send_goal','/move_base_simple/goal','/initialpose','/rviz/switch_map_initial_pose','/swj_robot_pose_run_state')
 interval=0 if important else (.5 if 'plan' in topic or topic in ('/path','/tf','/tf_static') else .1)
 if now-last.get(topic,0)<interval:return
 last[topic]=now
 try:
  cls=roslib.message.get_message_class(wire._connection_header['type']);m=cls();m.deserialize(wire._buff)
  emit({'event':'message','topic':topic,'source_unix':now,'type':wire._connection_header['type'],'callerid':wire._connection_header.get('callerid'),'data':plain(m)})
 except Exception as e:emit({'event':'decode_error','topic':topic,'error':str(e)})
topics=['/robot_pose','/robot_pose_correct','/odom','/cmd_vel','/plt_cmd_vel','/robot_cmd_vel','/move_base/NavfnROS/plan','/move_base/TrajectoryPlannerROS/global_plan','/move_base/TrajectoryPlannerROS/local_plan','/move_base/TrajectoryPlannerROS/fem_global_plan','/move_base/TrajectoryPlannerROS/fem_local_plan','/path','/tf','/tf_static','/Local/Goal_reached','/SWJ/ApkReceiveResult','/UD/elevator/send_goal','/move_base_simple/goal','/initialpose','/rviz/switch_map_initial_pose','/swj_robot_pose_run_state']
subs=[rospy.Subscriber(t,rospy.AnyMsg,lambda m,t=t:callback(m,t),queue_size=10) for t in topics]
emit({'event':'ready','source_unix':time.time(),'topics':topics,'map':rospy.get_param('/multiple_map_switching_node/UD_map_yaml',None)})
end=time.time()+480
while not rospy.is_shutdown() and time.time()<end:time.sleep(.1)
emit({'event':'finished','source_unix':time.time()})
RECORDER
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    client=connect_navigation_host(timeout=5)
    sftp=client.open_sftp()
    for name in ('PathNode.json','GlobalPath.json','jxb_105.yaml','jxb_105.pgm'):
        sftp.get('/home/ubuntu/std_wr_demo/map/'+name,str(args.output/name))
    sftp.close()
    stdin,stdout,stderr=client.exec_command('bash -s');stdin.write(REMOTE);stdin.channel.shutdown_write()
    try:
        with (args.output/'telemetry.jsonl').open('a',buffering=1) as journal:
            for line in stdout:
                try:entry=json.loads(line)
                except ValueError:continue
                entry.update(received_unix=time.time(),received_monotonic=time.monotonic())
                journal.write(json.dumps(entry,separators=(',',':'))+'\n')
                if entry.get('event')=='ready':
                    (args.output/'ready.json').write_text(json.dumps(entry,indent=2));print('ready',flush=True)
        (args.output/'stderr.txt').write_text(stderr.read().decode())
    finally:
        stdin.close();stdout.close();stderr.close();client.close()


if __name__=='__main__':main()
