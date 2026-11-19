#!/usr/bin/python3
"""Read-only by default Android first-leg navigation monitor with latched stop."""
import datetime
import json
import math
import queue
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piper_handeye.android_agx_protocol import decode_packet, encode_packet, AcceptedPacketCache
from piper_handeye.navigation_gui_state import INITIAL_NAVIGATION_DELAY_SEC, NavigationState, telemetry_fresh, resume_allowed, charge_resume_allowed
from piper_handeye.external_call_mission import ExternalCallMission, direction_for
from piper_handeye.navigation_arm_client import NavigationArmClient
from piper_handeye.navigation_ssh import connect_navigation_host
from run_voice_assistant_gui import configure_crisp_ui

# Dock departure also runs in charging_manager. Changing the selected goal
# must never change the scope of the pause that this GUI owns.
NAVIGATION_PAUSE_SERVICES = (
    '/pause_robot_behavior',
    '/robot_pause_to_relocate_behavior',
    '/robot_pause_to_charging_behavior',
)
FULL_ROUTE = '外呼→内呼→回桩'
CALL_ONLY_ROUTE = '外呼→回桩（原流程）'

REMOTE_MONITOR = r'''
source /opt/ros/melodic/setup.bash
source /home/ubuntu/ud_ws/install/setup.bash
python -u - <<'MONITOR'
import rospy,json,time,threading,re
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry,Path
from rosgraph_msgs.msg import Log
from custom_msgs_srvs.msg import ChargeState
rospy.init_node('agx_navigation_gui_monitor',anonymous=True,disable_signals=True)
lock=threading.Lock(); data={}
def pose(m):
 with lock: data['pose']=[m.pose.position.x,m.pose.position.y,m.pose.orientation.z,m.pose.orientation.w];data['pose_time']=time.time()
def odom(m):
 with lock: data['velocity']=[m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.angular.z];data['velocity_time']=time.time()
def plan(m):
 if m.poses:
  p=m.poses[-1].pose
  with lock: data['goal']=[p.position.x,p.position.y,p.orientation.z,p.orientation.w];data['goal_time']=time.time();data['goal_revision']=data.get('goal_revision',0)+1
def charge_log(m):
 now=time.time()
 # rosout is latched; receiving an old message does not establish a new task.
 if not 0 <= now-m.header.stamp.to_sec() < 2: return
 key=None
 if m.name=='/charging_manager_node' and (
     ('[getGoalCB]' in m.msg and re.search(r'\bnew mode:\s*9(?:!|\s|$)',m.msg)) or
     ('[getGoalSrv]' in m.msg and re.search(r'\bmode:\s*9(?:!|\s|$)',m.msg))): key='charge_request'
 if m.name=='/udp_commit_node' and 'call charge goal service OK' in m.msg: key='charge_service'
 if key:
  with lock: data[key+'_revision']=data.get(key+'_revision',0)+1;data[key+'_time']=m.header.stamp.to_sec()
 if m.name=='/charging_manager_node' and any(x in m.msg for x in ('Wrong charging control mode','invaild control mode','get control mode is invaild')):
  with lock: data['charge_error']=m.msg;data['charge_error_time']=now
def charge_work(m):
 with lock: data['real_charging']=bool(m.is_real_charging_);data['charge_signal']=bool(m.charge_signal_);data['charge_work_time']=time.time()
rospy.Subscriber('/rosout',Log,charge_log,queue_size=100)
rospy.Subscriber('/charge_work_state',ChargeState,charge_work,queue_size=1)
rospy.Subscriber('/robot_pose',PoseStamped,pose,queue_size=1)
rospy.Subscriber('/odom',Odometry,odom,queue_size=1)
rospy.Subscriber('/move_base/NavfnROS/plan',Path,plan,queue_size=1)
while not rospy.is_shutdown():
 with lock:
  sample=dict(data);now=time.time();sample['remote_time']=now
  for key in ('pose','velocity','charge_request','charge_service','charge_work','charge_error'): sample[key+'_age']=max(0,now-data.get(key+'_time',0))
  print(json.dumps(sample))
 time.sleep(.2)
MONITOR
'''

class Gui:
    def __init__(self, root):
        self.root=root; configure_crisp_ui(root)
        root.title('AGX 导航任务 · 状态与停止');root.geometry('1200x680');root.minsize(900,540)
        self.events=queue.Queue();self.state=NavigationState();self.confirm_until=0
        self.point_catalog=json.loads((ROOT/'config/navigation_test_points.json').read_text())
        self.selected_point=tk.StringVar(value='liftout');self.pending_goal=None
        self.closed=threading.Event();self.sock=None;self.ssh=None;self.telemetry={};self.stop_since=None;self.stationary_since=None;self.goal_since=None;self.goal_revision_floor=0
        self.sequence=int(time.time());self.start_time=time.time();self.generation=0;self.last_verification=None;self.stop_logged=False;self.pause_lock=threading.Lock();self.resume_requested=False;self.navigation_sent_at=None
        self.resume_confirmed=False
        self.dispatch_sequence=None;self.charge_dispatch_time=None;self.charge_ack=False;self.charge_request_baseline=0;self.charge_service_baseline=0;self.charge_confirm_since=None;self.charge_complete=False
        self.run=ROOT/'navigation_gui_runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run.mkdir(parents=True,exist_ok=True);self.journal=(self.run/'journal.jsonl').open('a',encoding='utf8',buffering=1)
        self.mission=ExternalCallMission();self.armed_link_mode=False;self.arm_pause_confirmed=False
        self._mission_packets=AcceptedPacketCache();self._arrival_packets=AcceptedPacketCache();self._arm_client=NavigationArmClient(self.events,self.run)
        self._press_request_id=None
        self._arm_ownership=False;self._mission_check_started=None;self._chassis_last_good=None
        self.arm_debug=None
        self.arm_debug_status=tk.StringVar(value='机械臂调试：待命')
        self.settings_path=ROOT/'config/navigation_link_settings.json';self.saved_current_floor=None
        try:
            value=json.loads(self.settings_path.read_text()).get('current_floor')
            if type(value) is int and 1<=value<=16:self.saved_current_floor=value
        except (OSError,ValueError):pass
        self.current_floor_input=tk.StringVar(value='' if self.saved_current_floor is None else str(self.saved_current_floor))
        self.link_mode=tk.BooleanVar(value=False);self.route_mode=tk.StringVar(value=FULL_ROUTE)
        self.mission_status=tk.StringVar(value='联动关闭：仅导航测试')
        self.status=tk.StringVar(value='只监视：执行未开启');self.connection=tk.StringVar(value='连接中…');self.floor=tk.StringVar(value='目标楼层：—');self.countdown=tk.StringVar(value='延迟：—')
        self.pose=tk.StringVar(value='位姿：等待反馈');self.velocity=tk.StringVar(value='速度：等待反馈');self.arrival=tk.StringVar(value='厂商到位：未收到');self.precision=tk.StringVar(value='位置核验：等待本次目标坐标')
        frame=ttk.Frame(root,padding=12);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='导航任务监视',font=('Noto Sans CJK SC',18,'bold')).pack(anchor='w')
        ttk.Label(frame,textvariable=self.connection).pack(anchor='w',pady=4)
        row=ttk.Frame(frame);row.pack(fill='x',pady=8)
        self.enable_button=ttk.Button(row,text='开启导航测试（需再次确认）',command=self.enable);self.enable_button.pack(side='left')
        ttk.Label(row,text='  目标点：').pack(side='left')
        self.point_selector=ttk.Combobox(row,textvariable=self.selected_point,
            values=list(self.point_catalog['points']),state='readonly',width=12)
        self.point_selector.pack(side='left',padx=6)
        self.manual_button=ttk.Button(row,text=f'导航到所选点（{INITIAL_NAVIGATION_DELAY_SEC}秒后）',command=self.manual_start)
        self.manual_button.pack(side='left',padx=6)
        ttk.Button(row,text='自动回桩（双次确认）',command=self.select_charge).pack(side='left',padx=6)
        tk.Button(row,text='停止任务',command=self.stop,bg='#be2532',fg='white',font=('Noto Sans CJK SC',13,'bold'),padx=24,pady=6).pack(side='right')
        mission_row=ttk.Frame(frame);mission_row.pack(fill='x',pady=(0,6))
        ttk.Label(mission_row,text='当前楼层：').pack(side='left')
        self.current_floor_entry=ttk.Entry(mission_row,textvariable=self.current_floor_input,width=5)
        self.current_floor_entry.pack(side='left')
        self.current_floor_save=ttk.Button(mission_row,text='保存楼层',command=self.save_current_floor)
        self.current_floor_save.pack(side='left',padx=6)
        self.link_mode_button=ttk.Checkbutton(mission_row,text='启用机械臂联动',variable=self.link_mode)
        self.link_mode_button.pack(side='left',padx=10)
        self.route_selector=ttk.Combobox(mission_row,textvariable=self.route_mode,
            values=(FULL_ROUTE,CALL_ONLY_ROUTE),state='readonly',width=24)
        self.route_selector.pack(side='left',padx=4)
        ttk.Label(frame,textvariable=self.mission_status,wraplength=1100).pack(anchor='w')
        arm_row=ttk.Frame(frame);arm_row.pack(fill='x',pady=(0,6))
        ttk.Button(arm_row,text='终止机械臂任务',command=self.terminate_arm).pack(side='left')
        ttk.Button(arm_row,text='机械臂归位',command=self.return_arm).pack(side='left',padx=8)
        ttk.Label(arm_row,textvariable=self.arm_debug_status,wraplength=760).pack(side='left')
        ttk.Label(frame,textvariable=self.status,font=('Noto Sans CJK SC',13,'bold')).pack(anchor='w')
        stats=ttk.Frame(frame);stats.pack(fill='x',pady=8)
        for i,var in enumerate((self.floor,self.countdown,self.pose,self.velocity,self.arrival,self.precision)):
            ttk.Label(stats,textvariable=var).grid(row=i//2,column=i%2,sticky='w',padx=(0,25),pady=3)
        ttk.Label(frame,text='联动：保存当前楼层 → 选择流程并启用联动 → 双次确认开启 → Android选楼层；停车测试门槛：外呼%gcm，内呼%gcm。'%(self.mission.position_tolerance_m*100,self.mission.floor_position_tolerance_m*100)).pack(anchor='w')
        self.log=ScrolledText(frame,font=('Noto Sans CJK SC',11),height=8,wrap='word',state='disabled');self.log.pack(fill='both',expand=True,pady=8)
        ttk.Label(frame,text='消息记录：'+str(self.run)).pack(anchor='w')
        self.record('界面启动',detail='只监视，关闭窗口会请求停止；不会自动恢复导航',external_call_position_tolerance_m=self.mission.position_tolerance_m,floor_position_tolerance_m=self.mission.floor_position_tolerance_m,external_call_yaw_tolerance_rad=self.mission.yaw_tolerance_rad)
        threading.Thread(target=self.listen,daemon=True).start();threading.Thread(target=self.monitor,daemon=True).start()
        root.protocol('WM_DELETE_WINDOW',self.close);root.after(100,self.tick)

    def record(self,event,**data):
        entry=dict(time=time.time(),event=event,**data);self.journal.write(json.dumps(entry,ensure_ascii=False)+'\n')
        self.log.configure(state='normal');self.log.insert('end',datetime.datetime.now().strftime('%H:%M:%S')+'  '+event+'  '+json.dumps(data,ensure_ascii=False)+'\n');self.log.see('end');self.log.configure(state='disabled')
        # Bound display only; the complete evidence remains in the journal.
        if int(self.log.index('end-1c').split('.')[0])>700:self.log.configure(state='normal');self.log.delete('1.0','100.0');self.log.configure(state='disabled')

    def listen(self):
        try:
            sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind(('192.168.168.162',5005));sock.settimeout(.3);self.sock=sock;self.events.put(('ready',None))
            while not self.closed.is_set():
                try:wire,source=sock.recvfrom(65535)
                except socket.timeout:continue
                if source[0]!='192.168.168.110':continue
                try:self.events.put(('packet',decode_packet(wire,source)))
                except (ValueError,UnicodeError) as e:self.events.put(('error',str(e)))
        except Exception as e:self.events.put(('listen_error',str(e)))
        finally:
            if self.sock:self.sock.close()
            self.sock=None

    def monitor(self):
        try:
            client=connect_navigation_host(timeout=5);self.ssh=client
            i,o,e=client.exec_command('bash -s');i.write(REMOTE_MONITOR);i.channel.shutdown_write()
            for line in o:
                if self.closed.is_set():break
                try:
                    sample=json.loads(line);sample['received_monotonic']=time.monotonic();self.events.put(('telemetry',sample))
                except ValueError:pass
            if not self.closed.is_set():self.events.put(('error','底盘反馈连接中断：'+e.read().decode()[-400:]))
        except Exception as e:self.events.put(('error','底盘反馈失败：'+str(e)))

    def send(self,command,value):
        if self.sock is None:raise RuntimeError('监听端口未就绪，无法发送')
        self.sequence+=1;self.sock.sendto(encode_packet(self.sequence,'active_result',dict(command=command,value=value)),('192.168.168.110',5002))
        self.record('发往 Android',command=command,value=value,sequence=self.sequence)

    def save_current_floor(self):
        if self.mission.active or self.state.enabled:
            self.status.set('执行权限已开启，当前楼层已冻结；请先停止任务');return
        try:
            value=int(self.current_floor_input.get().strip())
            if not 1<=value<=16:raise ValueError('当前楼层必须为1–16')
            tmp=self.settings_path.with_suffix('.tmp');tmp.write_text(json.dumps({'current_floor':value},ensure_ascii=False));tmp.replace(self.settings_path)
            self.saved_current_floor=value;self.record('当前楼层已保存',floor=value)
            self.mission_status.set('当前楼层 '+str(value)+'；布景联调不会自动更新当前楼层')
        except (ValueError,OSError) as exc:self.status.set('楼层保存失败：'+str(exc))

    def lock_mission_controls(self, locked):
        for widget in (self.current_floor_entry,self.current_floor_save,self.link_mode_button):
            widget.configure(state='disabled' if locked else 'normal')
        if locked:self.point_selector.configure(state='disabled')
        if getattr(self,'route_selector',None) is not None:
            self.route_selector.configure(state='disabled' if locked else 'readonly')

    def enable(self):
        if getattr(self,'arm_debug',None):
            self.status.set('机械臂调试尚未结束，请等待或终止');return
        if self.mission.active or self._arm_client.busy:
            self.record('拒绝重复开启',reason='联动任务或机械臂安全停止尚未结束');return
        now=time.monotonic()
        if self.state.enabled or self.state.sent and not self.state.arrived and not self.state.stopped:
            self.record('拒绝重复开启',reason='请先停止当前任务');return
        if self.link_mode.get():
            if self.saved_current_floor is None or self.current_floor_input.get().strip()!=str(self.saved_current_floor):
                self.status.set('请填写并保存当前楼层，再开启外呼联动');return
            self.selected_point.set('liftout')
        if now>self.confirm_until:
            self.confirm_until=now+5;self.enable_button.configure(text='再次点击确认开启');self.status.set('请确认开启导航测试：目标 '+self.selected_point.get());return
        if self.sock is None:self.record('无法开启',reason='5005 端口未就绪');return
        if not self.fresh():self.record('无法开启',reason='底盘位姿/速度反馈不新鲜');return
        self.confirm_until=0
        name=self.selected_point.get()
        if name not in self.point_catalog['points']:
            self.record('无法开启',reason='目标点不在已记录点位表');return
        point=self.point_catalog['points'][name]
        self.pending_link_mode=self.link_mode.get();self.pending_current_floor=self.saved_current_floor
        self.pending_include_floor=self.pending_link_mode and self.route_mode.get()==FULL_ROUTE
        if self.pending_link_mode:
            required=('liftout','liftin','back') if self.pending_include_floor else ('liftout','back')
            try:
                for goal in required:
                    pose=self.point_catalog['points'][goal]
                    if not all(math.isfinite(float(pose[k])) for k in ('x','y','z','w')):
                        raise ValueError(goal+'坐标无效')
            except (KeyError,TypeError,ValueError) as exc:
                self.record('无法开启',reason='联动点位配置不完整：'+str(exc));return
        self.lock_mission_controls(True)
        self.pending_goal=(name,tuple(point[k] for k in ('x','y','z','w')))
        self.point_selector.configure(state='disabled')
        # Keep the already authorized test cap; read back before accepting floors.
        self.enable_button.configure(state='disabled');self.status.set('正在检查地图和速度参数（实际限速尚待厂商核验）…')
        self.generation+=1
        threading.Thread(target=self.preflight,args=(self.generation,),daemon=True).start()

    def select_charge(self):
        if self.mission.active or self._arm_client.busy:
            self.status.set('联动进行中，请先停止任务');return
        self.link_mode.set(False)
        if self.state.enabled or self.state.sent and not self.state.stopped and not self.state.arrived:
            self.status.set('请先停止当前任务，再开启自动回桩');return
        self.selected_point.set('back')
        self.enable()
        if self.state.enabled:return
        self.status.set('自动回桩：再次点击确认，检查通过后点击“导航到所选点”；使用厂商充电流程')

    def manual_start(self):
        if getattr(self,'armed_link_mode',False):
            self.status.set('外呼联动请在Android选择目标楼层');return
        if not self.state.enabled or self.state.stopped:
            self.status.set('请先选择目标点并确认开启导航测试');return
        if not self.fresh():
            self.record('拒绝手动导航',reason='底盘反馈已过期');return
        result=self.state.accept('manual',time.monotonic())
        if result!='accepted':
            self.record('拒绝重复手动导航',result=result);return
        self.floor.set('手动导航：'+self.state.goal_name)
        self.goal_revision_floor=self.telemetry.get('goal_revision',0)
        self.telemetry.pop('goal',None);self.goal_since=time.monotonic()
        self.arrival.set('厂商到位：未收到');self.status.set(f'目标 {self.state.goal_name}，等待{INITIAL_NAVIGATION_DELAY_SEC}秒')
        self.record('手动导航已排队',goal=self.state.goal_name,delay_sec=INITIAL_NAVIGATION_DELAY_SEC)

    def preflight(self,generation):
        try:
            if self.ssh is None:raise RuntimeError('底盘 SSH 未连接')
            script='source /opt/ros/melodic/setup.bash\nsource /home/ubuntu/ud_ws/install/setup.bash\npython - <<\'CHECK\'\nimport rospy\nrospy.init_node("agx_gui_preflight",anonymous=True)\nassert rospy.get_param("/multiple_map_switching_node/UD_map_yaml")=="jxb_105.yaml", "map mismatch"\nCHECK\n'
            if self.pending_link_mode:
                script+='if [ $? -ne 0 ]; then exit 1; fi\npython - <<\'LEASE\'\nimport json\np="/home/ubuntu/codex_chassis_control/smoother_runtime_state.json"\nstate=json.load(open(p))\nassert state.get("active") is False, "manual chassis controller still owns control"\nLEASE\n'
            # No velocity topic is introduced; runtime setting uses the vendor API.
            script+='if [ $? -ne 0 ]; then exit 1; fi\nrosrun dynamic_reconfigure dynparam set /move_base/TrajectoryPlannerROS max_vel_x 0.1\nrosparam get /move_base/TrajectoryPlannerROS/max_vel_x\n'
            i,o,e=self.ssh.exec_command('bash -s',timeout=15);i.write(script);i.channel.shutdown_write();output=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status()
            if code or not output.rstrip().endswith('0.1'):raise RuntimeError(err or output)
            self.events.put(('enabled',generation))
        except Exception as e:self.events.put(('enable_error',str(e)))

    def stop(self):
        if getattr(self,'arm_debug',None):
            self.arm_debug=None;self.arm_debug_status.set('机械臂调试已取消，等待停止确认')
        if getattr(self,'mission',None) is not None:
            self.mission.abort();self.lock_mission_controls(False)
            if self._arm_ownership or self._arm_client.busy:
                try:self._arm_client.request('stop',self.mission.mission_id or 'stop')
                except Exception as exc:self.record('机械臂停止请求失败',error=str(exc))
            self._arm_ownership=False
        self.generation+=1;self.state.stop();self.confirm_until=0;self.stop_since=time.monotonic();self.stationary_since=None;self.stop_logged=False
        self.enable_button.configure(text='开启导航测试（需再次确认）',state='normal');self.status.set('已取消待发任务；正在请求导航暂停…');self.record('停止任务请求')
        self.point_selector.configure(state='readonly')
        try:self.send(6,'1')
        except Exception as e:self.record('停止发送失败',error=str(e));self.status.set('停止请求未送达，请检查 Android 连接')
        # Repeat only the verified idempotent pause command; never resume automatically.
        self.root.after(300,self.repeat_stop);self.root.after(800,self.repeat_stop)
        threading.Thread(target=self.direct_pause,args=(True,self.generation),daemon=True).start()

    def terminate_arm(self):
        self.start_arm_debug(False)

    def return_arm(self):
        if getattr(self,'arm_debug',None):
            self.arm_debug_status.set('已有机械臂调试操作，请等待；需要取消时点击终止机械臂任务');return
        self.start_arm_debug(True)

    def start_arm_debug(self, return_home):
        # Invalidate navigation and late mission replies before touching the arm.
        self.stop()
        self.arm_debug=dict(mission_id='manual-'+str(uuid.uuid4()),phase='stopping',
                            return_home=return_home,started=time.monotonic(),paused=False,
                            stable_since=None,last_sample=None,request_id=None)
        self.arm_debug_status.set('正在终止机械臂任务并暂停底盘…')
        self.record('机械臂手动操作',operation='return_origin' if return_home else 'stop',manual_id=self.arm_debug['mission_id'])
        self.arm_debug_request('stop')

    def arm_debug_request(self, operation):
        try:self.arm_debug['request_id']=self._arm_client.request(operation,self.arm_debug['mission_id'])
        except Exception as exc:self.fail_arm_debug('机械臂请求失败：'+str(exc))

    def fail_arm_debug(self, reason):
        self.record('机械臂手动操作失败',reason=reason)
        self.stop();self.arm_debug_status.set('机械臂操作失败：'+str(reason))

    def handle_arm_debug(self, message):
        d=self.arm_debug
        if message.get('request_id')!=d['request_id']:return
        event=message.get('event');result=message.get('result') or {}
        if event=='error':self.fail_arm_debug(message.get('error','机械臂后端失败'));return
        if event!='completed':return
        if d['phase']=='stopping':
            if not d['return_home']:
                self.arm_debug=None
                self.arm_debug_status.set('当前GUI机械臂任务已终止；底盘暂停状态见日志');return
            d['phase']='waiting_pause';self.arm_debug_status.set('机械臂任务已退出，等待底盘暂停及停稳确认')
        elif d['phase']=='returning':
            self._arm_ownership=False
            if result.get('return_code')!=0:self.fail_arm_debug('归位执行失败，请查看日志');return
            d['phase']='checking';self.arm_debug_status.set('归位执行结束，正在读取实时编码器核验')
            self.arm_debug_request('check_origin')
        elif d['phase']=='checking':
            valid=(result.get('ok') is True and result.get('at_origin') is True
                   and result.get('hardware_ok')==1 and result.get('joint_state_source')=='orange_pi_ros_joint_states')
            if not valid:self.fail_arm_debug('实时编码器未确认原位');return
            self.arm_debug=None
            self.arm_debug_status.set('机械臂归位成功：实时编码器已核验；底盘保持暂停')
            self.record('手动归位已核验',result=result)

    def tick_arm_debug(self, now):
        d=getattr(self,'arm_debug',None)
        if not d:return
        if now-d['started']>120:self.fail_arm_debug('机械臂调试操作超时');return
        v=self.telemetry.get('velocity',[])
        valid=(self.fresh() and len(v)==3 and all(math.isfinite(float(x)) for x in v))
        stationary=valid and math.hypot(v[0],v[1])<.01 and abs(v[2])<.02
        if d['phase']=='waiting_pause':
            if not d['paused'] or not stationary:
                d['stable_since']=d['last_sample']=None;return
            if d['last_sample'] is None or now-d['last_sample']>.35:d['stable_since']=now
            d['last_sample']=now
            if now-d['stable_since']>=.5 and not self._arm_client.busy:
                d['phase']='returning';self._arm_ownership=True
                self.arm_debug_status.set('底盘暂停及停稳已确认，机械臂正在安全归位')
                self.arm_debug_request('return_origin')
        elif d['phase'] in {'returning','checking'} and not stationary:
            self.fail_arm_debug('归位期间底盘运动或反馈失效')

    def direct_pause(self, paused=True, generation=None):
        try:
            with self.pause_lock:
                if not paused and (generation!=self.generation or self.state.stopped):return
                client=connect_navigation_host(timeout=4)
                try:
                    value='true' if paused else 'false'
                    script="source /opt/ros/melodic/setup.bash\nsource /home/ubuntu/ud_ws/install/setup.bash\nset -e\n"
                    names=NAVIGATION_PAUSE_SERVICES
                    for name in names:
                        script+="rosservice call "+name+" '{data: {data: "+value+"}}'\n"
                    i,o,e=client.exec_command('bash -s',timeout=6);i.write(script);i.channel.shutdown_write()
                    output=o.read().decode();error=e.read().decode()
                    if o.channel.recv_exit_status()!=0 or output.count('res: True')!=len(names):
                        raise RuntimeError(error or output)
                    self.events.put(('pause_result' if paused else 'resume_result',
                                     {'generation':generation,'services':list(names),'paused':paused,
                                      'message':str(len(names))+'个厂商暂停接口均已确认'+('暂停' if paused else '解除暂停')}))
                finally:client.close()
        except Exception as e:self.events.put(('pause_error' if paused else 'resume_error',str(e)))

    def repeat_stop(self):
        if self.state.stopped and not self.closed.is_set():
            try:self.send(6,'1')
            except Exception as e:self.record('重复暂停失败',error=str(e))

    def fresh(self):
        return telemetry_fresh(self.telemetry,time.monotonic())

    def packet(self,p):
        self.record('收到 Android',topic=p.topic,sequence=p.sequence,data=p.data)
        if p.topic=='active_result' and self.state.goal_name=='back' and self.state.sent and not self.state.stopped and p.sequence==self.dispatch_sequence and isinstance(p.data,dict):
            if p.data.get('result') is True:
                self.charge_ack=True;self.record('回桩传输确认',detail='等待导航板本次充电指令接收及服务成功反馈')
            elif p.data.get('result') is False:
                self.record('回桩请求被拒绝')
                if getattr(self,'mission',None) is not None and self.mission.active:self.fail_mission('Android拒绝本次回桩请求')
                else:self.stop()
            return
        if p.topic!='bt_control' or not isinstance(p.data,dict):return
        cmd=p.data.get('command');value=p.data.get('value')
        if cmd==10 and isinstance(value,str) and value in {str(i) for i in range(1,17)}:
            if getattr(self,'armed_link_mode',False):
                self.accept_mission_target(p,int(value));return
            self.floor.set('目标楼层：'+value);result=self.state.accept(value,time.monotonic());self.record('楼层处理',result=result)
            if result in ('accepted','duplicate'):self.sock.sendto(b'accepted',p.source);self.record('传输确认',detail='accepted 仅停止重发，不表示完成')
            elif self.state.stopped:self.sock.sendto(b'null',p.source)
            if result=='accepted':self.goal_revision_floor=self.telemetry.get('goal_revision',0);self.telemetry.pop('goal',None);self.goal_since=time.monotonic();self.arrival.set('厂商到位：未收到');self.status.set('已接收任务，等待人员离开')
        elif cmd==1 and value=='arrive':
            if getattr(self,'mission',None) is not None and self.mission.active and self.state.goal_name in {'liftout','liftin'}:
                if getattr(self,'_arrival_packets',None) is None:self._arrival_packets=AcceptedPacketCache()
                replay=self._arrival_packets.classify(p)
                if replay=='replay':
                    self.sock.sendto(b'accepted',p.source);self.record('旧到位报文已去重',goal=self.state.goal_name);return
                if replay=='conflict':
                    self.record('拒绝冲突到位报文',sequence=p.sequence);return
                if self.mission.expected_navigation_goal!=self.state.goal_name or not self.state.sent:
                    self._arrival_packets.mark_accepted(p)
                    self.sock.sendto(b'accepted',p.source);self.record('忽略联动阶段之外的到位报文',phase=self.mission.phase);return
                self._arrival_packets.mark_accepted(p)
                self.mission.arrive(time.monotonic())
            if self.state.goal_name=='back':
                if self.state.sent and not self.state.stopped:self.sock.sendto(b'accepted',p.source)
                self.record('回桩到位报文',detail='不作为充电成功依据');return
            if not self.state.sent or self.state.stopped:self.record('忽略本次任务之外的到位报文');return
            self.state.arrived=True;self.state.enabled=False;self.sock.sendto(b'accepted',p.source)
            self.point_selector.configure(state='disabled' if getattr(self,'mission',None) is not None and self.mission.active else 'readonly')
            if not getattr(self,'mission',None) or not self.mission.active:self.lock_mission_controls(False)
            self.arrival.set('厂商到位：已报告（不等于位置准确）');self.status.set('已收到厂商到位，核验实际位姿；机械臂未启动')

    def accept_mission_target(self, packet, target):
        cached=self._mission_packets.classify(packet)
        if cached=='replay':
            self.sock.sendto(b'accepted',packet.source);self.record('旧任务报文重发已去重');return
        if cached=='conflict':
            self.record('拒绝冲突报文',sequence=packet.sequence);return
        if self.mission.active:
            if target==self.mission.target_floor:
                self._mission_packets.mark_accepted(packet);self.sock.sendto(b'accepted',packet.source)
                self.record('联动任务重复目标已去重',mission_id=self.mission.mission_id)
            else:self.record('联动执行中拒绝新目标',target=target)
            return
        if not self.state.enabled or self.state.stopped:
            self.record('楼层处理',result='monitor');return
        try:
            direction=direction_for(self.pending_current_floor,target)
        except ValueError as exc:
            self.status.set(str(exc));self.sock.sendto(b'null',packet.source);self.record('联动目标未接受',reason=str(exc));return
        result=self.state.accept(str(target),time.monotonic())
        self.record('楼层处理',result=result)
        if result!='accepted':return
        self.mission.begin(self.pending_current_floor,target,time.monotonic())
        self._press_request_id=None
        self.lock_mission_controls(True);self._mission_packets.mark_accepted(packet)
        self.sock.sendto(b'accepted',packet.source)
        self.floor.set('当前 '+str(self.mission.current_floor)+'楼 → 目标 '+str(target)+'楼；按'+('上行' if direction=='up' else '下行'))
        self.record('完整电梯联动任务已接收' if self.mission.include_floor_panel else '外呼联动任务已接收',mission_id=self.mission.mission_id,current_floor=self.mission.current_floor,target_floor=target,direction=direction,include_floor_panel=self.mission.include_floor_panel)
        self.status.set(f'检查机械臂原位；导航下发前等待人员离开{INITIAL_NAVIGATION_DELAY_SEC}秒')
        try:
            self._mission_check_started=time.monotonic()
            self._origin_request_id=self._arm_client.request('check_origin',self.mission.mission_id)
        except Exception as exc:self.fail_mission('机械臂原位检查无法启动：'+str(exc))

    def fail_mission(self, reason):
        failed_phase=self.mission.phase
        self.mission.fail(reason);self.record('完整电梯联动任务失败' if self.mission.include_floor_panel else '外呼联动任务失败',mission_id=self.mission.mission_id,phase=self.mission.phase,failed_phase=failed_phase,reason=reason)
        self.stop();self.status.set('联动任务失败：'+str(reason))

    def handle_arm_worker(self, message):
        self.record('机械臂后端反馈',data=message)
        if getattr(self,'arm_debug',None) and message.get('mission_id')==self.arm_debug['mission_id']:
            self.handle_arm_debug(message);return
        if message.get('mission_id')!=self.mission.mission_id:return
        if message.get('operation')=='stop':
            if message.get('event')=='error':self.status.set('机械臂停止尚未确认：'+message.get('error',''))
            return
        if not self.mission.active:return
        if message.get('operation') in {'press','press_floor'}:
            expected_phase='arm_floor' if message['operation']=='press_floor' else 'arm_call'
            expected_request=getattr(self,'_press_request_id',None)
            if self.mission.phase!=expected_phase or (expected_request is not None and message.get('request_id')!=expected_request):
                self.record('忽略旧机械臂阶段反馈',operation=message['operation'],phase=self.mission.phase);return
        if message.get('event')=='error':
            self.fail_mission(message.get('error','机械臂后端失败'));return
        result=message.get('result') or {}
        if message.get('operation')=='check_origin' and message.get('event')=='completed':
            if message.get('request_id')!=self._origin_request_id:return
            self._mission_check_started=None
            action=self.mission.origin_checked(message['mission_id'],result,time.monotonic())
            if action=='failed':self.fail_mission(self.mission.reason)
            elif action=='ready_navigation':self.status.set(f'机械臂原位已确认，等待{INITIAL_NAVIGATION_DELAY_SEC}秒导航延迟结束')
            elif action=='navigate_back':self.queue_mission_return()
            elif action=='navigate_floor':self.queue_mission_floor()
            elif action=='pause_for_origin':
                self.arm_pause_confirmed=False;self.stationary_since=None
                self.status.set('机械臂未在原位：先暂停底盘，再执行现有安全归位')
                threading.Thread(target=self.direct_pause,args=(True,self.generation),daemon=True).start()
        elif message.get('operation')=='return_origin':
            if message.get('event')=='completed':
                self._arm_ownership=False
                action=self.mission.start_return_finished(message['mission_id'],result)
                if action=='failed':self.fail_mission(self.mission.reason)
                elif action=='check_origin':
                    self._mission_check_started=time.monotonic()
                    try:self._origin_request_id=self._arm_client.request('check_origin',self.mission.mission_id)
                    except Exception as exc:self.fail_mission('出发前归位复核无法启动：'+str(exc))
        elif message.get('operation') in {'press','press_floor'}:
            button_name='楼层'+str(self.mission.target_floor)+'按钮' if message['operation']=='press_floor' else '外呼按钮'
            if message.get('event')=='progress':
                events=result.get('mission_status_events') or []
                phase=events[-1].get('phase') if events else ''
                text={'button_on_confirmed':button_name+'已亮，等待熄灭',
                      'external_call_timeout_release':'外呼等待熄灭满30秒，按测试规则超时放行；未确认熄灭',
                      'initial_observation_reacquiring':'正在重新识别'+button_name+'并确认灯态',
                      'initial_view_search':'正在小幅调整观察视角，寻找'+button_name,
                      'initial_observation_waiting':result.get('message','暂未确认'+button_name+'，保持静止等待恢复'),
                      'initial_observation_recovered':button_name+'识别已恢复，继续任务',
                      'waiting_for_floor':'持续观察'+button_name+'，等待熄灭',
                      'retry_required':button_name+'未亮，尝试再次按压',
                      'floor_arrival_observed':button_name+'已熄灭，等待机械臂归位'}.get(phase,result.get('message','机械臂闭环执行中'))
                self.status.set(text)
            elif message.get('event')=='completed':
                self._arm_ownership=False
                action=self.mission.arm_finished(message['mission_id'],result)
                if action=='failed':self.fail_mission(self.mission.reason)
                elif action=='check_origin':
                    timed_out = result.get('off_wait_timed_out') is True
                    if timed_out:
                        self.record('外呼等待熄灭超时放行',
                                    evidence=result.get('external_call_off_wait_timeout'),
                                    off_confirmed=result.get('off_confirmed'))
                    self.status.set('外呼超时放行，再次核验机械臂原位' if timed_out else
                                    '按键闭环完成，再次读取实时编码器核验原位')
                    self._mission_check_started=time.monotonic()
                    try:self._origin_request_id=self._arm_client.request('check_origin',self.mission.mission_id)
                    except Exception as exc:self.fail_mission('归位核验无法启动：'+str(exc))

    def queue_mission_return(self):
        if self.mission.phase!='returning_dock':return
        self.queue_mission_navigation('back')

    def queue_mission_floor(self):
        if self.mission.phase!='preparing_floor':return
        self.queue_mission_navigation('liftin')

    def queue_mission_navigation(self, goal):
        try:
            now=time.monotonic();point=self.point_catalog['points'][goal]
            pose=tuple(float(point[k]) for k in ('x','y','z','w'))
            if not all(math.isfinite(v) for v in pose):raise ValueError('目标坐标无效')
            self.state.enable(goal,pose)
        except (KeyError,TypeError,ValueError) as exc:
            self.fail_mission('后续导航点位不可用：'+str(exc));return
        self.generation+=1
        self.state.accept(str(self.mission.target_floor),now);self.state.deadline=now
        self.selected_point.set(goal);self.point_selector.configure(state='disabled')
        self.resume_requested=False;self.resume_confirmed=False;self.navigation_sent_at=None
        self.charge_ack=False;self.charge_complete=False;self.charge_confirm_since=None
        self.dispatch_sequence=None;self.charge_dispatch_time=None
        self.arm_pause_confirmed=False;self.stationary_since=None;self.last_verification=None
        self.goal_revision_floor=self.telemetry.get('goal_revision',0);self.telemetry.pop('goal',None)
        self.goal_since=now;self.arrival.set('厂商到位：未收到')
        self.status.set('机械臂实际归位已确认，自动下发'+('内呼导航' if goal=='liftin' else '回桩'))
        self.record('联动内呼导航已排队' if goal=='liftin' else '联动回桩已排队',mission_id=self.mission.mission_id,goal=goal,target_floor=self.mission.target_floor,delay_sec=0)

    def tick_mission(self, now):
        labels={'pausing_start_origin':'出发前暂停底盘','returning_start_origin':'出发前安全归位','idle':'联动待命','checking_start_origin':'检查出发前原位','waiting_delay':'等待导航延迟',
                'navigating_call':'导航到外呼机','pausing_chassis':'确认底盘暂停并停稳',
                'arm_call':'外呼按键与亮灭闭环','checking_end_origin':'核验实际归位',
                'preparing_floor':'准备导航到内呼机','navigating_floor':'导航到内呼机',
                'pausing_floor_chassis':'内呼停稳并暂停底盘','arm_floor':'目标楼层按键与亮灭闭环',
                'checking_floor_origin':'核验内呼完成后归位',
                'returning_dock':'准备自动回桩','charging':'回桩并等待充电','completed':'联动完成',
                'failed':'联动失败','aborted':'联动已中止'}
        if self.mission.mission_id:
            self.mission_status.set(labels[self.mission.phase]+'；'+str(self.mission.current_floor)+'→'+str(self.mission.target_floor)+'楼 '+('上行' if self.mission.direction=='up' else '下行'))
        if not self.mission.active:return
        if self._mission_check_started is not None and now-self._mission_check_started>20:
            self.fail_mission('机械臂原位检查超时');return
        action=self.mission.observe(self.telemetry,self.state.goal_pose,now,self.resume_confirmed)
        if action=='failed':self.fail_mission(self.mission.reason);return
        if action=='pause_for_arm':
            self.arm_pause_confirmed=False;self.stationary_since=None
            panel='内呼' if self.state.goal_name=='liftin' else '外呼'
            self.status.set(panel+'停车位置已核验，正在暂停底盘以交接机械臂')
            self.record(panel+'到位交接',mission_id=self.mission.mission_id,pose=self.telemetry.get('pose'),expected_pose=self.state.goal_pose,position_tolerance_m=self.mission.position_tolerance_for(self.state.goal_name),yaw_tolerance_rad=self.mission.yaw_tolerance_rad)
            threading.Thread(target=self.direct_pause,args=(True,self.generation),daemon=True).start()
        if self.mission.phase in {'pausing_chassis','pausing_start_origin','pausing_floor_chassis'}:
            stationary=(self.arm_pause_confirmed and self.fresh() and self.stationary_since is not None and now-self.stationary_since>=.5)
            action=self.mission.chassis_paused(now,stationary)
            if action=='failed':self.fail_mission(self.mission.reason);return
            if action=='return_origin':
                self._arm_ownership=True;self._chassis_last_good=now
                try:
                    self._arm_client.request('return_origin',self.mission.mission_id)
                    self.status.set('底盘已暂停，机械臂执行出发前安全归位')
                except Exception as exc:self.fail_mission('出发前安全归位无法启动：'+str(exc))
            if action in {'press','press_floor'}:
                self._arm_ownership=True;self._chassis_last_good=now
                try:
                    self._press_request_id=self._arm_client.request(action,self.mission.mission_id,current_floor=self.mission.current_floor,
                                             target_floor=self.mission.target_floor,direction=self.mission.direction)
                    self.record('联动机械臂已请求',mission_id=self.mission.mission_id,operation=action,direction=self.mission.direction,target_floor=self.mission.target_floor)
                except Exception as exc:self.fail_mission('机械臂任务无法启动：'+str(exc))
        if self.mission.phase in {'returning_start_origin','arm_call','checking_end_origin','arm_floor','checking_floor_origin'}:
            if self.fresh():
                v=self.telemetry['velocity']
                if math.hypot(v[0],v[1])>=.01 or abs(v[2])>=.02:
                    self.fail_mission('机械臂阶段底盘发生移动');return
                self._chassis_last_good=now
            elif self._chassis_last_good is not None and now-self._chassis_last_good>2:
                self.fail_mission('机械臂阶段底盘速度反馈中断');return
        if self.mission.phase=='charging' and self.mission.nav_started is not None and now-self.mission.nav_started>240:
            self.fail_mission('自动回桩充电超时')

    def tick(self):
        for _ in range(200):
            try:kind,data=self.events.get_nowait()
            except queue.Empty:break
            if kind=='arm_worker':self.handle_arm_worker(data)
            elif kind=='arm_worker_exit':
                self.record('机械臂后端退出',**data)
                if getattr(self,'arm_debug',None):self.fail_arm_debug('机械臂后端意外退出')
                if self.mission.active:self.fail_mission('机械臂后端意外退出')
            elif kind=='packet':self.packet(data)
            elif kind=='telemetry':
                if self.goal_since and data.get('goal_revision',0)<=self.goal_revision_floor:data.pop('goal',None)
                self.telemetry=data
            elif kind=='ready':self.connection.set('Android：192.168.168.110 ⇄ AGX 5005，监听正常');self.record('监听就绪')
            elif kind=='enabled':
                if data==self.generation:
                    self.stop_since=None;self.resume_requested=False;self.resume_confirmed=False;self.navigation_sent_at=None
                    self.charge_ack=False;self.charge_complete=False;self.charge_confirm_since=None;self.dispatch_sequence=None
                    self.state.enable(*self.pending_goal);self.armed_link_mode=self.pending_link_mode
                    self.mission.include_floor_panel=getattr(self,'pending_include_floor',False)
                    self.arm_pause_confirmed=False;self.lock_mission_controls(True)
                    mode=('full_elevator_return' if self.mission.include_floor_panel else 'external_call_return') if self.armed_link_mode else 'navigation_only'
                    self.status.set(('外呼→内呼→回桩' if self.mission.include_floor_panel else '外呼→回桩')+'已启用：当前'+str(self.pending_current_floor)+'楼，请在Android选择目标楼层' if self.armed_link_mode else '导航已启用：'+self.state.goal_name+'，可点击手动导航或下达Android任务')
                    self.record('执行权限开启',goal=self.state.goal_name,expected_pose=self.state.goal_pose,mode=mode,current_floor=self.pending_current_floor if self.armed_link_mode else None)
                self.enable_button.configure(state='normal',text='导航测试已开启' if self.state.enabled else '开启导航测试（需再次确认）')
            elif kind=='resume_result':
                if data['generation']==self.generation and not self.state.stopped:
                    self.resume_confirmed=True
                    self.record('导航恢复已确认',detail=data['message'],services=data['services']);self.status.set('暂停已解除，厂商自动回桩执行中；等待真实充电反馈' if self.state.goal_name=='back' else '导航、重定位及充电暂停已解除，执行下桩/导航至 '+self.state.goal_name)
            elif kind=='pause_result':
                self.record('暂停服务已确认',detail=data['message'],services=data['services'])
                if getattr(self,'arm_debug',None) and data['generation']==self.generation:
                    self.arm_debug['paused']=True
                if data['generation']==self.generation and self.mission.phase in {'pausing_chassis','pausing_start_origin','pausing_floor_chassis'}:
                    self.arm_pause_confirmed=True;self.stationary_since=None
            elif kind=='resume_error':
                self.record('导航恢复失败',error=data)
                if self.mission.active:self.fail_mission('导航恢复失败：'+str(data))
                else:self.stop()
            elif kind=='pause_error':
                self.record('暂停服务失败',error=data)
                if getattr(self,'arm_debug',None):self.fail_arm_debug('底盘暂停失败，禁止归位')
                if self.mission.active:self.fail_mission('底盘暂停接口失败')
            elif kind=='enable_error':self.record('开启失败',error=data);self.status.set('开启失败：'+data);self.enable_button.configure(state='normal');self.point_selector.configure(state='readonly');self.lock_mission_controls(False)
            else:
                self.record(kind,error=data);self.connection.set(str(data))
                if self.mission.active and kind in {'error','listen_error'}:self.fail_mission('联动通信异常：'+str(data))
        now=time.monotonic()
        if self.confirm_until and now>self.confirm_until:self.confirm_until=0;self.enable_button.configure(text='开启导航测试（需再次确认）')
        if self.state.deadline and not self.state.sent:self.countdown.set('延迟：%.1f 秒'%max(0,self.state.deadline-now))
        else:self.countdown.set('延迟：—')
        dispatch_allowed=(not self.mission.active or self.mission.phase in {'waiting_delay','preparing_floor','returning_dock'})
        if dispatch_allowed and self.state.due(now):
            try:
                if not self.fresh():raise RuntimeError('底盘反馈已过期')
                self.goal_revision_floor=self.telemetry.get('goal_revision',0);self.goal_since=time.monotonic();self.telemetry.pop('goal',None)
                self.charge_request_baseline=self.telemetry.get('charge_request_revision',0);self.charge_service_baseline=self.telemetry.get('charge_service_revision',0);self.charge_ack=False;self.dispatch_sequence=self.sequence+1
                self.charge_dispatch_time=self.telemetry.get('remote_time',0)+now-self.telemetry.get('received_monotonic',now)
                self.send(1,self.state.goal_name);self.navigation_sent_at=now;self.status.set('回桩请求已发，等待本次充电服务接收核验' if self.state.goal_name=='back' else '目标 '+self.state.goal_name+' 已下发，等待新目标规划后解除暂停');self.record('导航已下发',floor=self.state.floor,goal=self.state.goal_name)
                if self.mission.active:self.mission.sent(self.state.goal_name,now)
            except Exception as e:
                self.record('导航未成功下发',error=str(e))
                if self.mission.active:self.fail_mission('导航未成功下发：'+str(e))
                else:self.stop()
        d=self.telemetry
        if self.state.sent and not self.state.stopped and not self.resume_requested and self.fresh():
            goal=d.get('goal')
            # Resume only for a new plan matching the frozen selected point.
            allowed=(charge_resume_allowed(self.state,d,self.charge_request_baseline,self.charge_service_baseline,self.charge_ack,now,self.charge_dispatch_time)
                     if self.state.goal_name=='back' else resume_allowed(self.state,d,self.goal_revision_floor,now))
            if allowed:
                self.resume_requested=True
                self.record('回桩指令已核验' if self.state.goal_name=='back' else '新目标规划已核验',goal=None if self.state.goal_name=='back' else goal,detail='本次Android确认、充电指令9接收及厂商服务调用成功' if self.state.goal_name=='back' else '解除本次测试的暂停，不恢复未知旧目标')
                threading.Thread(target=self.direct_pause,args=(False,self.generation),daemon=True).start()
            elif self.navigation_sent_at and now-self.navigation_sent_at>8:
                self.status.set('回桩请求已发，但本次充电服务接收尚未核验；保持暂停' if self.state.goal_name=='back' else '目标已下发，但未收到匹配的新规划；保持暂停')
                if self.state.goal_name=='back' and now-self.navigation_sent_at>30:
                    self.record('回桩接收核验超时',detail='未放行；请检查充电管理反馈',android_ack=self.charge_ack,request_baseline=self.charge_request_baseline,service_baseline=self.charge_service_baseline,dispatch_source_time=self.charge_dispatch_time,evidence={k:v for k,v in d.items() if k.startswith('charge_')})
                    if self.mission.active:self.fail_mission('回桩接收核验超时')
                    else:self.stop()
        if self.state.goal_name=='back' and self.state.sent and not self.state.stopped:
            if d.get('charge_error_age',float('inf'))<1:
                self.record('厂商回桩拒绝',detail=d.get('charge_error'))
                if self.mission.active:self.fail_mission('厂商回桩拒绝：'+str(d.get('charge_error')))
                else:self.stop()
            confirmed=(not self.state.stopped and self.resume_confirmed and self.fresh() and d.get('charge_work_age',float('inf'))<1
                       and d.get('real_charging') is True and d.get('charge_signal') is True
                       and math.hypot(*d['velocity'][:2])<.01 and abs(d['velocity'][2])<.02)
            if confirmed:
                if self.charge_confirm_since is None:self.charge_confirm_since=now
                if now-self.charge_confirm_since>=1 and not self.charge_complete:
                    self.charge_complete=True;self.state.arrived=True;self.state.enabled=False
                    self.point_selector.configure(state='readonly');self.status.set('自动回桩：真实充电反馈及停稳已确认')
                    self.record('真实充电已确认',detail='充电管理真实充电与充电信号均为真，且停稳持续1秒')
                    if self.mission.charged():
                        self.lock_mission_controls(False);self.record('完整电梯联动任务完成' if self.mission.include_floor_panel else '外呼联动任务完成',mission_id=self.mission.mission_id,
                                                                     external_call_off_wait_timed_out=self.mission.call_off_wait_timed_out)
                        self.status.set(('联动及回桩充电已完成；外呼按30秒超时放行，未确认外呼熄灭'
                                         if self.mission.call_off_wait_timed_out else
                                         '外呼、内呼楼层按键、熄灭等待及回桩充电已完成' if self.mission.include_floor_panel else '外呼按键、熄灭等待及回桩充电已完成'))
            else:self.charge_confirm_since=None
            self.precision.set('自动回桩使用厂商充电点，不使用普通 back 坐标核验')
        if self.fresh():
            p=d['pose'];v=d['velocity'];self.pose.set('实际位姿：x %.3f，y %.3f'%(p[0],p[1]));self.velocity.set('速度：%.3f m/s，角速度 %.3f rad/s'%(math.hypot(v[0],v[1]),v[2]))
            stationary=math.hypot(v[0],v[1])<.01 and abs(v[2])<.02
            if stationary:
                if self.stationary_since is None:self.stationary_since=now
            else:self.stationary_since=None
            if self.state.stopped and self.stationary_since and now-self.stationary_since>.5:
                self.status.set('任务已停止：底盘速度反馈已连续 0.5 秒停稳')
                if not self.stop_logged:self.record('停稳已核验',velocity=v);self.stop_logged=True
            if 'goal' in d and self.state.goal_name!='back':
                linked=self.armed_link_mode and self.state.goal_name in {'liftout','liftin'}
                g=self.state.goal_pose if linked else d['goal']
                pos_limit=self.mission.position_tolerance_for(self.state.goal_name) if linked else .05
                yaw_limit=self.mission.yaw_tolerance_rad if linked else .1
                distance=math.hypot(p[0]-g[0],p[1]-g[1]);yaw=abs(math.atan2(math.sin(2*math.atan2(p[2],p[3])-2*math.atan2(g[2],g[3])),math.cos(2*math.atan2(p[2],p[3])-2*math.atan2(g[2],g[3]))))
                within=distance<=pos_limit and yaw<=yaw_limit
                if self.state.arrived and self.last_verification!=(self.state.floor,within):
                    self.last_verification=(self.state.floor,within)
                    self.record('到位坐标核验',distance_m=distance,yaw_error_rad=yaw,within_tolerance=within,position_tolerance_m=pos_limit,yaw_tolerance_rad=yaw_limit,stationary=stationary,reference='本次'+self.state.goal_name+'联动目标坐标' if linked else '厂商规划路径终点，非命名点数据库')
                self.precision.set('%s差：%.1f cm / %.1f° %s%s'%('内呼目标' if linked and self.state.goal_name=='liftin' else '外呼目标' if linked else '规划终点',distance*100,math.degrees(yaw),'（误差范围内）' if within else '（未到位）',(' 门槛%.0fcm'%(pos_limit*100)) if linked else ''))
            elif self.state.goal_name!='back':self.precision.set('位置核验：等待本次规划目标坐标')
        else:
            self.velocity.set('速度反馈过期：不能确认停止');self.pose.set('位姿反馈未就绪或过期')
        if self.stop_since and now-self.stop_since>5 and (not self.fresh() or not self.stationary_since):self.status.set('已请求暂停，但尚未确认底盘停稳')
        self.tick_mission(now)
        self.tick_arm_debug(now)
        if self.mission.phase=='failed':
            self.status.set('联动任务失败：'+self.mission.reason+'；'+('底盘停稳已确认' if self.stop_logged and self.fresh() and self.stationary_since is not None else '等待底盘停稳确认'))
        if not self.closed.is_set():self.root.after(100,self.tick)

    def close(self):
        self.stop();self.root.after(1200,self.finish_close)

    def finish_close(self):
        self.record('关闭界面',stationary_confirmed=bool(self.fresh() and self.stationary_since and time.monotonic()-self.stationary_since>.5))
        self.closed.set();self._arm_client.close()
        if self.ssh:self.ssh.close()
        self.journal.close();self.root.destroy()

if __name__=='__main__':
    Gui(tk.Tk()).root.mainloop()
