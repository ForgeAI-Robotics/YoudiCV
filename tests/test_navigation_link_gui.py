import json
import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock,patch
from tools.navigation_task_gui import Gui, NAVIGATION_PAUSE_SERVICES
from piper_handeye.navigation_gui_state import NavigationState
from piper_handeye.external_call_mission import ExternalCallMission
from piper_handeye.android_agx_protocol import AndroidPacket, AcceptedPacketCache
from tests.test_external_call_mission import ORIGIN, ARM


class Backend:
    busy=False
    def __init__(self):self.requests=[]
    def request(self,op,mid,**kwargs):
        rid=str(len(self.requests));self.requests.append(dict(request_id=rid,operation=op,mission_id=mid,**kwargs));return rid
    def close(self):pass


class InlineThread:
    def __init__(self,target,args=(),**kwargs):self.target=target;self.args=args
    def start(self):self.target(*self.args)


def gui():
    g=Gui.__new__(Gui);g.events=queue.Queue();g.state=NavigationState();g.state.enable('liftout',(-.528,3.01,-.019123,.999817))
    g.mission=ExternalCallMission();g.armed_link_mode=True;g.pending_current_floor=16
    g._mission_packets=AcceptedPacketCache();g._arm_client=Backend();g._arm_ownership=False
    g._mission_check_started=None;g._chassis_last_good=None;g.arm_pause_confirmed=False
    g.root=MagicMock();g.closed=threading.Event();g.sock=MagicMock();g.sequence=100;g.generation=1
    for n in ['status','connection','floor','countdown','pose','velocity','arrival','precision','mission_status','selected_point']:
        setattr(g,n,MagicMock())
    for n in ['enable_button','point_selector','current_floor_entry','current_floor_save','link_mode_button']:
        setattr(g,n,MagicMock())
    g.point_catalog=json.loads((Path(__file__).parents[1]/'config/navigation_test_points.json').read_text())
    g.telemetry={};g.confirm_until=0;g.goal_revision_floor=0;g.goal_since=None
    g.stop_since=None;g.stationary_since=None;g.stop_logged=False;g.last_verification=None
    g.resume_requested=False;g.resume_confirmed=False;g.navigation_sent_at=None
    g.dispatch_sequence=None;g.charge_dispatch_time=None;g.charge_ack=False;g.charge_request_baseline=0
    g.charge_service_baseline=0;g.charge_confirm_since=None;g.charge_complete=False
    g.log_rows=[];g.sent=[]
    def record(event,**values):g.log_rows.append(dict(event=event,**values))
    g.record=record
    def send(cmd,value):g.sequence+=1;g.sent.append((cmd,value,g.sequence))
    g.send=send
    def pause(paused=True,generation=None):
        g.events.put(('pause_result' if paused else 'resume_result',dict(generation=generation,services=list(NAVIGATION_PAUSE_SERVICES),message='confirmed')))
    g.direct_pause=pause
    return g


def packet(seq,cmd,value,topic='bt_control'):
    return AndroidPacket(('192.168.168.110',5002),seq,topic,{'command':cmd,'value':value},json.dumps({'command':cmd,'value':value}))


def worker_reply(g,index,result):
    r=g._arm_client.requests[index]
    g.handle_arm_worker(dict(channel='navigation_arm_worker',event='completed',result=result,**r))


def tick(g,now,**fields):
    g.telemetry.update(dict(pose=[-.528,3.01,-.019123,.999817],velocity=[0,0,0],
                           pose_age=0,velocity_age=0,received_monotonic=now,remote_time=1000+now))
    g.telemetry.update(fields)
    with patch('tools.navigation_task_gui.time.monotonic',return_value=now),patch('tools.navigation_task_gui.threading.Thread',InlineThread):g.tick()


def test_gui_full_cycle_initial_check_press_off_origin_then_back_and_charge():
    g=gui();p=packet(1,10,'8')
    with patch('tools.navigation_task_gui.time.monotonic',return_value=0):g.packet(p)
    assert g.mission.phase=='checking_start_origin'
    tick(g,10);assert not g.sent  # expired delay must not bypass origin check
    worker_reply(g,0,ORIGIN);tick(g,11);assert [r[1] for r in g.sent]==['liftout']
    tick(g,11.1,goal_revision=1,goal=list(g.state.goal_pose));tick(g,11.2)
    with patch('tools.navigation_task_gui.time.monotonic',return_value=12):g.packet(packet(2,1,'arrive'))
    for t in [12,12.2,12.4,12.6,12.7,12.9,13.1,13.3]:tick(g,t)
    assert g.mission.phase=='arm_call'
    assert g._arm_client.requests[1]['direction']=='down'
    g.packet(p);assert len(g._arm_client.requests)==2  # retransmission cannot restart
    for t in [100,100.2]:tick(g,t)
    assert [r[1] for r in g.sent]==['liftout']  # wait for OFF indefinitely
    worker_reply(g,1,ARM);assert g.mission.phase=='checking_end_origin'
    assert [r[1] for r in g.sent]==['liftout']
    with patch('tools.navigation_task_gui.time.monotonic',return_value=100.2):worker_reply(g,2,ORIGIN)
    tick(g,101)
    assert [r[1] for r in g.sent]==['liftout','back']
    g.packet(AndroidPacket(('192.168.168.110',5002),g.dispatch_sequence,'active_result',{'result':True},''))
    evidence=dict(charge_request_revision=1,charge_service_revision=1,
                  charge_request_time=1101.05,charge_service_time=1101.06,charge_request_age=.1,charge_service_age=.1,
                  charge_work_age=0,real_charging=True,charge_signal=True)
    tick(g,101.1,**evidence);tick(g,101.2,**evidence);tick(g,102.3,**evidence)
    assert g.mission.phase=='completed' and g.mission.current_floor==16


def test_gui_cancel_during_check_never_dispatches_navigation():
    g=gui()
    with patch('tools.navigation_task_gui.time.monotonic',return_value=0):g.packet(packet(1,10,'8'))
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):g.stop()
    worker_reply(g,0,ORIGIN);tick(g,20)
    assert not any(cmd==1 for cmd,_,_ in g.sent)
    assert g.mission.phase=='aborted'


def test_external_call_timeout_still_requires_origin_before_liftin():
    g=gui();g.mission.include_floor_panel=True
    mid=g.mission.begin(16,8,0);g.mission.phase='arm_call'
    result=dict(ARM, off_confirmed=False, off_wait_timed_out=True,
                press_execution_started=True, press_attempts=1,
                flow_status='external_call_off_timeout_after_verified_press',
                external_call_off_wait_timeout=dict(target_label='down', timeout_sec=30.,
                    elapsed_sec=30.1, off_confirmed=False,
                    release_policy='mock_scene_timeout_release'))
    g.handle_arm_worker(dict(event='completed', operation='press', request_id='p',
                             mission_id=mid, result=result))
    assert g.mission.phase=='checking_end_origin' and not g.sent
    assert '超时放行' in g.status.set.call_args.args[0]
    assert any(e['event']=='外呼等待熄灭超时放行' for e in g.log_rows)
    with patch('tools.navigation_task_gui.time.monotonic', return_value=31):
        worker_reply(g,0,ORIGIN)
    tick(g,32)
    assert [r[1] for r in g.sent]==['liftin']


def test_gui_failed_origin_after_press_never_returns_chassis():
    g=gui();g.mission.begin(16,8,0);g.mission.phase='arm_call';g._origin_request_id='not-yet'
    g.handle_arm_worker(dict(event='completed',operation='press',request_id='p',mission_id=g.mission.mission_id,result=ARM))
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):worker_reply(g,0,dict(ORIGIN,at_origin=False))
    tick(g,40)
    assert g.mission.phase=='failed' and not any(cmd==1 for cmd,_,_ in g.sent)


def test_same_floor_does_not_start_or_ack_a_motion_task():
    g=gui();g.packet(packet(1,10,'16'))
    assert not g.mission.active and not g._arm_client.requests and not g.sent
    g.sock.sendto.assert_called_with(b'null',('192.168.168.110',5002))


def test_stopped_press_does_not_queue_back_from_late_success():
    g=gui();g.mission.begin(16,8,0);g.mission.phase='arm_call';g._arm_ownership=True
    g._arm_client.request('press',g.mission.mission_id,direction='down',current_floor=16,target_floor=8)
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):g.stop()
    worker_reply(g,0,ARM);tick(g,20)
    assert g.mission.phase=='aborted' and g._arm_client.requests[-1]['operation']=='stop'
    assert not any(cmd==1 for cmd,_,_ in g.sent)


def test_start_recovery_waits_for_pause_and_live_origin_check():
    g=gui()
    with patch('tools.navigation_task_gui.time.monotonic',return_value=0):g.packet(packet(1,10,'8'))
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread),patch('tools.navigation_task_gui.time.monotonic',return_value=1):
        worker_reply(g,0,dict(ORIGIN,at_origin=False))
    for t in [1.1,1.3,1.5,1.7]:tick(g,t)
    assert g._arm_client.requests[1]['operation']=='return_origin' and not g.sent
    worker_reply(g,1,{'return_code':0})
    assert g._arm_client.requests[2]['operation']=='check_origin' and not g.sent
    worker_reply(g,2,ORIGIN);tick(g,10.1)
    assert [r[1] for r in g.sent]==['liftout']


def test_failure_keeps_reason_after_chassis_stop_and_records_source_phase():
    g=gui();g.mission.begin(16,8,0);g.mission.phase='arm_call'
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):g.fail_mission('灯态证据不完整')
    tick(g,10);tick(g,10.7)
    assert g.mission.phase=='failed'
    assert '灯态证据不完整' in g.status.set.call_args.args[0]
    failure=next(r for r in g.log_rows if r['event']=='外呼联动任务失败')
    assert failure['failed_phase']=='arm_call'


def test_android_back_rejection_is_failure_not_user_abort():
    g=gui();g.mission.begin(16,8,0);g.mission.phase='charging'
    g.state.goal_name='back';g.state.sent=True;g.dispatch_sequence=88
    p=AndroidPacket(('192.168.168.110',5002),88,'active_result',{'result':False},'')
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):g.packet(p)
    assert g.mission.phase=='failed' and '拒绝' in g.mission.reason


def debug_gui():
    g=gui();g.arm_debug=None;g.arm_debug_status=MagicMock()
    return g


def test_manual_return_serializes_stop_pause_return_and_encoder_verification():
    g=debug_gui()
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):g.return_arm()
    assert g._arm_client.requests[-1]['operation']=='stop'
    tick(g,1);tick(g,1.2);tick(g,1.4);tick(g,1.6)
    assert len(g._arm_client.requests)==1  # stop completion is required
    worker_reply(g,0,{'event':'idle'})
    for t in [2,2.2,2.4,2.6]:tick(g,t)
    assert g._arm_client.requests[-1]['operation']=='return_origin'
    worker_reply(g,1,{'return_code':0})
    assert g._arm_client.requests[-1]['operation']=='check_origin'
    worker_reply(g,2,ORIGIN)
    assert g.arm_debug is None and '归位成功' in g.arm_debug_status.set.call_args.args[0]
    assert not any(cmd==1 for cmd,_,_ in g.sent)


def test_manual_return_cannot_start_without_pause_confirmation():
    g=debug_gui();g.direct_pause=lambda *args:None
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):g.return_arm()
    worker_reply(g,0,{'event':'idle'})
    for t in [1,1.2,1.4,1.6,1.8]:tick(g,t)
    assert len(g._arm_client.requests)==1 and g.arm_debug['phase']=='waiting_pause'


def test_abort_cancels_pending_manual_return_even_after_late_stop_reply():
    g=debug_gui()
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):g.return_arm();g.stop()
    worker_reply(g,0,{'event':'idle'})
    tick(g,1);tick(g,1.6)
    assert g.arm_debug is None
    assert not any(r['operation']=='return_origin' for r in g._arm_client.requests)


def test_manual_return_failed_live_check_does_not_show_success():
    g=debug_gui()
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):g.return_arm()
    worker_reply(g,0,{'event':'idle'})
    for t in [1,1.2,1.4,1.6]:tick(g,t)
    worker_reply(g,1,{'return_code':0})
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):worker_reply(g,2,dict(ORIGIN,at_origin=False))
    assert g.arm_debug is None and '未确认原位' in g.arm_debug_status.set.call_args.args[0]


def test_terminate_arm_never_queues_return():
    g=debug_gui();g.mission.begin(16,8,0);g.mission.phase='arm_call';g._arm_ownership=True
    with patch('tools.navigation_task_gui.threading.Thread',InlineThread):g.terminate_arm()
    worker_reply(g,len(g._arm_client.requests)-1,{'event':'stopped'})
    tick(g,1)
    assert g.mission.phase=='aborted' and g.arm_debug is None
    assert not any(r['operation']=='return_origin' for r in g._arm_client.requests)
