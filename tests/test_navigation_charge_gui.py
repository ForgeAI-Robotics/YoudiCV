import ast
import threading
import re
from types import SimpleNamespace
from unittest.mock import MagicMock
from tools.navigation_task_gui import Gui, REMOTE_MONITOR
from piper_handeye.android_agx_protocol import AndroidPacket
from piper_handeye.navigation_gui_state import NavigationState


def test_monitor_rejects_latched_old_charge_logs_and_unrelated_modes():
    source=REMOTE_MONITOR.split("python -u - <<'MONITOR'\n",1)[1].rsplit('\nMONITOR',1)[0]
    tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='charge_log')
    env=dict(time=SimpleNamespace(time=lambda:100),lock=threading.Lock(),data={},re=re)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<charge_log>','exec'),env)
    def call(node,text,stamp):
        env['charge_log'](SimpleNamespace(name=node,msg=text,header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:stamp))))
    call('/charging_manager_node','[getGoalSrv] mode: 9 new_ctrl_mode_int: 9',90)
    call('/charging_manager_node','[getGoalSrv] mode: 10 new_ctrl_mode_int: 10',100)
    assert not env['data']
    call('/charging_manager_node','[getGoalSrv] mode: 9 new_ctrl_mode_int: 9',99.9)
    call('/udp_commit_node','call move base charge goal service OK',100)
    assert env['data']['charge_request_revision']==1 and 'charge_service_revision' not in env['data']
    call('/udp_commit_node','call charge goal service OK',100)
    assert env['data']['charge_service_revision']==1
    assert env['data']['charge_request_time']==99.9


def gui():
    g=Gui.__new__(Gui);g.state=NavigationState();g.state.enable('back',(1,2,0,1));g.state.accept('manual',0);g.state.due(10)
    g.dispatch_sequence=101;g.charge_ack=False;g.record=MagicMock();g.stop=MagicMock();g.sock=MagicMock()
    return g


def test_charge_ack_rejects_previous_dispatch_and_stopped_task():
    g=gui()
    def ack(seq,result=True):return AndroidPacket(('192.168.168.110',5002),seq,'active_result',{'result':result},'')
    g.packet(ack(100));assert not g.charge_ack
    g.packet(ack(101));assert g.charge_ack
    g.charge_ack=False;g.state.stop();g.packet(ack(101));assert not g.charge_ack
    g=gui();g.packet(ack(101,False));g.stop.assert_called_once()


def test_arrive_does_not_mean_charging_complete():
    g=gui();g.packet(AndroidPacket(('192.168.168.110',5002),202,'bt_control',{'command':1,'value':'arrive'},''))
    assert not g.state.arrived
    g.sock.sendto.assert_called_once_with(b'accepted',('192.168.168.110',5002))


def test_real_charge_request_logs_20261013_unlock_only_current_request():
    import json
    from pathlib import Path
    from piper_handeye.navigation_gui_state import charge_resume_allowed
    events=json.loads((Path(__file__).parent/'fixtures/charge_request_20261013.json').read_text())
    source=REMOTE_MONITOR.split("python -u - <<'MONITOR'\n",1)[1].rsplit('\nMONITOR',1)[0]
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='charge_log')
    now=max(e['stamp'] for e in events)+.1
    env=dict(time=SimpleNamespace(time=lambda:now),lock=threading.Lock(),data={},re=re)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<charge_log>','exec'),env)
    for e in events:
        env['charge_log'](SimpleNamespace(name=e['node'],msg=e['text'],header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda e=e:e['stamp']))))
    d=dict(env['data'],pose=[0,0,0,1],velocity=[0,0,0],pose_age=0,velocity_age=0,received_monotonic=100,
           charge_request_age=.1,charge_service_age=.1)
    state=NavigationState();state.enable('back',(6.9,1.3,0,1));state.accept('manual',0);state.due(10)
    sent=min(e['stamp'] for e in events)-.05
    assert charge_resume_allowed(state,d,0,0,True,100,sent)
    assert not charge_resume_allowed(state,d,0,0,False,100,sent)
    assert not charge_resume_allowed(state,d,0,0,True,100,now+1)
    state.stop();assert not charge_resume_allowed(state,d,0,0,True,100,sent)
