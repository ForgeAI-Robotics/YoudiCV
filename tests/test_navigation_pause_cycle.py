"""Simulate vendor pause services without opening SSH or moving hardware."""
import io
import queue
import threading
from unittest.mock import MagicMock, patch

from tools.navigation_task_gui import Gui
from piper_handeye.navigation_gui_state import NavigationState

SERVICES=('/pause_robot_behavior','/robot_pause_to_relocate_behavior','/robot_pause_to_charging_behavior')


class Vendor:
    def __init__(self, rejected=None):
        self.flags={s:True for s in SERVICES};self.scripts=[];self.rejected=rejected
        self.client=MagicMock();self.client.exec_command=self.execute

    def execute(self, command, timeout):
        vendor=self
        class Input(io.StringIO):
            channel=MagicMock()
            def write(self, script):
                vendor.scripts.append(script)
                replies=[]
                for s in SERVICES:
                    if 'rosservice call '+s+' ' in script:
                        if s==vendor.rejected:replies.append('res: False')
                        else:
                            vendor.flags[s]='data: true' in script;replies.append('res: True')
                output.write('\n'.join(replies).encode());output.seek(0)
                return super().write(script)
        output=io.BytesIO();output.channel=MagicMock();output.channel.recv_exit_status.return_value=0
        return Input(),output,io.BytesIO()

def gui():
    g=Gui.__new__(Gui);g.state=NavigationState();g.generation=1
    g.events=queue.Queue();g.pause_lock=threading.Lock()
    return g


def call(g,v,paused):
    with patch('tools.navigation_task_gui.connect_navigation_host', return_value=v.client):g.direct_pause(paused,g.generation)
    return g.events.get_nowait()


def test_full_cycle_restores_charging_when_liftout_follows_stopped_back():
    g=gui();v=Vendor()
    g.state.enable('liftout',(-.528,3.01,-.019,.999))
    event,_=call(g,v,False);assert event=='resume_result'
    assert not any(v.flags.values())
    g.state.stop();event,_=call(g,v,True);assert event=='pause_result';assert all(v.flags.values())
    g.state.enable('back',(6.9,1.3,0,1));call(g,v,False)
    assert not any(v.flags.values())
    g.state.arrived=True;g.state.enabled=False
    g.state.stop();call(g,v,True);assert all(v.flags.values())
    g.state.enable('liftout',(-.528,3.01,-.019,.999));event,_=call(g,v,False)
    assert event=='resume_result';assert not any(v.flags.values())
    # The same service scope applies to both operations and both goal names.
    assert all(all('rosservice call '+s+' ' in script for s in SERVICES) for script in v.scripts)


def test_charging_pause_failure_prevents_resume_success_report():
    g=gui();v=Vendor(rejected=SERVICES[2]);g.state.enable()
    event,message=call(g,v,False)
    assert event=='resume_error' and 'res: False' in message
    assert v.flags[SERVICES[2]]


def test_stop_and_stale_generation_never_resume_services():
    g=gui();v=Vendor();g.state.enable();g.state.stop()
    with patch('tools.navigation_task_gui.connect_navigation_host', return_value=v.client):g.direct_pause(False,g.generation)
    assert not v.scripts and g.events.empty()
    g.state.enable()
    with patch('tools.navigation_task_gui.connect_navigation_host', return_value=v.client):g.direct_pause(False,g.generation-1)
    assert not v.scripts and g.events.empty()
