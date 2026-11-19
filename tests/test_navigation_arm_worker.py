from unittest.mock import MagicMock
from tools.navigation_arm_worker import Worker, arm_result


def request(op='press',rid='p',direction='down'):
    return dict(request_id=rid,mission_id='m',operation=op,current_floor=16,target_floor=8,direction=direction)


def records():
    return [dict(producer='arm_pressing',phase='button_on_confirmed'),
            dict(producer='arm_pressing',phase='floor_arrival_observed'),
            dict(producer='arm_pressing',raw_event='task_terminal',evidence=dict(
                flow_status='arrived_after_verified_press',returned_to_origin=True,
                origin_verification=dict(verified=True,hardware_ok=1)))]


def test_exit_zero_alone_is_not_press_success():
    r=arm_result([],0)
    assert not r['off_confirmed'] and not r['origin_verified']
    r=arm_result(records(),0);assert r['on_confirmed'] and r['off_confirmed'] and r['origin_verified']
    bad=records();bad[-1]['evidence']['origin_verification']['hardware_ok']=0
    assert not arm_result(bad,0)['origin_verified']


def test_only_direction_command_is_dispatched_and_duplicates_do_not_restart():
    d=MagicMock();d.active=None;d.dispatch_motion.return_value={'event':'motion_started'}
    emitted=[];w=Worker(d,emitted.append)
    w.handle(request(direction='up'));d.dispatch_motion.assert_not_called()
    w.handle(request(rid='valid'));assert d.dispatch_motion.call_args.args[0].button=='down'
    w.handle(request(rid='valid'));assert d.dispatch_motion.call_count==1
    d.refresh.return_value={'event':'motion_completed','return_code':0,'mission_status_events':records()}
    w.refresh();assert emitted[-1]['result']['off_confirmed'] and emitted[-1]['event']=='completed'
    w.handle(request(rid='valid'));assert d.dispatch_motion.call_count==1


def test_stop_pending_keeps_press_ownership_until_process_exit():
    d=MagicMock();d.active=None;d.dispatch_motion.return_value={};w=Worker(d,lambda m:None)
    w.handle(request());d.active=object();d.stop.return_value={'event':'stop_pending'}
    w.handle(request('stop','s'))
    assert w.press is not None
    d.refresh.return_value={'event':'motion_completed','return_code':2,'mission_status_events':[]}
    w.refresh();assert w.press is None
