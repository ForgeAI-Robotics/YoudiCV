import ast,json,socket,struct,threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from piper_handeye.trajectory_transport import FramedTrajectoryServer,receive_frame,send_frame,MAX_TRAJECTORY_BYTES
from piper_handeye.timed_trajectory import TimedTrajectoryUdpClient,encode_trajectory_transport_payload,PIPER_JOINT_NAMES,validate_trajectory_payload

def payload(n=500,command='validate',task='test'):
    return dict(protocol='piper_timed_trajectory_v1',command=command,task_id=task,token='test',joint_names=PIPER_JOINT_NAMES,
        points=[dict(time_from_start_sec=15*i/(n-1),positions=[.1234567890123,.2345678901234,-.3456789012345,.1234567890123,.2345678901234,.1234567890123],velocities=[0.]*6,accelerations=[0.]*6) for i in range(n)])

def test_framing_handles_split_header_and_body():
    a,b=socket.socketpair()
    data=b'x'*90000;frame=struct.pack('!I',len(data))+data
    def writer():
        for offset in range(0,len(frame),3):a.sendall(frame[offset:offset+3])
        a.close()
    thread=threading.Thread(target=writer);thread.start()
    try:assert receive_frame(b,timeout_sec=5)==data
    finally:b.close();thread.join()

def test_oversize_header_and_truncated_body_rejected():
    for frame,error in [(struct.pack('!I',MAX_TRAJECTORY_BYTES+1),ValueError),(struct.pack('!I',50)+b'abc',ConnectionError)]:
        a,b=socket.socketpair();a.sendall(frame);a.close()
        with pytest.raises(error):receive_frame(b)
        b.close()

def test_full_trajectory_preserved_on_tcp():
    original=payload();encoded,wire,decimals=encode_trajectory_transport_payload(original)
    assert len(encoded)>60000 and decimals is None
    assert json.loads(encoded)==original and wire==original
    seen=[]
    def handler(p,address):
        validate_trajectory_payload(p);seen.append(p)
        return dict(task_id=p['task_id'],status='validated',point_count=len(p['points']))
    server=FramedTrajectoryServer(('127.0.0.1',0),'127.0.0.1',handler)
    client=TimedTrajectoryUdpClient('127.0.0.1',server.address[1],timeout_sec=3)
    client.health=lambda **kwargs:dict(trajectory_tcp_v1=True)
    try:
        result=client.execute(original['points'],token='test',task_id='test',dry_run=True)
        assert result['wire_transport']=='tcp' and result['wire_point_count']==500
        assert seen==[original]
    finally:server.close()

def test_incomplete_upload_never_calls_handler():
    seen=[];server=FramedTrajectoryServer(('127.0.0.1',0),'127.0.0.1',lambda p,a:seen.append(p))
    try:
        with socket.create_connection(server.address) as sock:
            sock.sendall(struct.pack('!I',1000)+b'{');sock.shutdown(socket.SHUT_WR)
            receive_frame(sock)
        assert not seen
    finally:server.close()

def test_legacy_bridge_capability_rejected_before_upload():
    client=TimedTrajectoryUdpClient('127.0.0.1',1)
    client.health=lambda **kwargs:dict(status='ok')
    with pytest.raises(RuntimeError,match='lacks bulk TCP'):
        client.execute(payload()['points'],token='test',dry_run=True)

def production_bridge():
    tree=ast.parse(Path('tools/orangepi_timed_trajectory_bridge.py').read_text())
    validate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_handle_tcp_request','_process_request')]
    cls.body=methods
    env=dict(np=np,json=json,hashlib=__import__('hashlib'),JOINT_NAMES=PIPER_JOINT_NAMES,
        JOINT_LIMITS=np.array([[-2.6179,2.6179],[0,3.14],[-2.967,.034906585],[-1.745,1.745],[-1.22,1.22],[-2.09439,2.09439]]),
        MAX_VELOCITIES=np.array([.6,1.,1.,1.,1.,2.]),VELOCITY_CAP_SCALE=.75,
        rospy=SimpleNamespace(logerr=lambda *args:None))
    exec(compile(ast.Module(body=[validate,cls],type_ignores=[]),'production_tcp_bridge','exec'),env)
    bridge=env['Bridge']();bridge.args=SimpleNamespace(token='test',allowed_client='127.0.0.1');bridge.busy=threading.Lock();bridge.tcp_results=OrderedDict()
    calls=[]
    bridge.collision=SimpleNamespace(validate=lambda points:calls.append('collision') or {})
    bridge.execute=lambda points:calls.append('execute') or (0.,0.,[],{})
    return bridge,calls

def test_motion_validates_whole_path_and_replay_does_not_execute_twice():
    bridge,calls=production_bridge();p=payload(command='execute')
    r=bridge._handle_tcp_request(p,('127.0.0.1',1))
    assert r['status']=='complete' and calls==['collision','execute']
    replay=bridge._handle_tcp_request(p,('127.0.0.1',1))
    assert replay['replayed'] and calls==['collision','execute'] and not bridge.busy.locked()
    p['points'][0]['positions'][0]+=.0001
    with pytest.raises(ValueError,match='reused'):bridge._handle_tcp_request(p,('127.0.0.1',1))
    assert not bridge.busy.locked()

def test_tcp_and_udp_share_busy_lock_and_authentication():
    bridge,calls=production_bridge();p=payload()
    p['token']='wrong'
    with pytest.raises(ValueError,match='authentication'):bridge._handle_tcp_request(p,('127.0.0.1',1))
    p['token']='test';bridge.busy.acquire()
    with pytest.raises(RuntimeError,match='busy'):bridge._handle_tcp_request(p,('127.0.0.1',1))
    bridge.busy.release();assert not calls

def test_repeated_validation_is_not_cached():
    bridge,calls=production_bridge();p=payload()
    for _ in range(2):assert bridge._handle_tcp_request(p,('127.0.0.1',1))['status']=='validated'
    assert calls==['collision','collision']

def test_tcp_does_not_bypass_limits_or_nan_timing():
    for field,value in [('time_from_start_sec',float('nan')),('positions',[0.,0.,1.,0.,0.,0.])]:
        bridge,calls=production_bridge();p=payload(command='execute');p['points'][1][field]=value
        with pytest.raises(ValueError):bridge._handle_tcp_request(p,('127.0.0.1',1))
        assert not calls
