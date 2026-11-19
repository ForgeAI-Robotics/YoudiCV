"""Delayed second-YOLO lamp frames require measured stationary coverage."""
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

from piper_handeye.stationary_light_freshness import StationaryRetryLightGuard
from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.paired_button_light import PairedButtonLight
from tests.test_paired_button_light import patch_image, snapshot


class Clock:
    def __init__(self):
        self.t = 0.; self.epoch = time.time()
    def time(self):return self.epoch+self.t
    def monotonic(self):return self.t
    def sleep(self, seconds):self.t += seconds


def setup_guard(target='2'):
    clock = Clock();q = np.zeros(6);flags = {'hardware_ok': 1}
    def read():return NS(joints=q.copy(), raw=flags.copy(), timestamp=clock.t)
    guard = StationaryRetryLightGuard(read, lambda: None, target, clock=clock)
    return guard,clock,q,flags


def wire(clock, seq=1, *, captured=.1, retry=True, target='2'):
    return dict(fresh=True,frame_sequence=seq,capture_time_unix_sec=clock.epoch+captured,
                target_detection=dict(label=target,confidence=.95,frame_sequence=seq),
                perception_timing={'spans':[dict(name='vision.detection',status='ok',
                    retry_inference_sec=.18 if retry else 0.,numeric_retry_skipped_by_layout=False)]})


def warm(guard,clock,position='before_press',end=.7):
    while clock.t < end:
        guard.poll(position);clock.sleep(.01)
    guard.poll(position)


def test_real_slow_local_retry_admitted_without_changing_capture_time():
    g,c,_,_=setup_guard();s=wire(c);stamp=s['capture_time_unix_sec']
    warm(g,c);a=g.admission(s,'before_press')
    assert a['max_rgb_age_sec']==1. and .59<a['image_age_sec']<.62
    assert a['encoder_samples']>20 and a['max_joint_spread_rad']==0
    assert s['capture_time_unix_sec']==stamp


@pytest.mark.parametrize('mode',['old','before_entry','wrong_target','mixed_frame',
                                 'hardware','moving','feedback_gap','no_feedback',
                                 'phase_change','regressed_clock','duplicate_encoder'])
def test_no_slow_admission_without_same_frame_continuous_hold(mode):
    g,c,q,flags=setup_guard();s=wire(c);warm(g,c)
    if mode=='old':s['capture_time_unix_sec']=c.time()-1.01
    if mode=='before_entry':s['capture_time_unix_sec']=c.epoch-.01
    if mode=='wrong_target':s['target_detection']['label']='3'
    if mode=='mixed_frame':s['target_detection']['frame_sequence']=99
    if mode=='hardware':flags['hardware_ok']=0
    if mode=='moving':q[2]=.0031
    if mode=='feedback_gap':c.sleep(.251)
    if mode=='no_feedback':g.read_state=lambda: (_ for _ in ()).throw(TimeoutError())
    if mode=='phase_change':g.poll('near')
    if mode=='regressed_clock':g.read_state=lambda:NS(joints=q.copy(),raw=flags,timestamp=c.t-.1)
    if mode=='duplicate_encoder':
        g.reset();g.read_state=lambda:NS(joints=q.copy(),raw=flags,timestamp=c.t)
        g.poll('before_press')
    assert g.admission(s,'before_press') is None


@pytest.mark.parametrize('phase',['press_monitor','waiting_retreat','approach','near_moving'])
def test_moving_phases_cannot_use_one_second_allowance(phase):
    g,c,_,_=setup_guard();warm(g,c)
    assert g.admission(wire(c),phase) is None
    g.poll(phase);assert g.first is None


@pytest.mark.parametrize('target',['open','close','unknown'])
def test_unconfigured_other_targets_unchanged(target):
    g,c,_,_=setup_guard(target);warm(g,c)
    assert g.admission(wire(c,target=target),'before_press') is None


@pytest.mark.parametrize('target',['up','down'])
@pytest.mark.parametrize('retry,age,allowed',[(False,.6,True),(False,.71,False),
                                           (True,.9,True),(True,1.01,False)])
def test_call_button_static_rgb_requires_the_same_encoder_proof(target,retry,age,allowed):
    g,c,_,_=setup_guard(target);warm(g,c,end=1.2)
    s=wire(c,captured=c.t-age,retry=retry,target=target)
    admission=g.admission(s,'before_press')
    assert (admission is not None) is allowed
    if allowed:
        assert admission['policy'].startswith('stationary_call_')
        assert admission['max_rgb_age_sec']==(1. if retry else .7)


@pytest.mark.parametrize('target',['up','down'])
@pytest.mark.parametrize('fault',['moving','gap','hardware','wrong_frame','wrong_label','motion_phase','stop'])
def test_call_button_rgb_allowance_does_not_bypass_existing_guards(target,fault):
    g,c,q,flags=setup_guard(target);warm(g,c,end=.9);s=wire(c,captured=.2,target=target)
    if fault=='moving':q[2]=.0031
    if fault=='gap':c.sleep(.251)
    if fault=='hardware':flags['hardware_ok']=0
    if fault=='wrong_frame':s['target_detection']['frame_sequence']=99
    if fault=='wrong_label':s['target_detection']['label']='down' if target=='up' else 'up'
    if fault=='stop':
        g.check_safety=lambda:(_ for _ in ()).throw(RuntimeError('new stop'))
        with pytest.raises(RuntimeError,match='new stop'):g.admission(s,'before_press')
        return
    assert g.admission(s,'press_monitor' if fault=='motion_phase' else 'before_press') is None


@pytest.mark.parametrize('target',['up','down'])
def test_call_off_reference_collector_accepts_900ms_local_retry_with_full_hold(tmp_path,monkeypatch,target):
    import piper_handeye.button_light_hardware as hardware
    import piper_handeye.paired_button_light as paired_module
    g,c,_,_=setup_guard(target)
    monkeypatch.setattr(hardware,'time',c);monkeypatch.setattr(paired_module,'time',c)
    class Vision:
        sequence=0
        def request(self,*a,**kw):
            c.sleep(.001)
            if self.sequence>=2 or c.t<[1.0,1.4][self.sequence]:return {'fresh':False}
            self.sequence+=1;p=patch_image(self.sequence)
            p.captured=c.epoch+[.1,.5][self.sequence-1]
            s=snapshot(tmp_path,p);s['target_detection']['label']=target
            s['perception_timing']=wire(c,target=target)['perception_timing']
            return s
    paired=PairedButtonLight(require_stationary_reference=True,stationary_check=lambda:True)
    paired.begin_attempt()
    observer=FreshLightObserver(Vision(),NS(observe=lambda *a,**kw:{'state':'OFF'}),target,
        paired_light=paired,stationary_retry_light_guard=g)
    result=observer.capture_stationary_off_reference()
    assert result['sequences']==[1,2] and paired.stationary_reference_ready
    assert all(r['visibility']['rgb_age_limit_sec']==1. for r in observer.audit)
    assert all(.89<=r['visibility']['stationary_retry_rgb_admission']['image_age_sec']<=.93
               for r in observer.audit)


def test_stop_is_not_converted_to_permission():
    g,c,_,_=setup_guard();warm(g,c)
    g.check_safety=lambda: (_ for _ in ()).throw(RuntimeError('new stop'))
    with pytest.raises(RuntimeError,match='new stop'):
        g.admission(wire(c),'before_press')


@pytest.mark.parametrize('target,source',[
    ('2','retry'),('2','raw'),('2','layout'),
    ('up','retry'),('up','raw'),('down','retry'),('down','raw')])
def test_actual_reference_collector_accepts_slow_supported_frames(tmp_path,monkeypatch,target,source):
    import piper_handeye.button_light_hardware as hardware
    import piper_handeye.paired_button_light as paired_module
    g,c,_,_=setup_guard(target);monkeypatch.setattr(hardware,'time',c);monkeypatch.setattr(paired_module,'time',c)
    class Vision:
        seq=0
        def request(self,*a,**kw):
            c.sleep(.001)
            if self.seq>=2 or c.t < [.7,1.05][self.seq]:return {'fresh':False}
            self.seq+=1
            p=patch_image(self.seq);p.captured=c.epoch+[.1,.45][self.seq-1]
            s=snapshot(tmp_path,p);s.update(perception_timing=wire(c)['perception_timing'])
            s['target_detection']['label']=target
            if source=='raw':s['perception_timing']=wire(c,retry=False)['perception_timing']
            if source=='layout':
                from tests.test_numeric_layout_handoff import decorate,evidence
                monkeypatch.setattr('piper_handeye.numeric_layout_handoff.time.time',c.time)
                out=decorate(evidence(self.seq,capture=p.captured,box=p.box.tolist(),image_size=[128,128]))
                out['path']=s['path'];return out
            return s
    paired=PairedButtonLight(require_stationary_reference=True,stationary_check=lambda:True);paired.begin_attempt()
    o=FreshLightObserver(Vision(),NS(observe=lambda *a,**kw:{'state':'OFF'}),target,
                         paired_light=paired,stationary_retry_light_guard=g)
    result=o.capture_stationary_off_reference()
    assert result['sequences']==[1,2] and paired.stationary_reference_ready
    assert all(x['visibility']['rgb_age_limit_sec']==(1. if source=='retry' else .7) for x in o.audit)
    assert all(x['visibility']['stationary_retry_rgb_admission'] for x in o.audit)


def test_motion_during_recognition_invalidates_slow_result(tmp_path,monkeypatch):
    import piper_handeye.button_light_hardware as hardware
    g,c,q,_=setup_guard();warm(g,c,position='near');monkeypatch.setattr(hardware,'time',c)
    p=patch_image(1);p.captured=c.epoch+.1;s=snapshot(tmp_path,p)
    s.update(perception_timing=wire(c)['perception_timing']);s['target_detection']['label']='2'
    def recognition(*a,**kw):q[1]=.004;return {'state':'ON'}
    o=FreshLightObserver(NS(request=lambda *a,**kw:s),NS(observe=recognition),'2',stationary_retry_light_guard=g)
    assert o.observe('near')=='UNCERTAIN'
    assert o.audit[-1]['recognition']['freshness_rejection']=='stationary_retry_rgb_no_longer_current'


@pytest.mark.parametrize('target',['2','up','down'])
def test_moving_monitor_retains_old_350ms_rejection(tmp_path,monkeypatch,target):
    import piper_handeye.button_light_hardware as hardware
    g,c,_,_=setup_guard(target);warm(g,c,position='near');monkeypatch.setattr(hardware,'time',c)
    calls=[]
    class Vision:
        sequence=0
        def request(self,*a,**kw):
            c.sleep(.01);self.sequence+=1
            p=patch_image(self.sequence);p.captured=c.time()-.6
            s=snapshot(tmp_path,p);s['target_detection']['label']=target
            s.update(perception_timing=wire(c)['perception_timing']);return s
    o=FreshLightObserver(Vision(),NS(observe=lambda *a,**kw:calls.append(1)), target,stationary_retry_light_guard=g)
    assert o.observe('press_monitor',timeout_sec=.1)=='UNCERTAIN'
    assert not calls and o.audit[-1]['recognition']['visibility']['rgb_age_limit_sec']==.35


@pytest.mark.parametrize('retry',[True,False])
def test_slow_post_press_image_uses_actual_pair_and_latest_only_fallback(tmp_path,monkeypatch,retry):
    import piper_handeye.button_light_hardware as hardware
    import piper_handeye.paired_button_light as paired_module
    g,c,_,_=setup_guard();monkeypatch.setattr(hardware,'time',c);monkeypatch.setattr(paired_module,'time',c)
    paired=PairedButtonLight(require_stationary_reference=True,stationary_check=lambda:True)
    paired.begin_attempt()
    for seq in (1,2):
        p=patch_image(seq);p.captured=c.epoch-.3+seq*.1
        s=snapshot(tmp_path,p);s['target_detection']['label']='2';paired.collect(s,'OFF')
    assert len(paired.references)==2
    paired.stationary_reference_ready=True;paired.freeze()
    warm(g,c,position='near',end=.8)
    p=patch_image(3,increase=100);p.captured=c.epoch+.2
    s=snapshot(tmp_path,p);s['target_detection']['label']='2'
    s.update(perception_timing=wire(c,retry=retry)['perception_timing'])
    recog=NS(observe=lambda *a,**kw:{'state':'ON','legacy_state':'ON',
             'local_on_contrast_threshold':100.,
             'local_regions':[dict(contrast=120.,fraction=.10,area_px=20)]})
    o=FreshLightObserver(NS(request=lambda *a,**kw:s),recog,'2',paired_light=paired,
                        stationary_retry_light_guard=g)
    assert o.observe('near')=='ON'
    entry=o.audit[-1];assert entry['recognition']['paired_comparison']['state']=='ON'
    # Exercise the existing fallback's current-image clock check separately.
    entry['recognition']['recognizer']='opencv_self_contrast'
    entry['recognition']['paired_comparison']=dict(state='UNCERTAIN',
        reason='nonuniform_background_change',reference_count=2)
    assert o.post_press_opencv_result('near')['state']=='ON'
    c.sleep(.41)
    assert o.post_press_opencv_result('near') is None


def test_late_expiry_cannot_leave_a_new_off_reference(tmp_path,monkeypatch):
    import piper_handeye.button_light_hardware as hardware
    import piper_handeye.paired_button_light as paired_module
    g,c,_,_=setup_guard();warm(g,c);monkeypatch.setattr(hardware,'time',c);monkeypatch.setattr(paired_module,'time',c)
    paired=PairedButtonLight(require_stationary_reference=True);paired.begin_attempt()
    previous=patch_image(1);previous.captured=c.epoch+.05
    prev=snapshot(tmp_path,previous);prev['target_detection']['label']='2';paired.collect(prev,'OFF')
    current=patch_image(2);current.captured=c.epoch+.1
    s=snapshot(tmp_path,current);s['target_detection']['label']='2'
    s.update(perception_timing=wire(c)['perception_timing'])
    collect=paired.collect
    def slow(*a,**kw):
        collect(*a,**kw);c.sleep(.5)
    paired.collect=slow
    o=FreshLightObserver(NS(request=lambda *a,**kw:s),NS(observe=lambda *a,**kw:{'state':'OFF'}),
        '2',paired_light=paired,stationary_retry_light_guard=g)
    assert o.observe('before_press')=='UNCERTAIN'
    assert not paired.stationary_reference_ready
    assert [p.sequence for p in paired.references]==[1]
    assert paired.pending is None
