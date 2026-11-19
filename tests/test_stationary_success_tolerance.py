"""Failure-driven stationary retries and shared deadlines; no hardware."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from piper_handeye.button_light_hardware import FreshLightObserver
from piper_handeye.paired_button_light import compare_patches, _RING
from piper_handeye.numeric_layout_handoff import LayoutTaskPolicy, selected_button_region
from tests.test_stationary_retry_rgb_freshness import setup_guard, warm, wire
from tests.test_numeric_layout_handoff import evidence, decorate, SETTINGS
from tests.test_reference_wait_1s_total6s import scheduled_observer
from tests.test_paired_button_light import patch_image
from tests.test_up_safety_handoff import setup as safety_setup


@pytest.mark.parametrize('age,allowed', [(.6, True), (.69, True), (.71, False)])
def test_raw_lamp_admission_requires_continuous_hold(age, allowed):
    g,c,_,_=setup_guard();warm(g,c,end=.9)
    s=wire(c,captured=c.t-age,retry=False)
    a=g.admission(s,'before_press')
    assert (a is not None) is allowed
    if allowed:assert a['max_rgb_age_sec']==.7


def test_layout_candidate_is_lamp_only_depth_stays_350ms(monkeypatch):
    g,c,_,_=setup_guard();warm(g,c,end=.9)
    e=evidence(capture=c.time()-.6)
    monkeypatch.setattr('piper_handeye.numeric_layout_handoff.time.time',c.time)
    s=decorate(e)
    assert 'layout_target_region' not in s and 'stationary_light_layout_candidate' in s
    assert selected_button_region(s) is None
    policy=LayoutTaskPolicy(SETTINGS,'2')
    assert policy.sample(e,now=c.time()) is None
    assert policy.evidence(e,now=c.time(),depth=True,max_frame_age_sec=.7) is None
    a=g.admission(s,'before_press')
    assert a['region_source']=='panel_layout_inferred' and a['max_rgb_age_sec']==.7
    assert selected_button_region(s,max_frame_age_sec=.7)['confidence'] is None
    s['stationary_light_layout_candidate']['bbox_xyxy'][0]+=1
    assert g.admission(s,'before_press') is None


def test_late_reference_and_longer_single_wait(tmp_path,monkeypatch):
    o,p,c=scheduled_observer(tmp_path,monkeypatch,[7.2,8.4])
    result=o.capture_stationary_off_reference()
    assert result['sequences']==[1,2] and p.stationary_reference_ready
    assert 8.4<=c.elapsed<8.5


def test_reference_cannot_outlive_shared_deadline(tmp_path,monkeypatch):
    o,p,c=scheduled_observer(tmp_path,monkeypatch,[])
    o.stationary_budget_remaining=lambda:2.-c.elapsed
    with pytest.raises(RuntimeError):o.capture_stationary_off_reference()
    assert 2.<=c.elapsed<2.04 and not p.references


def test_pose_can_settle_after_previous_half_second_budget():
    from piper_handeye.observation_pose import wait_for_stable_observation_pose
    from tests.test_observation_pose import Clock
    c=Clock()
    def read():return NS(joints=[.009 if c()<10.65 else .003]*6,timestamp=c(),raw={'hardware_ok':1})
    _,a=wait_for_stable_observation_pose([0.]*6,read,lambda:None,clock=c,sleep=c.sleep)
    assert .73<=a['elapsed_sec']<.85 and a['timeout_sec']==1.


def test_postpress_recheck_waits_past_old_budget(monkeypatch):
    import piper_handeye.button_light_hardware as hardware
    _,c,_,_=setup_guard();monkeypatch.setattr(hardware,'time',c)
    pair=NS(frozen=True,require_stationary_reference=True,stationary_reference_ready=True,stationary_check=lambda:True)
    o=FreshLightObserver(None,None,'2',paired_light=pair)
    def observe(position,timeout_sec,**kwargs):
        c.sleep(timeout_sec)
        if c.t<1.4:raise TimeoutError()
        o.audit.append(dict(position=position,state='ON',frame_sequence=4,
            capture_time_unix_sec=c.time(),image='simulated',recognition=dict(
                absolute_state='ON',paired_comparison=dict(state='ON'))))
        return 'ON'
    o.observe=observe
    result=o.reobserve_post_press('near')
    assert result['confirmed_state']=='ON' and 1.4<=c.t<=2.5


def handoff(monkeypatch,delay=.3):
    import piper_handeye.button_light_hardware as hardware
    s,v,m,g,metrics,calls,state=safety_setup(monkeypatch)
    class Clock:
        t=100.
        def monotonic(self):return self.t
        def sleep(self,dt):self.t+=dt
    c=Clock();monkeypatch.setattr(hardware,'time',c);s.near_pause.started=98.
    reads=[]
    def read(**kw):
        reads.append(c.t)
        return NS(base_to_gripper=np.eye(4),timestamp=c.t,raw={'hardware_ok':1})
    s.robot.state=read;s.vision=NS(request=lambda *a,**kw:(c.sleep(delay) or {'ok':True}))
    return s,c,reads,g


def test_lease_reads_after_rpc_without_renewal(monkeypatch):
    s,c,reads,g=handoff(monkeypatch)
    assert s._fresh_reference_lease()>c.t
    assert reads==[100.,100.3] and s.near_pause.started==98.
    assert s.depth_degraded_commit_events[-1]['encoder_age_sec']==0.


def test_publisher_also_uses_post_rpc_feedback(monkeypatch):
    s,c,reads,g=handoff(monkeypatch)
    g.armed.set()
    s.closed.wait=lambda _timeout:s.closed.set()
    s._publish_pose()
    assert s.publisher_error is None
    assert s.near_pause.active and s.near_pause.started==98.
    assert reads==[100.,100.3]


def test_handoff_cannot_accept_after_its_one_second_recheck_budget(monkeypatch):
    s,c,reads,g=handoff(monkeypatch,1.1)
    with pytest.raises(RuntimeError,match='bounded recheck'):s._fresh_reference_lease()
    assert s.near_pause.started==98.


def test_transient_lease_rechecked_at_same_stationary_pose(monkeypatch):
    s,c,reads,g=handoff(monkeypatch,.01);real=s.near_pause.update;n=[0]
    def update(*a,**kw):
        n[0]+=1
        return 0. if n[0]<3 else real(*a,**kw)
    s.near_pause.update=update
    assert s._fresh_reference_lease()>c.t
    assert n[0]==3 and c.t<101 and s.near_pause.started==98.


@pytest.mark.parametrize('fault',['expired','stopped','hardware','feedback','far'])
def test_handoff_cannot_bypass_failures(monkeypatch,fault):
    s,c,reads,g=handoff(monkeypatch)
    if fault=='expired':s.near_pause.started=75.1
    if fault=='stopped':g.stop_active.set()
    if fault=='hardware':s.robot.state=lambda **kw:NS(base_to_gripper=np.eye(4),timestamp=c.t,raw={'hardware_ok':0})
    if fault=='feedback':s.robot.state=lambda **kw:NS(base_to_gripper=np.eye(4),timestamp=c.t-.251,raw={'hardware_ok':1})
    if fault=='far':s.target[0,3]=.3
    with pytest.raises(RuntimeError):s._fresh_reference_lease()
    assert not any(x.get('reason')=='stationary_reference_fresh_encoder_lease' for x in s.depth_degraded_commit_events)


def noisy_background():
    a,b=patch_image(),patch_image(2);rng=np.random.default_rng(1)
    a.gray[:]=rng.uniform(60,180,(64,64));b.gray[:]=a.gray
    b.gray[_RING]+=rng.normal(0,13,np.count_nonzero(_RING))
    return a,b


def test_mean15_is_explicit_off_only_and_other_limits_unchanged():
    a,b=noisy_background()
    old=compare_patches(a,b,'OFF',stationary=lambda:True)
    new=compare_patches(a,b,'OFF',stationary=lambda:True,allow_off_background_mean15=True)
    assert old['state']=='UNCERTAIN' and 10<old['background_error']<=15
    assert new['state']=='OFF'
    assert new['background_max_mean_error']==15 and new['background_max_p90_error']==25
    assert new['background_max_spatial_error']==12 and new['background_inlier_max_error']==15
    for stationary in [False,True,lambda:False]:
        assert compare_patches(a,b,'OFF',stationary=stationary,allow_off_background_mean15=True)['state']=='UNCERTAIN'


def test_mean15_cannot_hide_lamp_rise_or_relax_on():
    a,b=noisy_background();b.gray[25:39,25:39]+=100
    assert compare_patches(a,b,'OFF',stationary=lambda:True,allow_off_background_mean15=True)['state']=='UNCERTAIN'
    r=compare_patches(a,b,'ON',stationary=lambda:True,allow_off_background_mean15=True)
    assert r.get('background_tolerance_profile')!='numeric_stationary_aligned_off_mean15_v1'


def test_real_collector_uses_mean15_only_after_aligning_numeric_off(tmp_path):
    from piper_handeye.paired_button_light import PairedButtonLight
    from tests.test_paired_button_light import snapshot
    a,b=noisy_background()
    p=PairedButtonLight(require_stationary_reference=True,stationary_check=lambda:True)
    p.begin_attempt()
    for patch in [a,b]:
        image=snapshot(tmp_path,patch);image['target_detection']['label']='2'
        p.collect(image,'OFF',stationary=True)
    assert len(p.references)==2
    result=p.last_reference_comparison
    assert result['state']=='OFF'
    assert result['comparison_method']=='common_pixel_crop'
    assert result['background_max_mean_error']==15
    assert result['off_reference_strict_comparison']['background_max_mean_error']==10
