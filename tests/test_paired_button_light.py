import time
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
import pytest
from piper_handeye.paired_button_light import (
    ButtonPatch, PairedButtonLight, compare_patches, extract_patch,
)
from piper_handeye.button_light_hardware import FreshLightObserver


def patch_image(sequence=1, increase=0, exposure=0, box=None):
    y,x=np.mgrid[:64,:64]
    gray=(35+(x%11)*2+(y%7)).astype(np.float32)
    gray[25:39,25:39] += increase
    gray=np.clip(gray+exposure,0,255)
    return ButtonPatch(gray, np.array(box or [30,30,94,94]), sequence,
                       time.time(), f'/tmp/test-{sequence}.png')


def snapshot(tmp_path, patch):
    image=np.zeros((128,128,3),np.uint8)
    x1,y1,x2,y2=patch.box.astype(int)
    image[y1:y2,x1:x2]=cv2.cvtColor(patch.gray.astype(np.uint8),cv2.COLOR_GRAY2BGR)
    path=tmp_path/f'frame_{patch.sequence}.png'
    cv2.imwrite(str(path),image)
    return dict(fresh=True,path=str(path),frame_sequence=patch.sequence,
                capture_time_unix_sec=patch.captured,image_size=[128,128],
                target_detection=dict(frame_sequence=patch.sequence,label='up',
                                      confidence=.95,bbox_xyxy=patch.box.tolist()))


def test_unchanged_white_mark_cannot_confirm_on():
    a=patch_image(increase=100)
    assert compare_patches(a,a,'ON')['state']=='UNCERTAIN'


def test_real_local_increase_and_unchanged_off():
    a=patch_image()
    assert compare_patches(a,patch_image(2,increase=80),'ON')['state']=='ON'
    assert compare_patches(a,patch_image(2),'OFF')['state']=='OFF'


@pytest.mark.parametrize('exposure',[10,40,-30])
def test_global_exposure_change_cannot_be_mistaken_for_lamp(exposure):
    assert compare_patches(patch_image(),patch_image(2,exposure=exposure),'ON')['state']=='UNCERTAIN'


def test_view_change_and_occlusion_are_unknown():
    a=patch_image()
    b=patch_image(2,increase=80,box=[70,30,134,94])
    assert compare_patches(a,b,'ON')['reason']=='view_geometry_changed'
    b=patch_image(2,increase=80)
    b.gray[:,:]=200
    assert compare_patches(a,b,'ON')['state']=='UNCERTAIN'


def test_saturation_is_unknown():
    a=patch_image(); b=patch_image(2)
    b.gray[8:56,8:56]=255
    assert compare_patches(a,b,'ON')['state']=='UNCERTAIN'


def test_small_translation_registered_on_background():
    a=patch_image(); b=patch_image(2,increase=80)
    b.gray=np.roll(b.gray,(1,-1),axis=(0,1))
    result=compare_patches(a,b,'ON')
    assert result['state']=='ON'
    assert result['alignment_px']==[-1,1]


def test_mismatched_frame_rejected_before_reading(tmp_path):
    data=snapshot(tmp_path,patch_image())
    data['target_detection']['frame_sequence']=0
    with pytest.raises(ValueError,match='mismatch'):extract_patch(data)


def test_reference_requires_two_new_off_and_is_frozen(tmp_path):
    pair=PairedButtonLight(tmp_path/'evidence');pair.begin_attempt()
    first=snapshot(tmp_path,patch_image())
    pair.collect(first,'OFF');pair.collect(first,'OFF')
    assert not pair.references
    second=snapshot(tmp_path,patch_image(2));pair.collect(second,'OFF')
    assert len(pair.references)==2
    pair.freeze()
    with pytest.raises(RuntimeError):pair.collect(second,'OFF')
    after=snapshot(tmp_path,patch_image(3,increase=80))
    result=pair.classify(after,'ON')
    assert result['state']=='ON'
    assert result['reference_crop'] and result['current_crop']
    pair.begin_attempt()
    assert not pair.references and not pair.frozen


def test_invalid_or_on_frame_breaks_reference_streak(tmp_path):
    pair=PairedButtonLight();pair.begin_attempt()
    pair.collect(snapshot(tmp_path,patch_image()),'OFF')
    pair.collect(snapshot(tmp_path,patch_image(2)),'ON')
    pair.collect(snapshot(tmp_path,patch_image(3)),'OFF')
    assert not pair.references
    pair.reset_pending()
    pair.collect(snapshot(tmp_path,patch_image(4)),'OFF')
    assert not pair.references


def test_no_reference_and_prepress_image_cannot_confirm_success(tmp_path):
    pair=PairedButtonLight();pair.freeze()
    assert pair.classify(snapshot(tmp_path,patch_image(4,increase=80)),'ON')['state']=='UNCERTAIN'
    a=patch_image(5); a.captured=pair.press_started-.01
    assert pair.classify(snapshot(tmp_path,a),'ON')['reason']=='image_predates_press'


def test_post_press_observer_cannot_bypass_pairing_with_absolute_on(tmp_path):
    data=snapshot(tmp_path,patch_image())
    vision=SimpleNamespace(request=lambda *a,**k:data)
    recognizer=SimpleNamespace(observe=lambda *a,**k: {'state':'ON'})
    pair=PairedButtonLight();pair.freeze()
    observer=FreshLightObserver(vision,recognizer,'up',paired_light=pair)
    assert observer.observe('near')=='UNCERTAIN'
    assert observer.audit[-1]['recognition']['absolute_state']=='ON'
    # Initial already-on observation remains absolute, before any press.
    data['frame_sequence']=2;data['target_detection']['frame_sequence']=2
    assert observer.observe('waiting')=='ON'


def test_runtime_freezes_reference_before_press_and_recollects_on_retry():
    import threading
    from piper_handeye.button_light_hardware import assemble_button_light_runtime
    collected=threading.Event();confirmed=threading.Event()
    class Observer:
        def __init__(self):
            self.paired_light=PairedButtonLight();self.audit=[];self.event_sink=None
            self.busy=False;self.last_sequence=0
        def observe(self,position,**kwargs):
            assert not self.busy
            self.busy=True
            try:
                assert not self.paired_light.frozen
                self.last_sequence+=1
                self.paired_light.references.append(patch_image(self.last_sequence))
                collected.set()
                time.sleep(.001)
                return 'OFF'
            finally:self.busy=False
        def observe_during_motion(self):
            assert not self.busy and self.paired_light.frozen
            self.last_sequence+=1
            self.audit.append({'frame_sequence':self.last_sequence,'state':'ON'})
            if len(self.audit)%2==0:confirmed.set()
            return 'ON'
        def wait_for_next_frame(self):pass
    class Executor:
        def set_before_guarded_press_callback(self,callback):self.hook=callback
        def approach_and_press(self):
            assert collected.wait(1)
            self.hook()
            assert observer.paired_light.frozen
            assert confirmed.wait(1)
        def retreat_to_observation(self,distance):raise AssertionError('early ON should avoid retreat')
        def return_to_waiting(self):pass
    observer=Observer();executor=Executor()
    runtime=assemble_button_light_runtime(observer,executor,lambda:None,lambda:None,
                                         waiting_position_is_origin=True)
    for attempt in (1,2):
        collected.clear();confirmed.clear()
        runtime.press_button(attempt)
        assert runtime.move_to_near_observation(.05)['confirmed_state']=='ON'
        assert observer.paired_light.attempt==attempt
    assert not any(t.name in ('pre-press-off-reference','post-press-light-monitor') for t in threading.enumerate())


def test_approach_exception_stops_reference_consumer():
    import threading
    from piper_handeye.button_light_hardware import assemble_button_light_runtime
    class Observer:
        paired_light=PairedButtonLight()
        def observe(self,*a,**k):time.sleep(.001)
        def wait_for_next_frame(self):pass
    class Executor:
        def set_before_guarded_press_callback(self,cb):pass
        def approach_and_press(self):raise RuntimeError('approach failed')
        def return_to_waiting(self):pass
    runtime=assemble_button_light_runtime(Observer(),Executor(),lambda:None,lambda:None,
                                         waiting_position_is_origin=True)
    with pytest.raises(RuntimeError,match='approach failed'):runtime.press_button(1)
    assert not any(t.name=='pre-press-off-reference' for t in threading.enumerate())


@pytest.mark.parametrize('lights_on',[True,False])
@pytest.mark.parametrize('stationary',[True,False])
@pytest.mark.parametrize('lighting', [(1., 0), (1., 24), (1.3, 12)])
@pytest.mark.parametrize('lamp_increase',[80,120])
def test_real_observer_comparison_and_flow_without_hardware(tmp_path,lights_on,stationary,lighting,lamp_increase):
    """Exercise real crop/classifier/comparator/temporal flow; only motion is fake."""
    import threading
    from piper_handeye.button_light_hardware import assemble_button_light_runtime
    from piper_handeye.button_light_flow import ButtonLightFlow
    from tools.opencv_light_worker import RelativeLuminance
    class Vision:
        def __init__(self):self.sequence=0;self.lit=False;self.changed=False
        def request(self,command,**kwargs):
            assert command=='light_snapshot'
            self.sequence+=1
            time.sleep(.002)
            image=patch_image(self.sequence,increase=lamp_increase if self.lit else 0)
            if self.changed:image.gray=np.clip(lighting[0]*image.gray+lighting[1],0,255)
            return snapshot(tmp_path,image)
    class Recognizer:
        def observe(self,path,target,bbox_xyxy=None):
            return RelativeLuminance().predict(path,target,bbox_xyxy)
    vision=Vision();observer=FreshLightObserver(vision,Recognizer(),'up',
            paired_light=PairedButtonLight(tmp_path/'evidence',require_stationary_reference=stationary))
    class Executor:
        presses=0
        def set_before_guarded_press_callback(self,callback):self.hook=callback
        def approach_and_press(self):
            if stationary:
                assert not observer.paired_light.references
                assert not any(t.name=='pre-press-off-reference' for t in threading.enumerate())
                observer.capture_stationary_off_reference()
            else:
                limit=time.monotonic()+1
                while len(observer.paired_light.references)<2 and time.monotonic()<limit:time.sleep(.002)
            self.hook()
            self.presses+=1;vision.lit=lights_on;vision.changed=True
            time.sleep(.03)
        def retreat_to_observation(self,distance):pass
        def return_to_waiting(self):vision.lit=False
    executor=Executor();returned=[]
    runtime=assemble_button_light_runtime(observer,executor,lambda:None,lambda:returned.append(True),
               waiting_position_is_origin=False,hold_at_prepress_if_visible=lambda:True)
    result=ButtonLightFlow(runtime).run()
    if lights_on:
        # Raising the local ON gate to85 deliberately leaves this dim80.6
        # synthetic lamp uncertain unless gain makes its contrast exceed85.
        # Keep the weak-light case: it must neither announce success nor retry.
        if lamp_increase==80 and lighting[0]==1.:
            assert result.status=='post_press_confirmation_uncertain'
        else:
            assert result.status=='arrived_after_verified_press'
        assert executor.presses==1
    else:
        assert result.status!='arrived_after_verified_press'
        assert executor.presses==3
    assert returned==[True]
    assert not any(t.name in ('pre-press-off-reference','post-press-light-monitor') for t in threading.enumerate())


def test_thin_lit_stroke_not_required_to_fill_whole_button():
    before=patch_image();after=patch_image(2)
    after.gray[18:46,31:33]+=90
    result=compare_patches(before,after,'ON')
    assert result['state']=='ON'
    assert .03 <= result['changed_fraction'] < .06


def test_upsampling_one_source_pixel_does_not_make_an_on_region():
    before=patch_image(box=[30,30,44,44]);after=patch_image(2,box=[30,30,44,44])
    after.gray[18:46,31:33]+=90
    result=compare_patches(before,after,'ON')
    assert result['estimated_changed_source_pixels'] < 3
    assert result['state']=='UNCERTAIN'
