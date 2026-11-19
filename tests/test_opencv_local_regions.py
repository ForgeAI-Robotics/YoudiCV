import cv2
import numpy as np
from tools.opencv_light_worker import RelativeLuminance


def image(tmp_path,stroke_value=225,speckle=False):
    g=np.full((100,100),35,dtype=np.uint8)
    if speckle:g[50,50]=255
    else:g[29:70,48:52]=stroke_value
    path=tmp_path/'button.png';cv2.imwrite(str(path),g);return path


def test_thin_lit_stroke_is_not_diluted_by_black_background(tmp_path):
    p=image(tmp_path)
    old=RelativeLuminance(algorithm='legacy').predict(p,'1',[0,0,100,100])
    new=RelativeLuminance().predict(p,'1',[0,0,100,100])
    assert old['state']=='OFF';assert new['state']=='ON'
    assert new['center_bright_fraction']<.25


def test_dim_print_and_single_hot_pixel_do_not_trigger_on(tmp_path):
    assert RelativeLuminance().predict(image(tmp_path,75),'1',[0,0,100,100])['state']=='OFF'
    assert RelativeLuminance().predict(image(tmp_path,speckle=True),'1',[0,0,100,100])['state']=='OFF'


def test_missing_and_small_target_never_trigger_on(tmp_path):
    p=image(tmp_path)
    assert RelativeLuminance().predict(p,'1')['state']=='UNCERTAIN'
    assert RelativeLuminance().predict(p,'1',[0,0,5,5])['state']=='UNCERTAIN'
