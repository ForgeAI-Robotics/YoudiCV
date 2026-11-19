"""Saved images / synthetic shapes only. Never connect to camera or robot."""
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.offline_numeric_panel_layout_v2 import candidate,infer_image,verify_rim


@pytest.fixture
def scene():
    template={str(n):[0,1-n] for n in range(1,9)}
    template.update({str(n):[1,8-n] for n in range(9,17)})
    aa=[dict(label=l,center=[180+45*template[l][0],260+35*template[l][1]],diameter=22) for l in ['2','3','9','10']]
    image=np.full((360,640,3),180,np.uint8)
    for a in aa+[dict(center=[180,260])]:cv2.circle(image,tuple(a['center']),11,(15,15,15),-1)
    return template,aa,image


def test_four_anchors_require_independent_rim(scene):
    t,a,im=scene;r=infer_image(t,a,'1',im)
    assert r['accepted'] and r['four_anchor_mode']
    assert r['rim_verification']['accepted']
    assert np.linalg.norm(np.array(r['center'])-[180,260])<1
    assert r['purpose']=='offline_2d_research_only'


def test_blank_image_cannot_pass_exact_geometric_fit(scene):
    t,a,im=scene;im[:]=180
    r=infer_image(t,a,'1',im)
    assert not r['accepted'] and r['reason']=='no_complete_button_rim'


def test_missing_target_only_does_not_use_neighbor_circle(scene):
    t,a,im=scene;im[245:277,165:197]=180
    assert not infer_image(t,a,'1',im)['accepted']


@pytest.mark.parametrize('shift',[15,25,50])
def test_wrong_circle_location_rejected(scene,shift):
    t,a,im=scene;im[:]=180;cv2.circle(im,(180+shift,260),11,(15,15,15),-1)
    assert not infer_image(t,a,'1',im)['accepted']


def test_half_occluded_circle_not_complete_rim(scene):
    t,a,im=scene;im[240:280,180:205]=180
    assert not infer_image(t,a,'1',im)['accepted']


def test_single_wrong_label_fails_size_geometry(scene):
    t,a,im=scene;a=copy.deepcopy(a);a[1]['label']='5'
    r=infer_image(t,a,'1',im)
    assert not r['accepted']
    assert r['reason'] in ('anchor_size_geometry_conflict','model_disagreement','inverted_panel','target_scale_unstable')


def test_duplicate_label_rejected(scene):
    t,a,im=scene
    assert candidate(t,a+[a[1]],'1',[640,360])['reason']=='duplicate_anchor_label'


def test_three_anchors_still_rejected(scene):
    t,a,im=scene
    assert candidate(t,a[:3],'1',[640,360])['reason']=='fewer_than_four_anchors'


def test_target_never_used_in_geometry(scene):
    t,a,im=scene
    a.append(dict(label='1',center=[999,999],diameter=900))
    assert infer_image(t,a,'1',im)['accepted']


@pytest.mark.parametrize('center,diameter', [([float('nan'),20],22),([20,20],float('nan')),([20,20],0)])
def test_invalid_rim_inputs_rejected(center,diameter):
    assert not verify_rim(np.zeros((100,100,3),np.uint8),center,diameter)['accepted']


def test_frozen_source_unchanged_before_holdout():
    root=Path(__file__).resolve().parents[1]
    b=root/'tests/fixtures/numeric_panel_layout_offline/20261027_numeric_panel_layout_v2'
    frozen=json.loads((b/'frozen_policy.json').read_text())
    assert hashlib.sha256((root/frozen['source']).read_bytes()).hexdigest()==frozen['source_sha256']


def test_development_examples_not_misreported_as_holdout():
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'tests/fixtures/numeric_panel_layout_offline/20261027_numeric_panel_layout/manifest.json').read_text())
    new=json.loads((root/'tests/fixtures/numeric_panel_layout_offline/20261027_numeric_panel_layout_v2/holdout_manifest.json').read_text())
    oldtasks={f['task'] for f in old['frames']}
    assert not oldtasks & {f['task'] for f in new['frames'] if f['split']=='validation'}
