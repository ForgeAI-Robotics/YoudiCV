"""Practical inferred-position policy only; does not authorize motion."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from piper_handeye.button_pose import ButtonDetection
from tools.offline_numeric_panel_layout_v3 import infer_image,sparse_candidate
from tools.replay_numeric_layout_layers import run_layers,recover_one_conflicting_anchor


@pytest.fixture
def scene():
    t={str(n):[0,1-n] for n in range(1,9)};t.update({str(n):[1,8-n] for n in range(9,17)})
    anchors=[dict(label=l,center=[180+45*p[0],260+35*p[1]],diameter=22,confidence=.95) for l,p in t.items()]
    return t,anchors,np.full((360,640,3),180,np.uint8)


def subset(anchors,labels):return [a for a in anchors if a['label'] in labels]


def test_strong_geometry_not_blocked_by_incomplete_rim(scene):
    t,a,im=scene;r=infer_image(t,a,'2',im)
    assert r['accepted'] and not r['rim_verification']['accepted']
    assert r['target_detected'] is False
    assert r['light_state']=='not_evaluated'
    assert r['requires_live_depth_and_motion_validation']


def test_four_references_accepted_without_mandatory_circle(scene):
    t,a,im=scene;r=infer_image(t,subset(a,['2','3','9','10']),'1',im)
    assert r['accepted']
    assert np.linalg.norm(np.array(r['center'])-[180,260])<.001


def test_one_three_and_nine_can_interpolate_two(scene):
    t,a,im=scene;r=infer_image(t,subset(a,['1','3','9']),'2',im)
    assert r['accepted']
    assert r['method']=='three_anchor_bracket_and_row_partner'
    assert np.linalg.norm(np.array(r['center'])-[180,225])<.001


def test_three_references_need_same_row_partner(scene):
    t,a,im=scene;r=infer_image(t,subset(a,['1','3','10']),'2',im)
    assert not r['accepted']


def test_three_reference_extrapolation_rejected(scene):
    t,a,im=scene;r=infer_image(t,subset(a,['2','3','9']),'1',im)
    assert not r['accepted']


def test_three_reference_weak_label_rejected(scene):
    t,a,im=scene;a=copy.deepcopy(subset(a,['1','3','9']));a[0]['confidence']=.75
    assert not infer_image(t,a,'2',im)['accepted']


def test_two_references_not_enough_for_standalone_candidate(scene):
    t,a,im=scene
    assert not infer_image(t,subset(a,['1','3']),'2',im)['accepted']


def test_wrong_sparse_label_does_not_become_target(scene):
    t,a,im=scene;a=copy.deepcopy(subset(a,['2','3','9','10']));next(x for x in a if x['label']=='3')['label']='8'
    assert not infer_image(t,a,'1',im)['accepted']


def test_duplicate_sparse_labels_rejected(scene):
    t,a,im=scene;a=subset(a,['1','3','9']);a.append(a[0])
    assert not sparse_candidate(t,a,'2',[640,360])['accepted']


def test_layer_audit_uses_no_extra_inference_on_good_raw(scene):
    t,a,im=scene
    class Detector:
        def detect(self,image):raise AssertionError('unexpected extra inference')
    raw=[ButtonDetection(0,'2',.95,(169,214,191,236))]
    r=run_layers(Detector(),im,raw,'2',t)
    assert r['stage']=='existing_raw_yolo' and r['extra_inferences']==0


def test_local_failure_can_reach_geometry_without_truth(scene):
    t,a,im=scene
    class Detector:
        def detect(self,image):return []
    aa=subset(a,['2','3','9','10']);raw=[ButtonDetection(0,x['label'],.95,(x['center'][0]-11,x['center'][1]-11,x['center'][0]+11,x['center'][1]+11)) for x in aa]
    r=run_layers(Detector(),im,raw,'1',t)
    assert r['stage']=='layout_candidate_only'
    assert r['extra_inferences']==1
    assert r['geometry']['target_detected'] is False


def test_four_strong_references_can_reject_one_wrong_reference(scene):
    t,a,im=scene;aa=copy.deepcopy(subset(a,['1','2','9','10']))
    aa.append(dict(label='5',center=[180,190],diameter=22,confidence=.8))
    original=copy.deepcopy(aa);r=recover_one_conflicting_anchor(t,aa,'3',im)
    assert r['accepted'] and r['target_detected'] is False
    assert r['excluded_conflicting_anchor']['label']=='5'
    assert np.linalg.norm(np.array(r['center'])-[180,190])<.001
    assert aa==original


def test_single_reference_recovery_does_not_relax_all_confidences(scene):
    t,a,im=scene;aa=copy.deepcopy(subset(a,['1','2','9','10']))
    aa[0]['confidence']=.71;aa.append(dict(label='5',center=[180,190],diameter=22,confidence=.8))
    assert not recover_one_conflicting_anchor(t,aa,'3',im)['accepted']


def test_not_remove_two_bad_references_to_make_solution(scene):
    t,a,im=scene;aa=copy.deepcopy(subset(a,['1','2','9','10']))
    aa.extend([dict(label='5',center=[180,190],diameter=22,confidence=.8),dict(label='6',center=[180,155],diameter=22,confidence=.8)])
    assert not recover_one_conflicting_anchor(t,aa,'3',im)['accepted']


def test_prior_raw_geometry_successes_preserved_in_replays():
    root=Path(__file__).resolve().parents[1];b=root/'tests/fixtures/numeric_panel_layout_offline/20261027_numeric_panel_layout_v3'
    for name,old in [('group_a','tests/fixtures/numeric_panel_layout_offline/20261027_numeric_panel_layout/results/evaluation.json'),('group_b','tests/fixtures/numeric_panel_layout_offline/20261027_numeric_panel_layout_v2/holdout_v1/evaluation.json'),('new_v3','tests/fixtures/numeric_panel_layout_offline/20261027_numeric_panel_layout_v3/new_v1/evaluation.json')]:
        before=json.loads((root/old).read_text())['results'];after=json.loads((b/name/'evaluation.json').read_text())['results']
        for prev in before:
            if prev['mode']!='raw_yolo_cpu_jpeg' or not prev['accepted']:continue
            r=next(x for x in after if (x['mode'],x['frame'],x['target'])==(prev['mode'],prev['frame'],prev['target']))
            assert r['accepted'] and r['within_provisional_2d_tolerance']


def test_frozen_policy_did_not_change_after_new_images():
    root=Path(__file__).resolve().parents[1];f=json.loads((root/'tests/fixtures/numeric_panel_layout_offline/20261027_numeric_panel_layout_v3/frozen_policy.json').read_text())
    assert hashlib.sha256((root/f['source']).read_bytes()).hexdigest()==f['source_sha256']
