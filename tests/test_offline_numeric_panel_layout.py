"""No-motion checks for isolated panel-layout research, not production policy."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.offline_numeric_panel_layout import evaluate, hide_target, infer, project, yolo_anchors


@pytest.fixture
def scene():
    template = {str(n): [0, 1-n] for n in range(1, 9)}
    template.update({str(n): [1, 8-n] for n in range(9, 17)})
    h = np.array([[70,0,180],[0,30,260],[0,0,1]],float)
    anchors = [dict(label=l, center=project(h,[p])[0].tolist(),diameter=22) for l,p in template.items()]
    return template, anchors


@pytest.mark.parametrize('target',list('12345'))
def test_hide_and_predict_without_target(scene,target):
    template, anchors = scene
    result = infer(template,anchors,target,[640,360])
    assert result['accepted']
    assert target not in result['anchors']
    expected = next(a['center'] for a in anchors if a['label']==target)
    assert np.linalg.norm(np.array(result['center'])-expected) < 1e-5
    assert result['purpose']=='offline_2d_research_only'


def test_wrong_anchor_not_silently_discarded(scene):
    t,a=scene; a=copy.deepcopy(a); a[2]['center'][1] += 15
    result=infer(t,a,'1',[640,360])
    assert not result['accepted']
    assert result['reason']=='anchor_layout_conflict'


def test_duplicate_label_rejected(scene):
    t,a=scene
    assert infer(t,a+[a[1]],'1',[640,360])['reason']=='duplicate_anchor_label'


def test_same_physical_button_two_labels_rejected(scene):
    t,a=scene; a=copy.deepcopy(a); a[2]['center']=a[1]['center']
    assert infer(t,a,'1',[640,360])['reason']=='overlapping_anchor_labels'


def test_hide_removes_wrong_label_on_target_box(scene):
    _,a=scene; truth={v['label']:v for v in a}
    wrong=copy.deepcopy(a[2]); wrong['label']='99'
    hidden=hide_target(a+[wrong],truth,'3')
    assert not any(v['label'] in ('3','99') for v in hidden)


def test_hide_does_not_clean_other_wrong_labels(scene):
    _,a=scene; truth={v['label']:v for v in a}
    wrong=copy.deepcopy(a[2]); wrong['label']='99'
    assert any(v['label']=='99' for v in hide_target(a+[wrong],truth,'1'))


def test_four_points_not_called_robust(scene):
    t,a=scene
    assert infer(t,a[1:3]+a[8:10],'1',[640,360])['reason']=='fewer_than_five_anchors'


def test_single_column_rejected(scene):
    t,a=scene
    assert infer(t,a[:8],'1',[640,360])['reason']=='insufficient_two_column_row_coverage'


def test_offscreen_prediction_rejected(scene):
    t,a=scene
    assert infer(t,a,'1',[190,360])['reason']=='predicted_target_not_fully_visible'


def test_bad_geometry_rejected(scene):
    t,a=scene; a=copy.deepcopy(a); a[2]['center']=[float('nan'),20]
    assert infer(t,a,'1',[640,360])['reason']=='invalid_anchor_geometry'


def test_unknown_template_target_rejected(scene):
    t,a=scene
    assert not infer(t,a,'up',[640,360])['accepted']


def test_yolo_filter_needs_confidence_and_full_box(scene):
    t,_=scene
    good=dict(label='4',confidence=.9,bbox_xyxy=[100,100,125,125])
    ds=[good,dict(good,label='5',confidence=.69),dict(good,label='6',bbox_xyxy=[0,100,25,125]),dict(good,label='open')]
    assert [v['label'] for v in yolo_anchors(ds,t,[640,360])]==['4']


def test_split_leakage_rejected(tmp_path):
    manifest=dict(template=dict(coordinates={}),frames=[dict(task='same',split='reference'),dict(task='same',split='validation')])
    with pytest.raises(ValueError,match='task_leakage'):
        evaluate(manifest,tmp_path)


def test_preserved_dataset_and_source_hashes():
    base=Path(__file__).resolve().parents[1]/'tests/fixtures/numeric_panel_layout_offline/20261027_numeric_panel_layout'
    manifest=json.loads((base/'manifest.json').read_text())
    rows=evaluate(manifest,base)
    assert len(rows)==28
    assert set(r['target'] for r in rows)==set('12345')
    assert any(not r['accepted'] for r in rows)
    # This fixture deliberately records imperfect predictions, not a fake 100% score.
    assert any(r['accepted'] and not r['within_provisional_2d_tolerance'] for r in rows)
