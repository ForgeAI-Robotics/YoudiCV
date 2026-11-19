"""Same-frame local 3/5 consistency; no model reload, robot IO, or motion."""
import copy

import numpy as np
import pytest
from piper_handeye.button_pose import ButtonDetection
from piper_handeye.numeric_yolo_consistency import inspect_numeric_target, target_conflicted, admitted_detections
from tests.test_numeric_yolo_consistency import COORDINATES, button, make_worker


NEIGHBOURS = {'3': ('4', '2', '10'), '5': ('6', '4', '12')}


def triplet(claimed, actual=None):
    actual = actual or claimed
    return tuple([button(label) for label in NEIGHBOURS[actual]] + [button(claimed, actual, .99)])


def inspect(ds, target):
    return inspect_numeric_target(COORDINATES, ds, target, [640, 360])


@pytest.mark.parametrize('target', ['3', '5'])
def test_three_local_neighbours_confirm_correct_label(target):
    ds = triplet(target); before = copy.deepcopy(ds)
    r = inspect(ds, target)
    assert r['decision'] == 'consistent' and r['global_check']['decision'] == 'unverified'
    assert r['local_35_check']['version'] == 1
    assert r['anchors'] == list(NEIGHBOURS[target])
    assert ds == before and admitted_detections(ds, r) is ds


@pytest.mark.parametrize('claimed,actual', [('3', '5'), ('5', '3')])
def test_three_local_neighbours_reject_mutual_misclassification(claimed, actual):
    ds = triplet(claimed, actual); before = copy.deepcopy(ds)
    r = inspect(ds, claimed)
    assert r['decision'] == 'conflict' and r['alternative_label'] == actual
    assert r['global_check']['decision'] == 'unverified'
    assert r['claimed_cell_distance'] > 1.9
    assert target_conflicted(r) and not any(d.label == claimed for d in admitted_detections(ds, r))
    assert ds == before  # no relabel, invented detection, confidence or ON


@pytest.mark.parametrize('fault', ['missing', 'weak', 'duplicate', 'clipped', 'overlap', 'reversed', 'degenerate', 'giant', 'wrong_row', 'weak_target', 'duplicate_target'])
def test_bad_local_evidence_never_vetoes(fault):
    ds = list(triplet('5', '3'))
    ref = ds[0]
    if fault == 'missing': ds.pop(0)
    if fault == 'weak': ds[0] = ButtonDetection(0, ref.label, .84, ref.bbox_xyxy)
    if fault == 'duplicate': ds.append(ref)
    if fault == 'clipped': ds[0] = ButtonDetection(0, ref.label, .99, (0, 0, 24, 24))
    if fault == 'overlap': ds[0] = ButtonDetection(0, ref.label, .99, ds[-1].bbox_xyxy)
    if fault == 'reversed': ds[0], ds[1] = ButtonDetection(0, '4', .99, ds[1].bbox_xyxy), ButtonDetection(0, '2', .99, ds[0].bbox_xyxy)
    if fault == 'degenerate': ds[2] = ButtonDetection(0, '10', .99, button('1').bbox_xyxy)
    if fault == 'giant': ds[0] = ButtonDetection(0, '4', .99, (140, 160, 300, 230))
    if fault == 'wrong_row': ds[-1] = button('5', '1')
    if fault == 'weak_target': ds[-1] = button('5', '3', .69)
    if fault == 'duplicate_target': ds.append(button('5'))
    r = inspect(tuple(ds), '5')
    assert r['decision'] == 'unverified', r


def test_different_layout_cannot_activate_special_case():
    coords = copy.deepcopy(COORDINATES); coords['10'] = [2, -2]
    assert inspect_numeric_target(coords, triplet('5', '3'), '5', [640, 360])['decision'] == 'unverified'


@pytest.mark.parametrize('target', ['1', '2', '4', '6'])
def test_other_labels_keep_original_no_three_anchor_veto(target):
    ds = tuple([button(label) for label in ['2', '4', '10'] if label != target] + [button(target, '3')])
    r = inspect(ds, target)
    assert 'local_35_check' not in r and r['decision'] == 'unverified'


def test_known_global_decision_is_preserved_without_local_override():
    from tests.test_numeric_yolo_consistency import scene
    r = inspect(scene(wrong=True), '5')
    assert r['decision'] == 'conflict' and 'local_35_check' not in r


@pytest.mark.parametrize('claimed,actual', [('3', '3'), ('5', '5'), ('3', '5'), ('5', '3')])
def test_small_view_scale_shear_noise_keeps_cell_decision(claimed, actual):
    rng = np.random.default_rng(20261027)
    for _ in range(25):
        ds = []
        scale = rng.uniform(.9, 1.1)
        for d in triplet(claimed, actual):
            b = np.asarray(d.bbox_xyxy).reshape(2,2); b = (b-[220,200])*scale+[220,200]
            b[:,0] += .10*(b[:,1]-200)
            b += rng.uniform(-.5,.5,2)
            ds.append(ButtonDetection(0,d.label,d.confidence,tuple(b.ravel())))
        r = inspect(tuple(ds), claimed)
        assert r['decision'] == ('consistent' if claimed == actual else 'conflict'), r


def test_real_perception_never_creates_pose_or_early_lamp_package_from_local_wrong_label(monkeypatch):
    import tests.test_numeric_yolo_consistency as fixtures
    monkeypatch.setattr(fixtures, 'scene', lambda **kw: triplet('5', '3'))
    # Existing real producer fixture runs target 5, including localizer/depth path.
    worker, frame = make_worker()
    sample = worker.latest_sample()
    assert target_conflicted(sample.numeric_yolo_consistency)
    assert sample.numeric_yolo_consistency['alternative_label'] == '3'
    assert sample.pose is None and 'numeric_yolo_layout_conflict' in sample.error
    assert worker.latest_valid() is None and worker.latest_light_sample() is sample
    assert sample.publication_stage == 'completed_rgbd'
