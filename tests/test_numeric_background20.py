"""Frozen real button-6 replay and adverse-light guards; no hardware."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from piper_handeye.paired_button_light import (
    ButtonPatch, PairedButtonLight, compare_patches,
)
from piper_handeye.button_light_hardware import FreshLightObserver
from tests.test_paired_button_light import patch_image, snapshot
from tests.test_panel_background_reference import FIXTURES, GLARE, snapshot as glare_snapshot
from piper_handeye.paired_button_light import extract_patch


FIXTURE = Path(__file__).parent/'fixtures/numeric_background20'
RECORD = json.loads((FIXTURE/'manifest.json').read_text())


def real_patches():
    path = FIXTURE/'button6_patches.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RECORD['npz_sha256']
    with np.load(path, allow_pickle=False) as data:
        return [(f,ButtonPatch(data[f'gray_{f["sequence"]}'].copy(),np.array(f['box']),
                               f['sequence'],f['captured'],f['source_image']))
                for f in RECORD['frames']]


def replay(monkeypatch, *, strong_on=True, stationary=True):
    patches = real_patches()
    pair = PairedButtonLight(require_stationary_reference=True, stationary_check=lambda: stationary)
    pair.begin_attempt()
    pair.references.extend(p for f,p in patches if f['phase']=='before_press')
    pair.stationary_reference_ready=True
    pair.frozen=True
    pair.press_started=pair.references[-1].captured+.01
    pair.last_reference_sequence=pair.references[-1].sequence
    monkeypatch.setattr('piper_handeye.paired_button_light.extract_patch',lambda s:s['patch'])
    rows=[]
    for f,p in patches:
        if f['phase']=='before_press':continue
        result=pair.classify({'patch':p},f['absolute_state'],stationary=True,
                             strong_on=strong_on and f['absolute_state']=='ON')
        rows.append((f,result))
    return rows


def test_actual_button6_now_has_two_consecutive_qualified_on_before_it_goes_off(monkeypatch):
    rows=replay(monkeypatch)
    assert [r['state'] for _,r in rows] == ['UNCERTAIN','ON','ON','ON','ON','UNCERTAIN','UNCERTAIN']
    for f,r in rows[1:5]:
        assert r['background_tolerance_profile']=='numeric_stationary_strong_on_mean20_v1'
        assert r['background_max_mean_error']==20 and r['background_max_p90_error']==50
        assert r['background_inlier_max_error']==30 and r['background_inlier_fraction']>=.75
        assert .08 <= r['changed_fraction'] <= .70
    assert rows[0][1]['background_error']>20  # no unconditional success


@pytest.mark.parametrize('fault',['no_strong_on','not_stationary'])
def test_actual_button6_does_not_gain_relaxation_without_its_extra_guards(monkeypatch,fault):
    rows=replay(monkeypatch,strong_on=fault!='no_strong_on',stationary=fault!='not_stationary')
    assert all(r['state']=='UNCERTAIN' for _,r in rows)


@pytest.mark.parametrize('pair',GLARE)
@pytest.mark.parametrize('size_factor',[1,2])
def test_recorded_unlit_glare_stays_unconfirmed_even_with_a_wrong_strong_on_flag(pair,size_factor):
    before,after=(extract_patch(glare_snapshot(pair[k])) for k in ('before','after'))
    before.box=before.box*size_factor;after.box=after.box*size_factor
    assert compare_patches(before,after,'ON',stationary=True,strong_on=True)['state']=='UNCERTAIN'


@pytest.mark.parametrize('fault',['uniform_offset','uniform_gain','shadow','covered','saturated',
                                   'unchanged','view_shift','view_scale'])
def test_global_changes_and_invalid_views_do_not_become_success(fault):
    before=patch_image(increase=70);after=patch_image(2,increase=70)
    if fault=='uniform_offset':after.gray+=35
    if fault=='uniform_gain':after.gray*=1.5
    if fault=='shadow':after.gray[:,:32]+=50
    if fault=='covered':after.gray[:]=200
    if fault=='saturated':after.gray[8:56,8:56]=255
    if fault=='view_shift':after.box=before.box+np.array([10,0,10,0])
    if fault=='view_scale':after.box=before.box+np.array([0,0,20,20])
    assert compare_patches(before,after,'ON',stationary=True,strong_on=True)['state']=='UNCERTAIN'


def test_off_reference_comparison_is_identical_even_if_strong_on_was_supplied():
    before=patch_image();after=patch_image(2,exposure=35)
    assert compare_patches(before,after,'OFF',stationary=True,strong_on=True)==compare_patches(before,after,'OFF',stationary=True)


@pytest.mark.parametrize('target,legacy,fraction,eligible',[
    ('6','ON',.4,True),('6','OFF',.4,False),('6','ON',.04,False),('up','ON',.4,False)])
def test_real_observer_only_opts_in_for_agreeing_numeric_whole_and_local_on(tmp_path,target,legacy,fraction,eligible):
    data=snapshot(tmp_path,patch_image())
    data['target_detection']['label']=target
    pair=Mock();pair.classify.return_value={'state':'ON'}
    result=dict(state='ON',legacy_state=legacy,
                local_regions=[dict(contrast=160,area_px=100,fraction=fraction)])
    observer=FreshLightObserver(SimpleNamespace(request=lambda *a,**k:data),
        SimpleNamespace(observe=lambda *a,**k:result),target,paired_light=pair)
    assert observer.observe('near')=='ON'
    assert bool(pair.classify.call_args.kwargs.get('strong_on')) is eligible
    assert pair.classify.call_args.kwargs['stationary'] is True
