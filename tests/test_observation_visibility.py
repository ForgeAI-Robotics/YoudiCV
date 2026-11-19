from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from piper_handeye.observation_visibility import complete_button_box, stable_target_visibility


SIZE = [640, 360]


def estimate(low=10, high=13, count=4, stable=True):
    return SimpleNamespace(stable=stable, sequence_range=(low, high), sample_count=count)


def accepted(sequence, box=None, source='yolo'):
    return dict(accepted=True, sequence=sequence, source=source,
                bbox_xyxy=box if box is not None else [310., 120., 345., 155.])


def rows():
    return [accepted(s) for s in range(10, 14)]


def test_rejected_rows_are_retained_without_becoming_button_geometry():
    audit=rows();audit.insert(3,dict(accepted=False,sequence=12,
        reason='sample_age_out_of_range'))
    original=deepcopy(audit)
    assert not all(complete_button_box(r.get('bbox_xyxy'),SIZE) for r in audit[-2:])
    result=stable_target_visibility(estimate(),audit,SIZE)
    assert result['complete'] is True
    assert [r['sequence'] for r in result['checked_frames']]==[12,13]
    assert result['ignored_rejected_reasons']=={'sample_age_out_of_range':1}
    assert audit==original


def test_real_button1_round23_rejection_is_not_mistaken_for_clipping():
    fixture=json.loads((Path(__file__).parent/'fixtures/stationary_visibility/button1_round23.json').read_text())
    audit=fixture['audit'];value=SimpleNamespace(**fixture['estimate'])
    assert not all(complete_button_box(r.get('bbox_xyxy'),SIZE) for r in audit[-2:])
    result=stable_target_visibility(value,audit,SIZE)
    assert result['complete'] is True
    assert result['window_sequences']==[906938,906941,906945,906953]
    assert [r['sequence'] for r in result['checked_frames']]==[906945,906953]


def test_previously_successful_complete_window_still_passes():
    assert stable_target_visibility(estimate(),rows(),SIZE)['complete'] is True


@pytest.mark.parametrize('box',[[0,120,35,155],[310,0,345,35],
    [610,120,641,155],[310,330,345,361],[310,120,320,130],None])
def test_actual_incomplete_or_missing_accepted_box_still_blocks(box):
    audit=rows();audit[-1]['bbox_xyxy']=box
    result=stable_target_visibility(estimate(),audit,SIZE)
    assert result['complete'] is False
    assert result['reason']=='stable_target_box_incomplete'


def test_cannot_borrow_a_complete_box_from_an_older_window():
    audit=[accepted(1),accepted(2)]+rows()[:-1]
    result=stable_target_visibility(estimate(),audit,SIZE)
    assert result['complete'] is False
    assert result['reason']=='stable_target_window_evidence_missing'


def test_rolled_out_incomplete_box_does_not_change_current_window():
    audit=[accepted(1,[0,0,20,20]),accepted(2,[0,0,20,20])]+rows()
    assert stable_target_visibility(estimate(),audit,SIZE)['complete'] is True


@pytest.mark.parametrize('value',[estimate(stable=False),estimate(count=1),
    estimate(low=13,high=10),estimate(low=10,high=10),
    SimpleNamespace(stable=True,sequence_range=None,sample_count=4)])
def test_unstable_or_missing_window_metadata_never_passes(value):
    assert stable_target_visibility(value,rows(),SIZE)['complete'] is False


@pytest.mark.parametrize('audit',[[],rows()[:-1],rows()+[accepted(13)],
    [accepted(10),accepted(12),accepted(11),accepted(13)]])
def test_missing_duplicate_or_unordered_window_evidence_never_passes(audit):
    assert stable_target_visibility(estimate(),audit,SIZE)['complete'] is False


def test_rejected_row_with_a_good_box_cannot_fill_the_window():
    audit=rows();audit[2]['accepted']=False;audit[2]['reason']='sample_age_out_of_range'
    assert stable_target_visibility(estimate(),audit,SIZE)['complete'] is False


def test_independently_validated_layout_window_uses_its_original_boxes():
    audit=[accepted(s,source='panel_layout_inferred') for s in range(10,14)]
    result=stable_target_visibility(estimate(),audit,SIZE)
    assert result['complete'] is True
    assert all(r['source']=='panel_layout_inferred' for r in result['checked_frames'])
