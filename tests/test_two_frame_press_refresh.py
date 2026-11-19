"""Short stationary refresh retains freshness and geometry safeguards."""
import json
from pathlib import Path

import numpy as np
import pytest

from run_online_replan_no_contact import make_button_window
from tests.test_stationary_sampling_timing import SamplingScene


RECORDED = json.loads((Path(__file__).parent / 'fixtures' / 'stationary_sampling'
                      / 'down_refresh_failure_20261021.json').read_text())


def failed_refresh_scene():
    events = []
    start = RECORDED['start_unix_sec']
    for row in RECORDED['samples']:
        captured = row['capture_time_unix_sec']-start
        arrived = row['timing']['response_received_unix_sec']-start
        transform = np.eye(4)
        if row.get('base_to_button_position_m'):
            transform[:3,3] = row['base_to_button_position_m']
        # The pre-epoch row is deliberately kept and must be rejected, even
        # though it has no archived pose in the original failure audit.
        sample = dict(sequence=row['sequence'],
                      capture_time_unix_sec=100+captured,
                      camera_to_button=transform.tolist(),
                      confidence=row.get('confidence',.95),
                      plane_rms_m=row.get('plane_rms_m',.001),
                      bbox_xyxy=row.get('bbox_xyxy',[189,64,222,98]),
                      processing_sec=row['timing']['source_processing_sec'])
        events.append((arrived,sample))
    return SamplingScene(events=events)


def test_recorded_three_sample_timeout_recovers_with_two_qualified_fresh_samples():
    scene = failed_refresh_scene()
    estimate,audit = scene.run(frame_count=2)
    assert estimate.stable and estimate.sequence_range == (815896,815906)
    assert estimate.sample_count == 2
    assert estimate.latest_age_sec <= .35
    assert estimate.position_rms_m < .0015 and estimate.max_position_jump_m < .003
    assert 1.7 < scene.elapsed < 1.9
    assert audit[0]['reason'] == 'captured_before_stationary_pose'
    summary = audit[-1]['acquisition_summary']
    assert summary['policy']['frame_count'] == 2
    assert summary['accepted_frames_total'] == 2


def test_same_recording_still_fails_default_four_frame_initial_reference_policy():
    scene = failed_refresh_scene()
    with pytest.raises(RuntimeError,match='no stable fresh button target') as caught:
        scene.run()
    failure = caught.value.stationary_target_failure
    assert failure['acquisition_summary']['policy']['frame_count'] == 4
    assert failure['acquisition_summary']['accepted_frames_total'] == 3


def test_default_initial_collection_still_collects_four_frames():
    estimate,audit = SamplingScene().run()
    assert estimate.sample_count == 4
    assert audit[-1]['acquisition_summary']['policy']['frame_count'] == 4


@pytest.mark.parametrize('options', [
    {'count':1}, {'age':.4}, {'confidence':.69}, {'plane_rms':.02},
    {'jitter':True}, {'changing_normals':True}, {'moving':True},
])
def test_two_frames_do_not_bypass_quality_or_encoder_safeguards(options):
    with pytest.raises(RuntimeError,match='no stable fresh button target'):
        SamplingScene(**options).run(frame_count=2)


def test_duplicate_deliveries_do_not_turn_one_frame_into_two():
    pose=np.eye(4).tolist()
    events=[(.05+i*.05,dict(sequence=1,capture_time_unix_sec=100.02,
            camera_to_button=pose,confidence=.95,plane_rms_m=.001,
            bbox_xyxy=[100,60,140,100])) for i in range(30)]
    with pytest.raises(RuntimeError,match='no stable fresh button target'):
        SamplingScene(events=events).run(frame_count=2)


@pytest.mark.parametrize('count',[0,1,3,6,True,2.0])
def test_only_explicit_two_or_default_four_frames_are_supported(count):
    with pytest.raises(ValueError,match='frame_count must be 2 or 4'):
        SamplingScene().run(frame_count=count)


def test_moving_window_keeps_six_frames_and_original_freshness():
    window=make_button_window({'detector':{},'button_geometry':{}})
    assert window.size == 6
    assert window.max_input_age_sec == .18 and window.max_latest_age_sec == .18
