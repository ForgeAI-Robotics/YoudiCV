"""Static sampling tolerates measured cadence while retaining quality gates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json

import numpy as np
import pytest

import run_online_replan_no_contact as production
from piper_handeye.arm_report import capture_failure


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/stationary_sampling/down_failure_20261019.json').read_text())
CONFIG = {'detector': {}, 'button_geometry': {}}


class SamplingScene:
    """Deterministic camera deliveries and encoders, with no device clients."""
    def __init__(self, *, interval=.1, age=.23, confidence=.93, plane_rms=.0013,
                 count=20, jitter=False, changing_normals=False, moving=False,
                 events=None, processing_sec=.2):
        self.elapsed = 0.
        self.origin = 100.
        self.calls = []
        self.latest = None
        self.moving = moving
        self.source_rate = 1 / interval
        self.events = []
        if events is not None:
            self.events = deepcopy(events)
        else:
            for i in range(count):
                # Deliver deliberately future-dated samples after epoch start,
                # rather than pre-populating the first one before the test.
                captured = max(.01, .01-age) + interval * i
                transform = np.eye(4)
                if jitter:
                    transform[0, 3] = .01 * (i % 2)
                if changing_normals:
                    angle = np.deg2rad(10 if i % 2 else -10)
                    c, s = np.cos(angle), np.sin(angle)
                    transform[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
                sample = dict(sequence=i+1, capture_time_unix_sec=self.origin+captured,
                    camera_to_button=transform.tolist(), confidence=confidence,
                    plane_rms_m=plane_rms, bbox_xyxy=[100, 60, 140, 100],
                    processing_sec=processing_sec)
                self.events.append((captured+age, sample))
        self.events.sort(key=lambda row: row[0])

    def request(self, command, **kwargs):
        # This task deliberately retains the existing latest-only API.
        assert command == 'latest'
        assert kwargs == {'target_label': 'down', 'window_size': 6}
        self.calls.append(command)
        self.elapsed += .001
        while self.events and self.events[0][0] <= self.elapsed:
            _, self.latest = self.events.pop(0)
        return dict(sample=deepcopy(self.latest), server_time_unix_sec=self.origin+self.elapsed,
                    realtime=dict(sample_rate_hz=self.source_rate))

    def state(self, **kwargs):
        assert kwargs == {'max_age': .35}
        self.elapsed += .001
        q = np.zeros(6)
        if self.moving and self.latest:
            q[0] = .005 * self.latest['sequence']
        return SimpleNamespace(joints=q, base_to_gripper=np.eye(4))

    def run(self, module=production, **kwargs):
        with patch.object(module.time, 'time', lambda: self.origin+self.elapsed), \
             patch.object(module.time, 'monotonic', lambda: self.elapsed), \
             patch.object(module.time, 'perf_counter', lambda: self.elapsed), \
             patch.object(module.time, 'sleep', lambda sec: setattr(self, 'elapsed', self.elapsed+sec)), \
             patch.object(module, 'save_missing_button_view', return_value={'offline_scene': True}):
            return module.collect_stationary_target(self, self, np.eye(4), 'down', CONFIG, **kwargs)


def recorded_scene():
    events = []
    accepted = [row for row in FIXTURE['samples'] if row['accepted']]
    for i, row in enumerate(accepted):
        # Keep the fifth recorded sample fresh at the three-second deadline,
        # so the old policy reproduces span rejection rather than starvation.
        captured = 2.68 - (len(accepted)-1-i) * FIXTURE['reported_window_span_sec'] / 3
        transform = np.eye(4);transform[:3, 3] = row['base_to_button_position_m']
        sample = dict(sequence=row['sequence'], capture_time_unix_sec=100+captured,
                      camera_to_button=transform.tolist(), confidence=row['confidence'],
                      plane_rms_m=row['plane_rms_m'], bbox_xyxy=row['bbox_xyxy'])
        # Missing historical processing timings stay unknown, not fabricated.
        events.append((captured+row['sample_age_sec'], sample))
    for i, row in enumerate(FIXTURE['samples'][:2]):
        events.append((.01+i*.01, dict(sequence=row['sequence'], capture_time_unix_sec=99.5+i*.1,
            camera_to_button=np.eye(4).tolist(), confidence=.95, plane_rms_m=.001,
            bbox_xyxy=[100, 60, 140, 100])))
    return SamplingScene(interval=FIXTURE['reported_window_span_sec']/3, events=events)


def test_recorded_slow_stable_samples_now_pass_without_relaxing_age_or_count():
    scene = recorded_scene();estimate, audit = scene.run()
    assert estimate.stable and estimate.sample_count == 4
    assert estimate.span_sec == pytest.approx(1.677)
    assert estimate.latest_age_sec <= .35
    assert estimate.position_rms_m < .0015 and estimate.max_position_jump_m < .003
    assert any(row.get('reason') == 'captured_before_stationary_pose' for row in audit)
    assert all(row.get('bbox_xyxy') for row in audit[-2:])
    summary = audit[-1]['acquisition_summary']
    assert summary['status'] == 'ready' and summary['policy']['max_span_sec'] == 2.
    assert summary['source_processing_mean_sec'] is None
    assert scene.elapsed < 3.


def test_fast_frames_return_immediately_instead_of_waiting_two_seconds():
    scene = SamplingScene();estimate, audit = scene.run()
    assert estimate.sample_count == 4 and scene.elapsed < 1.
    assert audit[-1]['acquisition_summary']['elapsed_sec'] == pytest.approx(scene.elapsed)


@pytest.mark.parametrize('options,reason', [
    ({'age': .4}, 'empty'),
    ({'age': -.08}, 'empty'),
    ({'confidence': .69}, 'empty'),
    ({'plane_rms': .01}, 'empty'),
    ({'jitter': True}, 'position RMS'),
    ({'changing_normals': True}, 'normal RMS'),
    ({'moving': True}, 'empty'),
    ({'interval': .7}, 'window span'),
    ({'count': 3}, 'window 3/4'),
])
def test_invalid_samples_are_still_rejected(options, reason):
    scene = SamplingScene(**options)
    with pytest.raises(RuntimeError, match='no stable fresh button target before motion') as caught:
        scene.run()
    details = caught.value.stationary_target_failure
    assert any(reason in item for item in details['window_reasons'])
    assert details['acquisition_summary']['status'] == 'timeout'
    assert details['acquisition_summary']['policy']['timeout_sec'] == 3.
    assert scene.elapsed < 3.1


def test_stale_input_records_rejection_instead_of_being_used_as_new_evidence():
    scene = SamplingScene(age=.4)
    with pytest.raises(RuntimeError) as caught:
        scene.run()
    samples = caught.value.stationary_target_failure['samples']
    assert any('sample_age_out_of_range' in row.get('rejection_reasons', []) for row in samples)
    assert not any(row['accepted'] for row in samples)


def test_repeated_latest_sample_cannot_fill_four_frame_window():
    scene = SamplingScene(count=1)
    with pytest.raises(RuntimeError) as caught:
        scene.run()
    summary = caught.value.stationary_target_failure['acquisition_summary']
    assert summary['accepted_frames_total'] == 1 and summary['unchanged_reads'] > 1
    assert summary['window_sample_count'] == 1


def test_future_stationary_barrier_cannot_reuse_earlier_pictures():
    scene = SamplingScene()
    with pytest.raises(RuntimeError):
        scene.run(min_capture_unix_sec=200.)


def test_source_request_and_encoder_timings_are_separate_and_json_serializable():
    scene = SamplingScene();_, audit = scene.run()
    summary = audit[-1]['acquisition_summary']
    assert summary['source_processing_mean_sec'] == pytest.approx(.2)
    assert summary['request_max_sec'] == pytest.approx(.001)
    assert summary['robot_state_reads_total_sec'] > 0
    assert summary['source_valid_rate_hz'] == pytest.approx(10.)
    assert summary['observed_accepted_rate_hz'] == pytest.approx(10.)
    assert summary['empty_reads'] > 0 and summary['unchanged_reads'] > 0
    row = next(row for row in audit if row.get('accepted'))
    assert row['timing']['age_outside_processing_sec'] == pytest.approx(.03, abs=.012)
    assert row['timing']['request_roundtrip_sec'] == pytest.approx(.001)
    json.dumps(audit)


def test_failure_timings_reach_authoritative_report_without_changing_reason():
    scene = SamplingScene(jitter=True)
    with pytest.raises(RuntimeError) as caught:
        scene.run()
    report = {'active_stage': 'initial_light_recheck_post_motion_target'}
    capture_failure(report, caught.value, 'original stack')
    assert report['failure_code'] == 'BUTTON_LOCALIZATION_TIMEOUT'
    assert report['failure']['message'] == 'no stable fresh button target before motion'
    assert report['stationary_target_failure']['acquisition_summary']['status'] == 'timeout'


def test_optional_diagnostic_error_never_replaces_primary_failure():
    error = RuntimeError('no stable fresh button target before motion')
    error.stationary_target_failure = {'invalid_optional_field': object()}
    report = {};capture_failure(report, error, 'original stack')
    assert report['failure']['message'] == str(error)
    assert report['stationary_target_diagnostics_error']


def test_moving_window_configuration_is_unchanged():
    window = production.make_button_window(CONFIG)
    assert window.size == 6 and window.max_span_sec == .7
    assert window.max_input_age_sec == window.max_latest_age_sec == .18
    assert window.min_confidence == .7 and window.max_plane_rms_m == .006
    assert window.max_position_rms_m == .0015 and window.max_position_jump_m == .003
