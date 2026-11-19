from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from run_online_replan_no_contact import collect_stationary_target, make_button_window
from piper_handeye.realtime_planner import TimedButtonTransform


def acquire(*, age=0.23, interval=0.10, move=False, jitter=False):
    clock = [100.0]
    sequence = [0]
    config = {"detector": {}, "button_geometry": {}}

    class Vision:
        def request(self, *_args, **_kwargs):
            clock[0] += interval
            sequence[0] += 1
            pose = np.eye(4)
            if jitter:
                pose[0, 3] = 0.01 * (sequence[0] % 2)
            return {"sample": {
                "sequence": sequence[0], "capture_time_unix_sec": clock[0] - age,
                "camera_to_button": pose, "confidence": 0.93,
                "plane_rms_m": 0.0013, "bbox_xyxy": [0, 0, 20, 20],
            }}

    class Robot:
        def state(self, **_kwargs):
            joints = np.zeros(6)
            if move and sequence[0] >= 6:
                joints[0] = 0.01
            return SimpleNamespace(joints=joints, base_to_gripper=np.eye(4))

    with patch("run_online_replan_no_contact.time.time", side_effect=lambda: clock[0]), \
         patch("run_online_replan_no_contact.time.monotonic", side_effect=lambda: clock[0]):
        return collect_stationary_target(Vision(), Robot(), np.eye(4), "15", config)


def test_stationary_inference_latency_does_not_starve_window():
    estimate, audit = acquire()
    assert estimate.stable and estimate.sample_count == 4
    assert estimate.latest_age_sec == pytest.approx(0.23)
    assert any(x.get("reason") == "captured_before_stationary_pose" for x in audit)
    moving = make_button_window({"detector": {}, "button_geometry": {}})
    assert not moving.add(TimedButtonTransform(1, 100.0, np.eye(4), 0.93, 0.0013), now=100.23)


def test_stationary_window_allows_measured_slow_inference_cadence():
    estimate, _audit = acquire(interval=0.31)
    assert estimate.stable
    assert estimate.span_sec == pytest.approx(0.93)


def test_encoder_motion_discards_old_acquisition_epoch():
    estimate, audit = acquire(move=True)
    assert estimate.stable
    assert estimate.sequence_range[0] > 6
    assert any(x.get("reason") == "encoder_motion_during_stationary_capture" for x in audit)


@pytest.mark.parametrize("options", [{"age": 0.40}, {"jitter": True}])
def test_stale_or_spatially_unstable_samples_still_fail(options):
    with pytest.raises(RuntimeError, match="no stable fresh button target"):
        acquire(**options)
