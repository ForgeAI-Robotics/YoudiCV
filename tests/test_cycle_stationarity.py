import pytest
from piper_handeye.cycle_stationarity import StationarityGuard, ChassisNotStationary


def sample(seq=1, speed=0., **kw):
    return dict(available=True,error=None,age_sec=.01,velocity=[speed,0.,0.,0.,0.,0.],
                sequence=seq,source_stamp_sec=100.+seq*.04,pose_xy_yaw=[0.,0.,0.],**kw)


def test_single_small_spike_clears_only_with_distinct_new_stationary_frame():
    g=StationarityGuard()
    assert g.observe(sample(),100.)['decision']=='stationary'
    assert g.observe(sample(2,.008),100.04)['decision']=='transient_pending'
    assert g.observe(sample(2,.008),100.06)['distinct_suspect_samples']==1
    assert g.observe(sample(3),100.08)['decision']=='transient_cleared'


def test_persistent_small_speed_stops_bounded_and_records_trigger_velocity():
    g=StationarityGuard()
    for i in range(3): g.observe(sample(i+1,.008),100+i*.04)
    with pytest.raises(ChassisNotStationary) as caught:
        g.observe(sample(4,.008),100.13)
    assert caught.value.audit['sample']['velocity'][0]==.008
    assert caught.value.audit['recent']


def test_repeated_suspect_cache_frame_cannot_wait_forever():
    g=StationarityGuard(); g.observe(sample(1,.008),100.)
    with pytest.raises(ChassisNotStationary):g.observe(sample(1,.008),100.13)


def test_late_good_frame_cannot_hide_long_unchecked_suspect_gap():
    g=StationarityGuard();g.observe(sample(1,.008),100.)
    with pytest.raises(ChassisNotStationary):g.observe(sample(2),100.3)


@pytest.mark.parametrize('change', [dict(velocity=[.011,0,0,0,0,0]),dict(velocity=[0,0,0,0,.041,0]),
    dict(available=False),dict(error='lost'),dict(age_sec=.501),dict(velocity=[float('nan')]*6),
    dict(sequence=0),dict(source_stamp_sec=90.),dict(pose_xy_yaw=[.0021,0,0]),dict(pose_xy_yaw=[0,0,.0101])])
def test_motion_displacement_invalid_feedback_and_source_reset_stop_immediately(change):
    g=StationarityGuard();g.observe(sample(),100.)
    next_sample=sample(2);next_sample.update(change)
    with pytest.raises(ChassisNotStationary):g.observe(next_sample,100.04)


def test_healthy_sample_counter_stall_not_hidden_by_new_network_packets():
    g=StationarityGuard();g.observe(sample(),100.)
    with pytest.raises(ChassisNotStationary):g.observe(sample(),100.51)


def test_suspect_is_not_shared_across_tasks():
    a=StationarityGuard();a.observe(sample(1,.008),100.)
    b=StationarityGuard()
    assert b.observe(sample(10),101.)['decision']=='stationary'
