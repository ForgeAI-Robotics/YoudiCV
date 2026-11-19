import numpy as np
import pytest

from piper_handeye.guarded_press import (
    CartesianForceKalman,
    CartesianNormalForceKalman,
    GuardedPressConfig,
    GuardedPressDecision,
    GuardedPressMonitor,
    MotorCurrentSample,
    PiperCurrentFrameAssembler,
    RobustCurrentBaseline,
    PIPER_TORQUE_PER_AMP_NM,
    piper_tcp_linear_jacobian,
    decode_piper_motor_current,
    generate_bounded_retract_trajectory,
)
from piper_handeye.kinematics import forward_kinematics_with_jacobian


def test_lightweight_press_jacobian_matches_project_kinematics():
    joints = np.asarray([0.1, 1.5, -1.4, 0.2, -0.3, 0.1])
    expected = forward_kinematics_with_jacobian(joints)[1][:3]
    np.testing.assert_allclose(piper_tcp_linear_jacobian(joints), expected, atol=1e-12)


def test_cartesian_normal_force_kalman_recovers_force_from_motor_current():
    joints = np.asarray([0.1, 1.5, -1.4, 0.2, -0.3, 0.1])
    normal = np.asarray([-1.0, 0.0, 0.0])
    signature = piper_tcp_linear_jacobian(joints).T @ normal
    force_n = -0.8
    current_residual = signature * force_n / PIPER_TORQUE_PER_AMP_NM
    observer = CartesianNormalForceKalman(process_noise_n2_per_sec=80.0)
    estimate = None
    for index in range(30):
        estimate = observer.update(
            signed_current_residual_a=current_residual,
            current_noise_a=np.full(6, 0.01),
            joints_rad=joints,
            contact_normal_base=normal,
            now=1.0 + index * 0.005,
        )
    assert estimate is not None
    assert estimate.force_n == pytest.approx(force_n, abs=0.03)
    assert estimate.score > 4.0
    assert estimate.alignment == pytest.approx(1.0, abs=1e-9)


def test_cartesian_force_kalman_recovers_tangential_force():
    joints = np.asarray([0.1, 1.5, -1.4, 0.2, -0.3, 0.1])
    force = np.asarray([0.0, 0.0, 0.9])
    torque = piper_tcp_linear_jacobian(joints).T @ force
    observer = CartesianForceKalman(process_noise_n2_per_sec=80.0)
    estimate = covariance = None
    for index in range(30):
        estimate, covariance = observer.update(
            joint_torque_residual_nm=torque,
            joint_torque_noise_nm=np.full(6, 0.01),
            joints_rad=joints,
            now=1.0 + index * 0.005,
        )
    np.testing.assert_allclose(estimate, force, atol=0.04)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_sustained_tangential_contact_stops_without_normal_alignment():
    joints = np.asarray([0.1, 1.5, -1.4, 0.2, -0.3, 0.1])
    normal = np.asarray([-1.0, 0.0, 0.0])
    tangential_force = np.asarray([0.0, 0.0, 0.9])
    current = (
        piper_tcp_linear_jacobian(joints).T @ tangential_force
        / PIPER_TORQUE_PER_AMP_NM
    )
    cfg = GuardedPressConfig(
        contact_delta_a=np.full(6, 0.05), hard_delta_a=np.full(6, 1.0),
        absolute_current_limit_a=np.full(6, 3.0), median_window=5,
        persistence_samples=5, persistence_sec=0.40,
        dynamic_reference_window_samples=20, contact_arm_travel_m=0.008,
        max_forward_travel_m=0.022,
        contact_estimator="cartesian_normal_force_kalman",
        normal_force_threshold_n=0.50,
        normal_force_sigma_multiplier=4.0,
        normal_force_min_alignment=0.20,
        tangential_force_threshold_n=0.60,
        tangential_force_sigma_multiplier=4.0,
        tangential_force_persistence_sec=0.10,
        relative_current_hard_stop_enabled=False,
        absolute_current_hard_stop_enabled=False,
    )
    base = RobustCurrentBaseline(
        median_a=np.zeros(6), sigma_a=np.full(6, 0.01), sample_count=60
    )
    monitor = GuardedPressMonitor(base, cfg, contact_normal_base=normal)
    now = 1.0
    for _ in range(30):
        monitor.update(
            MotorCurrentSample(now, np.zeros(6)), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.0079,
            joints_rad=joints,
        )
        now += 0.005
    result = None
    for _ in range(30):
        result = monitor.update(
            MotorCurrentSample(now, current), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.010,
            joints_rad=joints,
        )
        if result.decision is GuardedPressDecision.CONTACT:
            break
        now += 0.005
    assert result.decision is GuardedPressDecision.CONTACT
    assert result.reason == "sustained_cartesian_tangential_force_kalman"
    assert result.persistence_sec < 0.40
    assert result.estimated_tangential_force_n >= 0.60


def test_cartesian_force_contact_requires_continuous_persistence():
    joints = np.asarray([0.1, 1.5, -1.4, 0.2, -0.3, 0.1])
    normal = np.asarray([-1.0, 0.0, 0.0])
    signature = piper_tcp_linear_jacobian(joints).T @ normal
    contact_current = signature * -0.8 / PIPER_TORQUE_PER_AMP_NM
    cfg = GuardedPressConfig(
        contact_delta_a=np.full(6, 0.05),
        hard_delta_a=np.full(6, 1.0),
        absolute_current_limit_a=np.full(6, 3.0),
        noise_sigma_multiplier=5.0,
        hard_noise_sigma_multiplier=8.0,
        median_window=5,
        persistence_samples=5,
        persistence_sec=0.10,
        dynamic_reference_window_samples=40,
        contact_arm_travel_m=0.005,
        contact_estimator="cartesian_normal_force_kalman",
        normal_force_threshold_n=0.35,
        normal_force_sigma_multiplier=4.0,
        normal_force_min_alignment=0.60,
    )
    baseline = RobustCurrentBaseline(
        median_a=np.zeros(6), sigma_a=np.full(6, 0.01), sample_count=60
    )
    monitor = GuardedPressMonitor(
        baseline, cfg, contact_normal_base=normal
    )
    now = 1.0
    result = None
    for _ in range(45):
        result = monitor.update(
            MotorCurrentSample(now, np.zeros(6)), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.006,
            joints_rad=joints,
        )
        now += 0.005
    contact_started = now
    for _ in range(30):
        result = monitor.update(
            MotorCurrentSample(now, contact_current), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.010,
            joints_rad=joints,
        )
        if result.decision is GuardedPressDecision.CONTACT:
            break
        now += 0.005
    assert result.decision is GuardedPressDecision.CONTACT
    assert result.reason == "sustained_cartesian_normal_force_kalman"
    assert result.persistence_sec >= 0.10
    assert now - contact_started >= 0.10


def test_cartesian_observer_keeps_learning_motion_load_until_arm_boundary():
    """A smooth free-motion load must not become contact at the arm boundary."""
    joints = np.asarray([0.1, 1.5, -1.4, 0.2, -0.3, 0.1])
    normal = np.asarray([-1.0, 0.0, 0.0])
    signature = piper_tcp_linear_jacobian(joints).T @ normal
    cfg = GuardedPressConfig(
        contact_delta_a=np.full(6, 0.05),
        hard_delta_a=np.full(6, 1.0),
        absolute_current_limit_a=np.full(6, 3.0),
        median_window=5, persistence_samples=5, persistence_sec=0.10,
        dynamic_reference_window_samples=40, contact_arm_travel_m=0.008,
        max_forward_travel_m=0.022,
        contact_estimator="cartesian_normal_force_kalman",
        relative_current_hard_stop_enabled=False,
        absolute_current_hard_stop_enabled=False,
    )
    base = RobustCurrentBaseline(
        median_a=np.zeros(6), sigma_a=np.full(6, 0.01), sample_count=60
    )
    monitor = GuardedPressMonitor(base, cfg, contact_normal_base=normal)
    now = 1.0
    result = None
    # Simulate the normal-direction motor load ramp observed during the real
    # press trajectory, without any environmental contact.
    for index in range(180):
        free_force = -0.15 - 0.003 * index
        currents = signature * free_force / PIPER_TORQUE_PER_AMP_NM
        travel = 0.0079 if index < 160 else 0.0085
        result = monitor.update(
            MotorCurrentSample(now, currents), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=travel,
            joints_rad=joints,
        )
        assert result.decision is GuardedPressDecision.CONTINUE
        now += 0.005

    # A new external-force step after arming remains detectable.
    for _ in range(40):
        contact_force = free_force - 0.8
        currents = signature * contact_force / PIPER_TORQUE_PER_AMP_NM
        result = monitor.update(
            MotorCurrentSample(now, currents), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.012,
            joints_rad=joints,
        )
        if result.decision is GuardedPressDecision.CONTACT:
            break
        now += 0.005
    assert result.decision is GuardedPressDecision.CONTACT


def test_cartesian_observer_does_not_absorb_slow_contact_ramp_after_arming():
    joints = np.asarray([0.1, 1.5, -1.4, 0.2, -0.3, 0.1])
    normal = np.asarray([-1.0, 0.0, 0.0])
    signature = piper_tcp_linear_jacobian(joints).T @ normal
    cfg = GuardedPressConfig(
        contact_delta_a=np.full(6, 0.05), hard_delta_a=np.full(6, 1.0),
        absolute_current_limit_a=np.full(6, 3.0), median_window=5,
        persistence_samples=5, persistence_sec=0.10,
        dynamic_reference_window_samples=40, contact_arm_travel_m=0.008,
        max_forward_travel_m=0.022,
        contact_estimator="cartesian_normal_force_kalman",
        relative_current_hard_stop_enabled=False,
        absolute_current_hard_stop_enabled=False,
    )
    base = RobustCurrentBaseline(
        median_a=np.zeros(6), sigma_a=np.full(6, 0.01), sample_count=60
    )
    monitor = GuardedPressMonitor(base, cfg, contact_normal_base=normal)
    now = 1.0
    for index in range(100):
        currents = signature * (-0.2 - 0.002 * index) / PIPER_TORQUE_PER_AMP_NM
        result = monitor.update(
            MotorCurrentSample(now, currents), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.0079,
            joints_rad=joints,
        )
        now += 0.005
    # A compliant button builds force gradually.  The frozen pre-contact model
    # must retain that ramp long enough for the persistence test to see it.
    for index in range(100):
        force = -0.4 - 0.012 * index
        currents = signature * force / PIPER_TORQUE_PER_AMP_NM
        result = monitor.update(
            MotorCurrentSample(now, currents), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.008 + index * 0.00005,
            joints_rad=joints,
        )
        if result.decision is GuardedPressDecision.CONTACT:
            break
        now += 0.005
    assert result.decision is GuardedPressDecision.CONTACT


def test_cartesian_opposite_direction_force_cannot_claim_panel_contact():
    joints = np.asarray([0.1, 1.5, -1.4, 0.2, -0.3, 0.1])
    normal = np.asarray([-1.0, 0.0, 0.0])
    signature = piper_tcp_linear_jacobian(joints).T @ normal
    cfg = GuardedPressConfig(
        contact_delta_a=np.full(6, 0.05), hard_delta_a=np.full(6, 1.0),
        absolute_current_limit_a=np.full(6, 3.0), median_window=5,
        persistence_samples=5, persistence_sec=0.10,
        dynamic_reference_window_samples=20, contact_arm_travel_m=0.008,
        max_forward_travel_m=0.022,
        contact_estimator="cartesian_normal_force_kalman",
        normal_force_threshold_n=0.50,
        normal_force_sigma_multiplier=4.0,
        normal_force_min_alignment=0.20,
        relative_current_hard_stop_enabled=False,
        absolute_current_hard_stop_enabled=False,
    )
    base = RobustCurrentBaseline(
        median_a=np.zeros(6), sigma_a=np.full(6, 0.01), sample_count=60
    )
    monitor = GuardedPressMonitor(base, cfg, contact_normal_base=normal)
    now = 1.0
    for _ in range(30):
        result = monitor.update(
            MotorCurrentSample(now, np.zeros(6)), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.0079,
            joints_rad=joints,
        )
        now += 0.005
    opposite_current = signature * 1.6 / PIPER_TORQUE_PER_AMP_NM
    for _ in range(40):
        result = monitor.update(
            MotorCurrentSample(now, opposite_current), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.010,
            joints_rad=joints,
        )
        assert result.decision is GuardedPressDecision.CONTINUE
        now += 0.005


def test_cartesian_only_mode_ignores_current_hard_limits_but_keeps_travel_limit():
    joints = np.asarray([0.1, 1.5, -1.4, 0.2, -0.3, 0.1])
    normal = np.asarray([-1.0, 0.0, 0.0])
    signature = piper_tcp_linear_jacobian(joints).T @ normal
    contact_current = signature * -0.8 / PIPER_TORQUE_PER_AMP_NM
    cfg = GuardedPressConfig(
        contact_delta_a=np.full(6, 0.01),
        hard_delta_a=np.full(6, 0.02),
        absolute_current_limit_a=np.full(6, 0.03),
        median_window=5, persistence_samples=5, persistence_sec=0.10,
        dynamic_reference_window_samples=20, contact_arm_travel_m=0.005,
        max_forward_travel_m=0.022,
        contact_estimator="cartesian_normal_force_kalman",
        relative_current_hard_stop_enabled=False,
        absolute_current_hard_stop_enabled=False,
    )
    base = RobustCurrentBaseline(
        median_a=np.zeros(6), sigma_a=np.full(6, 0.01), sample_count=60
    )
    monitor = GuardedPressMonitor(base, cfg, contact_normal_base=normal)
    now = 1.0
    for _ in range(25):
        result = monitor.update(
            MotorCurrentSample(now, np.zeros(6)), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.004,
            joints_rad=joints,
        )
        now += 0.005
    for _ in range(30):
        result = monitor.update(
            MotorCurrentSample(now, contact_current), now=now,
            elapsed_sec=now - 1.0, forward_travel_m=0.006,
            joints_rad=joints,
        )
        if result.decision is GuardedPressDecision.CONTACT:
            break
        now += 0.005
    assert result.decision is GuardedPressDecision.CONTACT

    travel_monitor = GuardedPressMonitor(base, cfg, contact_normal_base=normal)
    result = travel_monitor.update(
        MotorCurrentSample(now, np.zeros(6)), now=now,
        elapsed_sec=0.1, forward_travel_m=0.023,
        joints_rad=joints,
    )
    assert result.decision is GuardedPressDecision.HARD_STOP
    assert result.reason == "maximum_forward_travel"


def config():
    return GuardedPressConfig(
        contact_delta_a=np.full(6, 0.20),
        hard_delta_a=np.full(6, 0.40),
        absolute_current_limit_a=np.full(6, 3.0),
        median_window=3,
        persistence_samples=3,
        max_forward_travel_m=0.020,
        max_press_time_sec=2.0,
    )


def baseline():
    values = np.tile([0.07, 0.41, -1.00, -0.07, -0.48, 0.03], (50, 1))
    return RobustCurrentBaseline.fit(values)


def sample(t, joint2_delta=0.0):
    values = baseline().median_a.copy()
    values[1] += joint2_delta
    return MotorCurrentSample(t, values)


def test_decodes_signed_big_endian_milliamps():
    joint, current = decode_piper_motor_current(0x252, bytes.fromhex("0000019800000069"))
    assert joint == 1
    assert current == pytest.approx(0.408)
    joint, current = decode_piper_motor_current(0x253, bytes.fromhex("0000FC1500000000"))
    assert joint == 2
    assert current == pytest.approx(-1.003)


def test_assembler_requires_one_fresh_frame_from_every_joint():
    assembler = PiperCurrentFrameAssembler()
    result = None
    for index in range(6):
        payload = b"\x00\x00\x00\x64\x00\x00\x00\x00"
        result = assembler.add(0x251 + index, payload, 1.0 + index * 0.001)
    assert result is not None
    np.testing.assert_allclose(result.currents_a, 0.1)
    assert assembler.add(0x251, payload, 1.010) is None


def test_single_current_spike_does_not_claim_contact():
    monitor = GuardedPressMonitor(baseline(), config())
    deltas = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0]
    decisions = [
        monitor.update(sample(i * 0.005, delta), now=i * 0.005,
                       elapsed_sec=i * 0.005, forward_travel_m=0.001).decision
        for i, delta in enumerate(deltas)
    ]
    assert GuardedPressDecision.CONTACT not in decisions


def test_relative_thresholds_are_ignored_before_arm_travel():
    base_config = config()
    armed_config = GuardedPressConfig(
        contact_delta_a=base_config.contact_delta_a,
        hard_delta_a=base_config.hard_delta_a,
        absolute_current_limit_a=base_config.absolute_current_limit_a,
        median_window=3,
        persistence_samples=3,
        contact_arm_travel_m=0.007,
        max_forward_travel_m=0.020,
        max_press_time_sec=2.0,
    )
    monitor = GuardedPressMonitor(baseline(), armed_config)
    for i in range(8):
        result = monitor.update(
            sample(i * 0.005, 0.3), now=i * 0.005,
            elapsed_sec=i * 0.005, forward_travel_m=0.003,
        )
    assert result.decision is GuardedPressDecision.CONTINUE
    assert result.reason == "contact_detector_not_yet_armed"


def test_absolute_current_limit_remains_active_before_arm_travel():
    base_config = config()
    armed_config = GuardedPressConfig(
        contact_delta_a=base_config.contact_delta_a,
        hard_delta_a=base_config.hard_delta_a,
        absolute_current_limit_a=base_config.absolute_current_limit_a,
        median_window=3,
        persistence_samples=3,
        contact_arm_travel_m=0.007,
        max_forward_travel_m=0.020,
        max_press_time_sec=2.0,
    )
    monitor = GuardedPressMonitor(baseline(), armed_config)
    values = baseline().median_a.copy()
    values[2] = -3.1
    result = monitor.update(
        MotorCurrentSample(1.0, values), now=1.0,
        elapsed_sec=0.1, forward_travel_m=0.003,
    )
    assert result.decision is GuardedPressDecision.HARD_STOP
    assert result.reason == "absolute_current_limit"


def test_sustained_residual_claims_contact_after_filter_and_persistence():
    monitor = GuardedPressMonitor(baseline(), config())
    decision = None
    for i in range(10):
        decision = monitor.update(
            sample(i * 0.005, 0.3 if i >= 2 else 0.0), now=i * 0.005,
            elapsed_sec=i * 0.005, forward_travel_m=0.003,
        )
        if decision.decision is GuardedPressDecision.CONTACT:
            break
    assert decision.decision is GuardedPressDecision.CONTACT
    assert decision.persistence == 3


def test_non_contact_joint_residual_cannot_end_press_but_hard_limits_remain():
    base = baseline()
    guarded = GuardedPressConfig(
        contact_delta_a=np.full(6, 0.10),
        hard_delta_a=np.full(6, 0.40),
        absolute_current_limit_a=np.full(6, 3.0),
        median_window=3, persistence_samples=3,
        contact_joint_indices=(4,),
        max_forward_travel_m=0.020, max_press_time_sec=2.0,
    )
    monitor = GuardedPressMonitor(base, guarded)
    for index in range(8):
        values = base.median_a.copy()
        values[0] += 0.16
        result = monitor.update(
            MotorCurrentSample(index * 0.005, values), now=index * 0.005,
            elapsed_sec=index * 0.005, forward_travel_m=0.012,
        )
    assert result.decision is GuardedPressDecision.CONTINUE

    values[0] = base.median_a[0] + 0.45
    for index in range(3):
        stamp = 0.05 + index * 0.005
        hard = monitor.update(
            MotorCurrentSample(stamp, values), now=stamp,
            elapsed_sec=stamp, forward_travel_m=0.013,
        )
    assert hard.decision is GuardedPressDecision.HARD_STOP
    assert hard.reason == "relative_current_limit"


def test_gravity_axis_can_be_excluded_from_relative_hard_stop_only():
    base = baseline()
    guarded = GuardedPressConfig(
        contact_delta_a=np.full(6, 0.20),
        hard_delta_a=np.full(6, 0.40),
        absolute_current_limit_a=np.full(6, 3.0),
        median_window=3, persistence_samples=3,
        contact_joint_indices=(4,),
        hard_relative_joint_indices=(0, 3, 4, 5),
        max_forward_travel_m=0.020, max_press_time_sec=2.0,
    )
    monitor = GuardedPressMonitor(base, guarded)
    values = base.median_a.copy()
    values[1] += 0.70
    for index in range(5):
        stamp = index * 0.005
        result = monitor.update(
            MotorCurrentSample(stamp, values), now=stamp,
            elapsed_sec=stamp, forward_travel_m=0.012,
        )
    assert result.decision is GuardedPressDecision.CONTINUE

    # Exclusion applies only to relative motion current.  The same axis still
    # stops immediately at its absolute electrical safety limit.
    values[1] = 3.1
    absolute = monitor.update(
        MotorCurrentSample(0.030, values), now=0.030,
        elapsed_sec=0.030, forward_travel_m=0.013,
    )
    assert absolute.decision is GuardedPressDecision.HARD_STOP
    assert absolute.reason == "absolute_current_limit"


def test_relative_hard_stop_reports_finite_local_ratio():
    monitor = GuardedPressMonitor(baseline(), config())
    values = baseline().median_a.copy()
    values[1] += 0.45
    result = monitor.update(
        MotorCurrentSample(1.0, values), now=1.0,
        elapsed_sec=0.1, forward_travel_m=0.001,
    )
    assert result.decision is GuardedPressDecision.HARD_STOP
    assert np.isfinite(result.contact_ratio).all()
    assert result.contact_ratio[1] > 1.0
    assert result.trigger_joint_index == 1


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"elapsed_sec": 2.01, "forward_travel_m": 0.001}, "press_timeout"),
        ({"elapsed_sec": 0.1, "forward_travel_m": 0.021}, "maximum_forward_travel"),
    ],
)
def test_hard_geometric_limits(kwargs, reason):
    monitor = GuardedPressMonitor(baseline(), config())
    result = monitor.update(sample(1.0), now=1.0, **kwargs)
    assert result.decision is GuardedPressDecision.HARD_STOP
    assert result.reason == reason


def test_absolute_current_limit_stops_without_persistence():
    monitor = GuardedPressMonitor(baseline(), config())
    values = baseline().median_a.copy()
    values[2] = -3.1
    result = monitor.update(
        MotorCurrentSample(1.0, values), now=1.0,
        elapsed_sec=0.1, forward_travel_m=0.001,
    )
    assert result.decision is GuardedPressDecision.HARD_STOP
    assert result.reason == "absolute_current_limit"


def test_relative_current_limit_stops_without_persistence():
    monitor = GuardedPressMonitor(baseline(), config())
    values = baseline().median_a.copy()
    values[1] += 0.41
    result = monitor.update(
        MotorCurrentSample(1.0, values), now=1.0,
        elapsed_sec=0.1, forward_travel_m=0.001,
    )
    assert result.decision is GuardedPressDecision.HARD_STOP
    assert result.reason == "relative_current_limit"


def test_relative_current_limit_is_armed_by_forward_travel():
    base_config = config()
    armed_config = GuardedPressConfig(
        contact_delta_a=base_config.contact_delta_a,
        hard_delta_a=base_config.hard_delta_a,
        absolute_current_limit_a=base_config.absolute_current_limit_a,
        median_window=3,
        persistence_samples=3,
        contact_arm_travel_m=0.007,
        max_forward_travel_m=0.020,
        max_press_time_sec=2.0,
    )
    values = baseline().median_a.copy()
    values[1] += 0.41

    monitor = GuardedPressMonitor(baseline(), armed_config)
    before_arm = monitor.update(MotorCurrentSample(1.0, values), now=1.0,
                                elapsed_sec=0.1, forward_travel_m=0.0069)
    assert before_arm.decision is GuardedPressDecision.CONTINUE
    assert before_arm.reason == "contact_detector_not_yet_armed"

    # The pre-arm motion current becomes the local reference, so the same
    # current at the boundary is not mistaken for contact.
    after_arm = monitor.update(MotorCurrentSample(1.005, values), now=1.005,
        elapsed_sec=0.105, forward_travel_m=0.007,
    )
    assert after_arm.decision is GuardedPressDecision.CONTINUE

    contact_values = values.copy()
    contact_values[1] += 0.41
    for index in range(3):
        stamp = 1.010 + index * 0.005
        hard_contact = monitor.update(
            MotorCurrentSample(stamp, contact_values), now=stamp,
            elapsed_sec=0.11 + index * 0.005, forward_travel_m=0.007,
        )
    assert hard_contact.decision is GuardedPressDecision.HARD_STOP
    assert hard_contact.reason == "relative_current_limit"


def test_prearm_reference_tracks_free_space_pose_current():
    base_config = config()
    armed_config = GuardedPressConfig(
        contact_delta_a=base_config.contact_delta_a,
        hard_delta_a=base_config.hard_delta_a,
        absolute_current_limit_a=base_config.absolute_current_limit_a,
        median_window=3,
        persistence_samples=3,
        contact_arm_travel_m=0.007,
        max_forward_travel_m=0.020,
        max_press_time_sec=2.0,
    )
    monitor = GuardedPressMonitor(baseline(), armed_config)
    moving = baseline().median_a.copy()
    moving[2] += 0.35
    for index in range(5):
        result = monitor.update(
            MotorCurrentSample(1.0 + index * 0.005, moving),
            now=1.0 + index * 0.005, elapsed_sec=0.1,
            forward_travel_m=0.006,
        )
    assert result.decision is GuardedPressDecision.CONTINUE
    np.testing.assert_allclose(monitor.relative_reference_a, moving)


def test_dynamic_free_space_noise_expands_contact_threshold_at_arm_boundary():
    base_config = config()
    armed_config = GuardedPressConfig(
        contact_delta_a=base_config.contact_delta_a,
        hard_delta_a=base_config.hard_delta_a,
        absolute_current_limit_a=base_config.absolute_current_limit_a,
        noise_sigma_multiplier=5.0,
        median_window=3,
        persistence_samples=3,
        dynamic_reference_window_samples=20,
        contact_arm_travel_m=0.010,
        max_forward_travel_m=0.020,
        max_press_time_sec=2.0,
    )
    monitor = GuardedPressMonitor(baseline(), armed_config)
    for index in range(20):
        moving = baseline().median_a.copy()
        moving[0] += 0.12 if index % 2 else -0.12
        monitor.update(
            MotorCurrentSample(index * 0.005, moving), now=index * 0.005,
            elapsed_sec=index * 0.005, forward_travel_m=0.009,
        )
    for index in range(8):
        moving = baseline().median_a.copy()
        moving[0] += 0.12
        result = monitor.update(
            MotorCurrentSample(0.1 + index * 0.005, moving),
            now=0.1 + index * 0.005, elapsed_sec=0.1 + index * 0.005,
            forward_travel_m=0.010,
        )
    assert result.decision is GuardedPressDecision.CONTINUE
    assert monitor.relative_sigma_a[0] > 0.09


def test_dynamic_noise_envelope_also_prevents_premature_relative_hard_stop():
    base_config = config()
    armed_config = GuardedPressConfig(
        contact_delta_a=base_config.contact_delta_a,
        hard_delta_a=base_config.hard_delta_a,
        absolute_current_limit_a=base_config.absolute_current_limit_a,
        noise_sigma_multiplier=5.0,
        hard_noise_sigma_multiplier=8.0,
        median_window=3,
        persistence_samples=3,
        dynamic_reference_window_samples=20,
        contact_arm_travel_m=0.010,
        max_forward_travel_m=0.020,
        max_press_time_sec=2.0,
    )
    monitor = GuardedPressMonitor(baseline(), armed_config)
    for index in range(20):
        moving = baseline().median_a.copy()
        moving[0] += 0.12 if index % 2 else -0.12
        monitor.update(
            MotorCurrentSample(index * 0.005, moving), now=index * 0.005,
            elapsed_sec=index * 0.005, forward_travel_m=0.009,
        )
    # 0.23 A exceeds a 0.20 A fixed hard delta, but remains ordinary motion
    # variation relative to the measured local sigma and must keep searching.
    moving = monitor.relative_reference_a
    moving[0] += 0.23
    for index in range(5):
        stamp = 0.1 + index * 0.005
        result = monitor.update(
            MotorCurrentSample(stamp, moving), now=stamp,
            elapsed_sec=stamp, forward_travel_m=0.012,
        )
    assert result.decision is GuardedPressDecision.CONTINUE


def test_rolling_reference_tracks_free_motion_drift_without_false_contact():
    base = baseline()
    rolling = GuardedPressConfig(
        contact_delta_a=np.asarray([0.05, 0.10, 0.10, 0.05, 0.08, 0.06]),
        hard_delta_a=np.asarray([0.25, 0.35, 0.35, 0.25, 0.35, 0.30]),
        absolute_current_limit_a=np.full(6, 3.0),
        median_window=5, persistence_samples=5,
        dynamic_reference_window_samples=40,
        contact_arm_travel_m=0.005,
        max_forward_travel_m=0.022, max_press_time_sec=5.0,
    )
    monitor = GuardedPressMonitor(base, rolling)
    drift = np.asarray([0.002, 0.008, -0.004, 0.001, 0.003, -0.002])
    for index in range(180):
        stamp = index * 0.005
        values = base.median_a + index * drift
        result = monitor.update(
            MotorCurrentSample(stamp, values), now=stamp,
            elapsed_sec=stamp, forward_travel_m=min(0.020, index * 0.00012),
        )
        assert result.decision is GuardedPressDecision.CONTINUE
    np.testing.assert_allclose(
        monitor.relative_reference_a, base.median_a + 180 * drift, atol=0.025
    )


def test_rolling_reference_freezes_at_sustained_preimpact_limit_crossing():
    base = baseline()
    rolling = GuardedPressConfig(
        contact_delta_a=np.asarray([0.05, 0.10, 0.10, 0.05, 0.08, 0.06]),
        hard_delta_a=np.asarray([0.25, 0.35, 0.35, 0.25, 0.35, 0.30]),
        absolute_current_limit_a=np.full(6, 3.0),
        median_window=5, persistence_samples=5,
        dynamic_reference_window_samples=40,
        contact_arm_travel_m=0.005,
        max_forward_travel_m=0.022, max_press_time_sec=5.0,
    )
    monitor = GuardedPressMonitor(base, rolling)
    drift = np.asarray([0.0002, 0.0008, -0.0004, 0.0001, 0.0003, -0.0002])
    for index in range(100):
        stamp = index * 0.005
        values = base.median_a + index * drift
        result = monitor.update(
            MotorCurrentSample(stamp, values), now=stamp,
            elapsed_sec=stamp, forward_travel_m=min(0.012, index * 0.00012),
        )
    reference_before_impact = monitor.relative_reference_a.copy()
    for offset in range(20):
        index = 100 + offset
        stamp = index * 0.005
        values = base.median_a + index * drift
        values[4] += 0.14
        result = monitor.update(
            MotorCurrentSample(stamp, values), now=stamp,
            elapsed_sec=stamp, forward_travel_m=0.013 + offset * 0.00005,
        )
        if result.decision is GuardedPressDecision.CONTACT:
            break
    assert result.decision is GuardedPressDecision.CONTACT
    assert result.reason == "sustained_rolling_current_limit"
    assert result.trigger_joint_index == 4
    # The causal reference is the prediction immediately before impact; it
    # does not follow the contact load upward while persistence is counted.
    np.testing.assert_allclose(
        monitor.relative_reference_a, reference_before_impact, atol=0.015
    )


def test_ready_rolling_limit_uses_preimpact_noise_not_old_pose_noise():
    noisy_old_pose = RobustCurrentBaseline(
        baseline().median_a, np.full(6, 0.20), 50
    )
    rolling = GuardedPressConfig(
        contact_delta_a=np.asarray([0.05, 0.10, 0.10, 0.05, 0.08, 0.06]),
        hard_delta_a=np.asarray([0.25, 0.35, 0.35, 0.25, 0.35, 0.30]),
        absolute_current_limit_a=np.full(6, 3.0),
        median_window=5, persistence_samples=5,
        dynamic_reference_window_samples=40,
        contact_arm_travel_m=0.005,
        contact_joint_indices=(4,),
        max_forward_travel_m=0.022, max_press_time_sec=5.0,
    )
    monitor = GuardedPressMonitor(noisy_old_pose, rolling)
    for index in range(80):
        stamp = index * 0.005
        monitor.update(
            MotorCurrentSample(stamp, noisy_old_pose.median_a), now=stamp,
            elapsed_sec=stamp, forward_travel_m=min(0.010, index * 0.00012),
        )
    for offset in range(20):
        index = 80 + offset
        stamp = index * 0.005
        values = noisy_old_pose.median_a.copy()
        values[4] += 0.10
        result = monitor.update(
            MotorCurrentSample(stamp, values), now=stamp,
            elapsed_sec=stamp, forward_travel_m=0.011 + offset * 0.00005,
        )
        if result.decision is GuardedPressDecision.CONTACT:
            break
    assert result.decision is GuardedPressDecision.CONTACT
    assert monitor.relative_sigma_a[4] == pytest.approx(0.005)


def test_short_impact_spike_does_not_satisfy_timed_persistence():
    base = baseline()
    timed = GuardedPressConfig(
        contact_delta_a=np.asarray([0.05, 0.10, 0.10, 0.05, 0.08, 0.06]),
        hard_delta_a=np.asarray([0.25, 0.35, 0.35, 0.25, 0.35, 0.30]),
        absolute_current_limit_a=np.full(6, 3.0),
        median_window=5, persistence_samples=5, persistence_sec=0.10,
        dynamic_reference_window_samples=40,
        contact_arm_travel_m=0.005, contact_joint_indices=(4,),
        max_forward_travel_m=0.022, max_press_time_sec=5.0,
    )
    monitor = GuardedPressMonitor(base, timed)
    for index in range(80):
        stamp = index * 0.005
        monitor.update(
            MotorCurrentSample(stamp, base.median_a), now=stamp,
            elapsed_sec=stamp, forward_travel_m=min(0.010, index * 0.00012),
        )
    # A 50 ms impact spike satisfies the sample count after median filtering,
    # but it must not satisfy the 100 ms continuous-load requirement.
    decisions = []
    for offset in range(10):
        stamp = (80 + offset) * 0.005
        values = base.median_a.copy(); values[4] += 0.14
        result = monitor.update(
            MotorCurrentSample(stamp, values), now=stamp,
            elapsed_sec=stamp, forward_travel_m=0.011,
        )
        decisions.append(result.decision)
    assert GuardedPressDecision.CONTACT not in decisions
    assert result.persistence_sec < 0.10


def test_sustained_load_satisfies_timed_persistence():
    base = baseline()
    timed = GuardedPressConfig(
        contact_delta_a=np.asarray([0.05, 0.10, 0.10, 0.05, 0.08, 0.06]),
        hard_delta_a=np.asarray([0.25, 0.35, 0.35, 0.25, 0.35, 0.30]),
        absolute_current_limit_a=np.full(6, 3.0),
        median_window=5, persistence_samples=5, persistence_sec=0.10,
        dynamic_reference_window_samples=40,
        contact_arm_travel_m=0.005, contact_joint_indices=(4,),
        max_forward_travel_m=0.022, max_press_time_sec=5.0,
    )
    monitor = GuardedPressMonitor(base, timed)
    for index in range(80):
        stamp = index * 0.005
        monitor.update(
            MotorCurrentSample(stamp, base.median_a), now=stamp,
            elapsed_sec=stamp, forward_travel_m=min(0.010, index * 0.00012),
        )
    for offset in range(40):
        stamp = (80 + offset) * 0.005
        values = base.median_a.copy(); values[4] += 0.14
        result = monitor.update(
            MotorCurrentSample(stamp, values), now=stamp,
            elapsed_sec=stamp, forward_travel_m=0.011 + offset * 0.00005,
        )
        if result.decision is GuardedPressDecision.CONTACT:
            break
    assert result.decision is GuardedPressDecision.CONTACT
    assert result.persistence_sec >= 0.10


def test_stale_feedback_stops():
    monitor = GuardedPressMonitor(baseline(), config())
    result = monitor.update(
        sample(1.0), now=1.04, elapsed_sec=0.1, forward_travel_m=0.001
    )
    assert result.decision is GuardedPressDecision.HARD_STOP
    assert result.reason == "stale_current_sample"


def test_retract_trajectory_respects_limits_and_endpoints():
    q0 = np.array([0.1, 0.4, -0.6, 0.2, -0.1, 0.05])
    q1 = q0 + np.array([0.02, -0.03, 0.04, 0.0, 0.01, -0.02])
    vmax = np.full(6, 0.3)
    amax = np.full(6, 1.0)
    points = generate_bounded_retract_trajectory(
        q0, q1, max_velocities=vmax, max_accelerations=amax
    )
    np.testing.assert_allclose(points[0]["positions"], q0)
    np.testing.assert_allclose(points[-1]["positions"], q1)
    assert np.max(np.abs([p["velocities"] for p in points]), axis=0).max() <= 0.3
    assert np.max(np.abs([p["accelerations"] for p in points]), axis=0).max() <= 1.0


def test_scalar_contact_requires_independent_total_force_for_entire_hold():
    from types import SimpleNamespace
    from piper_handeye.guarded_press import NormalForceEstimate
    cfg=GuardedPressConfig(
        contact_delta_a=np.full(6,.05),hard_delta_a=np.full(6,1.0),
        absolute_current_limit_a=np.full(6,3.0),median_window=3,
        persistence_samples=3,persistence_sec=.4,contact_arm_travel_m=0,
        contact_estimator='cartesian_normal_force_kalman',normal_force_threshold_n=.5,
        normal_force_total_floor_n=.5,relative_current_hard_stop_enabled=False,
        absolute_current_hard_stop_enabled=False)
    monitor=GuardedPressMonitor(RobustCurrentBaseline(np.zeros(6),np.full(6,.01),60),cfg,
                                contact_normal_base=np.array([1.,0,0]))
    scalar=NormalForceEstimate(-2.,.1,20.,.9,np.zeros(6),np.ones(6))
    monitor._force_observer=SimpleNamespace(update=lambda **kwargs:scalar)
    force=np.array([-.3,0,0])
    monitor._cartesian_force_observer=SimpleNamespace(update=lambda **kwargs:(force.copy(),np.eye(3)*.001))
    now=1.
    for _ in range(60):
        result=monitor.update(MotorCurrentSample(now,np.zeros(6)),now=now,
                              elapsed_sec=now-1,forward_travel_m=.012,joints_rad=np.zeros(6))
        assert result.decision is GuardedPressDecision.CONTINUE
        now+=.01
    force[0]=-.8
    for _ in range(39):
        result=monitor.update(MotorCurrentSample(now,np.zeros(6)),now=now,
                              elapsed_sec=now-1,forward_travel_m=.014,joints_rad=np.zeros(6))
        assert result.decision is GuardedPressDecision.CONTINUE
        now+=.01
    for _ in range(5):
        result=monitor.update(MotorCurrentSample(now,np.zeros(6)),now=now,
                              elapsed_sec=now-1,forward_travel_m=.014,joints_rad=np.zeros(6))
        now+=.01
    assert result.decision is GuardedPressDecision.CONTACT


def test_invalid_total_force_floor_is_rejected():
    for value in (-.1,float('nan'),float('inf')):
        with pytest.raises(ValueError):
            GuardedPressConfig(contact_delta_a=np.full(6,.05),hard_delta_a=np.full(6,1.),
                               absolute_current_limit_a=np.full(6,3.),normal_force_total_floor_n=value)
