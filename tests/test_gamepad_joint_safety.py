from tools.gamepad_joint_safety import allowed_by_soft_limit


def test_joint_at_upper_physical_limit_can_move_into_soft_interval():
    assert allowed_by_soft_limit(0.0, -0.009, -2.947, -0.020)
    assert not allowed_by_soft_limit(0.0, 0.009, -2.947, -0.020)


def test_joint_at_lower_margin_can_only_move_inward():
    assert allowed_by_soft_limit(-2.96, -2.95, -2.947, -0.020)
    assert not allowed_by_soft_limit(-2.96, -2.97, -2.947, -0.020)


def test_joint_inside_interval_stays_inside():
    assert allowed_by_soft_limit(-1.0, -1.1, -2.947, -0.020)
    assert allowed_by_soft_limit(-1.0, -0.9, -2.947, -0.020)
    assert not allowed_by_soft_limit(-0.021, -0.010, -2.947, -0.020)
