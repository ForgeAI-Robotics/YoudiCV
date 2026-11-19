import numpy as np

from run_clearance_return_to_origin import build_direct_return_trajectory


def test_return_has_no_legacy_elbow_lift_or_six_degree_swing_constraint():
    origin = np.array([0.04, 0.105, 0.0, 0.051, 0.204, 0.0])
    start = np.array([0.40, 0.30, -0.20, 0.10, 0.30, 0.0])
    points, _audit, projection = build_direct_return_trajectory(start, origin)
    positions = np.asarray([point["positions"] for point in points])
    assert np.allclose(projection, 0.0)
    assert np.allclose(positions[0], start)
    assert np.allclose(positions[-1], origin)
    assert np.max(np.abs(positions[:, 0] - origin[0])) > np.deg2rad(6.0)
    phase = (positions[:, 0] - start[0]) / (origin[0] - start[0])
    expected = start + phase[:, None] * (origin - start)[None, :]
    assert np.allclose(positions, expected)
