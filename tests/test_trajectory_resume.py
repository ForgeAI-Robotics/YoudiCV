import importlib.util
from pathlib import Path

import numpy as np

from piper_handeye.timed_trajectory import generate_timed_joint_trajectory


spec = importlib.util.spec_from_file_location(
    "piper_trajectory_resume",
    Path(__file__).parents[1] / "tools" / "piper_trajectory_resume.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rejoin_starts_held_and_matches_future_original_path():
    start = np.asarray([0.1, 0.4, -0.5, 0.1, 0.0, 0.0])
    end = np.asarray([0.3, 0.8, -1.0, 0.4, 0.2, 0.1])
    points, _ = generate_timed_joint_trajectory(start, end)
    original = np.asarray([point["positions"] for point in points])
    stopped = original[len(points) // 3] + np.asarray([0.001, 0, 0, 0, 0, 0])
    resumed, audit = module.generate_rejoin_trajectory(
        points,
        stopped,
        elapsed_sec=points[len(points) // 3]["time_from_start_sec"],
        max_velocities=np.asarray([0.45, 0.75, 0.75, 0.75, 0.75, 1.5]),
        max_accelerations=np.full(6, 2.0),
    )
    assert np.allclose(resumed[0]["positions"], stopped)
    assert np.allclose(resumed[0]["velocities"], 0.0)
    assert np.allclose(resumed[-1]["positions"], points[-1]["positions"])
    assert audit["join_position_error_rad"] < 1e-9
    assert audit["join_velocity_error_rad_s"] < 1e-8
    assert max(audit["peak_connector_acceleration_rad_s2"]) <= 2.0 + 1e-8


def test_rejoin_uses_elapsed_time_to_disambiguate_return_path():
    outbound, _ = generate_timed_joint_trajectory(
        np.asarray([0.1, 0.4, -0.5, 0.0, 0.0, 0.0]),
        np.asarray([0.2, 0.7, -0.8, 0.2, 0.1, 0.0]),
    )
    duration = outbound[-1]["time_from_start_sec"]
    loop = list(outbound)
    for point in reversed(outbound[:-1]):
        loop.append(
            {
                "time_from_start_sec": 2 * duration - point["time_from_start_sec"],
                "positions": point["positions"],
                "velocities": (-np.asarray(point["velocities"])).tolist(),
                "accelerations": point["accelerations"],
            }
        )
    stopped = np.asarray(outbound[len(outbound) // 2]["positions"])
    _, audit = module.generate_rejoin_trajectory(
        loop,
        stopped,
        elapsed_sec=1.5 * duration,
        max_velocities=np.asarray([0.45, 0.75, 0.75, 0.75, 0.75, 1.5]),
        max_accelerations=np.full(6, 2.0),
    )
    assert audit["nearest_index"] >= len(outbound)
def test_only_tiny_goal_tolerance_violation_is_accepted():
    assert module.accept_near_goal_tolerance_violation(-5, 0.000195)
    assert not module.accept_near_goal_tolerance_violation(-5, 0.004)
    assert not module.accept_near_goal_tolerance_violation(-4, 0.000195)

