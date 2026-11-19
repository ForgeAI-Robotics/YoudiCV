"""Bounded press tolerance on derived geometry, without robot I/O.

The fixture combines historical XYZ/joints and a later panel orientation.
It demonstrates a solver boundary; it is not an exact historical RGB-D replay.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from piper_handeye.kinematics import forward_kinematics
from piper_handeye.realtime_planner import (
    FixedRollFastIkPlanner,
    PRESS_SEGMENT_MAX_JOINT_DELTA_RAD,
)
from piper_handeye.robot import PIPER_JOINT_LIMITS
from run_online_replan_no_contact import (
    refine_precomputed_ik,
    solve_independent_press_ik,
)


DATA = json.loads((Path(__file__).parent / 'fixtures' / 'press_joint_tolerance'
                   / 'derived_geometry.json').read_text())


def solve_case(case, limit):
    return FixedRollFastIkPlanner(
        roll_deg=DATA['roll_deg'], stand_off_m=DATA['guard_distance_m'],
        max_joint_delta_rad=1.0, max_perpendicular_deviation_deg=35.0,
        downstream_base_x_distances_m=(.01, -.012),
        downstream_max_segment_joint_deltas_rad=(None, limit),
    ).solve(np.array(case['base_to_button']), np.array(case['seed_joints']))


@pytest.mark.parametrize('case', DATA['cases'], ids=lambda c: c['source_run'])
def test_guard_preview_and_actual_press_solve_accept_the_same_geometry(case):
    result = solve_case(case, PRESS_SEGMENT_MAX_JOINT_DELTA_RAD)
    prepress = result.downstream_preview_solutions[0].joints
    target = np.array(case['base_to_button'])[:3, 3].copy()
    target[0] += .012
    pressed = solve_independent_press_ik(prepress, target)
    assert pressed.max_joint_delta_rad <= PRESS_SEGMENT_MAX_JOINT_DELTA_RAD
    assert np.all(pressed.joints >= PIPER_JOINT_LIMITS[:, 0])
    assert np.all(pressed.joints <= PIPER_JOINT_LIMITS[:, 1])
    assert pressed.orientation_error_deg <= .5
    np.testing.assert_allclose(forward_kinematics(pressed.joints)[:3, 3], target,
                               atol=.001)
    # Per-waypoint continuity is still bounded separately from the stroke.
    assert result.downstream_preview_max_joint_step_rad <= .25


def test_borderline_press_was_rejected_by_legacy_cap_but_passes_shared_cap():
    case = DATA['cases'][1]
    seed = np.array(case['prepress_joints'])
    target = np.array(case['base_to_button'])[:3, 3].copy()
    target[0] += .012
    with pytest.raises(RuntimeError, match='exceeds total joint delta'):
        refine_precomputed_ik(forward_kinematics(seed), target, seed,
                              max_joint_delta_rad=.25)
    result = solve_independent_press_ik(seed, target)
    assert .25 < result.max_joint_delta_rad <= PRESS_SEGMENT_MAX_JOINT_DELTA_RAD


def test_more_demanding_press_is_still_rejected_by_bounded_total_change():
    case = DATA['cases'][1]
    seed = np.array(case['prepress_joints'])
    target = forward_kinematics(seed)[:3, 3].copy()
    target[0] += .024
    with pytest.raises(RuntimeError, match='exceeds total joint delta'):
        solve_independent_press_ik(seed, target)


def test_previously_accepted_geometry_keeps_the_same_selected_pose():
    case = DATA['cases'][0]
    old = solve_case(case, .25)
    new = solve_case(case, PRESS_SEGMENT_MAX_JOINT_DELTA_RAD)
    np.testing.assert_allclose(new.joints, old.joints, atol=1e-10)
    np.testing.assert_allclose(new.orientation_offset_deg,
                               old.orientation_offset_deg, atol=1e-10)
    for before, after in zip(old.downstream_preview_solutions,
                             new.downstream_preview_solutions):
        np.testing.assert_allclose(after.joints, before.joints, atol=1e-10)
