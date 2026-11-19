from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import piper_handeye.kinematics as kin
from piper_handeye.realtime_planner import _FlexiblePoseObjective


def numerical(fun, x, step=1e-6):
    eye = np.eye(len(x))*step
    return np.column_stack([(fun(x+v)-fun(x-v))/(2*step) for v in eye])


def test_geometric_jacobian_matches_central_differences():
    rng = np.random.default_rng(914)
    for _ in range(30):
        q = rng.uniform(kin.PIPER_JOINT_LIMITS[:, 0]+.05, kin.PIPER_JOINT_LIMITS[:, 1]-.05)
        pose, jac = kin.forward_kinematics_with_jacobian(q)
        np.testing.assert_allclose(pose, kin.forward_kinematics(q), atol=1e-14)
        def local(x):
            p = kin.forward_kinematics(x)
            return np.r_[p[:3, 3], Rotation.from_matrix(p[:3, :3]@pose[:3, :3].T).as_rotvec()]
        np.testing.assert_allclose(jac, numerical(local, q), atol=2e-8, rtol=2e-6)


def test_flexible_nine_variable_jacobian_matches_central_differences():
    rng = np.random.default_rng(33)
    for _ in range(30):
        q = rng.uniform(kin.PIPER_JOINT_LIMITS[:, 0]+.05, kin.PIPER_JOINT_LIMITS[:, 1]-.05)
        objective = _FlexiblePoseObjective(Rotation.random(random_state=rng).as_matrix(),
                                          rng.uniform(-.3, .3, 3), q, np.zeros(3))
        value = np.r_[q, rng.uniform(-.6, .6, 3)]
        np.testing.assert_allclose(objective.jacobian(value), numerical(objective, value),
                                   atol=3e-6, rtol=3e-5)


def test_fixed_joint_and_elbow_residual_jacobian():
    original = kin.least_squares
    checks = []
    def inspect(fun, x, **kw):
        np.testing.assert_allclose(kw['jac'](x), numerical(fun, x), atol=3e-6, rtol=3e-5)
        checks.append(True)
        return original(fun, x, **kw)
    seed = np.array([.04, .6, -.8, .1, .2, 0.])
    goal = seed.copy(); goal[1] += .03
    with patch.object(kin, 'least_squares', inspect):
        result, position, angle = kin.inverse_kinematics(
            kin.forward_kinematics(goal), seed, fixed_joint_indices=(0,),
            minimum_elbow_z_m=kin.elbow_position(seed)[2]+.001,
        )
    assert checks and result[0] == seed[0]
    assert position < .001 and angle < np.deg2rad(.5)
