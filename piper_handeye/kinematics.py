from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .robot import PIPER_JOINT_LIMITS
from .pipeline_timing import traced


# Official Piper URDF (firmware >= S-V1.6-3), base_link -> link6.
_ORIGINS = (
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416)),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939)),
    ((-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),
    ((8.8259e-05, -0.091, 0.0), (1.5708, 0.0, 0.0)),
)


def _rotation_matrix(value: Rotation) -> np.ndarray:
    """Support both SciPy's legacy ``as_dcm`` and current ``as_matrix`` API."""
    as_matrix = getattr(value, "as_matrix", None)
    return as_matrix() if callable(as_matrix) else value.as_dcm()


def _origin(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = _rotation_matrix(Rotation.from_euler("xyz", rpy))
    value[:3, 3] = xyz
    return value


_FIXED = tuple(_origin(*item) for item in _ORIGINS)

# The onsite controller's reported pose is 145.44 mm beyond link6 along its Z
# axis (confirmed against the live state stream).  This is the frame used by
# hand-eye acquisition and therefore the frame targeted by IK here.
PIPER_CONTROL_TCP_Z = 0.14544
PIPER_ELBOW_LINK_INDEX = 2


def _rotation_matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Fast SO(3) logarithm with a robust fallback near 180 degrees."""
    value = np.asarray(matrix, dtype=float)
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    skew = np.asarray(
        [
            value[2, 1] - value[1, 2],
            value[0, 2] - value[2, 0],
            value[1, 0] - value[0, 1],
        ],
        dtype=float,
    )
    skew_norm = float(np.linalg.norm(skew))
    angle = float(np.arctan2(0.5 * skew_norm, cosine))
    if angle < 1e-8:
        return 0.5 * skew
    # The skew vector tends to zero at pi, so use SciPy's carefully handled
    # branch only for this rare singular neighbourhood.
    if np.pi - angle < 1e-5 or skew_norm < 1e-10:
        return Rotation.from_matrix(value).as_rotvec()
    return skew * (angle / skew_norm)


def forward_link_frames(joints: np.ndarray) -> np.ndarray:
    """Return base transforms of link1..link6; link3 origin is the elbow."""
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (6,):
        raise ValueError("Piper link FK expects six joint angles")
    result = np.eye(4)
    frames = []
    for fixed, angle in zip(_FIXED, joints):
        rz = np.eye(4)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rz[0, 0] = cosine
        rz[0, 1] = -sine
        rz[1, 0] = sine
        rz[1, 1] = cosine
        result = result @ fixed @ rz
        frames.append(result.copy())
    return np.asarray(frames)


def elbow_position(joints: np.ndarray) -> np.ndarray:
    """Return the joint-3/link-3 origin, i.e. the physical elbow center."""
    return forward_link_frames(joints)[PIPER_ELBOW_LINK_INDEX, :3, 3].copy()


def forward_kinematics(joints: np.ndarray, tcp_z: float = PIPER_CONTROL_TCP_Z) -> np.ndarray:
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (6,):
        raise ValueError("Piper FK expects six joint angles")
    result = np.eye(4)
    for fixed, angle in zip(_FIXED, joints):
        rz = np.eye(4)
        # Every Piper joint rotates about its local Z axis.  Constructing this
        # elementary matrix directly avoids twelve SciPy Rotation objects per
        # numerical-Jacobian evaluation, which dominates online IK latency.
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rz[0, 0] = cosine
        rz[0, 1] = -sine
        rz[1, 0] = sine
        rz[1, 1] = cosine
        result = result @ fixed @ rz
    result[:3, 3] += result[:3, :3] @ np.array([0.0, 0.0, tcp_z])
    return result


def _chain_pose_and_jacobian(joints, link_count, tcp_z):
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (6,):
        raise ValueError("Piper differential FK expects six joint angles")
    result = np.eye(4)
    axes, origins = [], []
    for fixed, angle in zip(_FIXED[:link_count], joints[:link_count]):
        before = result @ fixed
        origins.append(before[:3, 3].copy())
        axes.append(before[:3, 2].copy())
        cosine, sine = np.cos(angle), np.sin(angle)
        rz = np.array([[cosine, -sine, 0., 0.], [sine, cosine, 0., 0.],
                       [0., 0., 1., 0.], [0., 0., 0., 1.]])
        result = before @ rz
    result[:3, 3] += result[:3, :3] @ np.array([0., 0., tcp_z])
    jacobian = np.zeros((6, 6))
    jacobian[:3, :link_count] = np.cross(
        np.asarray(axes), result[:3, 3] - np.asarray(origins)
    ).T
    jacobian[3:, :link_count] = np.asarray(axes).T
    return result, jacobian


def forward_kinematics_with_jacobian(joints, tcp_z=PIPER_CONTROL_TCP_Z):
    """TCP pose and geometric Jacobian, expressed in the base frame."""
    return _chain_pose_and_jacobian(joints, 6, tcp_z)


def rotation_log_left_jacobian_inverse(rotvec):
    """Derivative of Log(exp(delta) R) with respect to a left perturbation."""
    x, y, z = np.asarray(rotvec, dtype=float)
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    theta2 = x*x + y*y + z*z
    if theta2 < 1e-8:
        coefficient = 1./12. + theta2/720. + theta2*theta2/30240.
    else:
        theta = np.sqrt(theta2)
        coefficient = (1. - .5*theta/np.tan(.5*theta)) / theta2
    return np.eye(3) - .5*skew + coefficient*(skew @ skew)


def forward_kinematics_batch(
    joints: np.ndarray, tcp_z: float = PIPER_CONTROL_TCP_Z
) -> np.ndarray:
    """Vectorized equivalent of ``forward_kinematics`` for N joint rows."""
    values = np.asarray(joints, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Piper batch FK expects an Nx6 array")
    count = values.shape[0]
    result = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    for joint_index, fixed in enumerate(_FIXED):
        angles = values[:, joint_index]
        cosine = np.cos(angles)
        sine = np.sin(angles)
        rz = np.zeros((count, 4, 4), dtype=float)
        rz[:, 0, 0] = cosine
        rz[:, 0, 1] = -sine
        rz[:, 1, 0] = sine
        rz[:, 1, 1] = cosine
        rz[:, 2, 2] = 1.0
        rz[:, 3, 3] = 1.0
        result = (result @ fixed) @ rz
    result[:, :3, 3] += result[:, :3, :3] @ np.asarray([0.0, 0.0, tcp_z])
    return result


def elbow_positions_batch(joints: np.ndarray) -> np.ndarray:
    """Vectorized elbow-center positions for an Nx6 joint trajectory."""
    values = np.asarray(joints, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Piper batch elbow FK expects an Nx6 array")
    count = values.shape[0]
    result = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    for joint_index, fixed in enumerate(_FIXED[: PIPER_ELBOW_LINK_INDEX + 1]):
        angles = values[:, joint_index]
        cosine = np.cos(angles)
        sine = np.sin(angles)
        rz = np.zeros((count, 4, 4), dtype=float)
        rz[:, 0, 0] = cosine
        rz[:, 0, 1] = -sine
        rz[:, 1, 0] = sine
        rz[:, 1, 1] = cosine
        rz[:, 2, 2] = 1.0
        rz[:, 3, 3] = 1.0
        result = (result @ fixed) @ rz
    return result[:, :3, 3].copy()


@traced('ik.inverse_kinematics', 'ik')
def inverse_kinematics(
    target: np.ndarray,
    seed: np.ndarray,
    limits: np.ndarray = PIPER_JOINT_LIMITS,
    regularization: float = 0.01,
    minimum_elbow_z_m: float | None = None,
    max_nfev: int = 500,
    solver_tolerance: float = 1e-12,
    fixed_joint_indices: tuple[int, ...] = (),
) -> tuple[np.ndarray, float, float]:
    target = np.asarray(target, dtype=float)
    seed = np.asarray(seed, dtype=float)
    limits = np.asarray(limits, dtype=float)
    fixed_indices = tuple(sorted({int(value) for value in fixed_joint_indices}))
    if any(value < 0 or value >= 6 for value in fixed_indices):
        raise ValueError("fixed joint indices must be in 0..5")
    free_indices = np.asarray(
        [index for index in range(6) if index not in fixed_indices], dtype=int
    )
    if free_indices.size == 0:
        raise ValueError("inverse kinematics requires at least one free joint")
    if int(max_nfev) < 20:
        raise ValueError("max_nfev must be at least 20")
    if not 1e-12 <= float(solver_tolerance) <= 1e-4:
        raise ValueError("solver_tolerance must be in 1e-12..1e-4")

    cached = {}

    def evaluate_free(free_q: np.ndarray):
        if "q" in cached and np.array_equal(cached["q"], free_q):
            return cached["residual"], cached["jacobian"]
        q = seed.copy()
        q[free_indices] = free_q
        actual, geometric = forward_kinematics_with_jacobian(q)
        position = (actual[:3, 3] - target[:3, 3]) / 0.01
        rotvec = _rotation_matrix_to_rotvec(
            target[:3, :3].T @ actual[:3, :3]
        )
        rotation = rotvec / np.deg2rad(2.0)
        stay_near = np.sqrt(regularization) * (q - seed)
        terms = [position, rotation, stay_near]
        jacobians = [geometric[:3] / .01,
                     rotation_log_left_jacobian_inverse(rotvec)
                     @ target[:3, :3].T @ geometric[3:] / np.deg2rad(2.),
                     np.sqrt(regularization) * np.eye(6)]
        if minimum_elbow_z_m is not None:
            # A strong one-sided residual guides the solver away from IK
            # branches that lower the elbow while raising the eye-in-hand
            # camera.  Callers still verify every sampled path point.
            elbow, elbow_jac = _chain_pose_and_jacobian(q, 3, 0.)
            elbow_violation = max(0., float(minimum_elbow_z_m) - float(elbow[2, 3]))
            terms.append(np.asarray([elbow_violation / 0.0002]))
            jacobians.append((-elbow_jac[2:3] / .0002)
                             if elbow_violation > 0 else np.zeros((1, 6)))
        residual = np.concatenate(terms)
        jacobian = np.vstack(jacobians)[:, free_indices]
        cached.update(q=free_q.copy(), residual=residual, jacobian=jacobian)
        return residual, jacobian

    def residual_free(free_q):
        return evaluate_free(free_q)[0]

    def jacobian_free(free_q):
        return evaluate_free(free_q)[1]

    solution = least_squares(
        residual_free,
        np.clip(
            seed[free_indices],
            limits[free_indices, 0] + 1e-5,
            limits[free_indices, 1] - 1e-5,
        ),
        bounds=(
            limits[free_indices, 0] + 1e-5,
            limits[free_indices, 1] - 1e-5,
        ),
        jac=jacobian_free,
        max_nfev=int(max_nfev),
        xtol=float(solver_tolerance),
        ftol=float(solver_tolerance),
        gtol=float(solver_tolerance),
    ).x
    full_solution = seed.copy()
    full_solution[free_indices] = solution
    solution = full_solution
    actual = forward_kinematics(solution)
    position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    angle_error = float(np.linalg.norm(_rotation_matrix_to_rotvec(
        target[:3, :3].T @ actual[:3, :3]
    )))
    return solution, position_error, angle_error


def relative_target(base_pose: np.ndarray, translation_base=(0.0, 0.0, 0.0), rotation_tool=(0.0, 0.0, 0.0)) -> np.ndarray:
    target = np.array(base_pose, dtype=float, copy=True)
    target[:3, 3] += np.asarray(translation_base, dtype=float)
    target[:3, :3] = target[:3, :3] @ _rotation_matrix(
        Rotation.from_rotvec(rotation_tool)
    )
    return target
