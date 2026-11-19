from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _rotation_matrix(value: Rotation) -> np.ndarray:
    """Support both SciPy's legacy ``as_dcm`` and current ``as_matrix`` API."""
    as_matrix = getattr(value, "as_matrix", None)
    return as_matrix() if callable(as_matrix) else value.as_dcm()


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    out[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return out


def pose_to_transform(pose: dict) -> np.ndarray:
    p = pose["position"]
    q = pose["orientation"]
    rotation = _rotation_matrix(
        Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])
    )
    return transform(rotation, [p["x"], p["y"], p["z"]])


def matrix_to_vector(value: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(value[:3, :3]).as_rotvec(), value[:3, 3]]


def vector_to_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return transform(_rotation_matrix(Rotation.from_rotvec(value[:3])), value[3:6])


def rotation_angle(rotation: np.ndarray) -> float:
    return float(Rotation.from_matrix(rotation).magnitude())


def average_transforms(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("cannot average an empty transform list")
    quats = np.stack([Rotation.from_matrix(v[:3, :3]).as_quat() for v in values])
    reference = quats[0]
    quats[(quats @ reference) < 0] *= -1
    scatter = quats.T @ quats
    quat = np.linalg.eigh(scatter)[1][:, -1]
    return transform(
        _rotation_matrix(Rotation.from_quat(quat)),
        np.mean([v[:3, 3] for v in values], axis=0),
    )
