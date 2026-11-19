from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import average_transforms, matrix_to_vector, transform, vector_to_matrix
from .vision import Detection, require_cv2


@dataclass
class CalibrationResult:
    gripper_to_camera: np.ndarray
    base_to_target: np.ndarray
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rms_px: float
    per_view_rms_px: list[float]
    condition_number: float
    samples: int
    corners: int


def calibrate_intrinsics(detections: list[Detection], image_size: tuple[int, int]):
    cv2 = require_cv2()
    object_points = [d.object_points.astype(np.float32) for d in detections]
    image_points = [d.image_points.astype(np.float32) for d in detections]
    # Five coefficients are more stable for this 7x5 board coverage.  The
    # rational model produced large mutually-cancelling coefficients on the
    # onsite wide-angle camera despite only a tiny RMS improvement.
    flags = 0
    rms, camera, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags
    )
    return float(rms), camera, distortion


def board_poses(detections: list[Detection], camera: np.ndarray, distortion: np.ndarray):
    cv2 = require_cv2()
    output = []
    for detection in detections:
        ok, rvec, tvec = cv2.solvePnP(
            detection.object_points, detection.image_points, camera, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed for an accepted calibration sample")
        output.append(transform(cv2.Rodrigues(rvec)[0], tvec))  # camera <- target
    return output


def initialize_handeye(base_to_gripper: list[np.ndarray], camera_to_target: list[np.ndarray]):
    cv2 = require_cv2()
    rotations_g = [value[:3, :3] for value in base_to_gripper]
    translations_g = [value[:3, 3] for value in base_to_gripper]
    rotations_t = [value[:3, :3] for value in camera_to_target]
    translations_t = [value[:3, 3] for value in camera_to_target]
    candidates = []
    for method in (cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_DANIILIDIS):
        try:
            rotation, translation = cv2.calibrateHandEye(
                rotations_g, translations_g, rotations_t, translations_t, method=method
            )
            value = transform(rotation, translation)
            if np.isfinite(value).all():
                targets = [g @ value @ t for g, t in zip(base_to_gripper, camera_to_target)]
                center = average_transforms(targets)
                score = sum(np.linalg.norm((np.linalg.inv(center) @ t)[:3, 3]) for t in targets)
                candidates.append((score, value, center))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("both Park and Daniilidis hand-eye initializers failed")
    _, x, z = min(candidates, key=lambda item: item[0])
    return x, z


def optimize_reprojection(
    detections: list[Detection],
    base_to_gripper: list[np.ndarray],
    camera: np.ndarray,
    distortion: np.ndarray,
    initial_x: np.ndarray,
    initial_z: np.ndarray,
) -> CalibrationResult:
    cv2 = require_cv2()
    initial = np.r_[matrix_to_vector(initial_x), matrix_to_vector(initial_z)]

    def view_residuals(parameters: np.ndarray, index: int) -> np.ndarray:
        x, z = vector_to_matrix(parameters[:6]), vector_to_matrix(parameters[6:12])
        camera_to_target = np.linalg.inv(base_to_gripper[index] @ x) @ z
        rvec, _ = cv2.Rodrigues(camera_to_target[:3, :3])
        projected, _ = cv2.projectPoints(
            detections[index].object_points, rvec, camera_to_target[:3, 3], camera, distortion
        )
        return (projected.reshape(-1, 2) - detections[index].image_points).ravel()

    def residuals(parameters: np.ndarray) -> np.ndarray:
        return np.concatenate([view_residuals(parameters, i) for i in range(len(detections))])

    fit = least_squares(residuals, initial, method="trf", loss="huber", f_scale=1.0,
                        x_scale="jac", max_nfev=500)
    if not fit.success:
        raise RuntimeError(f"joint reprojection optimization failed: {fit.message}")
    per_view = [float(np.sqrt(np.mean(view_residuals(fit.x, i) ** 2)))
                for i in range(len(detections))]
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], np.finfo(float).eps))
    all_residuals = residuals(fit.x)
    return CalibrationResult(
        gripper_to_camera=vector_to_matrix(fit.x[:6]),
        base_to_target=vector_to_matrix(fit.x[6:12]),
        camera_matrix=camera,
        distortion=np.asarray(distortion),
        rms_px=float(np.sqrt(np.mean(all_residuals ** 2))),
        per_view_rms_px=per_view,
        condition_number=condition,
        samples=len(detections),
        corners=sum(len(d.ids) for d in detections),
    )
