from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import transform
from .vision import require_cv2


@dataclass(frozen=True)
class ArucoPoseDetection:
    camera_to_marker: np.ndarray
    corners: np.ndarray
    reprojection_rms_px: float
    side_length_px: float
    image: np.ndarray


class ArucoPoseTarget:
    """Single-marker 6D pose detector with an explicit right-handed frame.

    Marker frame: origin at center, +X printed right, +Y printed up and +Z
    out of the printed face toward an observing camera.
    """

    def __init__(
        self,
        dictionary_name: str,
        marker_id: int,
        marker_length_m: float,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        min_side_px: float = 100.0,
        max_reprojection_rms_px: float = 1.0,
    ):
        cv2 = require_cv2()
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"unknown ArUco dictionary {dictionary_name}")
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dictionary_name)
        )
        self.marker_id = int(marker_id)
        self.marker_length_m = float(marker_length_m)
        self.camera_matrix = np.asarray(camera_matrix, dtype=float)
        self.distortion = np.asarray(distortion, dtype=float)
        self.min_side_px = float(min_side_px)
        self.max_reprojection_rms_px = float(max_reprojection_rms_px)
        self.object_points = self._object_points(self.marker_length_m)
        try:
            self.params = cv2.aruco.DetectorParameters()
        except TypeError:
            self.params = cv2.aruco.DetectorParameters_create()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementWinSize = 7
        self.params.cornerRefinementMaxIterations = 50
        self.params.cornerRefinementMinAccuracy = 0.01

    @staticmethod
    def _object_points(length: float) -> np.ndarray:
        half = length / 2.0
        # Required order for SOLVEPNP_IPPE_SQUARE: TL, TR, BR, BL.
        return np.array(
            [[-half, half, 0], [half, half, 0],
             [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )

    def detect(self, image: np.ndarray) -> ArucoPoseDetection | None:
        cv2 = require_cv2()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.params
        )
        if ids is None:
            return None
        matches = np.flatnonzero(np.asarray(ids).reshape(-1) == self.marker_id)
        if len(matches) != 1:
            return None
        points = np.asarray(corners[int(matches[0])], dtype=np.float32).reshape(4, 2)
        sides = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        side_px = float(np.mean(sides))
        if side_px < self.min_side_px:
            return None
        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            points,
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok or float(np.asarray(tvec).reshape(3)[2]) <= 0:
            return None
        projected, _ = cv2.projectPoints(
            self.object_points, rvec, tvec, self.camera_matrix, self.distortion
        )
        residual = projected.reshape(4, 2) - points
        rms = float(np.sqrt(np.mean(residual ** 2)))
        if rms > self.max_reprojection_rms_px:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        # With the declared corner order, a visible front face has marker +Z
        # pointing approximately toward the camera (camera -Z direction).
        if float(rotation[2, 2]) >= -0.1:
            return None
        return ArucoPoseDetection(
            transform(rotation, tvec), points, rms, side_px, image
        )


def marker_to_facing_tcp(stand_off_m: float) -> np.ndarray:
    """Marker <- TCP: TCP above marker, with TCP +Z pointing at its face."""
    if stand_off_m < 0.08:
        raise ValueError("stand-off below 0.08 m is unsafe for automatic alignment")
    return transform(Rotation.from_euler("x", np.pi).as_matrix(), [0, 0, stand_off_m])


def pose_error(actual: np.ndarray, desired: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(actual) @ desired
    position_m = float(np.linalg.norm(delta[:3, 3]))
    angle_rad = float(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()))
    return position_m, angle_rad
