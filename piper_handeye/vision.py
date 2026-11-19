from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run: pip install -r requirements-handeye.txt"
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("opencv-contrib-python is required for ChArUco support")
    return cv2


@dataclass
class Detection:
    image: np.ndarray
    image_points: np.ndarray
    object_points: np.ndarray
    ids: np.ndarray
    sharpness: float


class Camera:
    def __init__(self, config: dict[str, Any]):
        cv2 = require_cv2()
        source = config.get("source", 0)
        if isinstance(source, str) and source.isdecimal():
            source = int(source)
        backend_name = str(config.get("backend", "any")).lower()
        backend = {
            "any": cv2.CAP_ANY,
            "v4l2": cv2.CAP_V4L2,
            "gstreamer": cv2.CAP_GSTREAMER,
            "ffmpeg": cv2.CAP_FFMPEG,
        }.get(backend_name)
        if backend is None:
            raise ValueError(f"unsupported camera backend: {backend_name}")
        self.capture = cv2.VideoCapture(source, backend)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open camera source {source!r}")
        for key, prop in (("width", cv2.CAP_PROP_FRAME_WIDTH), ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                          ("fps", cv2.CAP_PROP_FPS), ("exposure", cv2.CAP_PROP_EXPOSURE)):
            if key in config:
                self.capture.set(prop, float(config[key]))
        self.flush_frames = int(config.get("flush_frames", 8))

    def close(self) -> None:
        self.capture.release()

    def frame(self) -> np.ndarray:
        frame = None
        for _ in range(max(1, self.flush_frames)):
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError("camera frame acquisition failed")
        assert frame is not None
        return frame


class CharucoTarget:
    def __init__(self, config: dict[str, Any]):
        cv2 = require_cv2()
        dictionary_name = str(config.get("dictionary", "DICT_5X5_100"))
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"unknown ArUco dictionary {dictionary_name}")
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        sx, sy = int(config["squares_x"]), int(config["squares_y"])
        square = float(config["square_length_m"])
        marker = float(config["marker_length_m"])
        try:
            self.board = cv2.aruco.CharucoBoard((sx, sy), square, marker, self.dictionary)
        except TypeError:  # OpenCV 4.6 and older Python API
            self.board = cv2.aruco.CharucoBoard_create(sx, sy, square, marker, self.dictionary)
        self.min_corners = int(config.get("min_corners", 12))
        self.min_sharpness = float(config.get("min_sharpness", 60.0))
        try:
            params = cv2.aruco.DetectorParameters()
        except TypeError:
            params = cv2.aruco.DetectorParameters_create()
        self.params = params

    def _corners(self) -> np.ndarray:
        if hasattr(self.board, "getChessboardCorners"):
            return np.asarray(self.board.getChessboardCorners(), dtype=np.float32)
        return np.asarray(self.board.chessboardCorners, dtype=np.float32)

    def detect(self, image: np.ndarray) -> Detection | None:
        cv2 = require_cv2()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.params
        )
        if marker_ids is None or len(marker_ids) < 2:
            return None
        cv2.aruco.refineDetectedMarkers(gray, self.board, marker_corners, marker_ids, [])
        count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.board
        )
        if charuco_ids is None or int(count) < self.min_corners:
            return None
        points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
        ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.min_sharpness:
            return None
        return Detection(image, points, self._corners()[ids], ids, sharpness)

    def acquire_best(self, camera: Camera, attempts: int = 6) -> Detection | None:
        best = None
        for _ in range(attempts):
            detected = self.detect(camera.frame())
            if detected is not None and (best is None or len(detected.ids) > len(best.ids) or
                                         (len(detected.ids) == len(best.ids) and detected.sharpness > best.sharpness)):
                best = detected
            time.sleep(0.03)
        return best
