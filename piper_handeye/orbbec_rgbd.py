from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RgbdFrame:
    color: np.ndarray
    depth_m: np.ndarray


def depth_inside_marker(
    depth_m: np.ndarray,
    corners: np.ndarray,
    min_valid_pixels: int = 200,
) -> tuple[float, float, int]:
    """Robust center-plane depth from the inner 65% of a detected marker."""
    points = np.asarray(corners, dtype=float).reshape(4, 2)
    center = np.mean(points, axis=0)
    inner = center + 0.65 * (points - center)
    mask = np.zeros(depth_m.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(inner).astype(np.int32), 1)
    values = np.asarray(depth_m, dtype=float)[(mask != 0) & np.isfinite(depth_m) & (depth_m > 0.08) & (depth_m < 5.0)]
    if len(values) < min_valid_pixels:
        raise RuntimeError(
            f"only {len(values)} valid aligned depth pixels inside marker; require {min_valid_pixels}"
        )
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad, int(len(values))


class OrbbecRgbdCamera:
    """Synchronized RGB + software D2C-aligned depth using Orbbec SDK v2."""

    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30, flush_frames: int = 8):
        try:
            from pyorbbecsdk import (
                AlignFilter,
                Config,
                Context,
                OBFormat,
                OBFrameAggregateOutputMode,
                OBSensorType,
                OBStreamType,
                Pipeline,
            )
        except ImportError as exc:
            raise RuntimeError("install pyorbbecsdk2 in the piper-handeye environment") from exc

        context = Context()
        devices = context.query_devices()
        if devices.get_count() < 1:
            raise RuntimeError("Orbbec SDK found no accessible RGB-D camera")
        self._pipeline = Pipeline()
        self._config = Config()
        color_profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_profiles.get_video_stream_profile(
                int(width), int(height), OBFormat.RGB, int(fps)
            )
        except Exception:
            color_profile = color_profiles.get_video_stream_profile(
                int(width), int(height), OBFormat.RGB, 0
            )
        depth_profiles = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        self._config.enable_stream(color_profile)
        self._config.enable_stream(depth_profile)
        self._config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )
        try:
            self._pipeline.enable_frame_sync()
        except Exception:
            pass
        self._align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self._pipeline.start(self._config)
        self.width = int(width)
        self.height = int(height)
        self.flush_frames = int(flush_frames)

    @staticmethod
    def _color_to_bgr(frame) -> np.ndarray:
        from pyorbbecsdk import OBFormat

        width, height = frame.get_width(), frame.get_height()
        data = np.asanyarray(frame.get_data())
        fmt = frame.get_format()
        if fmt == OBFormat.RGB:
            return cv2.cvtColor(np.resize(data, (height, width, 3)), cv2.COLOR_RGB2BGR)
        if fmt == OBFormat.BGR:
            return np.resize(data, (height, width, 3)).copy()
        if fmt == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("failed to decode Orbbec MJPG color frame")
            return image
        if fmt == OBFormat.YUYV:
            return cv2.cvtColor(
                np.resize(data, (height, width, 2)), cv2.COLOR_YUV2BGR_YUY2
            )
        raise RuntimeError(f"unsupported Orbbec color format: {fmt}")

    def capture_rgbd(self) -> RgbdFrame:
        frames = None
        for _ in range(max(1, self.flush_frames)):
            raw = self._pipeline.wait_for_frames(1500)
            if raw is None:
                continue
            frames = self._align.process(raw)
        if frames is None:
            raise RuntimeError("Orbbec RGB-D frame acquisition timed out")
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            raise RuntimeError("Orbbec aligned frame set is incomplete")
        color = self._color_to_bgr(color_frame)
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width()
        )
        depth_m = depth.astype(np.float32) * float(depth_frame.get_depth_scale()) / 1000.0
        if color.shape[:2] != (self.height, self.width):
            raise RuntimeError(
                f"Orbbec color resolution {color.shape[1]}x{color.shape[0]} does not match calibrated {self.width}x{self.height}"
            )
        if depth_m.shape != color.shape[:2]:
            raise RuntimeError(
                f"aligned depth shape {depth_m.shape} does not match color {color.shape[:2]}"
            )
        return RgbdFrame(color, depth_m)

    def frame(self) -> np.ndarray:
        return self.capture_rgbd().color

    def close(self) -> None:
        self._pipeline.stop()
