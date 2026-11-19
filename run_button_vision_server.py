#!/usr/bin/env python3
"""Persistent Orbbec/YOLO service for the OrangePi ROS proxy and local clients."""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import socketserver
import threading
import time

import numpy as np
import cv2
import yaml

from piper_handeye.retention import prune_capture_groups
from scipy.spatial.transform import Rotation

from piper_handeye.button_pose import YoloButtonDetector, select_button
from piper_handeye.depth_safety import DepthPanelSafetyMonitor, DepthSafetyConfig
from piper_handeye.geometry import average_transforms
from piper_handeye.live_button import LiveButtonPerception
from piper_handeye.perception_activity import PerceptionActivity
from piper_handeye.pipeline_timing import mark, interval
from piper_handeye.orbbec_v1_stream import OrbbecV1Stream
from piper_handeye.numeric_layout_localization import NumericLayoutLocalizer
from piper_handeye.numeric_yolo_consistency import target_conflicted, LOCAL_35_CHECK_VERSION


ROOT = Path(__file__).resolve().parent


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_handeye(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return (
        np.asarray(data["gripper_to_camera"], dtype=float),
        np.asarray(data["camera_matrix"], dtype=float),
        np.asarray(data["distortion"], dtype=float),
    )


def scaled_camera_matrix(matrix: np.ndarray, calibrated_size, capture_size):
    old_w, old_h = map(float, calibrated_size)
    new_w, new_h = map(float, capture_size)
    sx, sy = new_w / old_w, new_h / old_h
    if abs(sx - sy) > 1e-9:
        raise RuntimeError(
            "capture aspect ratio differs from calibrated RGB aspect ratio"
        )
    scaled = np.asarray(matrix, dtype=float).copy()
    scaled[0, :] *= sx
    scaled[1, :] *= sy
    return scaled


def numeric_consistency_payload(sample):
    record=getattr(sample,'numeric_yolo_consistency',None)
    if not isinstance(record,dict) or not record:return None
    if (record.get('frame_sequence')!=sample.sequence or sample.frame.sequence!=sample.sequence
            or record.get('target_label')!=sample.target_label):
        raise RuntimeError('numeric consistency frame/target mismatch')
    return deepcopy(record)


def layout_localization_payload(sample):
    """Read-only shadow evidence from exactly this completed inference frame."""
    record=getattr(sample,'numeric_layout_localization',None) if sample is not None else None
    if not isinstance(record,dict) or not record:return None
    if (sample.sequence!=sample.frame.sequence
            or record.get('frame_sequence')!=sample.sequence
            or record.get('rgbd_bundle_sequence')!=sample.sequence
            or record.get('target_label')!=sample.target_label):
        raise RuntimeError('layout localization bundle has inconsistent frame/target')
    out=deepcopy(record)
    age=time.time()-float(sample.frame.received_unix)
    out['sample_age_sec']=age
    out['fresh']=bool(-.05<=age<=.35)
    out['geometry_valid_3d_at_processing']=bool(out.get('valid_3d'))
    out['valid_3d']=bool(out.get('valid_3d') and out['fresh'])
    out['motion_eligible']=False
    return out


def pose_payload(sample) -> dict:
    pose = sample.pose
    assert pose is not None
    matrix = pose.camera_to_button
    return {
        "sequence": sample.sequence,
        "target_label": sample.target_label,
        "received_monotonic": sample.frame.received_monotonic,
        "capture_time_unix_sec": sample.frame.received_unix,
        "sample_age_sec": max(0.0, time.time() - sample.frame.received_unix),
        "color_timestamp": sample.frame.color_timestamp,
        "depth_timestamp": sample.frame.depth_timestamp,
        "confidence": pose.detection.confidence,
        "bbox_xyxy": list(pose.detection.bbox_xyxy),
        "p_camera_m": pose.p_camera.tolist(),
        "normal_camera": pose.normal_camera.tolist(),
        "camera_to_button": matrix.tolist(),
        "orientation_xyzw": Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
        "plane_rms_m": pose.plane_rms_m,
        "plane_inliers": pose.plane_inliers,
        "panel_plane_fit": getattr(pose, "panel_plane_fit", None),
        "panel_support_detections": pose.panel_support_detections,
        "local_global_depth_delta_m": pose.local_global_depth_delta_m,
        "processing_sec": sample.processing_sec,
        "timing": getattr(sample, "timing", {}),
        "detection_enhancement": getattr(sample, "detection_enhancement", {}),
        "numeric_yolo_consistency": numeric_consistency_payload(sample),
    }


def realtime_payload(perception, label: str, window_size: int = 6) -> dict:
    samples = perception.valid_window(label, window_size)
    if not samples:
        return {
            "window_size": 0,
            "requested_window_size": int(window_size),
            "stable": False,
            "position_rms_m": None,
            "sample_rate_hz": 0.0,
            "latest_age_sec": None,
        }
    positions = np.asarray([sample.pose.p_camera for sample in samples])
    center = np.mean(positions, axis=0)
    rms = float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))
    span = samples[-1].frame.received_monotonic - samples[0].frame.received_monotonic
    rate = (len(samples) - 1) / span if len(samples) > 1 and span > 0 else 0.0
    latest_age = max(0.0, time.time() - samples[-1].frame.received_unix)
    return {
        "window_size": len(samples),
        "requested_window_size": int(window_size),
        "stable": len(samples) >= int(window_size) and rms <= 0.003,
        "position_rms_m": rms,
        "mean_position_m": center.tolist(),
        "sample_rate_hz": float(rate),
        "latest_age_sec": latest_age,
        "first_sequence": samples[0].sequence,
        "latest_sequence": samples[-1].sequence,
    }


class VisionService:
    def __init__(self, args):
        self.args = args
        self.started_monotonic = time.monotonic()
        self.collect_lock = threading.Lock()
        config_path = args.config.expanduser().resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        depth_cfg = config["depth"]
        detector_cfg = config["detector"]
        self.tcp_to_camera, full_matrix, _ = load_handeye(
            resolve_path(config["calibration"]["handeye_yaml"], config_path)
        )
        camera_matrix = scaled_camera_matrix(
            full_matrix,
            config["calibration"].get("rgb_calibration_size", [1920, 1080]),
            tuple(depth_cfg.get("capture_size", [640, 360])),
        )
        self.numeric_layout_localizer=NumericLayoutLocalizer.from_config(
            config.get('numeric_layout_localization'),config_path,camera_matrix,
            config.get('button_geometry',{}),float(detector_cfg.get('target_confidence',.70)))
        self.camera = OrbbecV1Stream(
            resolve_path(depth_cfg["stream_executable"], config_path),
            resolve_path(depth_cfg["sdk_work_directory"], config_path),
        )
        self.numeric_layout_task_settings=dict(config.get('numeric_layout_localization') or {})
        camera_start = time.perf_counter()
        self.camera.start()
        self.camera_start_sec = time.perf_counter() - camera_start
        model_start = time.perf_counter()
        self.detector = YoloButtonDetector(
            resolve_path(detector_cfg["model_path"], config_path),
            conf=float(detector_cfg.get("inference_confidence", 0.15)),
            iou=float(detector_cfg.get("iou", 0.5)),
            imgsz=int(detector_cfg.get("imgsz", 640)),
            device=str(detector_cfg.get("device", "auto")),
        )
        self.model_load_sec = time.perf_counter() - model_start
        self.activity = PerceptionActivity()
        self.perception = LiveButtonPerception(
            self.camera,
            self.detector,
            camera_matrix,
            target_label=args.target_label,
            target_confidence=float(detector_cfg.get("target_confidence", 0.70)),
            geometry_options=dict(config.get("button_geometry", {})),
            enhancement_fallback_enabled=bool(detector_cfg.get("enhancement_fallback_enabled", False)),
            activity=self.activity,
            numeric_layout_localizer=self.numeric_layout_localizer,
            layout_before_local_yolo_enabled=(
                self.numeric_layout_task_settings.get('task_fallback_enabled') is True),
        )
        self.perception.start()
        safety_cfg = dict(config.get("visual_safety", {}))
        if safety_cfg.get("hazard_snapshot_directory"):
            safety_cfg["hazard_snapshot_directory"] = str(
                resolve_path(safety_cfg["hazard_snapshot_directory"], config_path)
            )
        self.safety = DepthPanelSafetyMonitor(
            self.camera,
            lambda: self.perception.latest_sample(),
            DepthSafetyConfig(**safety_cfg),
            camera_matrix=camera_matrix,
            numeric_layout_task_settings=self.numeric_layout_task_settings,
        )
        self.safety.start(args.target_label)

    def dispatch(self, request: dict) -> dict:
        if request.get("token") != self.args.token:
            raise PermissionError("invalid vision service token")
        command = request.get("command")
        activity = getattr(self, "activity", None)
        if activity is not None and command in {
            "set_target", "set_perception_target", "latest", "visible_buttons",
            "collect", "light_snapshot", "safety_capture_matrix_reference",
            "safety_disable_matrix_reference", "safety_set_depth_degraded_commit",
            "safety_motion_state", "safety_set_mode", "safety_camera_pose",
            "layout_localization_snapshot",
        }:
            activity.touch()
        if command == "health":
            latest = self.perception.latest_valid()
            processed = self.perception.latest_sample()
            light = self.perception.latest_light_sample()
            return {
                "ok": True,
                "inference_activity": activity.status() if activity is not None else None,
                "detection_enhancement": {} if processed is None else getattr(processed, "detection_enhancement", {}),
                "status": "ready",
                "uptime_sec": time.monotonic() - self.started_monotonic,
                "camera_sequence": self.camera.latest_sequence,
                "target_label": self.perception.target_label,
                "latest_valid_sequence": latest.sequence if latest else None,
                "realtime": realtime_payload(
                    self.perception, self.perception.target_label
                ),
                "camera_start_sec": self.camera_start_sec,
                "camera_diagnostics": list(self.camera.diagnostics),
                "model_load_sec": self.model_load_sec,
                "perception_timing_version": 1,
                "light_publication": dict(version=1,yolo_2d_before_depth=True,
                    pairing='same_inference_frame',motion_cache='completed_rgbd_only',
                    inferred_roi=('layout_2d_before_depth' if getattr(self.perception,
                        'layout_before_local_yolo_enabled', False) is True
                        else 'completed_same_frame_fallback'),
                    latest_sequence=None if light is None else int(light.sequence),
                    latest_stage=None if light is None else light.publication_stage,
                    latest_age_sec=None if light is None else time.time()-light.frame.received_unix,
                    latest_completed_sequence=None if processed is None else int(processed.sequence)),
                "numeric_yolo_consistency": dict(version=1,
                    local_35_triplet_check_version=LOCAL_35_CHECK_VERSION,
                    enabled=getattr(self,'numeric_layout_task_settings',{}).get('yolo_consistency_enabled') is True,
                    profile_sha256=getattr(self,'numeric_layout_task_settings',{}).get('profile_sha256')),
                "numeric_layout_task_handoff": dict(version=1,
                    enabled=getattr(self,'numeric_layout_task_settings',{}).get('task_fallback_enabled') is True,
                    profile_sha256=getattr(self,'numeric_layout_task_settings',{}).get('profile_sha256')),
                "numeric_layout_before_local_yolo": dict(version=1,
                    enabled=getattr(self.perception,'layout_before_local_yolo_enabled',False) is True,
                    scope='configured_numeric_targets_only',max_rgb_age_sec=.35,
                    motion_cache='completed_rgbd_only'),
                "numeric_layout_localization": (
                    self.numeric_layout_localizer.status()
                    if isinstance(getattr(self,'numeric_layout_localizer',None),NumericLayoutLocalizer)
                    else {'mode':'off','motion_eligible':False}),
                "tcp_to_camera": {
                    "position_m": self.tcp_to_camera[:3, 3].tolist(),
                    "orientation_xyzw": Rotation.from_matrix(
                        self.tcp_to_camera[:3, :3]
                    ).as_quat().tolist(),
                    "matrix": self.tcp_to_camera.tolist(),
                },
                "visual_safety": self.safety.status(),
            }
        if command == "set_target":
            barrier = self.perception.set_target(str(request["target_label"]))
            self.safety.reset(str(request["target_label"]))
            return {"ok": True, "barrier_sequence": barrier}
        if command == "set_perception_target":
            # During an upward visual search the depth-matrix guard may be
            # anchored to another visible button.  Change only the detector
            # label so that the already captured safety model stays armed.
            barrier = self.perception.set_target(str(request["target_label"]))
            return {"ok": True, "barrier_sequence": barrier}
        if command == "safety":
            return {"ok": True, "visual_safety": self.safety.status()}
        if command == "safety_pose_timing":
            return {"ok": True, "evidence": self.safety.pose_timing_evidence()}
        if command == "safety_probe":
            frame = self.camera.latest_frame()
            return {
                "ok": True,
                "frame_sequence": int(frame.sequence),
                "probes": self.safety.probe_roi_scales(
                    frame, request.get("roi_scales", [3.2, 5.0, 7.0, 9.0])
                ),
            }
        if command == "depth_failure_snapshot":
            # Explicit diagnostic only: reuse one immutable failed RGB-D bundle.
            # Never invoked by the ordinary task or the background producer.
            label = str(request.get("target_label", self.perception.target_label))
            sample = self.perception.latest_sample(label)
            if sample is None or sample.sequence <= int(request.get("after_sequence", -1)):
                raise RuntimeError("no new processed depth-failure sample")
            failures = [x for x in sample.timing.get("spans", [])
                        if x.get("name") == "vision.depth_pose" and x.get("status") == "error"]
            if sample.pose is not None or not failures or not sample.error:
                raise RuntimeError("sample is not a recorded depth-stage failure")
            frame = sample.frame
            if sample.sequence != frame.sequence or not -.05 <= time.time()-frame.received_unix <= .35:
                raise RuntimeError("depth-failure RGB-D bundle is stale or mismatched")
            directory = ROOT / "artifacts/captures/depth_failure_frames" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            directory.mkdir(parents=True, exist_ok=False)
            bundle = directory / "same_frame_rgbd.npz"
            np.savez_compressed(bundle, color=frame.color.copy(), depth_m=frame.depth_m.copy(),
                                camera_matrix=np.asarray(self.perception.camera_matrix).copy())
            metadata = dict(frame_sequence=int(sample.sequence), target_label=label,
                capture_time_unix_sec=float(frame.received_unix),
                color_timestamp=int(frame.color_timestamp), depth_timestamp=int(frame.depth_timestamp),
                error_message=sample.error, error_type=failures[-1].get("error_type"),
                geometry_options=dict(self.perception.geometry_options),
                detections=[dict(label=x.label, confidence=float(x.confidence), bbox_xyxy=list(x.bbox_xyxy))
                            for x in sample.detections], timing=sample.timing,
                pairing="same_completed_rgbd_bundle", raw_pixels_unmodified=True)
            metadata_path = directory / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return dict(ok=True, frame_sequence=int(sample.sequence), target_label=label,
                        error_message=sample.error, bundle_path=str(bundle), metadata_path=str(metadata_path))
        if command == "snapshot":
            frame = self.camera.latest_frame()
            output_dir = ROOT / "camera_snapshots"
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = output_dir / f"{stem}.jpg"
            depth_mm_path = output_dir / f"{stem}_depth_mm.png"
            depth_color_path = output_dir / f"{stem}_depth_color.png"
            if not cv2.imwrite(
                str(output_path), frame.color, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError("failed to save camera snapshot")
            depth = np.asarray(frame.depth_m, dtype=float)
            depth_mm = np.zeros(depth.shape, dtype=np.uint16)
            valid = np.isfinite(depth) & (depth > 0.0)
            depth_mm[valid] = np.clip(
                np.rint(depth[valid] * 1000.0), 0, 65535
            ).astype(np.uint16)
            if not cv2.imwrite(str(depth_mm_path), depth_mm):
                raise RuntimeError("failed to save 16-bit depth snapshot")
            depth_color = self.safety._colorize_depth(depth)
            if not cv2.imwrite(str(depth_color_path), depth_color):
                raise RuntimeError("failed to save colorized depth snapshot")
            prune_capture_groups(output_dir)
            height, width = frame.color.shape[:2]
            return {
                "ok": True,
                "path": str(output_path),
                "depth_mm_path": str(depth_mm_path),
                "depth_color_path": str(depth_color_path),
                "frame_sequence": int(frame.sequence),
                "capture_time_unix_sec": float(frame.received_unix),
                "image_size": [int(width), int(height)],
            }
        if command == 'layout_localization_snapshot':
            label=str(request.get('target_label',self.perception.target_label))
            sample=self.perception.latest_sample(label)
            if sample is None or sample.sequence<=int(request.get('after_sequence',-1)):
                return dict(ok=True,fresh=False,reason='waiting_for_new_inference_frame')
            evidence=layout_localization_payload(sample)
            if evidence is None:
                return dict(ok=True,fresh=False,reason='layout_localization_disabled_or_unavailable')
            response=dict(ok=True,fresh=evidence['fresh'],frame_sequence=int(sample.sequence),
                          capture_time_unix_sec=float(sample.frame.received_unix),
                          valid_3d=evidence['valid_3d'],localization=evidence)
            if request.get('include_rgb',False):
                ok,encoded=cv2.imencode('.jpg',sample.frame.color,[cv2.IMWRITE_JPEG_QUALITY,90])
                if not ok:raise RuntimeError('failed to encode paired layout RGB')
                response['jpeg_base64']=base64.b64encode(encoded).decode('ascii')
            return response
        if command == "recognition_snapshot":
            from piper_handeye.detection_enhancement import detection_payload, enhance_detection_image
            from piper_handeye.recognition_image import draw_recognition
            sample = self.perception.latest_sample()
            if sample is None:
                raise RuntimeError("no processed YOLO frame available")
            directory = ROOT / "recognition_snapshots"
            directory.mkdir(parents=True, exist_ok=True)
            stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            audit = getattr(sample, "detection_enhancement", {})
            raw = audit.get("raw_detections", detection_payload(sample.detections))
            frame = sample.frame
            response = {"ok": True, "frame_sequence": int(sample.sequence),
                        "capture_time_unix_sec": float(frame.received_unix),
                        "target_label": str(sample.target_label),
                        "detection_enhancement": audit, "detections": detection_payload(sample.detections)}
            layout=layout_localization_payload(sample)
            if layout is not None:response['numeric_layout_localization']=layout
            consistency=numeric_consistency_payload(sample)
            if consistency is not None:response['numeric_yolo_consistency']=consistency
            images = {"input_path": frame.color,
                      "raw_yolo_path": draw_recognition(frame.color, raw,
                          caption=f"raw seq={sample.sequence}", target_label=sample.target_label)}
            if (audit.get("source") in {"call_tile", "numeric_tile"}
                    or audit.get('algorithm') == 'numeric_panel_local_scale'
                    or (str(sample.target_label) == 'down'
                        and audit.get('algorithm') == 'call_button_local_scale')):
                local_candidates = audit.get('tile_detections',detection_payload(sample.detections))
                caption = f"local retry seq={sample.sequence} accepted={audit.get('accepted')}"
                if str(sample.target_label) == 'down':
                    local_candidates = audit.get('local_candidates',local_candidates)
                    caption += f" target_accepted={audit.get('down_recovery',{}).get('target_accepted')}"
                images["tile_yolo_path"] = draw_recognition(
                    frame.color, local_candidates, caption=caption, target_label=sample.target_label)
            if "enhanced_detections" in audit:
                enhanced = enhance_detection_image(frame.color)
                images["enhanced_yolo_path"] = draw_recognition(
                    enhanced, audit["enhanced_detections"],
                    caption=f"enhanced seq={sample.sequence}", target_label=sample.target_label)
            for key, image in images.items():
                destination = directory / f"{stem}_{key}.jpg"
                if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"failed to save recognition image: {key}")
                response[key] = str(destination)
            prune_capture_groups(directory)
            return response
        if command == "preview":
            # Lightweight GUI preview: return only the latest RGB frame, with
            # no disk writes and no additional camera or inference worker.
            frame = self.camera.latest_frame()
            height, width = frame.color.shape[:2]
            # Keep the sensor's native 640x360 RGB resolution. The GUI scales
            # it to the available camera pane, whose size varies by display.
            preview = frame.color
            ok, encoded = cv2.imencode(
                ".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 68]
            )
            if not ok:
                raise RuntimeError("failed to encode camera preview")
            return {
                "ok": True,
                "frame_sequence": int(frame.sequence),
                "capture_time_unix_sec": float(frame.received_unix),
                "image_size": [int(width), int(height)],
                "jpeg_base64": base64.b64encode(encoded).decode("ascii"),
            }
        if command == "light_snapshot":
            # Ordinary admitted YOLO results need not await depth. The cache
            # still owns the exact RGB input, never camera.latest_frame().
            label = str(request.get("target_label", self.perception.target_label))
            sample = self.perception.latest_light_sample(label)
            after_sequence = int(request.get("after_sequence", -1))
            if sample is None or int(sample.sequence) <= after_sequence:
                return {"ok": True, "fresh": False,
                        "frame_sequence": None if sample is None else int(sample.sequence),
                        "reason": "waiting_for_new_inference_frame"}
            frame = sample.frame
            if int(frame.sequence) != int(sample.sequence):
                raise RuntimeError("inference bundle has inconsistent frame sequence")
            output_dir = ROOT / "button_light_snapshots"
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = output_dir / f"{stem}.jpg"
            if not cv2.imwrite(
                str(output_path), frame.color, [cv2.IMWRITE_JPEG_QUALITY, 90]
            ):
                raise RuntimeError("failed to save button-light RGB snapshot")
            prune_capture_groups(output_dir)
            response = {
                "ok": True,
                "fresh": True,
                "path": str(output_path),
                "frame_sequence": int(frame.sequence),
                "capture_time_unix_sec": float(frame.received_unix),
                "image_size": [
                    int(frame.color.shape[1]), int(frame.color.shape[0])
                ],
            }
            response["pairing"] = "same_inference_frame"
            response['light_publication_stage']=getattr(sample,'publication_stage','completed_rgbd')
            response['depth_complete']=response['light_publication_stage']=='completed_rgbd'
            response['sample_age_at_response_sec']=time.time()-float(frame.received_unix)
            layout=layout_localization_payload(sample)
            if layout is not None:response['numeric_layout_localization']=layout
            consistency=numeric_consistency_payload(sample)
            if consistency is not None:response['numeric_yolo_consistency']=consistency
            response["perception_timing"] = getattr(sample, "timing", {})
            try:
                if target_conflicted(numeric_consistency_payload(sample),target=label,sequence=sample.sequence):
                    raise RuntimeError('numeric_yolo_layout_conflict')
                detection = select_button(list(sample.detections), label, 0.70)
            except RuntimeError:
                detection = None
            if detection is not None:
                response["target_detection"] = {
                    "label": label,
                    "bbox_xyxy": list(detection.bbox_xyxy),
                    "confidence": float(detection.confidence),
                    "frame_sequence": int(frame.sequence),
                }
            return response
        if command == "safety_capture_matrix_reference":
            started = time.perf_counter()
            timing = {}
            event = {"event": "safety_reference_build", "status": "failed"}
            try:
                frames = self.camera.recent_frames(int(request.get("frame_count", 25)))
                selected = time.perf_counter()
                timing["select_frames"] = selected - started
                stack = np.stack([frame.depth_m for frame in frames])
                stacked = time.perf_counter()
                timing["stack_frames"] = stacked - selected
                audit = self.safety.capture_matrix_reference(stack)
                timing["capture_reference"] = time.perf_counter() - stacked
                event.update(status="complete", frame_count=len(frames),
                             last_frame_sequence=int(frames[-1].sequence), **audit)
                return {"ok": True, **audit, "request_timing_sec": timing}
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                timing["total"] = time.perf_counter() - started
                event["request_timing_sec"] = timing
                print(json.dumps(event, ensure_ascii=False), flush=True)
        if command == "safety_disable_matrix_reference":
            self.safety.disable_matrix_reference()
            return {"ok": True}
        if command == "safety_set_depth_degraded_commit":
            return {
                "ok": True,
                **self.safety.set_depth_degraded_commit(bool(request["enabled"])),
            }
        if command == "safety_motion_state":
            paused = bool(request["paused"])
            self.safety.set_motion_paused(paused)
            return {
                "ok": True,
                "paused": paused,
                "visual_safety": self.safety.status(),
            }
        if command == "safety_set_mode":
            return {"ok": True, **self.safety.set_mode(str(request["mode"]))}
        if command == "safety_camera_pose":
            evidence = self.safety.update_camera_pose(
                float(request["timestamp_monotonic"]),
                np.asarray(request["base_to_camera"], dtype=float),
            )
            return {"ok": True, "pose_timing": evidence}
        if command == "latest":
            label = str(request.get("target_label", self.perception.target_label))
            sample = self.perception.latest_valid(label)
            processed = self.perception.latest_sample(label)
            window_size = int(request.get("window_size", 6))
            response = {
                "ok": True,
                "server_time_unix_sec": time.time(),
                "sample": pose_payload(sample) if sample else None,
            }
            layout=layout_localization_payload(processed)
            if layout is not None:response['numeric_layout_localization']=layout
            consistency=numeric_consistency_payload(processed)
            if consistency is not None:response['numeric_yolo_consistency']=consistency
            # Diagnostic-only: expose failed/no-target processing as well as
            # valid 3D samples, without promoting it to a motion target.
            if processed is not None and isinstance(getattr(processed, 'timing', None), dict):
                response['latest_processed_timing'] = dict(
                    sequence=processed.sequence, timing=processed.timing,
                    pose_valid=processed.pose is not None)
            if request.get("include_realtime", True):
                response["realtime"] = realtime_payload(self.perception, label, window_size)
            if "after_sequence" in request:
                # Bounded cache delivery; the consumer still validates age and
                # matches each observation to its own encoder timestamp.
                barrier = int(request["after_sequence"])
                max_age = float(request.get("max_batch_age_sec", float("inf")))
                now = time.time()
                response["samples"] = [
                    pose_payload(item)
                    for item in self.perception.valid_window(label, 6)
                    if item.sequence > barrier
                    and -0.05 <= now - item.frame.received_unix <= max_age
                ]
            return response
        if command == "visible_buttons":
            sample = self.perception.latest_sample()
            detections = [] if sample is None else sample.detections
            frame=sample.frame if sample is not None else self.camera.latest_frame()
            response = {
                "ok": True,
                "detection_enhancement": {} if sample is None else getattr(sample, "detection_enhancement", {}),
                "frame_sequence": None if sample is None else int(sample.sequence),
                "capture_time_unix_sec": (
                    None if sample is None else float(sample.frame.received_unix)
                ),
                "image_size": [
                    int(frame.color.shape[1]),
                    int(frame.color.shape[0]),
                ],
                "detections": [
                    {
                        "label": item.label,
                        "confidence": float(item.confidence),
                        "bbox_xyxy": list(item.bbox_xyxy),
                        "center_xy": list(item.center_pixel),
                    }
                    for item in detections
                    if item.confidence
                    >= float(request.get("min_confidence", 0.50))
                ],
            }
            layout=layout_localization_payload(sample)
            if layout is not None:response['numeric_layout_localization']=layout
            consistency=numeric_consistency_payload(sample)
            if consistency is not None:response['numeric_yolo_consistency']=consistency
            return response
        if command == "collect":
            with self.collect_lock:
                label = str(request.get("target_label", self.perception.target_label))
                barrier = self.perception.set_target(label)
                batch = self.perception.collect_after(
                    barrier,
                    valid_results=int(request.get("valid_results", 6)),
                    max_attempts=int(request.get("max_attempts", 12)),
                    timeout=float(request.get("timeout_sec", 8.0)),
                )
                transforms = [item.pose.camera_to_button for item in batch.accepted]
                averaged = average_transforms(transforms)
                positions = np.asarray([item.pose.p_camera for item in batch.accepted])
                center = np.mean(positions, axis=0)
                rms = float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))
                return {
                    "ok": True,
                    "target_label": label,
                    "attempts": batch.attempts,
                    "rejected": list(batch.rejected),
                    "collection_sec": batch.collection_sec,
                    "sequence_range": [batch.first_sequence, batch.last_sequence],
                    "average": {
                        "position_m": averaged[:3, 3].tolist(),
                        "orientation_xyzw": Rotation.from_matrix(
                            averaged[:3, :3]
                        ).as_quat().tolist(),
                        "camera_to_button": averaged.tolist(),
                        "position_rms_m": rms,
                    },
                    "samples": [pose_payload(item) for item in batch.accepted],
                }
        raise ValueError(f"unsupported command: {command!r}")

    def close(self):
        self.safety.close()
        self.perception.close()
        self.camera.close()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        dispatch_started = None
        try:
            raw = self.rfile.readline(1024 * 1024)
            request = json.loads(raw)
            dispatch_started = mark()
            response = self.server.vision.dispatch(request)
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if dispatch_started is not None:
            response['server_request_timing'] = interval('vision.server_dispatch',
                'server', dispatch_started, status='ok' if response.get('ok') else 'error',
                scope='dispatch only; excludes response serialization and socket transfer')
        self.wfile.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5020)
    parser.add_argument("--token", required=True)
    parser.add_argument("--target-label", default="4")
    args = parser.parse_args()
    vision = VisionService(args)
    server = Server((args.bind, args.port), Handler)
    server.vision = vision
    print(
        f"button vision service ready on {args.bind}:{args.port}; "
        f"camera={vision.camera_start_sec:.3f}s model={vision.model_load_sec:.3f}s",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        vision.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
