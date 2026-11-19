from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .calibration import (
    CalibrationResult,
    board_poses,
    calibrate_intrinsics,
    initialize_handeye,
    optimize_reprojection,
)
from .robot import PIPER_JOINT_LIMITS, PiperUdpRobot
from .vision import Camera, CharucoTarget, require_cv2


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def validate_config(config: dict[str, Any]) -> list[np.ndarray]:
    for section in ("robot", "camera", "target", "sampling", "output"):
        if section not in config:
            raise ValueError(f"missing configuration section: {section}")
    poses = [np.asarray(p, dtype=float) for p in config["sampling"].get("joint_poses_rad", [])]
    if len(poses) < 3 or any(p.shape != (6,) for p in poses):
        raise ValueError("sampling.joint_poses_rad must contain at least three 6-axis poses")
    limits = np.asarray(config["robot"].get("joint_limits_rad", PIPER_JOINT_LIMITS), dtype=float)
    for index, pose in enumerate(poses):
        if not np.isfinite(pose).all() or np.any(pose < limits[:, 0]) or np.any(pose > limits[:, 1]):
            raise ValueError(f"sampling pose {index + 1} exceeds configured Piper limits")
    return poses


def matrix_list(value: np.ndarray) -> list[list[float]]:
    return np.asarray(value, dtype=float).tolist()


def save_result(path: Path, result: CalibrationResult, kept_indices: list[int]) -> None:
    value = {
        "convention": {
            "gripper_to_camera": "p_gripper = T_gripper_camera * p_camera",
            "base_to_target": "p_base = T_base_target * p_target",
            "quaternion_order": "xyzw",
            "length_unit": "metre",
        },
        "gripper_to_camera": matrix_list(result.gripper_to_camera),
        "base_to_target": matrix_list(result.base_to_target),
        "camera_matrix": matrix_list(result.camera_matrix),
        "distortion": np.asarray(result.distortion).reshape(-1).tolist(),
        "quality": {
            "joint_reprojection_rms_px": result.rms_px,
            "per_view_rms_px": result.per_view_rms_px,
            "jacobian_condition_number": result.condition_number,
            "samples": result.samples,
            "corners": result.corners,
            "kept_original_indices": kept_indices,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)


def fit(detections, transforms, camera_config, image_size):
    intrinsics = camera_config.get("intrinsics")
    if intrinsics:
        camera = np.asarray(intrinsics["camera_matrix"], dtype=float)
        distortion = np.asarray(intrinsics.get("distortion", []), dtype=float)
        intrinsic_rms = None
    else:
        intrinsic_rms, camera, distortion = calibrate_intrinsics(detections, image_size)
    pnp = board_poses(detections, camera, distortion)
    x, z = initialize_handeye(transforms, pnp)
    result = optimize_reprojection(detections, transforms, camera, distortion, x, z)
    return intrinsic_rms, result


def reject_outliers(detections, transforms, camera_config, image_size, max_view_rms):
    indices = list(range(len(detections)))
    intrinsic_rms, result = fit(detections, transforms, camera_config, image_size)
    errors = np.asarray(result.per_view_rms_px)
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    robust_limit = median + 3.5 * max(1.4826 * mad, 0.05)
    limit = min(float(max_view_rms), robust_limit)
    keep = [i for i, error in enumerate(errors) if error <= limit]
    if len(keep) >= 6 and len(keep) < len(detections):
        intrinsic_rms, result = fit(
            [detections[i] for i in keep], [transforms[i] for i in keep], camera_config, image_size
        )
        indices = keep
    return intrinsic_rms, result, indices


def plan(config: dict[str, Any], poses: list[np.ndarray]) -> None:
    robot = config["robot"]
    print("Piper Eye-in-Hand calibration plan")
    print(f"  controller : {robot['controller_ip']}:{robot.get('controller_port', 5001)}")
    print(f"  local UDP  : {robot['local_ip']}:{robot.get('result_port', 5002)} / state {robot.get('state_port', 5004)}")
    print(f"  camera     : {config['camera'].get('source', 0)!r}")
    print(f"  poses      : {len(poses)}")
    print(f"  output     : {config['output']['directory']}")
    for index, pose in enumerate(poses, 1):
        print(f"  {index:02d}: " + " ".join(f"{x:+.4f}" for x in pose))
    print("Dry run only. Add --execute after independently checking every pose in simulation/RViz.")


def execute(config: dict[str, Any], poses: list[np.ndarray]) -> Path:
    if config["sampling"].get("safety_acknowledged") is not True:
        raise RuntimeError("set sampling.safety_acknowledged: true only after checking every pose")
    cv2 = require_cv2()
    output_dir = Path(config["output"]["directory"]).expanduser().resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    target = CharucoTarget(config["target"])
    camera = Camera(config["camera"])
    detections = []
    transforms = []
    metadata = []
    sampling = config["sampling"]
    min_samples = int(sampling.get("min_samples", 12))
    settle = float(sampling.get("settle_time_sec", 1.0))
    attempts = int(sampling.get("capture_attempts", 6))

    try:
        with PiperUdpRobot(config["robot"]) as robot:
            state = robot.state(max_age=2.0)
            print("Piper state online; joints:", np.round(state.joints, 5).tolist())
            if sampling.get("enable_before_run", True):
                robot.enable()
                print("Piper enabled")
            for index, pose in enumerate(poses, 1):
                print(f"[{index}/{len(poses)}] moving to calibration pose")
                robot.move_joints(pose)
                time.sleep(settle)
                detected = target.acquire_best(camera, attempts=attempts)
                sample = robot.state(max_age=1.0)
                if detected is None:
                    print("  rejected: target missing, too few corners, or image blurred")
                    continue
                image_path = images_dir / f"sample_{len(detections):03d}.png"
                if not cv2.imwrite(str(image_path), detected.image):
                    raise RuntimeError(f"failed to save {image_path}")
                detections.append(detected)
                transforms.append(sample.base_to_gripper)
                metadata.append({
                    "candidate_index": index - 1,
                    "image": str(image_path),
                    "joints_rad": sample.joints.tolist(),
                    "base_to_gripper": matrix_list(sample.base_to_gripper),
                    "corner_ids": detected.ids.tolist(),
                    "sharpness": detected.sharpness,
                })
                print(f"  accepted: {len(detected.ids)} corners, sharpness={detected.sharpness:.1f}")
            if sampling.get("return_home_after_run", False):
                robot.home()
                print("Piper returned home")
    finally:
        camera.close()

    if len(detections) < min_samples:
        raise RuntimeError(f"only {len(detections)} valid samples; require at least {min_samples}")
    height, width = detections[0].image.shape[:2]
    intrinsic_rms, result, kept = reject_outliers(
        detections, transforms, config["camera"], (width, height),
        sampling.get("max_view_rms_px", 1.5),
    )
    if len(kept) < min_samples:
        raise RuntimeError(
            f"robust fit kept only {len(kept)} views; require at least {min_samples}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "samples.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    result_path = output_dir / "handeye.yaml"
    save_result(result_path, result, kept)
    print(f"Calibration complete: RMS={result.rms_px:.4f}px, samples={result.samples}")
    if intrinsic_rms is not None:
        print(f"Camera intrinsic RMS={intrinsic_rms:.4f}px")
    print(f"Result: {result_path}")
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Automatic eye-in-hand calibration for Piper/RK3588")
    parser.add_argument("--config", type=Path, default=Path("piper_handeye.yaml"))
    parser.add_argument("--execute", action="store_true", help="allow physical robot motion")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        poses = validate_config(config)
        if args.execute:
            execute(config, poses)
        else:
            plan(config, poses)
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
