#!/usr/bin/env python3
"""Record the existing vision service RGB stream without opening the camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

import cv2
import yaml

from piper_handeye.vision_rpc import VisionRpcClient


running = True


def stop_recording(*_):
    global running
    running = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()
    if not 1.0 <= args.fps <= 15.0:
        raise ValueError("fps must be in 1..15")

    config = yaml.safe_load(args.config.read_text())
    rpc = config["vision_service"]
    client = VisionRpcClient(
        rpc["host"], int(rpc["port"]), rpc["token"], timeout=2.0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial" + args.output.suffix)
    metadata_path = args.output.with_suffix(".json")
    writer = None
    frames = []
    errors = []
    last_sequence = -1
    started_unix = time.time()
    period = 1.0 / args.fps
    signal.signal(signal.SIGTERM, stop_recording)
    signal.signal(signal.SIGINT, stop_recording)
    try:
        while running:
            cycle = time.monotonic()
            try:
                response = client.request(
                    "light_snapshot", after_sequence=last_sequence,
                    target_label=str(args.target_label),
                )
                if response.get("fresh"):
                    sequence = int(response["frame_sequence"])
                    image = cv2.imread(str(response["path"]))
                    if image is None:
                        raise RuntimeError("recording snapshot cannot be read")
                    if writer is None:
                        height, width = image.shape[:2]
                        writer = cv2.VideoWriter(
                            str(partial), cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps, (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("MP4 writer failed to open")
                    writer.write(image)
                    last_sequence = sequence
                    frames.append({
                        "video_frame": len(frames),
                        "camera_sequence": sequence,
                        "capture_time_unix_sec": response.get(
                            "capture_time_unix_sec"
                        ),
                        "recorded_time_unix_sec": time.time(),
                    })
            except Exception as exc:
                errors.append({
                    "time_unix_sec": time.time(),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                errors[:] = errors[-50:]
            delay = period - (time.monotonic() - cycle)
            if delay > 0:
                time.sleep(delay)
    finally:
        if writer is not None:
            writer.release()
            partial.replace(args.output)
        metadata_path.write_text(json.dumps({
            "output": str(args.output),
            "target_label": str(args.target_label),
            "fps": args.fps,
            "started_unix_sec": started_unix,
            "ended_unix_sec": time.time(),
            "frame_count": len(frames),
            "frames": frames,
            "errors": errors,
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
