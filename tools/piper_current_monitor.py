#!/usr/bin/env python3
"""Read-only Piper 0x251..0x256 current telemetry from SocketCAN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import struct
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from piper_handeye.guarded_press import PiperCurrentFrameAssembler


CAN_FRAME = struct.Struct("=IB3x8s")


def collect(interface: str, duration_sec: float, samples_output: str | None = None) -> dict:
    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((str(interface),))
    sock.settimeout(0.1)
    assembler = PiperCurrentFrameAssembler()
    samples = []
    started = time.monotonic()
    sample_file = None
    if samples_output:
        sample_path = Path(samples_output)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_file = sample_path.open("w", encoding="utf-8")
    try:
        while time.monotonic() - started < duration_sec:
            try:
                packet = sock.recv(CAN_FRAME.size)
            except socket.timeout:
                continue
            can_id, length, payload = CAN_FRAME.unpack(packet)
            can_id &= socket.CAN_EFF_MASK
            if not 0x251 <= can_id <= 0x256 or length != 8:
                continue
            sample = assembler.add(can_id, payload, time.monotonic())
            if sample is not None:
                samples.append(sample.currents_a)
                if sample_file is not None:
                    sample_file.write(json.dumps({
                        "monotonic_sec": sample.timestamp,
                        "wall_time_unix_sec": time.time(),
                        "currents_a": sample.currents_a.tolist(),
                    }, separators=(",", ":")) + "\n")
    finally:
        sock.close()
        if sample_file is not None:
            sample_file.close()
    if not samples:
        raise RuntimeError("no complete Piper current samples received")
    values = np.asarray(samples, dtype=float)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    elapsed = time.monotonic() - started
    return {
        "interface": str(interface),
        "duration_sec": elapsed,
        "complete_sample_count": len(values),
        "sample_rate_hz": len(values) / elapsed,
        "median_current_a": median.tolist(),
        "mad_current_a": mad.tolist(),
        "minimum_current_a": np.min(values, axis=0).tolist(),
        "maximum_current_a": np.max(values, axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--duration-sec", type=float, default=1.0)
    parser.add_argument("--output")
    parser.add_argument("--samples-output", help="optional 200 Hz JSONL sample log")
    args = parser.parse_args()
    if not 0.1 <= args.duration_sec <= 60.0:
        parser.error("--duration-sec must be between 0.1 and 60")
    report = collect(args.interface, args.duration_sec, args.samples_output)
    encoded = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
