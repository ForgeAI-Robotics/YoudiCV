#!/usr/bin/env python3
"""Print normalized DualSense input only; never connects to robot outputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time

from piper_handeye.dualsense_evdev import DualSenseEvdevAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    args = parser.parse_args()
    if not 1.0 <= args.rate_hz <= 200.0:
        raise ValueError("rate-hz must be in 1..200")
    period = 1.0 / args.rate_hz
    with DualSenseEvdevAdapter(args.device) as adapter:
        print(json.dumps({"controller": adapter.identity}, ensure_ascii=False))
        while True:
            started = time.monotonic()
            sample = adapter.read_sample(timeout_sec=min(period, 0.02))
            print(json.dumps(asdict(sample), ensure_ascii=False), flush=True)
            if not sample.connected:
                return 2
            time.sleep(max(0.0, period - (time.monotonic() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
