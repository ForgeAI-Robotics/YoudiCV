#!/usr/bin/env python3
"""Run complete button press/return cycles sequentially in one command."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "piper_button_pose.yaml"
DEFAULT_HOLD_REPORT = ROOT / "button_fast_chain_runs/20260826_150725/chain_report.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+", help="button labels in execution order")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hold-report", type=Path, default=DEFAULT_HOLD_REPORT)
    args = parser.parse_args()
    labels = [str(value) for value in args.labels]
    valid = {
        value.casefold() in {"up", "down"}
        or (value.isdigit() and 1 <= int(value) <= 16)
        for value in labels
    }
    if not all(valid):
        raise ValueError("button labels must be integers in 1..16, up, or down")

    base = [
        sys.executable,
        str(ROOT / "run_button_light_closed_loop.py"),
        "--hold-report", str(args.hold_report.resolve()),
        "--config", str(args.config.resolve()),
        "--max-press-attempts", "3",
        "--depth-degraded-entry-distance-m", "0.15",
        "--pause-depth-near-button",
        "--execute",
        "--acknowledge-clear-workspace",
    ]
    for index, label in enumerate(labels, 1):
        print(
            f"batch_button_start: {label} ({index}/{len(labels)})",
            flush=True,
        )
        result = subprocess.run(
            [*base, label],
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"batch_stopped_after_failure: {label}; "
                f"return_code={result.returncode}",
                flush=True,
            )
            return result.returncode
        print(f"batch_button_complete: {label}", flush=True)
    print(f"batch_complete: {','.join(labels)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
