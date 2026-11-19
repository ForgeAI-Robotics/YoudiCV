#!/usr/bin/env python3
"""Generate and validate the segmented press motion without robot I/O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from piper_handeye.segmented_button_motion import plan_segmented_press_attempt
from piper_handeye.timed_trajectory import PIPER_JOINT_NAMES, validate_trajectory_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    waiting = np.array([0.23, 1.63, -0.98, 0.39, -0.55, -0.03])
    prepress = np.array([0.19, 1.99, -1.46, 0.37, -0.48, -0.04])
    pressed = prepress + np.array([-0.003, 0.006, -0.006, 0.0, 0.0, 0.0])
    panel = np.eye(4)
    panel[:3, 3] = [0.616, 0.077, 0.277]
    panel[:3, 2] = [-0.9998, -0.0172, -0.0071]
    panel[:3, 3] -= 0.10 * panel[:3, 2]
    first = plan_segmented_press_attempt(
        waiting, waiting, prepress, pressed, panel
    )
    retry = plan_segmented_press_attempt(
        first.observation_joints, waiting, prepress, pressed, panel
    )
    names = (
        "approach",
        "press_cycle",
        "retreat_to_observation",
        "observation_to_waiting",
    )
    segments = []
    for name in names:
        points = getattr(first, name)
        validate_trajectory_payload(
            {
                "protocol": "piper_timed_trajectory_v1",
                "command": "validate",
                "joint_names": PIPER_JOINT_NAMES,
                "points": points,
            }
        )
        segments.append(
            {
                "name": name,
                "duration_sec": points[-1]["time_from_start_sec"],
                "point_count": len(points),
                "starts_stopped": max(abs(x) for x in points[0]["velocities"]) < 1e-12,
                "ends_stopped": max(abs(x) for x in points[-1]["velocities"]) < 1e-12,
            }
        )
    report = {
        "mode": "synthetic_geometry_dry_run",
        "robot_commands_sent": False,
        "observation_distance_m": first.observation_distance_m,
        "segments": segments,
        "retry_approach_duration_sec": retry.approach_audit.duration_sec,
        "first_approach_duration_sec": first.approach_audit.duration_sec,
        "retry_starts_at_observation": bool(
            np.allclose(retry.approach[0]["positions"], first.observation_joints)
        ),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
