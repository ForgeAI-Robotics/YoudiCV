#!/usr/bin/env python3
"""Offline simulation of the DualSense chassis command pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime

from piper_handeye.gamepad_teleop import (
    ChassisMotionLimiter,
    DualModeGamepad,
    GamepadSample,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class VendorSmootherModel:
    """Deterministic model of the configured filter-free velocity limiter."""

    def __init__(self) -> None:
        self.linear = 0.0
        self.angular = 0.0

    @staticmethod
    def approach(current: float, target: float, step: float) -> float:
        if target > current:
            return min(target, current + step)
        return max(target, current - step)

    def update(self, linear: float, angular: float, dt: float) -> tuple[float, float]:
        linear_rate = 0.10 if abs(linear) > abs(self.linear) else 0.12
        angular_rate = 0.40 if abs(angular) > abs(self.angular) else 0.50
        self.linear = self.approach(self.linear, linear, linear_rate * dt)
        self.angular = self.approach(self.angular, angular, angular_rate * dt)
        return self.linear, self.angular


def input_at(t: float) -> tuple[str, GamepadSample]:
    if 1.0 <= t < 3.0:
        return "forward", GamepadSample(right_trigger=1.0)
    if 4.5 <= t < 6.5:
        return "reverse", GamepadSample(left_trigger=1.0)
    if 7.5 <= t < 9.5:
        return "forward_turn", GamepadSample(right_trigger=1.0, left_x=1.0)
    return "released", GamepadSample()


def main() -> int:
    clock = Clock()
    mapper = DualModeGamepad(clock=clock)
    limiter = ChassisMotionLimiter(clock=clock)
    smoother = VendorSmootherModel()
    rows = []
    dt = 0.02
    while clock.now <= 10.5 + 1e-9:
        phase, sample = input_at(clock.now)
        intent = mapper.update(sample)
        gui_linear, gui_angular = limiter.update(
            intent.chassis_linear_x_m_s,
            intent.chassis_angular_z_rad_s,
        )
        output_linear, output_angular = smoother.update(
            gui_linear, gui_angular, dt
        )
        rows.append({
            "time_sec": round(clock.now, 3),
            "phase": phase,
            "trigger_target_linear": intent.chassis_linear_x_m_s,
            "trigger_target_angular": intent.chassis_angular_z_rad_s,
            "gui_limited_linear": gui_linear,
            "gui_limited_angular": gui_angular,
            "vendor_output_linear": output_linear,
            "vendor_output_angular": output_angular,
        })
        clock.now += dt

    max_linear = max(abs(row["vendor_output_linear"]) for row in rows)
    max_angular = max(abs(row["vendor_output_angular"]) for row in rows)
    max_linear_step = max(
        abs(b["vendor_output_linear"] - a["vendor_output_linear"])
        for a, b in zip(rows, rows[1:])
    )
    first_reverse = next(
        i for i, row in enumerate(rows) if row["vendor_output_linear"] < -1e-6
    )
    last_forward = max(
        i for i, row in enumerate(rows[:first_reverse])
        if row["vendor_output_linear"] > 1e-6
    )
    zero_gap = rows[first_reverse]["time_sec"] - rows[last_forward]["time_sec"]
    post_release = [
        row["vendor_output_linear"]
        for row in rows if 3.0 <= row["time_sec"] <= 4.5
    ]
    post_release_monotonic = all(
        b <= a + 1e-12 for a, b in zip(post_release, post_release[1:])
    )

    checks = {
        "linear_speed_at_or_below_0_10_m_s": max_linear <= 0.1000001,
        "angular_speed_at_or_below_0_40_rad_s": max_angular <= 0.4000001,
        "linear_step_respects_0_12_m_s2": max_linear_step <= 0.12 * dt + 1e-9,
        "release_has_no_delayed_speed_spike": post_release_monotonic,
        "reversal_passes_through_zero_hold": zero_gap >= 0.30,
        "final_output_is_zero": (
            abs(rows[-1]["vendor_output_linear"]) <= 1e-9
            and abs(rows[-1]["vendor_output_angular"]) <= 1e-9
        ),
    }
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parents[1] / "chassis_simulation_runs" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "timeline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "dataflow": [
            "DualSense input",
            "DualModeGamepad target",
            "ChassisMotionLimiter",
            "/cmd_vel geometry_msgs/Twist",
            "/velocity_smoother filter-free rate limiter",
            "/plt_cmd_vel geometry_msgs/TwistStamped",
            "/chassis_server_node",
            "/dev/ttyTHS1 at 115200 baud",
        ],
        "configuration": {
            "linear_speed_limit_m_s": 0.10,
            "angular_speed_limit_rad_s": 0.40,
            "linear_accel_m_s2": 0.10,
            "linear_decel_m_s2": 0.12,
            "reversal_zero_hold_sec": 0.30,
            "calman_filter": False,
            "moving_filter": False,
            "limit_speedup_filter": False,
        },
        "measurements": {
            "max_abs_linear_m_s": max_linear,
            "max_abs_angular_rad_s": max_angular,
            "max_linear_step_per_20ms_m_s": max_linear_step,
            "forward_to_reverse_zero_gap_sec": zero_gap,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **report}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
