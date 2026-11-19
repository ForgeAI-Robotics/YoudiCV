#!/usr/bin/env python3
"""Summarize DualSense input, expected Twist, and published Twist journal data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "gamepad_control_runs"


def latest_journal() -> Path:
    candidates = sorted(RUN_ROOT.glob("*/control_journal.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no journal under {RUN_ROOT}")
    return candidates[-1]


def analyze(path: Path) -> dict[str, object]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    samples = [item for item in records if item.get("event") == "sample_compare"]
    comparisons = Counter(str(item.get("comparison")) for item in samples)
    active = [
        item for item in samples
        if abs(float(item["expected_twist"]["linear_x"])) > 1e-9
        or abs(float(item["expected_twist"]["angular_z"])) > 1e-9
    ]
    published = [
        item for item in samples
        if abs(float(item["published_twist"]["linear_x"])) > 1e-9
        or abs(float(item["published_twist"]["angular_z"])) > 1e-9
    ]
    smoothed = [
        item for item in active
        if item.get("smoothed_twist", {}).get("available")
        and abs(float(item["smoothed_twist"].get("linear_x", 0.0))) > 0.01
    ]
    moving = [
        item for item in active
        if item.get("odometry_twist", {}).get("available")
        and abs(float(item["odometry_twist"].get("linear_x", 0.0))) > 0.01
    ]
    max_publish_gap = max(
        (float(item.get("publish_gap_sec", 0.0)) for item in active), default=0.0
    )
    max_abs_expected_linear = max(
        (abs(float(item["expected_twist"]["linear_x"])) for item in active),
        default=0.0,
    )
    max_abs_smoothed_linear = max(
        (
            abs(float(item.get("smoothed_twist", {}).get("linear_x", 0.0)))
            for item in active
            if item.get("smoothed_twist", {}).get("available")
        ),
        default=0.0,
    )
    smoothed_to_expected_ratio = (
        max_abs_smoothed_linear / max_abs_expected_linear
        if max_abs_expected_linear > 1e-9
        else 0.0
    )
    if not active:
        diagnosis = "没有记录到有效扳机速度输入"
    elif not published and comparisons.get("blocked_control_gate"):
        diagnosis = "存在有效输入，但控制确认门未开启"
    elif not published and comparisons.get("blocked_arm_mode"):
        diagnosis = "存在有效输入，但当前处于机械臂模式"
    elif not published and comparisons.get("blocked_no_ros_subscriber"):
        diagnosis = "控制已开启，但 ROS 底盘订阅者未连接"
    elif comparisons.get("expected_published_mismatch"):
        diagnosis = "计算速度与实际发布速度不一致"
    elif max_publish_gap > 0.2:
        diagnosis = (
            f"速度发布最大间隔 {max_publish_gap:.3f}s，超过底盘 0.2s 超时"
        )
    elif published and not smoothed and any(
        item.get("smoothed_twist", {}).get("available") for item in active
    ):
        diagnosis = "GUI 已发布非零速度，但速度平滑器输出始终为零"
    elif (
        smoothed
        and not moving
        and max_abs_expected_linear >= 0.1
        and smoothed_to_expected_ratio < 0.1
        and any(item.get("odometry_twist", {}).get("available") for item in active)
    ):
        diagnosis = (
            "速度平滑器严重限幅："
            f"输入峰值 {max_abs_expected_linear:.3f}m/s，"
            f"输出峰值仅 {max_abs_smoothed_linear:.3f}m/s "
            f"({smoothed_to_expected_ratio:.1%})，底盘里程计为零"
        )
    elif smoothed and not moving and any(
        item.get("odometry_twist", {}).get("available") for item in active
    ):
        diagnosis = "平滑器已输出非零速度，但底盘里程计没有运动"
    elif moving:
        diagnosis = "手柄输入、发布、平滑器输出和底盘里程计均有响应"
    elif published:
        diagnosis = "输入和发布一致，但下游反馈数据不足"
    else:
        diagnosis = "数据不足，需再次操作手柄"
    report: dict[str, object] = {
        "journal": str(path),
        "record_count": len(records),
        "sample_count": len(samples),
        "active_input_samples": len(active),
        "nonzero_published_samples": len(published),
        "nonzero_smoothed_samples": len(smoothed),
        "nonzero_odometry_samples": len(moving),
        "max_publish_gap_sec": round(max_publish_gap, 6),
        "max_abs_expected_linear_m_s": round(max_abs_expected_linear, 6),
        "max_abs_smoothed_linear_m_s": round(max_abs_smoothed_linear, 6),
        "smoothed_to_expected_ratio": round(smoothed_to_expected_ratio, 6),
        "comparison_counts": dict(comparisons),
        "diagnosis": diagnosis,
        "latest_sample": samples[-1] if samples else None,
    }
    output = path.with_name("comparison_report.json")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report"] = str(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("journal", nargs="?", type=Path)
    args = parser.parse_args()
    report = analyze(args.journal or latest_journal())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
