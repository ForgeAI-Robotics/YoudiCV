#!/usr/bin/env python3
"""Audit closed-loop reports for unsupported claims and incomplete actions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


TERMINAL_MOTION_EVENTS = {
    "motion_segment_completed",
    "motion_segment_failed",
    "motion_segment_feedback_unconfirmed",
}
TERMINAL_LIFECYCLES = {"completed", "failed"}


def audit_report(path: Path) -> list[dict]:
    report = yaml.safe_load(path.read_text()) or {}
    journal = report.get("journal") or []
    events = report.get("events") or []
    issues = []

    starts = Counter()
    terminals = Counter()
    for row in journal:
        value = row.get("value") or {}
        segment = value.get("segment")
        if row.get("kind") == "motion_segment_started":
            starts[(segment, value.get("task_id"))] += 1
        elif row.get("kind") in TERMINAL_MOTION_EVENTS:
            # Legacy reports have no task ID, so segment remains the fallback key.
            terminals[(segment, value.get("task_id"))] += 1
    for key, count in starts.items():
        missing = count - terminals[key]
        if missing > 0:
            issues.append({
                "code": "motion_transaction_without_terminal_event",
                "segment": key[0], "task_id": key[1], "count": missing,
            })

    if any(event.get("event") == "press_completed" for event in events):
        issues.append({
            "code": "legacy_press_completed_without_outcome_contract",
            "detail": "the event does not prove contact or button actuation",
        })

    if report.get("returned_to_origin") is True:
        evidence = report.get("origin_verification") or next((
            event.get("verification") for event in reversed(events)
            if event.get("event") == "at_origin" and event.get("verification")
        ), None)
        if not isinstance(evidence, dict) or evidence.get("verified") is not True:
            issues.append({"code": "origin_claim_without_encoder_evidence"})

    lifecycle = report.get("lifecycle_status")
    if lifecycle is not None and lifecycle not in TERMINAL_LIFECYCLES:
        issues.append({
            "code": "nonterminal_lifecycle",
            "lifecycle_status": lifecycle,
            "phase": report.get("phase"),
        })
    elif lifecycle is None and report.get("status") in {"initializing", "running"}:
        issues.append({
            "code": "legacy_nonterminal_status",
            "status": report.get("status"),
            "phase": report.get("phase"),
        })

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path,
        default=[Path("button_light_closed_loop_runs")],
    )
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()
    reports = []
    for item in args.paths:
        candidates = [item] if item.is_file() else sorted(item.glob("*/closed_loop_report.yaml"))
        for path in candidates:
            issues = audit_report(path)
            if issues:
                reports.append({"path": str(path), "issues": issues})
    result = {
        "reports_with_issues": len(reports),
        "reports": reports,
    }
    print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    return 2 if args.fail_on_issues and reports else 0


if __name__ == "__main__":
    raise SystemExit(main())
