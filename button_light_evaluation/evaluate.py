#!/usr/bin/env python3
"""Large-scale offline evaluation for production YOLO + OpenCV light state."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATES = ("ON", "OFF", "UNCERTAIN")


@dataclass(frozen=True)
class Sample:
    row_id: int
    image: Path
    target: str
    state: str
    gt_bbox: tuple[float, float, float, float] | None


def _bbox(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    names = ("gt_x1", "gt_y1", "gt_x2", "gt_y2")
    present = [bool(str(row.get(name, "")).strip()) for name in names]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("gt_x1, gt_y1, gt_x2 and gt_y2 must be supplied together")
    value = tuple(float(row[name]) for name in names)
    if value[2] <= value[0] or value[3] <= value[1]:
        raise ValueError("invalid ground-truth bounding box")
    return value


def load_manifest(path: Path) -> list[Sample]:
    path = path.resolve()
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    samples = []
    for index, row in enumerate(rows, 1):
        missing = [name for name in ("image", "target", "state") if not str(row.get(name, "")).strip()]
        if missing:
            raise ValueError(f"row {index} missing fields: {', '.join(missing)}")
        state = str(row["state"]).strip().upper()
        if state not in STATES:
            raise ValueError(f"row {index} has invalid state {state!r}")
        image = Path(str(row["image"]).strip()).expanduser()
        if not image.is_absolute():
            image = path.parent / image
        if not image.is_file():
            raise FileNotFoundError(f"row {index} image not found: {image}")
        samples.append(Sample(index, image.resolve(), str(row["target"]).strip(), state, _bbox(row)))
    if not samples:
        raise ValueError("manifest contains no samples")
    return samples


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    a, b = tuple(map(float, first)), tuple(map(float, second))
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _wilson(successes: int, count: int) -> list[float] | None:
    if count == 0:
        return None
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(
    rows: list[dict], elapsed_sec: float, iou_threshold: float,
    processed_count: int | None = None,
) -> dict:
    total = len(rows)
    predicted = [row.get("predicted_state", "ERROR") for row in rows]
    correct = sum(predicted[i] == rows[i]["truth_state"] for i in range(total))
    covered = [i for i, state in enumerate(predicted) if state in ("ON", "OFF")]
    covered_correct = sum(predicted[i] == rows[i]["truth_state"] for i in covered)
    confusion = {
        truth: {prediction: 0 for prediction in (*STATES, "DETECTION_MISS", "ERROR")}
        for truth in STATES
    }
    for row, prediction in zip(rows, predicted):
        confusion[row["truth_state"]].setdefault(prediction, 0)
        confusion[row["truth_state"]][prediction] += 1
    bbox_rows = [row for row in rows if row.get("gt_bbox") is not None]
    detected = sum(bool(row.get("target_detected")) for row in rows)
    bbox_tp = sum(
        bool(row.get("target_detected")) and float(row.get("bbox_iou", 0.0)) >= iou_threshold
        for row in bbox_rows
    )
    per_class = {}
    for state in ("ON", "OFF"):
        true_positive = sum(row["truth_state"] == state and row.get("predicted_state") == state for row in rows)
        false_negative = sum(row["truth_state"] == state and row.get("predicted_state") != state for row in rows)
        false_positive = sum(row["truth_state"] != state and row.get("predicted_state") == state for row in rows)
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else None
        recall = true_positive / recall_denominator if recall_denominator else None
        per_class[state] = {
            "support": recall_denominator,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None,
        }
    recalls = [value["recall"] for value in per_class.values() if value["recall"] is not None]
    processed = total if processed_count is None else int(processed_count)
    return {
        "samples": total,
        "exact_state_accuracy": correct / total,
        "correct_state": correct,
        "exact_state_accuracy_wilson_95": _wilson(correct, total),
        "decisive_coverage": len(covered) / total,
        "decisive_accuracy": covered_correct / len(covered) if covered else None,
        "balanced_accuracy_on_off": sum(recalls) / len(recalls) if recalls else None,
        "per_class": per_class,
        "target_detection_rate": detected / total,
        "bbox_samples": len(bbox_rows),
        "bbox_iou_threshold": iou_threshold,
        "bbox_recall_at_iou": bbox_tp / len(bbox_rows) if bbox_rows else None,
        "elapsed_sec": elapsed_sec,
        "processed_this_run": processed,
        "images_per_sec_this_run": processed / elapsed_sec if elapsed_sec > 0 else None,
        "confusion_matrix": confusion,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV or JSONL with image,target,state and optional gt_x1..gt_y2")
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    # Heavy vision imports stay local so manifest validation and metric tests
    # also run in the lightweight control/test environment.
    from button_light_optimization.opencv_light_worker import RelativeLuminance
    from piper_handeye.button_pose import YoloButtonDetector, select_button

    args = parse_args()
    if args.batch_size < 1 or args.limit < 0 or not 0 < args.iou_threshold <= 1:
        raise ValueError("invalid batch size, limit or IoU threshold")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    detector_cfg = config["detector"]
    model_path = Path(detector_cfg["model_path"])
    if not model_path.is_absolute():
        model_path = (config_path.parent / model_path).resolve()
    samples = load_manifest(args.manifest)
    if args.limit:
        samples = samples[:args.limit]
    output = (args.output or ROOT / "button_light_evaluation" / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "predictions.jsonl"
    completed = set()
    rows = []
    if args.resume and result_path.is_file():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.append(row)
            completed.add((int(row["row_id"]), row["image"], row["target"]))
    pending = [sample for sample in samples if (sample.row_id, str(sample.image), sample.target) not in completed]
    load_started = time.perf_counter()
    detector = YoloButtonDetector(
        model_path,
        conf=float(detector_cfg["inference_confidence"]),
        iou=float(detector_cfg["iou"]),
        imgsz=int(detector_cfg["imgsz"]),
        device=detector_cfg.get("device", "auto"),
    )
    light = RelativeLuminance()
    model_load_sec = time.perf_counter() - load_started
    started = time.perf_counter()
    result_mode = "a" if args.resume else "w"
    if not args.resume:
        rows.clear()
    with result_path.open(result_mode, encoding="utf-8") as sink:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            try:
                detections = detector.detect_many(
                    [str(sample.image) for sample in batch], batch_size=args.batch_size
                )
            except Exception:
                # Isolate corrupt/unsupported images so one bad file does not
                # discard an otherwise useful long-running evaluation.
                detections = []
                for sample in batch:
                    try:
                        detections.append(detector.detect(str(sample.image)))
                    except Exception as exc:
                        detections.append(exc)
            for sample, found in zip(batch, detections):
                row = {
                    "row_id": sample.row_id, "image": str(sample.image), "target": sample.target,
                    "truth_state": sample.state,
                    "gt_bbox": list(sample.gt_bbox) if sample.gt_bbox else None,
                }
                if isinstance(found, Exception):
                    row.update(target_detected=False, predicted_state="ERROR",
                               error=f"{type(found).__name__}: {found}")
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    rows.append(row)
                    continue
                try:
                    selected = select_button(found, sample.target, float(detector_cfg["target_confidence"]))
                    row.update(target_detected=True, detection_confidence=selected.confidence,
                               predicted_bbox=list(selected.bbox_xyxy))
                    if sample.gt_bbox:
                        row["bbox_iou"] = box_iou(selected.bbox_xyxy, sample.gt_bbox)
                except RuntimeError as exc:
                    row.update(target_detected=False, predicted_state="DETECTION_MISS", error=str(exc))
                else:
                    try:
                        result = light.predict(sample.image, sample.target, selected.bbox_xyxy)
                        row.update(predicted_state=result["state"], opencv=result)
                    except Exception as exc:
                        row.update(predicted_state="ERROR",
                                   error=f"{type(exc).__name__}: {exc}")
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                rows.append(row)
    elapsed = time.perf_counter() - started
    # Keep only requested rows when resuming into an existing output folder.
    keys = {(sample.row_id, str(sample.image), sample.target) for sample in samples}
    rows = [row for row in rows if (int(row["row_id"]), row["image"], row["target"]) in keys]
    summary = summarize(rows, elapsed, args.iou_threshold, processed_count=len(pending))
    summary.update({"manifest": str(args.manifest.resolve()), "model": str(model_path),
                    "model_sha256": file_sha256(model_path),
                    "model_load_sec": model_load_sec,
                    "opencv_thresholds": {
                        "contrast_on": light.contrast_on,
                        "bright_fraction_on": light.bright_fraction_on,
                        "off_contrast_max": 5.0,
                        "off_bright_fraction_max": 0.15,
                    },
                    "detector_settings": {
                        "inference_confidence": float(detector_cfg["inference_confidence"]),
                        "target_confidence": float(detector_cfg["target_confidence"]),
                        "iou": float(detector_cfg["iou"]),
                        "imgsz": int(detector_cfg["imgsz"]),
                        "device": detector_cfg.get("device", "auto"),
                        "batch_size": args.batch_size,
                    },
                    "output": str(output), "new_samples_this_run": len(pending)})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mistakes = [row for row in rows if row.get("predicted_state") != row["truth_state"]]
    with (output / "mistakes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("row_id", "image", "target", "truth_state", "predicted_state", "target_detected", "detection_confidence", "bbox_iou", "error"))
        writer.writeheader()
        for row in mistakes:
            writer.writerow({name: row.get(name) for name in writer.fieldnames})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
