#!/usr/bin/env python3
"""Persistent OpenCV-only relative-luminance button state worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PREFIX = "BUTTON_LIGHT_RESPONSE "


def emit(value):
    print(PREFIX + json.dumps(value, ensure_ascii=False), flush=True)


class RelativeLuminance:
    def __init__(self, contrast_on=15.0, bright_fraction_on=0.25, algorithm="local_regions_v1"):
        if algorithm not in ("legacy", "local_regions_v1"):
            raise ValueError("unknown light algorithm")
        self.algorithm = algorithm
        self.contrast_on = float(contrast_on)
        self.bright_fraction_on = float(bright_fraction_on)
        self.load_sec = 0.0
        self.attention_mode = "none"

    def predict(self, path, target, bbox_xyxy=None, **_):
        started = time.perf_counter()
        if bbox_xyxy is None:
            return {"state": "UNCERTAIN", "reason": "target_bbox_required"}
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("cannot read light image")
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        left, top = max(0, round(x1)), max(0, round(y1))
        right, bottom = min(image.shape[1], round(x2)), min(image.shape[0], round(y2))
        if right-left < 10 or bottom-top < 10:
            return {"state": "UNCERTAIN", "reason": "target_crop_too_small"}
        gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        y1c, y2c = int(.275*h), max(int(.725*h), 1)
        x1c, x2c = int(.275*w), max(int(.725*w), 1)
        center = gray[y1c:y2c, x1c:x2c]
        ring_mask = np.ones(gray.shape, dtype=bool)
        ring_mask[y1c:y2c, x1c:x2c] = False
        ring = gray[ring_mask]
        center_ring_contrast = float(center.mean() - ring.mean())
        ring_median = float(np.median(ring))
        bright_fraction = float(np.mean(center > ring_median + 30.0))
        if center_ring_contrast >= self.contrast_on and bright_fraction >= self.bright_fraction_on:
            state = "ON"
        elif center_ring_contrast <= 5.0 or bright_fraction <= 0.15:
            state = "OFF"
        else:
            state = "UNCERTAIN"
        legacy_state = state
        # Connected bright strokes, rather than averaging their black background.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (center > ring_median + 30.0).astype(np.uint8), connectivity=8)
        min_area = max(3, int(np.ceil(center.size * .03)))
        regions = []
        on_labels = []
        # Numeral strokes on an unlit button may exceed85 from reflections.
        # Keep arrows/other labels on their existing independently tuned gate.
        numeric_target = str(target).isdigit() and 1 <= int(target) <= 16
        local_on_threshold = 100.0 if numeric_target else 85.0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            mean = float(center[labels == label].mean())
            regions.append({"area_px": area, "fraction": area / center.size,
                            "mean_gray": mean, "contrast": mean - ring_median})
            if mean - ring_median >= local_on_threshold:
                on_labels.append(label)
        local_on = any(r["contrast"] >= local_on_threshold for r in regions)
        # Local strokes may confirm ON, but must not turn an otherwise OFF
        # unlit symbol into UNCERTAIN merely for being below its ON threshold.
        # The legacy classifier and invalid-image uncertainty remain intact.
        if self.algorithm == "local_regions_v1" and local_on:
            state = "ON"
        arrow_audit = None
        if (self.algorithm == "local_regions_v1" and local_on and legacy_state != 'ON'
                and str(target).casefold() in {'up', 'down'}):
            from piper_handeye.arrow_light_region import resolve_arrow_highlight
            highlight_mask = np.zeros(gray.shape, dtype=bool)
            highlight_mask[y1c:y2c, x1c:x2c] = np.isin(labels, on_labels)
            if left != round(x1) or top != round(y1) or right != round(x2) or bottom != round(y2):
                state = 'UNCERTAIN'
                arrow_audit = dict(policy='arrow_diagonals_v1', applied=True,
                                   reason='clipped_button_box', raw_local_state='ON')
            else:
                state, arrow_audit = resolve_arrow_highlight(
                    gray, highlight_mask, str(target).casefold(), ring_median, legacy_state)
        return {
            "legacy_state": legacy_state, "algorithm": self.algorithm,
            "local_regions": regions, "local_min_area_px": min_area,
            "local_min_fraction": .03, "local_on_contrast_threshold": local_on_threshold,
            "local_uncertainty_enabled": False,
            "local_decision_policy": f"on_at_{local_on_threshold:g}_else_legacy",
            **({"arrow_highlight_guard": arrow_audit} if arrow_audit is not None else {}),
            "ring_median_gray": ring_median,
            "state": state, "target": str(target), "bbox": [x1,y1,x2,y2],
            "detection_source": "vision_service",
            "center_ring_contrast": center_ring_contrast,
            "center_bright_fraction": bright_fraction,
            "contrast_on_threshold": self.contrast_on,
            "bright_fraction_on_threshold": self.bright_fraction_on,
            "recognizer": "opencv_self_contrast", "total_sec": time.perf_counter()-started,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['legacy', 'local_regions_v1'], default='local_regions_v1')
    recognizer = RelativeLuminance(algorithm=parser.parse_args().algorithm)
    emit({"ready": True, "load_sec": 0.0,
          "recognizer": "opencv_self_contrast", "algorithm": recognizer.algorithm})
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "shutdown":
            emit({"shutdown": True}); return
        try:
            response = recognizer.predict(request["image"], request["target"], request.get("bbox_xyxy"))
            emit({"id": request.get("id"), "ok": True, **response})
        except Exception as exc:
            emit({"id": request.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__": main()
