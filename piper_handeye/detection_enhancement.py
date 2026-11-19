"""One photometric retry when raw RGB lacks the requested YOLO target."""
import time
import cv2
import numpy as np

from .button_pose import select_button
from .call_button_detection import retry_call_button_tile
from .numeric_button_detection import retry_numeric_button_tile


def enhance_detection_image(image):
    # Preserve pixel geometry: depth alignment and camera intrinsics stay valid.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    lut = np.rint(255 * (np.arange(256) / 255.0) ** 0.75).astype(np.uint8)
    lab[:, :, 0] = cv2.LUT(luminance, lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def detection_payload(detections):
    return [{"label": d.label, "confidence": float(d.confidence),
             "bbox_xyxy": list(d.bbox_xyxy)} for d in detections]


def detect_with_enhancement_fallback(detector, image, target_label, threshold, *, enabled=True,
                                     before_numeric_retry=None):
    started = time.perf_counter()
    raw = tuple(detector.detect(image))
    audit = {"source": "raw", "attempted": False, "accepted": False,
             "enhancement_sec": 0.0, "retry_inference_sec": 0.0,
             "raw_inference_sec": time.perf_counter()-started}
    if enabled and str(target_label).casefold() in {'up', 'down'}:
        # Keep one bounded retry per frame. Do not stack photometric retries on
        # call tiles: the RGB-D sample must remain inside its freshness budget.
        try:
            result, call_audit = retry_call_button_tile(
                detector, image, raw, str(target_label).casefold(), threshold)
            audit.update(call_audit)
            if audit['attempted']:
                audit['raw_detections'] = detection_payload(raw)
            return result, audit
        except Exception as exc:
            audit.update(attempted=True, error=f'{type(exc).__name__}: {exc}')
            return raw, audit
    # Ambiguous raw targets still go to the existing rejection logic. Never use
    # enhancement to override a raw target that meets the production threshold.
    if not enabled or any(d.label.casefold() == str(target_label).casefold()
                          and d.confidence >= threshold for d in raw):
        return raw, audit
    if str(target_label).isdigit() and 1 <= int(target_label) <= 16:
        if before_numeric_retry is not None:
            started = time.perf_counter()
            try:
                region = before_numeric_retry(raw)
                audit['pre_retry_layout_2d'] = region
                if (isinstance(region, dict) and region.get('accepted_2d') is True
                        and region.get('source') == 'panel_layout_inferred'
                        and region.get('target_label') == str(target_label)
                        and region.get('valid_3d') is False
                        and region.get('motion_eligible') is False):
                    audit.update(source='raw_with_layout_2d',
                                 numeric_retry_skipped_by_layout=True,
                                 raw_detections=detection_payload(raw))
                    return raw, audit
            except Exception as exc:
                audit['pre_retry_layout_error'] = f'{type(exc).__name__}: {exc}'
            finally:
                audit['layout_2d_sec'] = time.perf_counter()-started
            audit['raw_detections'] = detection_payload(raw)
        try:
            result, numeric_audit = retry_numeric_button_tile(
                detector, image, raw, str(target_label), threshold)
            if numeric_audit['attempted']:
                audit.update(numeric_audit, raw_detections=detection_payload(raw))
                # Never stack CLAHE and a panel inference on one RGB-D frame.
                return result, audit
        except Exception as exc:
            audit.update(attempted=True, error=f'{type(exc).__name__}: {exc}')
            return raw, audit
    audit.update(attempted=True, algorithm="lab_clahe2_gamma075",
                 raw_detections=detection_payload(raw))
    try:
        started = time.perf_counter()
        enhanced = enhance_detection_image(image)
        audit["enhancement_sec"] = time.perf_counter() - started
        started = time.perf_counter()
        retry = tuple(detector.detect(enhanced))
        audit["retry_inference_sec"] = time.perf_counter() - started
        audit["enhanced_detections"] = detection_payload(retry)
        select_button(list(retry), str(target_label), threshold)
    except Exception as exc:
        # A diagnostic enhancement failure cannot destroy raw detections.
        audit["error"] = f"{type(exc).__name__}: {exc}"
        return raw, audit
    audit.update(source="enhanced", accepted=True)
    return retry, audit
