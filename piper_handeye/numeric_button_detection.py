"""One same-frame panel-scale retry; labels must come from YOLO, never a grid."""
import math
import time

import numpy as np

from .button_pose import ButtonDetection, select_button
from .call_button_detection import overlap
from .observation_visibility import complete_button_box


def numeric_panel_crop(image, raw):
    height, width = image.shape[:2]
    buttons = []
    for d in sorted(raw, key=lambda d: d.confidence, reverse=True):
        if not str(d.label).isdigit() or not math.isfinite(d.confidence) or d.confidence < .50:
            continue
        b = d.bbox_xyxy
        if (not all(math.isfinite(v) for v in b)
                or not (0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height)
                or min(b[2]-b[0], b[3]-b[1]) < 8
                or any(overlap(b, a.bbox_xyxy) > .5 for a in buttons)):
            continue
        buttons.append(d)
    if len(buttons) < 3:
        return None
    pitch = float(np.median([max(d.bbox_xyxy[2]-d.bbox_xyxy[0],
                                     d.bbox_xyxy[3]-d.bbox_xyxy[1]) for d in buttons]))
    # Keep neighbouring labels and the lower control row as visual context.
    x = max(0, math.floor(min(d.bbox_xyxy[0] for d in buttons)-pitch))
    y = max(0, math.floor(min(d.bbox_xyxy[1] for d in buttons)-pitch))
    right = min(width, math.ceil(max(d.bbox_xyxy[2] for d in buttons)+pitch))
    bottom = min(height, math.ceil(max(d.bbox_xyxy[3] for d in buttons)+3*pitch))
    if min(right-x, bottom-y) < 64 or max(right-x, bottom-y) > .90*max(width, height):
        return None
    return x, y, right, bottom


def retry_numeric_button_tile(detector, image, raw, target, threshold):
    audit = dict(attempted=False, accepted=False, source='raw')
    if not str(target).isdigit() or not 1 <= int(target) <= 16:
        return raw, audit
    # Preserve both the normal path and the existing multiple-target rejection.
    if any(d.label == str(target) and d.confidence >= threshold for d in raw):
        return raw, audit
    tile = numeric_panel_crop(image, raw)
    if tile is None:
        return raw, audit
    x, y, right, bottom = tile
    audit.update(attempted=True, algorithm='numeric_panel_local_scale', tile_xyxy=list(tile))
    started = time.perf_counter()
    retry = detector.detect(image[y:bottom, x:right])
    audit['retry_inference_sec'] = time.perf_counter()-started
    mapped = []
    for d in retry:
        a, b, c, e = d.bbox_xyxy
        if (not str(d.label) or not math.isfinite(d.confidence)
                or not all(math.isfinite(v) for v in (a,b,c,e))
                or not (0 <= a < c <= right-x and 0 <= b < e <= bottom-y)):
            continue
        if ((x > 0 and a < 5) or (y > 0 and b < 5)
                or (right < image.shape[1] and c > right-x-5)
                or (bottom < image.shape[0] and e > bottom-y-5)):
            continue
        mapped.append(ButtonDetection(d.class_id, d.label, d.confidence,
                                      (a+x,b+y,c+x,e+y)))
    audit['tile_detections'] = [dict(label=d.label, confidence=d.confidence,
                                    bbox_xyxy=list(d.bbox_xyxy)) for d in mapped]
    try:
        target_detection = select_button(mapped, str(target), threshold)
    except RuntimeError:
        audit['reason'] = 'no_unique_qualified_numeric_tile_target'
        return raw, audit
    if not complete_button_box(target_detection.bbox_xyxy,
                               [image.shape[1],image.shape[0]]):
        audit['reason'] = 'numeric_tile_target_incomplete'
        return raw, audit
    competitors = [d for d in mapped if d.label != str(target)
                   and overlap(d.bbox_xyxy,target_detection.bbox_xyxy) > .5]
    if any(d.confidence > target_detection.confidence-.15 for d in competitors):
        audit['reason'] = 'conflicting_numeric_tile_labels'
        return raw, audit
    if any(not str(d.label).isdigit() and d.confidence >= threshold
           and overlap(d.bbox_xyxy,target_detection.bbox_xyxy) > .5 for d in raw):
        audit['reason'] = 'raw_non_numeric_button_conflict'
        return raw, audit
    raw_numeric_conflict = any(d.label != str(target) and str(d.label).isdigit()
                               and d.confidence >= threshold
                               and overlap(d.bbox_xyxy,target_detection.bbox_xyxy) > .5 for d in raw)
    if raw_numeric_conflict and target_detection.confidence < max(.90,threshold):
        audit['reason'] = 'numeric_scale_conflict_requires_strong_local_evidence'
        return raw, audit
    # A stronger scale-specific reading may replace a raw 14/4 confusion only
    # with agreement on two independent neighbouring labels in this SAME frame.
    supports = []
    for d in raw:
        if (d.label == str(target)
                or not math.isfinite(d.confidence) or d.confidence < threshold
                or overlap(d.bbox_xyxy,target_detection.bbox_xyxy) > .25):
            continue
        if any(d.label == a.label and a.confidence >= threshold
               and overlap(d.bbox_xyxy,a.bbox_xyxy) > .5 for a in mapped):
            if not any(overlap(d.bbox_xyxy,a.bbox_xyxy) > .5 for a in supports):
                supports.append(d)
    audit['independent_supports'] = [d.label for d in supports]
    if len(supports) < 2:
        audit['reason'] = 'insufficient_same_frame_numeric_support'
        return raw, audit
    # Remove the displaced competing label at the same physical button; all
    # other strong targets remain, so cropping cannot hide global ambiguity.
    displaced = [d for d in raw if str(d.label).isdigit()
                 and overlap(d.bbox_xyxy,target_detection.bbox_xyxy) > .5]
    merged = [d for d in raw if d not in displaced]+[target_detection]
    try:
        select_button(merged, str(target), threshold)
    except RuntimeError:
        audit['reason'] = 'global_numeric_target_ambiguous'
        return raw, audit
    audit.update(accepted=True, source='numeric_tile',
                 displaced_raw_labels=[d.label for d in displaced])
    return tuple(merged), audit
