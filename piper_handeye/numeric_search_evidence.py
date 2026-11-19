"""Same-frame neighbour evidence for upward search, never a pressing target."""
import math
import time

from .call_button_detection import overlap
from .observation_visibility import complete_button_box
from .visual_search import predict_button_center_from_grid


def _qualified(rows, image_size, threshold):
    result = []
    for row in rows:
        try:
            label = str(row['label'])
            confidence = float(row['confidence'])
            box = list(map(float, row['bbox_xyxy']))
            if (not label or not math.isfinite(confidence)
                    or not threshold <= confidence <= 1
                    or not complete_button_box(box, image_size)):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        result.append(dict(label=label, confidence=confidence, bbox_xyxy=box))
    return result


def numeric_search_evidence(response, target_label, min_confidence):
    """Keep working raw grids; otherwise validate the existing panel retry.

    The caller must still confirm independent fresh frames, reacquire a real
    anchor's depth, validate the bounded search, and detect the requested
    button after moving. Primary detections and lamp inputs are never changed.
    """
    target_label = str(target_label)
    raw = response.get('detections', [])
    try:
        prediction = predict_button_center_from_grid(target_label, raw)
        return raw, prediction, dict(source='raw')
    except RuntimeError as exc:
        raw_error = str(exc)
    if not target_label.isdigit() or not 1 <= int(target_label) <= 16:
        raise RuntimeError(raw_error)
    age = time.time() - float(response.get('capture_time_unix_sec', 0))
    if (not -.05 <= age <= .35 or response.get('frame_sequence') is None):
        raise RuntimeError('numeric search context is stale or has no frame')
    enhancement = response.get('detection_enhancement', {})
    if (enhancement.get('algorithm') != 'numeric_panel_local_scale'
            or enhancement.get('attempted') is not True
            or enhancement.get('error')):
        raise RuntimeError(raw_error)
    # This audit is produced from the same LiveButtonSample as the response;
    # reject an explicit different-frame marker rather than mixing evidence.
    if enhancement.get('frame_sequence', response['frame_sequence']) != response['frame_sequence']:
        raise RuntimeError('numeric search context frame mismatch')
    image_size = response['image_size']
    width, height = map(float, image_size)
    x, y, right, bottom = map(float, enhancement['tile_xyxy'])
    if not (0 <= x < right <= width and 0 <= y < bottom <= height
            and min(right-x, bottom-y) >= 64):
        raise RuntimeError('numeric search context has invalid crop geometry')
    # Never use neighbours to conceal a directly seen or ambiguous target.
    if any(str(d.get('label')) == target_label
           and float(d.get('confidence', 0)) >= min_confidence for d in raw):
        raise RuntimeError(raw_error)
    raw_qualified = _qualified(raw, image_size, min_confidence)
    local = _qualified(enhancement.get('tile_detections', []), image_size, min_confidence)
    if any(d['label'] == target_label for d in local):
        raise RuntimeError('local search context contains a target requiring direct reacquisition')
    supports = []
    for d in local:
        if any(a['label'] == d['label'] and overlap(a['bbox_xyxy'], d['bbox_xyxy']) > .5
               for a in raw_qualified):
            if not any(overlap(a['bbox_xyxy'], d['bbox_xyxy']) > .5 for a in supports):
                supports.append(d)
    if len(supports) < 2:
        raise RuntimeError('numeric search context lacks two independent same-frame supports')
    numeric = []
    corrections = []
    for d in sorted(local, key=lambda row: row['confidence'], reverse=True):
        if not d['label'].isdigit() or not 1 <= int(d['label']) <= 16:
            continue
        box = d['bbox_xyxy']
        # Both global-frame and artificial crop edges must preserve the button.
        if not (x <= box[0] < box[2] <= right and y <= box[1] < box[3] <= bottom):
            continue
        if ((x > 0 and box[0] < x+5) or (y > 0 and box[1] < y+5)
                or (right < width and box[2] > right-5)
                or (bottom < height and box[3] > bottom-5)):
            continue
        if any(a['label'] != d['label'] and a['confidence'] > d['confidence']-.15
               and overlap(a['bbox_xyxy'], box) > .5 for a in local):
            raise RuntimeError('numeric search context has competing local labels')
        conflicts = [a for a in raw_qualified if a['label'] != d['label']
                     and overlap(a['bbox_xyxy'], box) > .5]
        if any(not a['label'].isdigit() for a in conflicts):
            raise RuntimeError('numeric search context conflicts with a non-numeric button')
        if conflicts and d['confidence'] < max(.90, min_confidence):
            raise RuntimeError('numeric search correction needs strong local evidence')
        if any(a['label'] == d['label'] and overlap(a['bbox_xyxy'], box) <= .5 for a in numeric):
            raise RuntimeError('numeric search context has duplicate floor labels')
        if any(overlap(a['bbox_xyxy'], box) > .5 for a in numeric):
            continue
        numeric.append(d)
        if conflicts:
            corrections.append(dict(label=d['label'], raw_labels=[a['label'] for a in conflicts]))
    prediction = predict_button_center_from_grid(target_label, numeric)
    # This fallback only enables looking above the current view. It must never
    # turn a missing, in-view button into an inferred pressing destination.
    if prediction.center_xy[1] >= 5 or prediction.rows_above_visible <= 0:
        raise RuntimeError('numeric neighbour fallback is not an above-frame search')
    audit = dict(source='numeric_tile_context', frame_sequence=response['frame_sequence'],
                 capture_time_unix_sec=response['capture_time_unix_sec'],
                 target_label=target_label, raw_grid_error=raw_error,
                 independent_supports=[d['label'] for d in supports], corrections=corrections,
                 labels=list(prediction.labels), anchor_label=prediction.anchor_label,
                 predicted_center_xy=list(prediction.center_xy), fit_rms_px=prediction.fit_rms_px,
                 purpose='upward_search_only', target_detected=False)
    return numeric, prediction, audit
