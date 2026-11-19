"""One same-frame, local-scale retry for small call buttons.

Tile hints choose pixels to inspect, never a motion target. Every returned box
comes from inference on the current frame and is mapped back to RGB coordinates.
"""
import math
import time

from .button_pose import ButtonDetection

CALL_LABELS = {'up', 'down'}


def complete_box(box, width, height, margin=5):
    x1, y1, x2, y2 = box
    return (all(math.isfinite(v) for v in box) and
            margin <= x1 < x2 <= width-margin and
            margin <= y1 < y2 <= height-margin and
            x2-x1 >= 14 and y2-y1 >= 14)


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1)*max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/max(union, 1e-9)


def retry_call_button_tile(detector, image, raw, target, threshold):
    height, width = image.shape[:2]
    audit = {'attempted': False, 'accepted': False, 'source': 'raw'}
    strong = [d for d in raw if d.label == target and d.confidence >= threshold]
    # Never resolve ambiguity by selectively cropping out another strong target.
    if len(strong) > 1 or (strong and complete_box(strong[0].bbox_xyxy, width, height)):
        return raw, audit
    size = min(height, max(64, width//2))
    if size >= width or height < 64:
        return raw, audit
    xs = sorted({0, (width-size)//2, width-size})
    ys = sorted({0, height-size})
    tiles = [(x, y, x+size, y+size) for y in ys for x in xs]
    key = (width, height, target)
    hints = getattr(detector, '_call_tile_hints', {})
    hint = hints.get(key)
    candidates = [d for d in raw if d.label in CALL_LABELS
                  and math.isfinite(d.confidence)
                  and all(math.isfinite(v) for v in d.bbox_xyxy)
                  and 0 <= d.bbox_xyxy[0] < d.bbox_xyxy[2] <= width
                  and 0 <= d.bbox_xyxy[1] < d.bbox_xyxy[3] <= height]
    if hint in tiles:
        tile = hint
    elif candidates:
        candidate = max(candidates, key=lambda d: d.confidence)
        cx, cy = candidate.center_pixel
        tile = min(tiles, key=lambda t: (cx-(t[0]+t[2])/2)**2+(cy-(t[1]+t[3])/2)**2)
    else:
        index = int(getattr(detector, '_call_tile_cursor', 0))
        tile = tiles[index % len(tiles)]
        detector._call_tile_cursor = index+1
    x, y, right, bottom = tile
    audit.update(attempted=True, tile_xyxy=list(tile), algorithm='call_button_local_scale')
    started = time.perf_counter()
    retry = detector.detect(image[y:bottom, x:right])
    audit['retry_inference_sec'] = time.perf_counter()-started
    mapped = []
    for d in retry:
        if d.label not in CALL_LABELS or not math.isfinite(d.confidence) or d.confidence < threshold:
            continue
        a,b,c,e = d.bbox_xyxy
        if not all(math.isfinite(v) for v in (a,b,c,e)) or not (0<=a<c<=size and 0<=b<e<=size):
            continue
        # Reject clipping introduced by the crop. Real image-edge clipping is
        # retained as a search cue; downstream full-visibility gates still apply.
        if ((x>0 and a<5) or (y>0 and b<5) or
                (right<width and c>size-5) or (bottom<height and e>size-5)):
            continue
        box = (a+x,b+y,c+x,e+y)
        if c-a < 14 or e-b < 14:
            continue
        mapped.append(ButtonDetection(d.class_id,d.label,d.confidence,box))
    # Conflicting labels at the same place cannot authorize search or pressing.
    if any(a.label != b.label and a.confidence >= threshold and b.confidence >= threshold
           and overlap(a.bbox_xyxy,b.bbox_xyxy) > .5
           for a in (*raw,*mapped) for b in mapped):
        audit['reason'] = 'conflicting_call_labels'
        hints.pop(key,None); detector._call_tile_hints=hints
        return raw,audit
    if not mapped:
        hints.pop(key,None); detector._call_tile_hints=hints
        audit['reason'] = 'no_qualified_tile_detection'
        return raw,audit
    merged = list(raw)
    for d in mapped:
        matches = [i for i,a in enumerate(merged) if a.label == d.label
                   and overlap(a.bbox_xyxy,d.bbox_xyxy) > .5]
        if matches:
            i=max(matches,key=lambda i:merged[i].confidence)
            if d.confidence > merged[i].confidence: merged[i]=d
        else:
            merged.append(d)
    hints[key]=tile; detector._call_tile_hints=hints
    audit.update(accepted=True,source='call_tile',
                 tile_detections=[dict(label=d.label,confidence=d.confidence,
                                      bbox_xyxy=list(d.bbox_xyxy)) for d in mapped])
    return tuple(merged),audit
