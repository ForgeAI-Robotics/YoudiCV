"""DOWN-only retry scheduling. A cached crop is never a cached detection."""
import math
import time


def up_cues(detections):
    cues = []
    for d in detections:
        if d.label != 'up' or not math.isfinite(d.confidence) or d.confidence < .15:
            continue
        x1, y1, x2, y2 = d.bbox_xyxy
        if (all(math.isfinite(v) for v in d.bbox_xyxy)
                and x1 >= 0 and y1 >= 0 and x2-x1 >= 14 and y2-y1 >= 14):
            cues.append(d)
    return cues


class DownRetryTiles:
    """Try the existing tile twice, then other existing tiles one per frame.

    The original stationary/IK/search/light gates remain authoritative. This
    class only changes which current pixels receive the one local inference.
    """
    def __init__(self, context):
        self.context = context
        self.misses = 0
        self.cursor = 0
        self.last_call = None
        self.anchor = None
        self.used_alternative = False

    def choose(self, baseline, tiles, raw):
        now = time.monotonic()
        cues = up_cues(raw)
        cue = max(cues, key=lambda d: d.confidence) if cues else None
        center = cue.center_pixel if cue else None
        moved = (center is not None and self.anchor is not None
                 and math.dist(center, self.anchor) > 16)
        if (self.last_call is not None and now-self.last_call > 1.) or moved:
            self.misses = self.cursor = 0
        self.last_call, self.anchor = now, center
        self.used_alternative = False
        # Keep the old path until it has actually missed DOWN twice. Without
        # a current UP cue, retain the old scan rather than search unrelated UI.
        if self.misses < 2 or cue is None:
            return baseline
        cx, cy = center
        size = max(cue.bbox_xyxy[2]-cue.bbox_xyxy[0], cue.bbox_xyxy[3]-cue.bbox_xyxy[1])
        choices = [t for t in tiles if t != baseline
                   and t[0]-size <= cx <= t[2]+size
                   and t[1]-size <= cy <= t[3]+size]
        choices.sort(key=lambda t: ((cx-(t[0]+t[2])/2)**2
                                   + (cy+size-(t[1]+t[3])/2)**2, t))
        if self.cursor >= len(choices):
            self.misses = self.cursor = 0
            return baseline
        tile = choices[self.cursor]
        self.cursor += 1
        self.used_alternative = True
        return tile

    def finish(self, target_accepted):
        attempted = self.cursor
        if target_accepted:
            self.misses = self.cursor = 0
        else:
            self.misses += 1
        return dict(target_accepted=bool(target_accepted), consecutive_misses=self.misses,
                    alternative_attempt=attempted, used_alternative=self.used_alternative,
                    policy='legacy_first_then_existing_tiles')


def recovery_target_has_support(target, raw, local, threshold):
    """Require actual same-frame UP evidence near an alternative-tile DOWN.

    Geometry only rejects unrelated candidates; it never supplies a label or
    confidence, which must still come from YOLO on this frame.
    """
    for cue in up_cues([*raw, *local]):
        if cue.confidence < threshold:
            continue
        scale = max(cue.bbox_xyxy[2]-cue.bbox_xyxy[0], cue.bbox_xyxy[3]-cue.bbox_xyxy[1])
        dx = abs(target.center_pixel[0]-cue.center_pixel[0])
        dy = abs(target.center_pixel[1]-cue.center_pixel[1])
        if dx <= 2*scale and .5*scale <= dy <= 4*scale:
            return True
    return False
