"""Shared full-button geometry gate for stationary lamp observations."""
import math
from collections import Counter

OBSERVATION_EDGE_MARGIN_PX = 5.0


def complete_button_box(box, image_size, *, margin=OBSERVATION_EDGE_MARGIN_PX):
    try:
        x1,y1,x2,y2=map(float,box);width,height=map(float,image_size)
        return (all(math.isfinite(v) for v in (x1,y1,x2,y2,width,height))
                and x1>=margin and y1>=margin and x2<=width-margin and y2<=height-margin
                and 14<=x2-x1<=width*.85 and 14<=y2-y1<=height*.85)
    except (TypeError,ValueError):
        return False


def stable_target_visibility(estimate, audit, image_size):
    """Check boxes from the returned stable window, not rejected audit rows.

    Sampling has already validated freshness, depth, pose and spatial stability.
    Match its exact window count and endpoints before reading the last two
    accepted boxes. Never recover boxes from an earlier window or rejected row.
    """
    result = dict(complete=False, reason='stable_target_not_ready',
                  policy='accepted_samples_in_returned_stable_window',
                  required_box_frames=2, checked_frames=[])
    if not getattr(estimate, 'stable', False):
        return result
    sequence_range = getattr(estimate, 'sequence_range', None)
    count = getattr(estimate, 'sample_count', None)
    if (not isinstance(sequence_range, (list, tuple)) or len(sequence_range) != 2
            or any(type(v) is not int for v in sequence_range)
            or not 0 < sequence_range[0] < sequence_range[1]
            or type(count) is not int or count < 2):
        result['reason'] = 'stable_target_window_metadata_invalid'
        return result
    low, high = sequence_range
    result.update(window_sequence_range=list(sequence_range), window_sample_count=count)
    if not isinstance(audit, (list, tuple)) or any(not isinstance(r, dict) for r in audit):
        result['reason'] = 'stable_target_audit_invalid'
        return result
    result['ignored_rejected_reasons'] = dict(Counter(
        str(r.get('reason') or 'unspecified') for r in audit if r.get('accepted') is not True))
    selected = [r for r in audit if r.get('accepted') is True
                and type(r.get('sequence')) is int and low <= r['sequence'] <= high]
    sequences = [r['sequence'] for r in selected]
    result['window_sequences'] = sequences
    if len(set(sequences)) != len(sequences):
        result['reason'] = 'stable_target_window_duplicate_sequence'
        return result
    if (len(selected) != count or not sequences or sequences[0] != low
            or sequences[-1] != high or sequences != sorted(sequences)):
        result['reason'] = 'stable_target_window_evidence_missing'
        return result
    result['checked_frames'] = [dict(sequence=r['sequence'], source=r.get('source', 'yolo'),
        bbox_xyxy=r.get('bbox_xyxy'),
        complete=complete_button_box(r.get('bbox_xyxy'), image_size)) for r in selected[-2:]]
    result['complete'] = all(r['complete'] for r in result['checked_frames'])
    result['reason'] = 'complete' if result['complete'] else 'stable_target_box_incomplete'
    return result
