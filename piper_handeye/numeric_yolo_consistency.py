"""Cheap, same-frame checks of a seen numeric label against a known panel.

Only strong independent neighbours can veto a label. Insufficient evidence
leaves the original admission unchanged. Never relabel a detection or infer ON.
"""
import math
import cv2
import numpy as np


SOURCE = 'known_panel_neighbour_consistency'


def target_conflicted(record, *, target=None, sequence=None, digest=None):
    return (isinstance(record, dict) and record.get('version') == 1
            and record.get('source') == SOURCE and record.get('enabled') is True
            and record.get('decision') == 'conflict'
            and (target is None or str(record.get('target_label')) == str(target))
            and (sequence is None or record.get('frame_sequence') == sequence)
            and (digest is None or record.get('profile_sha256') == digest))


def admitted_detections(detections, record, *, target=None, sequence=None, digest=None):
    if not target_conflicted(record, target=target, sequence=sequence, digest=digest):
        return detections
    label = str(record['target_label'])
    return tuple(d for d in detections if d.label != label)


def _inspect_numeric_target_global(coordinates, detections, target, image_size, *,
                           min_target_confidence=.70):
    """Require agreement of projective and affine neighbour fits to veto.

    At least four >=.85 references, two in each column, independent of the
    suspect target. A veto requires a different known cell within the observed
    reference rows; extrapolation of the requested target cannot alone veto.
    """
    result = dict(version=1, source=SOURCE, enabled=True, target_label=str(target),
                  decision='unverified', reason='insufficient_independent_neighbours')
    try:
        if target not in coordinates:
            result['reason'] = 'outside_known_panel'; return result
        width, height = image_size
        def valid(d, confidence):
            b=np.asarray(d.bbox_xyxy,float)
            return (math.isfinite(d.confidence) and confidence<=d.confidence<=1.
                    and b.shape==(4,) and np.isfinite(b).all()
                    and 5<=b[0]<b[2]<=width-5 and 5<=b[1]<b[3]<=height-5
                    and min(b[2]-b[0],b[3]-b[1])>=14)
        candidates=[d for d in detections if d.label==target and valid(d,min_target_confidence)]
        if len(candidates)!=1:
            result['reason']='no_unique_complete_target'; return result
        candidate=candidates[0]; observed=np.asarray(candidate.center_pixel,float)
        result.update(target_bbox_xyxy=list(candidate.bbox_xyxy),
                      target_confidence=float(candidate.confidence))
        refs=[d for d in detections if d.label!=target and d.label in coordinates and valid(d,.85)]
        counts={d.label:sum(a.label==d.label for a in refs) for d in refs}
        refs=[d for d in refs if counts[d.label]==1
              and np.linalg.norm(np.asarray(d.center_pixel)-observed)>
                  .6*max(d.bbox_xyxy[2]-d.bbox_xyxy[0],candidate.bbox_xyxy[2]-candidate.bbox_xyxy[0])]
        if len(refs)<4:return result
        src=np.asarray([coordinates[d.label] for d in refs],float)
        dst=np.asarray([d.center_pixel for d in refs],float)
        if min(sum(src[:,0]==col) for col in (0,1))<2:return result
        diameter=float(np.median([(d.bbox_xyxy[2]-d.bbox_xyxy[0]+d.bbox_xyxy[3]-d.bbox_xyxy[1])/2 for d in refs]))
        limit=min(5.,max(2.,.12*diameter))
        h,mask=cv2.findHomography(src,dst,0 if len(refs)==4 else cv2.RANSAC,
                                 limit,maxIters=100,confidence=.995)
        if h is None or mask is None:return result
        keep=mask.ravel().astype(bool)
        if sum(keep)<4 or sum(keep)<.8*len(refs):return result
        src,dst=src[keep],dst[keep]; kept=[d for d,ok in zip(refs,keep) if ok]
        if min(sum(src[:,0]==col) for col in (0,1))<2:return result
        affine,_,rank,_=np.linalg.lstsq(np.column_stack((src,np.ones(len(src)))),dst,rcond=None)
        if rank!=3 or np.max(np.linalg.norm(np.column_stack((src,np.ones(len(src))))@affine-dst,axis=1))>limit:
            result['reason']='neighbour_models_disagree';return result
        if affine[0,0]<=0 or affine[1,1]<=0:return result
        inv=np.linalg.inv(h); point=inv@np.r_[observed,1.]
        if abs(point[2])<1e-9:return result
        canonical=point[:2]/point[2]
        affine_point=np.linalg.solve(affine[:2].T,observed-affine[2])
        if not np.isfinite(canonical).all() or np.linalg.norm(canonical-affine_point)>.15:
            result['reason']='target_prediction_unstable';return result
        labels=list(coordinates)
        distances={label:float(np.linalg.norm(canonical-coordinates[label])) for label in labels}
        nearest=min(distances,key=distances.get)
        result.update(anchors=[d.label for d in kept], canonical_xy=canonical.tolist(),
                      nearest_cell_label=nearest, nearest_cell_distance=distances[nearest],
                      claimed_cell_distance=distances[target])
        if distances[target]<=.25:
            result.update(decision='consistent',reason='label_matches_independent_neighbours')
        elif (nearest!=target and distances[nearest]<=.20 and distances[target]>=.65
              and min(src[:,1])-.1<=coordinates[nearest][1]<=max(src[:,1])+.1):
            expected=h@np.r_[coordinates[target],1.]
            if abs(expected[2])<1e-9:return result
            result.update(decision='conflict',reason='label_conflicts_with_independent_neighbours',
                          alternative_label=nearest,expected_target_center_xy=(expected[:2]/expected[2]).tolist())
        else:
            result['reason']='no_strong_alternative_cell'
    except (ValueError,TypeError,KeyError,cv2.error,np.linalg.LinAlgError) as exc:
        result.update(decision='unverified',reason='invalid_or_unstable_neighbour_geometry',error=str(exc))
    return result


LOCAL_35_CHECK_VERSION = 1
_LOCAL_35_NEIGHBOURS = {'3': ('4', '2', '10'), '5': ('6', '4', '12')}


def _inspect_local_35(coordinates, detections, target, image_size, min_confidence):
    """Check a measured cell between its upper/lower neighbours, beside its right neighbour.

    This is an affine, same-frame local check for the current known panel, not a
    general three-point projective calibration. Only a bracketed measured 3/5
    cell may veto its mutually confused label; no extrapolated motion ROI.
    """
    out = dict(version=LOCAL_35_CHECK_VERSION, decision='unverified',
               reason='local_35_neighbours_unavailable', cells=[])
    try:
        width, height = image_size
        def valid(d, confidence):
            b = np.asarray(d.bbox_xyxy, float)
            return (math.isfinite(d.confidence) and confidence <= d.confidence <= 1.
                    and b.shape == (4,) and np.isfinite(b).all()
                    and 5 <= b[0] < b[2] <= width-5
                    and 5 <= b[1] < b[3] <= height-5
                    and min(b[2]-b[0], b[3]-b[1]) >= 14)
        candidates = [d for d in detections if d.label == target and valid(d, min_confidence)]
        if len(candidates) != 1:
            out['reason'] = 'no_unique_complete_target'; return out
        candidate = candidates[0]
        observed = np.asarray(candidate.center_pixel, float)
        references = [d for d in detections if d.label != target and valid(d, .85)]
        matches = []
        for cell, labels in _LOCAL_35_NEIGHBOURS.items():
            audit = dict(cell_label=cell, anchor_labels=list(labels), accepted=False)
            out['cells'].append(audit)
            if any(label not in coordinates for label in (cell, *labels)):
                audit['reason'] = 'unknown_layout'; continue
            canonical_center = np.asarray(coordinates[cell], float)
            canonical_refs = np.asarray([coordinates[label] for label in labels], float)
            if not np.allclose(canonical_refs-canonical_center, [[0,-1],[0,1],[1,0]], atol=1e-8):
                audit['reason'] = 'different_panel_layout'; continue
            groups = [[d for d in references if d.label == label] for label in labels]
            if any(len(group) != 1 for group in groups):
                audit['reason'] = 'missing_or_duplicate_anchor'; continue
            refs = [group[0] for group in groups]
            centers = np.asarray([d.center_pixel for d in refs], float)
            upper, lower, right = centers
            midpoint = (upper+lower)/2
            row_vector = (lower-upper)/2
            column_vector = right-midpoint
            row_pitch, column_pitch = np.linalg.norm(row_vector), np.linalg.norm(column_vector)
            boxes = [np.asarray(d.bbox_xyxy, float) for d in (*refs, candidate)]
            diameters = np.asarray([((b[2]-b[0])+(b[3]-b[1]))/2 for b in boxes])
            median_diameter = float(np.median(diameters))
            # Independent, non-overlapping anchors with a plausible two-column view.
            independent = all(
                np.linalg.norm(centers[i]-centers[j]) > .6*max(diameters[i],diameters[j])
                for i in range(3) for j in range(i)) and all(
                np.linalg.norm(center-observed) > .6*max(diameters[i],diameters[-1])
                for i, center in enumerate(centers))
            if (not independent or not np.isfinite(centers).all()
                    or not 1.05*median_diameter <= row_pitch <= 3.5*median_diameter
                    or not 1.05*median_diameter <= column_pitch <= 4.*median_diameter
                    or diameters.max()/diameters.min() > 1.75
                    or row_vector[1] <= 0 or column_vector[0] <= 0
                    or abs(row_vector[0]) > .6*row_vector[1]
                    or abs(column_vector[1]) > .6*column_vector[0]):
                audit['reason'] = 'invalid_local_geometry'; continue
            basis = np.column_stack((column_vector, row_vector))
            sine = abs(float(np.linalg.det(basis)))/(row_pitch*column_pitch)
            if sine < .70:
                audit['reason'] = 'degenerate_anchor_triangle'; continue
            offset = np.linalg.solve(basis, observed-midpoint)
            canonical = canonical_center+offset
            distance = float(np.linalg.norm(offset))
            audit.update(accepted=True, reason='local_bracket_measured',
                         expected_center_xy=midpoint.tolist(),
                         canonical_xy=canonical.tolist(), cell_distance=distance,
                         row_pitch_px=float(row_pitch), column_pitch_px=float(column_pitch))
            if distance <= .20:
                matches.append((cell, labels, canonical, distance))
        if len(matches) != 1:
            out['reason'] = 'local_models_ambiguous' if matches else 'no_supported_local_cell'
            return out
        cell, labels, canonical, distance = matches[0]
        claimed_distance = float(np.linalg.norm(canonical-np.asarray(coordinates[target],float)))
        out.update(anchors=list(labels), nearest_cell_label=cell,
                   nearest_cell_distance=distance, claimed_cell_distance=claimed_distance,
                   canonical_xy=canonical.tolist(), target_bbox_xyxy=list(candidate.bbox_xyxy),
                   target_confidence=float(candidate.confidence))
        if cell == target:
            out.update(decision='consistent', reason='label_matches_local_three_neighbours')
        elif {cell, target} == {'3', '5'} and claimed_distance >= 1.5:
            out.update(decision='conflict', reason='label_conflicts_with_local_three_neighbours',
                       alternative_label=cell)
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError, ZeroDivisionError):
        out.update(decision='unverified', reason='invalid_local_geometry')
    return out


def inspect_numeric_target(coordinates, detections, target, image_size, *,
                           min_target_confidence=.70):
    result = _inspect_numeric_target_global(coordinates, detections, target, image_size,
                                           min_target_confidence=min_target_confidence)
    # Preserve all existing decisive global decisions and every other label.
    if target not in _LOCAL_35_NEIGHBOURS or result['decision'] != 'unverified':
        return result
    local = _inspect_local_35(coordinates, detections, target, image_size, min_target_confidence)
    result['local_35_check'] = local
    if local['decision'] in {'consistent', 'conflict'}:
        result['global_check'] = dict(decision=result['decision'], reason=result['reason'])
        result.update({key: value for key, value in local.items() if key not in {'version', 'cells'}})
    return result
