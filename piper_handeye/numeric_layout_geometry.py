"""Pure numeric-panel geometry frozen from the October 27 offline trials.

No file/model/camera/robot I/O. Counterpart tools retain their original source
for experiment reproducibility. All outputs are geometry estimates, never YOLO
labels or ON/OFF evidence. Production use is gated by numeric_layout_localization.
"""
import math
import cv2
import numpy as np

def project(h, points):
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    q = np.column_stack((p, np.ones(len(p)))) @ h.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-08):
        raise ValueError('projective_horizon')
    return q[:, :2] / q[:, 2:]

def strong_candidate(template, anchors, target, image_size):
    """Conservative experimental gate; returned pixels are NEVER press targets."""
    result = dict(accepted=False, purpose='offline_2d_research_only')
    try:
        if target not in template:
            raise ValueError('target_not_in_verified_template')
        anchors = [a for a in anchors if a['label'] != target]
        labels = [a['label'] for a in anchors]
        if len(labels) != len(set(labels)):
            raise ValueError('duplicate_anchor_label')
        if any((l not in template for l in labels)):
            raise ValueError('unknown_anchor')
        result['anchors'] = labels
        if len(anchors) < 5:
            raise ValueError('fewer_than_five_anchors')
        src = np.array([template[l] for l in labels], dtype=float)
        dst = np.array([a['center'] for a in anchors], dtype=float)
        diam = float(np.median([a['diameter'] for a in anchors]))
        if not np.isfinite(src).all() or not np.isfinite(dst).all() or (not math.isfinite(diam)) or (diam < 8):
            raise ValueError('invalid_anchor_geometry')
        if min((sum(src[:, 0] == x) for x in (0, 1))) < 2 or len(set(src[:, 1])) < 3:
            raise ValueError('insufficient_two_column_row_coverage')
        for i in range(len(dst)):
            if any((np.linalg.norm(dst[i] - dst[j]) < 0.35 * diam for j in range(i))):
                raise ValueError('overlapping_anchor_labels')
        limit = min(4.0, max(2.0, 0.08 * diam))
        cv2.setRNGSeed(0)
        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, limit)
        if h is None or mask is None:
            raise ValueError('degenerate_fit')
        result['ransac_outlier_labels'] = [l for l, ok in zip(labels, mask.ravel()) if not ok]
        all_h, _ = cv2.findHomography(src, dst, 0)
        if all_h is not None and max(np.linalg.norm(project(all_h, src) - dst, axis=1)) < max(np.linalg.norm(project(h, src) - dst, axis=1)):
            h = all_h
        residual = np.linalg.norm(project(h, src) - dst, axis=1)
        result.update(max_anchor_error_px=float(max(residual)), rms_anchor_error_px=float(np.sqrt(np.mean(residual ** 2))))
        if max(residual) > limit:
            raise ValueError('anchor_layout_conflict')
        loo = []
        for i in range(len(src)):
            s, d = (np.delete(src, i, 0), np.delete(dst, i, 0))
            if min((sum(s[:, 0] == x) for x in (0, 1))) < 2:
                continue
            hi, _ = cv2.findHomography(s, d, 0)
            if hi is None:
                raise ValueError('unstable_leave_one_out_fit')
            loo.append(float(np.linalg.norm(project(hi, [src[i]])[0] - dst[i])))
        result['max_leave_one_out_px'] = max(loo, default=None)
        if not loo or max(loo) > max(4.0, 0.12 * diam):
            raise ValueError('leave_one_out_unstable')
        t = np.array(template[target], dtype=float)
        if t[1] < min(src[:, 1]) - 1 or t[1] > max(src[:, 1]) + 1:
            raise ValueError('extrapolation_more_than_one_row')
        p, right, down = project(h, [t, t + [0.1, 0], t + [0, 0.1]])
        if right[0] <= p[0] or down[1] <= p[1]:
            raise ValueError('inverted_panel_orientation')
        w, hh = image_size
        if not (5 + diam / 2 <= p[0] <= w - 5 - diam / 2 and 5 + diam / 2 <= p[1] <= hh - 5 - diam / 2):
            raise ValueError('predicted_target_not_fully_visible')
        result.update(accepted=True, center=p.tolist(), estimated_diameter_px=diam, reason='qualified_offline_estimate')
    except (ValueError, cv2.error) as exc:
        result['reason'] = str(exc)
    return result

def local_scale(h, point):
    p = np.asarray(point, float)
    v = project(h, [p, p + [0.001, 0], p + [0, 0.001]])
    jac = np.column_stack(((v[1] - v[0]) / 0.001, (v[2] - v[0]) / 0.001))
    return math.sqrt(abs(float(np.linalg.det(jac))))

def qualified_candidate(template, anchors, target, size):
    result = dict(accepted=False, purpose='offline_2d_research_only', method='layout_plus_complete_rim_v2')
    try:
        if target not in template:
            raise ValueError('unknown_target')
        aa = [a for a in anchors if a['label'] != target]
        labels = [a['label'] for a in aa]
        result['anchors'] = labels
        if len(labels) != len(set(labels)):
            raise ValueError('duplicate_anchor_label')
        if len(aa) < 4:
            raise ValueError('fewer_than_four_anchors')
        if any((l not in template for l in labels)):
            raise ValueError('unknown_anchor')
        src = np.array([template[l] for l in labels], float)
        dst = np.array([a['center'] for a in aa], float)
        diameters = np.array([a['diameter'] for a in aa], float)
        diam = float(np.median(diameters))
        if not np.isfinite(src).all() or not np.isfinite(dst).all() or (not np.isfinite(diameters).all()) or (min(diameters) < 8):
            raise ValueError('invalid_geometry')
        if min((sum(src[:, 0] == x) for x in (0, 1))) < 2 or len(set(src[:, 1])) < 2:
            raise ValueError('insufficient_two_column_coverage')
        for i in range(len(dst)):
            if any((np.linalg.norm(dst[i] - dst[j]) < 0.35 * diam for j in range(i))):
                raise ValueError('overlapping_anchor_labels')
        h, _ = cv2.findHomography(src, dst, 0)
        if h is None:
            raise ValueError('degenerate_fit')
        errors = np.linalg.norm(project(h, src) - dst, axis=1)
        result['max_anchor_error_px'] = float(max(errors))
        if max(errors) > min(4, max(2, 0.08 * diam)):
            raise ValueError('anchor_layout_conflict')
        t = np.array(template[target], float)
        if t[1] < min(src[:, 1]) - 1 or t[1] > max(src[:, 1]) + 1:
            raise ValueError('extrapolation_more_than_one_row')
        p, right, down = project(h, [t, t + [0.1, 0], t + [0, 0.1]])
        if right[0] <= p[0] or down[1] <= p[1]:
            raise ValueError('inverted_panel')
        scales = np.array([local_scale(h, s) for s in src])
        if not np.isfinite(scales).all() or min(scales) < 1e-06:
            raise ValueError('invalid_local_scale')
        normalized = diameters / scales
        median = float(np.median(normalized))
        mismatch = float(max(np.abs(normalized / median - 1)))
        result['anchor_size_scale_mismatch'] = mismatch
        if mismatch > 0.25:
            raise ValueError('anchor_size_geometry_conflict')
        target_diameter = median * local_scale(h, t)
        if not math.isfinite(target_diameter) or not 0.6 * diam <= target_diameter <= 1.6 * diam:
            raise ValueError('target_scale_unstable')
        affine_src = np.column_stack((src, np.ones(len(src))))
        affine, _, rank, _ = np.linalg.lstsq(affine_src, dst, rcond=None)
        if rank < 3:
            raise ValueError('degenerate_affine')
        pa = np.append(t, 1) @ affine
        disagreement = float(np.linalg.norm(p - pa))
        result['affine_projective_disagreement_px'] = disagreement
        if len(aa) == 4 and disagreement > max(6, 0.15 * target_diameter):
            raise ValueError('model_disagreement')
        leave_predictions = []
        for i in range(len(src)):
            ss, dd = (np.delete(src, i, 0), np.delete(dst, i, 0))
            if len(ss) < 4 or min((sum(ss[:, 0] == x) for x in (0, 1))) < 2:
                continue
            hh, _ = cv2.findHomography(ss, dd, 0)
            if hh is not None:
                leave_predictions.append(project(hh, [t])[0])
        spread = max((float(np.linalg.norm(x - p)) for x in leave_predictions), default=0.0)
        result['leave_one_anchor_target_spread_px'] = spread
        if spread > max(6, 0.15 * target_diameter):
            raise ValueError('target_prediction_unstable')
        w, hh = size
        if not (5 + target_diameter / 2 <= p[0] <= w - 5 - target_diameter / 2 and 5 + target_diameter / 2 <= p[1] <= hh - 5 - target_diameter / 2):
            raise ValueError('target_not_fully_visible')
        result.update(accepted=True, center=p.tolist(), estimated_diameter_px=target_diameter, reason='candidate_requires_rim_check', four_anchor_mode=len(aa) == 4)
    except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        result['reason'] = str(exc)
    return result

def verify_rim(image, center, diameter):
    """No digit label/truth/ON state input: only bounded local edge geometry."""
    fail = dict(accepted=False, reason='no_complete_button_rim')
    if image is None or image.ndim not in (2, 3):
        return fail
    p = np.asarray(center, float)
    if not np.isfinite(p).all() or not math.isfinite(diameter) or diameter < 8:
        return fail
    h, w = image.shape[:2]
    rad = 0.85 * diameter
    x1 = max(0, int(math.floor(p[0] - rad)))
    y1 = max(0, int(math.floor(p[1] - rad)))
    x2 = min(w, int(math.ceil(p[0] + rad)))
    y2 = min(h, int(math.ceil(p[1] + rad)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return fail
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    edge = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90)
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    body_contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = list(contours) + list(body_contours)
    choices = []
    for contour in contours:
        if len(contour) < 16:
            continue
        (cx, cy), (a, b), angle = cv2.fitEllipse(contour)
        if min(a, b) < 0.65 * diameter or max(a, b) > 1.35 * diameter or max(a, b) / max(min(a, b), 1e-06) > 1.7:
            continue
        global_center = np.array([cx + x1, cy + y1])
        if np.linalg.norm(global_center - p) > min(8, 0.15 * diameter):
            continue
        if min(global_center[0] - max(a, b) / 2, global_center[1] - max(a, b) / 2, w - global_center[0] - max(a, b) / 2, h - global_center[1] - max(a, b) / 2) < 5:
            continue
        theta = np.deg2rad(angle)
        rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        points = (contour[:, 0, :] - [cx, cy]) @ rot.T
        normalized = points / [a / 2, b / 2]
        rho = np.linalg.norm(normalized, axis=1)
        angles = np.arctan2(normalized[:, 1], normalized[:, 0])
        bins = np.unique(np.floor((angles + np.pi) / (2 * np.pi) * 36).astype(int) % 36)
        gaps = np.diff(np.r_[np.sort(angles), min(angles) + 2 * np.pi])
        fit = float(np.percentile(abs(rho - 1), 90))
        area = cv2.contourArea(contour) / (np.pi * a * b / 4)
        if len(bins) < 28 or max(gaps) > np.pi / 3 or fit > 0.1 or (not 0.8 <= area <= 1.15):
            continue
        choices.append(dict(center=global_center.tolist(), axes=[a, b], angle=angle, fit_p90=fit, angular_bins=len(bins), area_fraction=area))
    if not choices:
        return fail
    centers = np.array([c['center'] for c in choices])
    median = np.median(centers, axis=0)
    if max(np.linalg.norm(centers - median, axis=1)) > max(2, 0.05 * diameter):
        return dict(accepted=False, reason='ambiguous_button_rims', candidates=choices)
    best = min(choices, key=lambda c: c['fit_p90'])
    return dict(accepted=True, reason='complete_rim_verified', center=median.tolist(), matching_contours=len(choices), rim=best, correction_px=float(np.linalg.norm(median - p)))

def sparse_candidate(template, anchors, target, size):
    result = dict(accepted=False, purpose='offline_inferred_2d_only', target_detected=False, method='sparse_geometry')
    try:
        if target not in template:
            raise ValueError('unknown_target')
        aa = [a for a in anchors if a['label'] != target]
        labels = [a['label'] for a in aa]
        result['anchors'] = labels
        if len(labels) != len(set(labels)):
            raise ValueError('duplicate_anchor_label')
        if any((l not in template for l in labels)):
            raise ValueError('unknown_anchor')
        if len(aa) not in (3, 4):
            raise ValueError('sparse_mode_requires_three_or_four')
        s = np.array([template[l] for l in labels], float)
        d = np.array([a['center'] for a in aa], float)
        ds = np.array([a['diameter'] for a in aa], float)
        median = float(np.median(ds))
        t = np.array(template[target], float)
        if not np.isfinite(s).all() or not np.isfinite(d).all() or (not np.isfinite(ds).all()) or (min(ds) < 8):
            raise ValueError('invalid_geometry')
        if min((sum(s[:, 0] == x) for x in (0, 1))) < 1:
            raise ValueError('single_column_only')
        for i in range(len(d)):
            if any((np.linalg.norm(d[i] - d[j]) < 0.35 * median for j in range(i))):
                raise ValueError('overlapping_anchor_labels')
        affine_s = np.column_stack((s, np.ones(len(s))))
        affine, _, rank, _ = np.linalg.lstsq(affine_s, d, rcond=None)
        if rank < 3:
            raise ValueError('degenerate_affine')
        ap = np.append(t, 1) @ affine
        if len(aa) == 3:
            own = s[s[:, 0] == t[0]]
            other = s[s[:, 0] != t[0]]
            if len(own) != 2 or len(other) != 1 or (not min(own[:, 1]) < t[1] < max(own[:, 1])) or (other[0, 1] != t[1]):
                raise ValueError('three_anchor_target_not_bracketed_with_row_partner')
            if max(own[:, 1]) - min(own[:, 1]) > 3:
                raise ValueError('bracketing_gap_too_large')
            if any((a.get('confidence', 1) < 0.85 for a in aa)):
                raise ValueError('three_anchor_confidence_low')
            h = np.vstack((affine.T, [0, 0, 1]))
            p = ap
            result['method'] = 'three_anchor_bracket_and_row_partner'
        else:
            if min((sum(s[:, 0] == x) for x in (0, 1))) < 2:
                raise ValueError('four_anchors_need_two_per_column')
            h, _ = cv2.findHomography(s, d, 0)
            if h is None:
                raise ValueError('degenerate_fit')
            if t[1] < min(s[:, 1]) - 1 or t[1] > max(s[:, 1]) + 1:
                raise ValueError('extrapolation_more_than_one_row')
            p = project(h, [t])[0]
        scales = np.array([local_scale(h, point) for point in s])
        ratios = ds / scales
        if not np.isfinite(scales).all() or min(scales) < 1e-06 or (not np.isfinite(ratios).all()):
            raise ValueError('invalid_scale')
        mismatch = float(max(abs(ratios / np.median(ratios) - 1)))
        result['anchor_size_scale_mismatch'] = mismatch
        if mismatch > 0.25:
            raise ValueError('anchor_size_geometry_conflict')
        diameter = float(np.median(ratios) * local_scale(h, t))
        if not 0.6 * median <= diameter <= 1.6 * median:
            raise ValueError('target_scale_unstable')
        disagreement = float(np.linalg.norm(ap - p))
        result['affine_projective_disagreement_px'] = disagreement
        if disagreement > max(12, 0.25 * diameter):
            raise ValueError('model_disagreement')
        pp, right, down = project(h, [t, t + [0.1, 0], t + [0, 0.1]])
        if right[0] <= pp[0] or down[1] <= pp[1]:
            raise ValueError('inverted_panel')
        w, hh = size
        if not (5 + diameter / 2 <= p[0] <= w - 5 - diameter / 2 and 5 + diameter / 2 <= p[1] <= hh - 5 - diameter / 2):
            raise ValueError('target_not_fully_visible')
        result.update(accepted=True, center=p.tolist(), estimated_diameter_px=diameter, reason='geometry_inferred_not_visual_detection')
    except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        result['reason'] = str(exc)
    return result

def infer_layout(template, anchors, target, image):
    size = [image.shape[1], image.shape[0]]
    result = strong_candidate(template, anchors, target, size)
    if result['accepted']:
        result['method'] = 'existing_five_plus_geometry'
    else:
        original_reason = result['reason']
        result = qualified_candidate(template, anchors, target, size)
        if not result['accepted']:
            result = sparse_candidate(template, anchors, target, size)
        result['previous_reason'] = original_reason
    result.update(target_detected=False, purpose='offline_inferred_2d_only', light_state='not_evaluated', requires_live_depth_and_motion_validation=True)
    if not result['accepted']:
        return result
    result['layout_center'] = result['center']
    result['rim_verification'] = verify_rim(image, result['center'], result['estimated_diameter_px'])
    rim = result['rim_verification']
    if rim['accepted']:
        result.update(center=rim['center'], reason='geometry_with_optional_rim_refinement')
    else:
        result['reason'] = 'geometry_accepted_without_complete_rim'
    return result

def recover_one_conflict(template, anchors, target, image):
    """Rare last resort: four strong agreeing labels may expose one wrong label.

    Never modify raw detections or convert this result to a YOLO detection.
    Requires >=80% retained references and a unique consistent target estimate.
    """
    aa = [a for a in anchors if a['label'] != target]
    if not 5 <= len(aa) <= 8:
        return dict(accepted=False, reason='single_outlier_recovery_needs_five_to_eight', target_detected=False)
    choices = []
    for i, bad in enumerate(aa):
        kept = aa[:i] + aa[i + 1:]
        if any((a.get('confidence', 0) < 0.85 for a in kept)):
            continue
        result = infer_layout(template, kept, target, image)
        if not result['accepted']:
            continue
        s = np.array([template[a['label']] for a in kept], float)
        d = np.array([a['center'] for a in kept], float)
        h, _ = cv2.findHomography(s, d, 0)
        if h is None:
            continue
        predicted = project(h, [template[bad['label']]])[0]
        error = float(np.linalg.norm(predicted - bad['center']))
        if error <= max(12, 0.75 * bad['diameter']):
            continue
        result['excluded_conflicting_anchor'] = dict(label=bad['label'], confidence=bad.get('confidence'), observed_center=bad['center'], predicted_center=predicted.tolist(), conflict_px=error)
        result['retained_fraction'] = len(kept) / len(aa)
        choices.append(result)
    if not choices:
        return dict(accepted=False, reason='no_unique_single_outlier_solution', target_detected=False)
    centers = np.array([r['center'] for r in choices])
    median = np.median(centers, axis=0)
    if max(np.linalg.norm(centers - median, axis=1)) > min(6, 0.2 * choices[0]['estimated_diameter_px']):
        return dict(accepted=False, reason='single_outlier_solutions_disagree', target_detected=False)
    result = choices[0]
    result['method'] = 'four_plus_strong_references_reject_one_conflict'
    result['consistent_solutions'] = len(choices)
    return result
