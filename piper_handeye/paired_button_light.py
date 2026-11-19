"""Conservative, bounded before/after evidence for one button (no extra model).

An OFF reference is usable only after two adjacent OFF observations of a
comparable view. Comparison is frozen before contact. View/exposure changes
which cannot be explained by a small translation are UNKNOWN, never success.
Thresholds are provisional; recorded replay is not a substitute for live tests.
"""
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import time
import shutil

import cv2
import numpy as np
from .pipeline_timing import traced, stage
from .numeric_layout_handoff import selected_button_region
from .button_background_registration import register_reference
from .off_reference_alignment import (recompare_stationary_off,
    recompare_stationary_post_press_off, supports_off_reference_alignment)


@dataclass
class ButtonPatch:
    gray: np.ndarray
    box: np.ndarray
    sequence: int
    captured: float
    image: str
    evidence: str | None = None
    region_source: str = 'yolo'
    target_label: str = ''


def extract_patch(snapshot, *, max_frame_age_sec=.35):
    detection = selected_button_region(snapshot, max_frame_age_sec=max_frame_age_sec)
    if detection is None:
        raise ValueError('qualified_button_region_required')
    if int(detection['frame_sequence']) != int(snapshot['frame_sequence']):
        raise ValueError('image_box_frame_mismatch')
    image = cv2.imread(str(snapshot['path']))
    if image is None:
        raise ValueError('image_unreadable')
    box = np.asarray(detection['bbox_xyxy'], dtype=float)
    x1, y1, x2, y2 = np.rint(box).astype(int)
    h, w = image.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('invalid_crop')
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ButtonPatch(gray, box, int(snapshot['frame_sequence']),
                       float(snapshot['capture_time_unix_sec']), str(snapshot['path']),
                       region_source=detection.get('source','yolo'),
                       target_label=str(detection.get('label','')))


def geometry_error(a, b):
    size_a, size_b = a.box[2:] - a.box[:2], b.box[2:] - b.box[:2]
    ratio = np.abs(np.log(size_a / size_b))
    shift = np.abs((a.box[:2] + a.box[2:] - b.box[:2] - b.box[2:]) / 2) / size_a
    if np.max(ratio) > .16 or np.max(shift) > .35:
        return None
    return float(np.sum(ratio) + .1 * np.sum(shift))


_Y, _X = np.mgrid[:64, :64]
_RADIUS = np.hypot(_X - 31.5, _Y - 31.5) / 64
_CENTER = _RADIUS < .36
_RING = (_RADIUS > .44) & (_RADIUS < .62) & (_X > 3) & (_X < 60) & (_Y > 3) & (_Y < 60)
# A tight YOLO button box includes the circular chrome bezel around radius .5.
# Its reflection changes when the lamp lights. These four outer corner patches
# stay outside that bezel; their location is fixed before inspecting any change.
_PANEL_RING = _RING & (_RADIUS > .53)
_STRICT_BACKGROUND_LIMITS = (10., 25.)
_POST_PRESS_BACKGROUND_LIMITS = (15., 35.)
_NUMERIC_STRONG_ON_BACKGROUND_LIMITS = (20., 50.)
_NUMERIC_STRONG_ON_INLIER_ERROR = 30.
_NUMERIC_STRONG_ON_MAX_SPATIAL_ERROR = 30.
_NUMERIC_POST_PRESS_MAX_CHANGED_FRACTION = .99
_CLIPPED_BACKGROUND_MIN_VALID_FRACTION = .70
_CLIPPED_BACKGROUND_MIN_QUADRANT_FRACTION = .50
_CLIPPED_BACKGROUND_MAX_DARK_FRACTION = .02


def normalize_background(reference_gray, current_gray, *, ring_mask=None,
                         residual_limits=_STRICT_BACKGROUND_LIMITS,
                         allow_clipped_background=False, inlier_max_error=15.,
                         max_spatial_error=12.):
    """Fit brightness only on the unlit ring, never on the changing arrow.

    A robust gain + offset model handles common brightness/contrast changes.
    Clipped, textureless or spatially inconsistent surroundings cannot grant
    permission to classify a large lighting change. This is image-derived
    compensation, not a measurement of the camera's exposure register.
    """
    ring = _RING if ring_mask is None else ring_mask
    max_mean_error, max_p90_error = residual_limits
    x, y = reference_gray[ring], current_gray[ring]
    valid = (x > 5) & (x < 250) & (y > 5) & (y < 250)
    offset = float(np.median(y - x))
    audit = dict(background_normalization='robust_ring_affine_v1',
                 background_valid_fraction=float(np.mean(valid)),
                 exposure_offset=offset,
                 background_max_mean_error=max_mean_error,
                 background_max_p90_error=max_p90_error,
                 background_max_spatial_error=float(max_spatial_error))
    clipped_fallback = np.mean(valid) < .80
    if clipped_fallback:
        if (not allow_clipped_background or ring_mask is not None
                or np.mean(valid) < _CLIPPED_BACKGROUND_MIN_VALID_FRACTION):
            return None, {**audit, 'normalization_rejection': 'clipped_background'}
        usable = ring & (reference_gray > 5) & (reference_gray < 250) & (current_gray > 5) & (current_gray < 250)
        support = [float(np.mean(usable[q])) for q in (
            ring & horizontal & vertical
            for horizontal in (_X < 32, _X >= 32)
            for vertical in (_Y < 32, _Y >= 32))]
        dark_fraction = float(np.mean((x <= 5) | (y <= 5)))
        audit.update(background_tolerance_profile='clipped_background_70_80_v1',
                     background_min_valid_fraction=_CLIPPED_BACKGROUND_MIN_VALID_FRACTION,
                     background_sample_pixels=int(np.count_nonzero(usable)),
                     background_quadrant_valid_fraction=support,
                     background_min_quadrant_valid_fraction=_CLIPPED_BACKGROUND_MIN_QUADRANT_FRACTION,
                     background_dark_fraction=dark_fraction)
        if dark_fraction > _CLIPPED_BACKGROUND_MAX_DARK_FRACTION:
            return None, {**audit, 'normalization_rejection': 'clipped_background_dark_pixels'}
        if min(support) < _CLIPPED_BACKGROUND_MIN_QUADRANT_FRACTION:
            return None, {**audit, 'normalization_rejection': 'insufficient_distributed_background'}
    if ring_mask is not None:
        usable = ring & (reference_gray > 5) & (reference_gray < 250) & (current_gray > 5) & (current_gray < 250)
        support = [float(np.mean(usable[q])) for q in (
            ring & horizontal & vertical
            for horizontal in (_X < 32, _X >= 32)
            for vertical in (_Y < 32, _Y >= 32))]
        audit.update(background_sample_pixels=int(np.count_nonzero(ring)),
                     background_quadrant_valid_fraction=support)
        if min(support) < .60:
            return None, {**audit, 'normalization_rejection': 'insufficient_panel_support'}
    x, y = x[valid].astype(float), y[valid].astype(float)
    span = float(np.percentile(x, 90) - np.percentile(x, 10))
    gain, intercept = 1.0, float(np.median(y - x))
    audit['reference_background_span'] = span
    if clipped_fallback and span < 12:
        return None, {**audit, 'normalization_rejection': 'insufficient_background_texture'}
    if span >= 12:
        # Huber reweighting prevents a small changing reflection from defining
        # the correction. Fitting is bounded to one aligned ring (~640 pixels).
        weights = np.ones_like(x)
        for _ in range(4):
            mean_x = float(np.average(x, weights=weights))
            mean_y = float(np.average(y, weights=weights))
            centered = x - mean_x
            variance = float(np.sum(weights * centered * centered))
            if variance < 1e-6:
                return None, {**audit, 'normalization_rejection': 'ambiguous_background'}
            gain = float(np.sum(weights * centered * (y - mean_y)) / variance)
            intercept = float(np.median(y - gain * x))
            error = np.abs(y - (gain * x + intercept))
            weights = np.minimum(1., 8. / np.maximum(error, 1e-6))
    elif abs(offset) > 15:
        return None, {**audit, 'normalization_rejection': 'insufficient_background_texture'}
    audit.update(background_gain=gain, background_offset=intercept)
    if not .5 <= gain <= 2.0:
        return None, {**audit, 'normalization_rejection': 'unsupported_background_gain'}
    residual = current_gray - (gain * reference_gray + intercept)
    ring_error = np.abs(residual[ring])
    quadrants = [ring & (horizontal) & (vertical)
                 for horizontal in (_X < 32, _X >= 32)
                 for vertical in (_Y < 32, _Y >= 32)]
    spatial_error = max(abs(float(np.median(residual[q]))) for q in quadrants)
    audit.update(background_error=float(np.mean(ring_error)),
                 background_error_p90=float(np.percentile(ring_error, 90)),
                 background_inlier_fraction=float(np.mean(ring_error <= inlier_max_error)),
                 background_inlier_max_error=float(inlier_max_error),
                 background_spatial_error=spatial_error)
    if (audit['background_error'] > max_mean_error
            or audit['background_error_p90'] > max_p90_error
            or audit['background_inlier_fraction'] < .75 or spatial_error > max_spatial_error):
        return None, {**audit, 'normalization_rejection': 'nonuniform_background_change'}
    return residual, audit


def current_patch_fresh(current, check=None):
    if callable(check):
        return bool(check())
    return -.05 <= time.time()-current.captured <= .35


@traced('light.compare_before_after', 'light')
def compare_patches(reference, current, absolute_state, *, stationary=False,
                    strong_on=False, allow_area_relaxation=False, current_freshness_check=None,
                    allow_off_background_mean15=False):
    """Register on the surrounding unlit area, never on the changing lamp."""
    if not stationary:
        current_freshness_check = None
    if geometry_error(reference, current) is None:
        return {'state': 'UNCERTAIN', 'reason': 'view_geometry_changed'}
    best = None
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            shifted = np.roll(reference.gray, (dy, dx), axis=(0, 1))
            delta = current.gray - shifted
            offset = float(np.median(delta[_RING]))
            residual = delta - offset
            score = float(np.mean(np.abs(residual[_RING])))
            if best is None or score + .00001*(abs(dx)+abs(dy)) < best[0] + .00001*(abs(best[3])+abs(best[4])):
                best = (score, offset, residual, dx, dy)
    score, offset, residual, dx, dy = best
    audit = {'state': 'UNCERTAIN', 'reason': 'incomparable_background',
             'reference_sequence': reference.sequence, 'current_sequence': current.sequence,
             'reference_image': reference.image, 'reference_crop': reference.evidence,
             'reference_box': reference.box.tolist(), 'current_box': current.box.tolist(),
             'reference_time': reference.captured, 'current_time': current.captured,
             'background_error': score, 'exposure_offset': offset, 'alignment_px': [dx, dy]}
    aligned = np.roll(reference.gray, (dy, dx), axis=(0, 1))
    residual, normalization = normalize_background(aligned, current.gray)
    if (residual is None and stationary
            and normalization.get('normalization_rejection') == 'clipped_background'
            and normalization.get('background_valid_fraction', 0) >= _CLIPPED_BACKGROUND_MIN_VALID_FRACTION):
        # This branch is shared by stationary OFF references and stationary
        # post-press comparisons. Moving monitors never provide this guard.
        size_a, size_b = reference.box[2:] - reference.box[:2], current.box[2:] - current.box[:2]
        shift = np.abs((reference.box[:2] + reference.box[2:] - current.box[:2] - current.box[2:]) / 2) / size_a
        checks = dict(close_view=bool(min(*size_a, *size_b) >= 48),
                      stable_size=bool(np.max(np.abs(np.log(size_a / size_b))) <= .06),
                      stable_center=bool(np.max(shift) <= .06),
                      independent_frames=bool(current.sequence > reference.sequence),
                      recent_pair=bool(0 < current.captured - reference.captured <= 30))
        checks['stationary_observation_verified'] = bool(
            all(checks.values()) and (stationary() if callable(stationary) else stationary))
        audit['clipped_background_view_checks'] = checks
        if all(checks.values()):
            audit['strict_background_attempt'] = dict(normalization)
            residual, normalization = normalize_background(
                aligned, current.gray, allow_clipped_background=True)
    if (residual is None and absolute_state == 'OFF'
            and allow_off_background_mean15 is True and callable(stationary)
            and normalization.get('normalization_rejection') == 'nonuniform_background_change'
            and 10. < normalization.get('background_error', float('inf')) <= 15.
            and stationary()):
        # Only the aligned panel-button pre-press OFF-reference caller opts in.
        # Keep P90=25, inlier error=15, quadrant=12 and all center-rise gates.
        strict = dict(normalization)
        residual, normalization = normalize_background(aligned, current.gray,
            residual_limits=(15., 25.),
            allow_clipped_background=bool(audit.get('clipped_background_view_checks')
                and all(audit['clipped_background_view_checks'].values())))
        normalization['background_tolerance_profile'] = (
            'call_stationary_aligned_off_mean15_v1'
            if str(current.target_label) in {'up', 'down'}
            else 'numeric_stationary_aligned_off_mean15_v1')
        audit['off_reference_strict_background_attempt'] = strict
    if (residual is None and absolute_state == 'ON'
            and normalization.get('normalization_rejection') == 'nonuniform_background_change'):
        # Only rescue close, adequately resolved views after the established
        # ring check fails. OFF-reference collection keeps its original checks.
        size_a, size_b = reference.box[2:] - reference.box[:2], current.box[2:] - current.box[:2]
        shift = np.abs((reference.box[:2] + reference.box[2:] - current.box[:2] - current.box[2:]) / 2) / size_a
        eligible = (min(*size_a, *size_b) >= 48
                    and np.max(np.abs(np.log(size_a / size_b))) <= .06
                    and np.max(shift) <= .06)
        if eligible:
            panel_residual, panel_audit = normalize_background(aligned, current.gray, ring_mask=_PANEL_RING)
            audit['button_ring_normalization'] = dict(normalization)
            if (panel_residual is None
                    and panel_audit.get('normalization_rejection') == 'nonuniform_background_change'):
                # A bounded tolerance for a clear, stationary post-press view.
                # Keep strict OFF references and every previously passing path.
                # Support, texture, gain, 75% inliers within 15 and quadrant
                # consistency stay strict, as does independent lamp evidence.
                audit['panel_background_strict_attempt'] = dict(panel_audit)
                panel_residual, panel_audit = normalize_background(
                    aligned, current.gray, ring_mask=_PANEL_RING,
                    residual_limits=_POST_PRESS_BACKGROUND_LIMITS)
                panel_audit['background_tolerance_profile'] = 'post_press_relaxed_v1'
            audit['panel_background_attempt'] = dict(panel_audit)
            if panel_residual is not None and panel_audit.get('reference_background_span', 0) >= 12:
                # Reference selection must still compare the original common
                # ring score, not favor the smaller panel just for its low MAE.
                audit['reference_selection_background_error'] = normalization['background_error']
                residual, normalization = panel_residual, panel_audit
                audit['background_region'] = 'outer_panel_ring_v1'
    if (residual is None and absolute_state == 'ON' and strong_on is True
            and stationary
            and normalization.get('normalization_rejection') == 'nonuniform_background_change'):
        # Numeric-only opt-in from the observer: both absolute classifiers must
        # already say ON. Never change OFF baselines, initial lamp detection,
        # moving monitors or any previously passing comparison. The 30-gray spatial
        # limit belongs only to this verified numeric strong-ON fallback;
        # every other background comparison retains its 12-gray default.
        size_a, size_b = reference.box[2:] - reference.box[:2], current.box[2:] - current.box[:2]
        shift = np.abs((reference.box[:2] + reference.box[2:]
                        - current.box[:2] - current.box[2:]) / 2) / size_a
        checks = dict(close_view=bool(min(*size_a, *size_b) >= 48),
                      stable_size=bool(np.max(np.abs(np.log(size_a / size_b))) <= .06),
                      stable_center=bool(np.max(shift) <= .06),
                      independent_frames=bool(current.sequence > reference.sequence),
                      recent_pair=bool(0 < current.captured-reference.captured <= 30))
        checks['stationary_observation_verified'] = bool(
            all(checks.values()) and (stationary() if callable(stationary) else stationary))
        audit['numeric_strong_on_view_checks'] = checks
        if all(checks.values()):
            strict_audit = dict(normalization)
            candidate, candidate_audit = normalize_background(
                aligned, current.gray,
                residual_limits=_NUMERIC_STRONG_ON_BACKGROUND_LIMITS,
                inlier_max_error=_NUMERIC_STRONG_ON_INLIER_ERROR,
                max_spatial_error=_NUMERIC_STRONG_ON_MAX_SPATIAL_ERROR,
                allow_clipped_background=bool(
                    audit.get('clipped_background_view_checks') and
                    all(audit['clipped_background_view_checks'].values())))
            candidate_audit['background_tolerance_profile'] = 'numeric_stationary_strong_on_mean20_v1'
            audit['numeric_strong_on_background_attempt'] = dict(candidate_audit)
            audit['numeric_strong_on_strict_attempt'] = strict_audit
            # Keep original reference ranking; do not choose a reference
            # merely because its relaxed comparison says ON. Report the
            # attempted limits even when the bounded fallback also rejects.
            audit['reference_selection_background_error'] = strict_audit['background_error']
            residual, normalization = candidate, candidate_audit
    audit.update(normalization)
    if residual is None:
        return audit
    if absolute_state == 'ON' and normalization['reference_background_span'] < 12:
        # A flat ring cannot distinguish gain from offset. A pre-existing white
        # arrow under increased contrast must not be accepted as newly lit.
        audit['normalization_rejection'] = 'insufficient_background_texture'
        return audit
    saturation = float(np.mean(current.gray[_CENTER] >= 250))
    audit['saturated_fraction'] = saturation
    if saturation > .35:
        audit['reason'] = 'saturated_button'
        return audit
    increased = (_CENTER & (residual >= 25)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(increased, 8)
    largest = int(max(stats[1:, cv2.CC_STAT_AREA], default=0))
    fraction = largest / int(np.count_nonzero(_CENTER))
    error = float(np.mean(np.abs(residual[_CENTER])))
    audit.update(changed_fraction=fraction, center_error=error)
    raw_changed_pixels = largest * float(np.prod(current.box[2:] - current.box[:2])) / (64 * 64)
    audit['estimated_changed_source_pixels'] = raw_changed_pixels
    max_changed_fraction = .70
    if (fraction > max_changed_fraction and allow_area_relaxation is True
            and absolute_state == 'ON'
            and strong_on is True and stationary
            and current_patch_fresh(current, current_freshness_check)
            and (audit.get('background_tolerance_profile') == 'numeric_stationary_strong_on_mean20_v1'
                 or audit.get('background_region') == 'outer_panel_ring_v1')):
        # Only the numeric observer supplies strong_on after both absolute
        # tests agree. Recheck the fixed observation pose before widening the
        # area guard; moving monitors, arrows and OFF references keep 70%.
        # A background gain/offset fit can manufacture a broad positive
        # residual even when the actual button stayed dark. The new allowance
        # additionally requires the original center to have really brightened;
        # this does not change any comparison already passing below 70%.
        raw_center_rise = float(np.mean(current.gray[_CENTER] - aligned[_CENTER]))
        audit['changed_area_relaxation_checks'] = {
            'raw_center_rise': raw_center_rise, 'min_raw_center_rise': 25.,
        }
        stationary_verified = bool(raw_center_rise >= 25.
            and (stationary() if callable(stationary) else stationary))
        if stationary_verified and current_patch_fresh(current, current_freshness_check):
            max_changed_fraction = _NUMERIC_POST_PRESS_MAX_CHANGED_FRACTION
            audit['changed_area_relaxation'] = {
                'policy': 'numeric_stationary_strong_on_area99_v1',
                'previous_max_changed_fraction': .70,
                'max_changed_fraction': max_changed_fraction,
                'stationary_observation_verified': True,
                'raw_center_rise': raw_center_rise,
            }
    if audit.get('background_tolerance_profile') == 'numeric_stationary_strong_on_mean20_v1':
        audit.update(numeric_required_changed_fraction=.08,
                     numeric_max_changed_fraction=max_changed_fraction,
                     numeric_required_changed_source_pixels=12)
        if fraction < .08 or raw_changed_pixels < 12 or fraction > max_changed_fraction:
            audit['reason'] = 'insufficient_local_rise_for_numeric_relaxation'
            return audit
    if audit.get('background_region') == 'outer_panel_ring_v1':
        # A smaller reference area must have stronger independent lamp evidence.
        # Never promote a small reflection just because the outer panel agrees.
        audit['panel_required_changed_fraction'] = .08
        audit['panel_max_changed_fraction'] = max_changed_fraction
        audit['panel_required_changed_source_pixels'] = 12
        if fraction < .08 or raw_changed_pixels < 12:
            audit['reason'] = 'insufficient_local_rise_for_panel_reference'
            return audit
        if fraction > max_changed_fraction:
            audit['reason'] = 'nonlocal_rise_for_panel_reference'
            return audit
    if fraction >= .03 and raw_changed_pixels >= 3 and absolute_state == 'ON':
        audit.update(state='ON', reason='localized_rise_from_confirmed_off')
    elif error <= 8 and fraction < .02 and absolute_state == 'OFF':
        audit.update(state='OFF', reason='unchanged_confirmed_off')
    else:
        audit['reason'] = 'inconclusive_change_or_absolute_disagreement'
        # Candidate evidence for a stationary multi-frame recheck, never a
        # single-frame OFF verdict or permission to replace the frozen baseline.
        size_a = reference.box[2:] - reference.box[:2]
        size_b = current.box[2:] - current.box[:2]
        shift = np.abs((reference.box[:2] + reference.box[2:]
                        - current.box[:2] - current.box[2:]) / 2) / size_a
        weak_rise = (_CENTER & (residual >= 15)).astype(np.uint8)
        _, _, weak_stats, _ = cv2.connectedComponentsWithStats(weak_rise, 8)
        weak_fraction = float(max(weak_stats[1:, cv2.CC_STAT_AREA], default=0)
                              / np.count_nonzero(_CENTER))
        audit.update(
            off_recheck_candidate=bool(
                absolute_state == 'OFF' and 8 < error <= 15
                and min(*size_a, *size_b) >= 48
                and np.max(np.abs(np.log(size_a / size_b))) <= .06
                and np.max(shift) <= .06
                and fraction < .01 and weak_fraction < .01
                and normalization['reference_background_span'] >= 12
                and 'background_region' not in audit
            ),
            off_recheck_max_center_error=15.,
            off_recheck_weak_rise_fraction=weak_fraction,
            off_recheck_max_rise_fraction=.01,
        )
    return audit


class PairedButtonLight:
    def __init__(self, evidence_dir=None, *, require_stationary_reference=False,
                 stationary_check=None):
        self.require_stationary_reference = bool(require_stationary_reference)
        self.stationary_check = stationary_check
        self.last_reference_comparison = None
        self.stationary_reference_ready = False
        self.references = deque(maxlen=64)
        self.pending = None
        self.frozen = False
        self.press_started = float('inf')
        self.last_reference_sequence = -1
        self.attempt = 0
        self.saved_current = 0
        self.saved_states = {"ON": 0, "OFF": 0}
        self.evidence_dir = Path(evidence_dir) if evidence_dir is not None else None
        self._registration_cache = {}

    def begin_attempt(self):
        self._registration_cache.clear()
        self.stationary_reference_ready = False
        self.last_reference_comparison = None
        self.references.clear()
        self.pending = None
        self.frozen = False
        self.press_started = float('inf')
        self.last_reference_sequence = -1
        self.attempt += 1
        self.saved_current = 0
        self.saved_states = {"ON": 0, "OFF": 0}

    def freeze(self):
        if self.require_stationary_reference and not self.stationary_reference_ready:
            raise RuntimeError("stationary observation reference has not been confirmed")
        self.frozen = True
        self.pending = None
        self.press_started = time.time()
        self.last_reference_sequence = max((p.sequence for p in self.references), default=-1)

    def reset_pending(self):
        self.pending = None
        self.last_reference_comparison = None

    def _save(self, patch, kind):
        if self.evidence_dir is None:
            return None
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        path = self.evidence_dir / f'attempt_{self.attempt}_{kind}_{patch.sequence}.png'
        if not cv2.imwrite(str(path), patch.gray.astype(np.uint8)):
            raise RuntimeError('cannot save paired light evidence')
        patch.evidence = str(path)
        raw_path = path.with_name(path.stem + "_full.jpg")
        shutil.copy2(patch.image, raw_path)
        patch.image = str(raw_path)
        return str(path)

    def collect(self, snapshot, absolute_state, *, stationary=False, current_freshness_check=None):
        if self.frozen:
            raise RuntimeError('cannot change OFF baseline after press starts')
        if absolute_state != 'OFF':
            self.reset_pending()
            return
        region_options = ({'max_frame_age_sec': .7} if stationary
            and self.require_stationary_reference and callable(self.stationary_check)
            and callable(current_freshness_check) and current_freshness_check() else {})
        patch = extract_patch(snapshot, **region_options)
        self.last_reference_comparison = None
        if self.pending is not None and patch.sequence > self.pending.sequence:
            result = compare_patches(self.pending, patch, 'OFF', stationary=(
                self.stationary_check if stationary and self.require_stationary_reference
                and callable(self.stationary_check) else False))
            if (result['state'] != 'OFF' and stationary and self.require_stationary_reference
                    and callable(self.stationary_check) and snapshot.get('fresh') is True
                    and supports_off_reference_alignment(patch.target_label)):
                original = dict(result)
                with stage('light.off_reference_alignment', 'reference'):
                    recovered, alignment = recompare_stationary_off(
                        self.pending, patch, compare_patches,
                        stationary_check=self.stationary_check,
                        freshness_check=lambda: current_patch_fresh(patch, current_freshness_check),
                        cache=self._registration_cache)
                if recovered is not None:
                    result = recovered
                result['off_reference_alignment'] = alignment
                result['unaligned_comparison'] = original
            self.last_reference_comparison = result
            if result['state'] == 'OFF':
                for item in (self.pending, patch):
                    if not self.references or item.sequence > self.references[-1].sequence:
                        self._save(item, 'before')
                        if len(self.references) == self.references.maxlen:
                            evicted = self.references[0]
                            if evicted.evidence is not None:
                                Path(evicted.evidence).unlink(missing_ok=True)
                                Path(evicted.image).unlink(missing_ok=True)
                        self.references.append(item)
        self.pending = patch

    def classify(self, snapshot, absolute_state, *, preserve_evidence=False,
                 stationary=False, strong_on=False, current_freshness_check=None,
                 allow_post_press_off_alignment=False):
        result = {'state': 'UNCERTAIN', 'reason': 'no_comparable_off_reference'}
        if not self.frozen:
            return result
        region_options = ({'max_frame_age_sec': .7} if stationary
            and self.require_stationary_reference and self.stationary_reference_ready
            and callable(self.stationary_check) and callable(current_freshness_check)
            and current_freshness_check() else {})
        current = extract_patch(snapshot, **region_options)
        if current.captured <= self.press_started or current.sequence <= self.last_reference_sequence:
            return {'state': 'UNCERTAIN', 'reason': 'image_predates_press'}
        candidates = []
        for ref in self.references:
            error = geometry_error(ref, current)
            if error is not None and 0 < current.captured - ref.captured <= 30:
                candidates.append((error, ref))
        # Select by surrounding appearance, never by which result says ON.
        guard = (self.stationary_check if stationary and self.require_stationary_reference
                 and self.stationary_reference_ready and callable(self.stationary_check) else False)
        label=str((selected_button_region(snapshot, **region_options) or {}).get('label',''))
        current_freshness_check = current_freshness_check if guard else None
        area_options = {'current_freshness_check': current_freshness_check,
                        'allow_area_relaxation': bool(
            absolute_state == 'ON' and strong_on is True and guard
            and label in tuple(map(str, range(1, 17)))
            and snapshot.get('fresh') is True
            and current_patch_fresh(current, current_freshness_check))}
        compared = [compare_patches(ref, current, absolute_state, stationary=guard,
                                    strong_on=strong_on, **area_options)
                    for _, ref in sorted(candidates, key=lambda x: x[0])[:4]]
        if compared:
            result = min(compared, key=lambda x: x.get('reference_selection_background_error',
                                                      x.get('background_error', float('inf'))))
        # Only completed observation poses explicitly opt in. Motion UNKNOWNs
        # do not enter this path, and the already selected reference stays fixed.
        if (allow_post_press_off_alignment and result['state'] == 'UNCERTAIN'
                and absolute_state == 'OFF' and guard and snapshot.get('fresh') is True
                and supports_off_reference_alignment(label)
                and current_patch_fresh(current, current_freshness_check)):
            reference = next((ref for _, ref in candidates
                if ref.sequence == result.get('reference_sequence')), None)
            if reference is not None:
                original = dict(result)
                with stage('light.post_press_off_alignment', 'light'):
                    recovered, alignment = recompare_stationary_post_press_off(
                        reference, current, compare_patches, stationary_check=guard,
                        freshness_check=lambda: current_patch_fresh(current, current_freshness_check),
                        cache=self._registration_cache)
                if recovered is not None:
                    result = recovered
                    result['reference_selection_background_error'] = original.get(
                        'reference_selection_background_error',
                        original.get('background_error', float('inf')))
                result['post_press_off_alignment'] = alignment
                result['unaligned_comparison'] = original
        # Rescue only a rejected, strong numeric ON at a verified stationary
        # post-press pose. Keep passing comparisons, OFF, arrows and moving
        # observations on their original path. Select the reference by the
        # ORIGINAL background score, never by which alignment returns ON.
        if (result['state']=='UNCERTAIN' and absolute_state=='ON' and strong_on
                and label in tuple(map(str,range(1,17))) and guard
                and snapshot.get('fresh') is True
                and current_patch_fresh(current, current_freshness_check)):
            reference=next((ref for _,ref in candidates
                            if ref.sequence==result.get('reference_sequence')),None)
            if (reference is not None and guard()
                    and current_patch_fresh(current, current_freshness_check)):
                admitted_at=time.time()
                with stage('light.background_registration','light'):
                    registered,audit=register_reference(reference,current,
                                                        cache=self._registration_cache)
                audit.update(target_label=label,stationary_encoder_verified=True,
                             strong_whole_and_local_on=True,
                             current_age_at_admission_sec=admitted_at-current.captured)
                original=dict(result)
                if registered is not None:
                    aligned=replace(reference,gray=registered,box=current.box.copy())
                    candidate=compare_patches(aligned,current,absolute_state,
                                              stationary=guard,strong_on=True, **area_options)
                    # The old selection score remains explicit, even though
                    # comparison pixels now share current-frame coordinates.
                    candidate.update(reference_box=reference.box.tolist(),
                        comparison_reference_box=current.box.tolist(),
                        reference_selection_background_error=original.get(
                            'reference_selection_background_error',
                            original.get('background_error',float('inf'))))
                    result=candidate
                result['background_registration']=audit
                result['unaligned_comparison']=original
        state = result['state']
        first_decisive = state in self.saved_states and self.saved_states[state] < 2
        if self.saved_current < 12 or first_decisive or preserve_evidence:
            result['current_crop'] = self._save(current, 'after')
            result['current_image'] = current.image
            self.saved_current += 1
            if state in self.saved_states:
                self.saved_states[state] += 1
        result.update(reference_count=len(self.references), absolute_state=absolute_state)
        result.update(current_region_source=current.region_source,
                      reference_region_sources=sorted({r.region_source for r in self.references}))
        return result
