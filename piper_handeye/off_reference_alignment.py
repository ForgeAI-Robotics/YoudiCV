"""Recover stationary numeric and call-button OFF pairs from crop jitter.

Only comparison pixels are resampled. Real RGB, detection boxes, frame IDs and
motion targets remain unchanged. Original lamp/background thresholds still run.
"""
from dataclasses import replace
import math
import time

import cv2
import numpy as np

from .button_background_registration import register_reference
from .observation_visibility import complete_button_box

POLICY = dict(version=1, scope='stationary_numeric_pre_press_off_only',
              min_button_size_px=24., max_center_shift_fraction=.10,
              max_log_size_change=.08, max_pair_span_sec=6.,
              processing_budget_sec=.20, methods=['common_pixel_crop','background_registration'])

POST_PRESS_POLICY = dict(version=1, scope='stationary_post_press_off_only',
    min_button_size_px=48., max_center_shift_fraction=.06,
    max_log_size_change=.06, max_pair_span_sec=30.,
    processing_budget_sec=.20, methods=['common_pixel_crop'])


def supports_off_reference_alignment(target_label):
    label = str(target_label)
    return label in {'up', 'down'} or label in tuple(map(str, range(1, 17)))


def _gray(path, cache):
    key=('off_reference_gray',str(path))
    if key in cache:return cache[key]
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('alignment_image_unreadable')
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if image.shape[0]>720 or image.shape[1]>1280:
        raise ValueError('alignment_image_outside_budget')
    cache[key]=image
    while len(cache)>4:cache.pop(next(iter(cache)))
    return image


def recompare_stationary_post_press_off(reference, current, compare, *,
                                        stationary_check, freshness_check, cache):
    """Retry a rejected OFF comparison on one common crop; never refit the lamp.

    The caller must have frozen a verified OFF bank and completed its retreat.
    No new reference, relaxed threshold, feature fit or movement is introduced.
    """
    began = time.perf_counter()
    audit = dict(policy=dict(POST_PRESS_POLICY), accepted=False,
        reference_sequence=reference.sequence, current_sequence=current.sequence,
        original_reference_box=reference.box.tolist(),
        original_current_box=current.box.tolist(), raw_images_modified=False)

    def finish(result, reason):
        audit.update(accepted=result is not None, reason=reason,
                     elapsed_sec=time.perf_counter()-began)
        return result, audit

    def valid():
        return (time.perf_counter()-began <= POST_PRESS_POLICY['processing_budget_sec']
            and bool(freshness_check()) and bool(stationary_check())
            and bool(freshness_check())
            and time.perf_counter()-began <= POST_PRESS_POLICY['processing_budget_sec'])

    try:
        if (not supports_off_reference_alignment(reference.target_label)
                or reference.target_label != current.target_label
                or current.sequence <= reference.sequence
                or not math.isfinite(current.captured-reference.captured)
                or not 0 < current.captured-reference.captured <= POST_PRESS_POLICY['max_pair_span_sec']):
            return finish(None, 'not_same_supported_post_press_pair')
        a, b = np.asarray(reference.box, float), np.asarray(current.box, float)
        if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
            return finish(None, 'invalid_box')
        sa, sb = a[2:]-a[:2], b[2:]-b[:2]
        if min(*sa, *sb) < POST_PRESS_POLICY['min_button_size_px']:
            return finish(None, 'button_too_small_for_post_press_crop_recovery')
        shift = np.abs((a[:2]+a[2:]-b[:2]-b[2:])/2)/sa
        size = np.abs(np.log(sa/sb))
        audit.update(center_shift_fraction=shift.tolist(), log_size_change=size.tolist())
        if (np.max(shift) > POST_PRESS_POLICY['max_center_shift_fraction']
                or np.max(size) > POST_PRESS_POLICY['max_log_size_change']):
            return finish(None, 'view_change_exceeds_post_press_crop_jitter_range')
        if not valid():
            return finish(None, 'pose_or_image_not_current')
        ga, gb = _gray(reference.image, cache), _gray(current.image, cache)
        if ga.shape != gb.shape:
            return finish(None, 'image_size_changed')
        h, w = ga.shape
        if not complete_button_box(a, [w, h]) or not complete_button_box(b, [w, h]):
            return finish(None, 'button_not_complete')
        box = np.r_[np.floor(np.minimum(a[:2], b[:2])),
                    np.ceil(np.maximum(a[2:], b[2:]))]
        x1, y1, x2, y2 = box.astype(int)
        ra = replace(reference, box=box.copy(), gray=cv2.resize(
            ga[y1:y2, x1:x2], (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32))
        rb = replace(current, box=box.copy(), gray=cv2.resize(
            gb[y1:y2, x1:x2], (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32))
        if not valid():
            return finish(None, 'pose_image_or_budget_expired')
        result = compare(ra, rb, 'OFF', stationary=stationary_check)
        audit.update(comparison_reference_box=box.tolist(),
            comparison_current_box=box.tolist(), state=result['state'],
            comparison_reason=result.get('reason'),
            background_error=result.get('background_error'),
            center_error=result.get('center_error'),
            changed_fraction=result.get('changed_fraction'))
        if not valid():
            return finish(None, 'pose_image_or_budget_expired')
        if result['state'] != 'OFF':
            return finish(None, 'common_crop_does_not_confirm_off')
        result.update(reference_box=reference.box.tolist(), current_box=current.box.tolist(),
            comparison_reference_box=box.tolist(), comparison_current_box=box.tolist(),
            comparison_method='post_press_common_pixel_crop')
        return finish(result, 'off_confirmed_after_common_crop')
    except (ValueError, cv2.error) as exc:
        audit['error'] = f'{type(exc).__name__}: {exc}'
        return finish(None, 'post_press_off_alignment_unavailable')


def recompare_stationary_off(reference,current,compare,*,stationary_check,freshness_check,cache):
    """Return the original-rule OFF comparison or None, plus a bounded audit."""
    began=time.perf_counter()
    audit=dict(policy=dict(POLICY),accepted=False,attempts=[],raw_images_modified=False,
               reference_sequence=reference.sequence,current_sequence=current.sequence,
               original_reference_box=reference.box.tolist(),original_current_box=current.box.tolist())
    if str(reference.target_label) in {'up', 'down'}:
        audit['policy']['scope'] = 'stationary_call_pre_press_off_only'
    def finish(result,reason):
        audit.update(accepted=result is not None,reason=reason,elapsed_sec=time.perf_counter()-began)
        return result,audit
    def still_valid():
        return (time.perf_counter()-began<=POLICY['processing_budget_sec']
                and bool(freshness_check()) and bool(stationary_check())
                and bool(freshness_check()))
    def examine(a,b,method):
        result=compare(a,b,'OFF',stationary=stationary_check)
        if result['state']!='OFF' and still_valid():
            strict=dict(result)
            result=compare(a,b,'OFF',stationary=stationary_check,
                           allow_off_background_mean15=True)
            result['off_reference_strict_comparison']=strict
        audit['attempts'].append(dict(method=method,state=result['state'],
            reason=result.get('reason'),background_error=result.get('background_error'),
            background_error_p90=result.get('background_error_p90'),
            center_error=result.get('center_error'),changed_fraction=result.get('changed_fraction')))
        if result['state']!='OFF' or not still_valid():return None
        # Keep detection provenance distinct from the comparison crop.
        result.update(reference_box=reference.box.tolist(),current_box=current.box.tolist(),
                      comparison_reference_box=a.box.tolist(),comparison_current_box=b.box.tolist(),
                      comparison_method=method)
        return result
    try:
        if (not supports_off_reference_alignment(reference.target_label)
                or reference.target_label!=current.target_label
                or current.sequence<=reference.sequence
                or not math.isfinite(current.captured-reference.captured)
                or not 0<current.captured-reference.captured<=POLICY['max_pair_span_sec']):
            return finish(None,'not_same_supported_button_off_pair')
        a,b=np.asarray(reference.box,float),np.asarray(current.box,float)
        if a.shape!=(4,) or b.shape!=(4,) or not np.isfinite([a,b]).all():
            return finish(None,'invalid_box')
        sa,sb=a[2:]-a[:2],b[2:]-b[:2]
        if min(*sa,*sb)<POLICY['min_button_size_px']:
            return finish(None,'button_too_small_for_crop_recovery')
        shift=np.abs((a[:2]+a[2:]-b[:2]-b[2:])/2)/sa
        size=np.abs(np.log(sa/sb))
        audit.update(center_shift_fraction=shift.tolist(),log_size_change=size.tolist())
        if np.max(shift)>POLICY['max_center_shift_fraction'] or np.max(size)>POLICY['max_log_size_change']:
            return finish(None,'view_change_exceeds_crop_jitter_range')
        if not still_valid():return finish(None,'pose_or_image_not_current')
        ga,gb=_gray(reference.image,cache),_gray(current.image,cache)
        if ga.shape!=gb.shape:return finish(None,'image_size_changed')
        h,w=ga.shape
        if not complete_button_box(a,[w,h]) or not complete_button_box(b,[w,h]):
            return finish(None,'button_not_complete')
        # The union keeps both independently detected buttons complete. Both
        # real images use exactly these integer coordinates and resize factor.
        box=np.r_[np.floor(np.minimum(a[:2],b[:2])),np.ceil(np.maximum(a[2:],b[2:]))]
        x1,y1,x2,y2=box.astype(int)
        ra=replace(reference,box=box.copy(),gray=cv2.resize(
            ga[y1:y2,x1:x2],(64,64),interpolation=cv2.INTER_AREA).astype(np.float32))
        rb=replace(current,box=box.copy(),gray=cv2.resize(
            gb[y1:y2,x1:x2],(64,64),interpolation=cv2.INTER_AREA).astype(np.float32))
        result=examine(ra,rb,'common_pixel_crop')
        if result is not None:return finish(result,'off_confirmed_after_common_crop')
        if not still_valid():return finish(None,'pose_image_or_budget_expired')
        # Reuse the existing bounded background-only ORB/RANSAC fit. It cannot
        # fit a changing glyph away, and all original OFF checks run again.
        registered,fit=register_reference(reference,current)
        fit={**fit, 'authorization_context':'pre_press_off_reference',
             'policy':{**fit.get('policy',{}),'scope':audit['policy']['scope']}}
        audit['background_registration']=fit
        if registered is None:return finish(None,'background_registration_unavailable')
        if not still_valid():return finish(None,'pose_image_or_budget_expired')
        aligned=replace(reference,gray=registered,box=current.box.copy())
        result=examine(aligned,current,'background_registered_crop')
        if result is not None:return finish(result,'off_confirmed_after_background_registration')
        return finish(None,'aligned_images_do_not_confirm_off')
    except (ValueError,cv2.error) as exc:
        audit['error']=f'{type(exc).__name__}: {exc}'
        return finish(None,'off_alignment_unavailable')
