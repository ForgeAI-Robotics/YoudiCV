"""Bounded stationary numeric reacquisition; no motion or guessed labels."""
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .button_pose import ButtonDetection, select_button
from .observation_visibility import complete_button_box
from .numeric_search_evidence import numeric_search_evidence
from .numeric_layout_handoff import selected_button_region, region_is_inferred
from .numeric_yolo_consistency import target_conflicted


def emit_numeric_progress(kind, **value):
    event = dict(kind=kind, value=value)
    print(json.dumps({'numeric_visibility_event': event}, ensure_ascii=False), flush=True)
    task_id = os.environ.get('YOUXIAOMEI_TASK_ID')
    outbox = os.environ.get('YOUXIAOMEI_MISSION_STATUS_OUTBOX')
    if not task_id or not outbox:
        return
    try:
        from .mission_status import append_status
        append_status(Path(outbox), dict(
            task_id=task_id, mission_id=os.environ.get('YOUXIAOMEI_MISSION_ID',task_id),
            segment_id=os.environ.get('YOUXIAOMEI_SEGMENT_ID'), producer='arm_pressing',
            phase=kind, outcome='degraded' if kind.endswith('waiting') else 'active',
            evidence=value, raw_event=kind), durable=False)
    except OSError:
        # Status mirroring must not change the perception or motion decision.
        pass


def wait_numeric_visibility(vision, *, target_label, min_confidence,
                            sequence_barrier, timeout_sec=4., required_consecutive=2,
                            check_hold=None, clock=time.monotonic, sleep=time.sleep):
    deadline = clock()+timeout_sec
    notify_at = clock()+.75
    last = int(sequence_barrier)
    previous_mode, previous_center, streak = None, None, 0
    previous_context = None
    previous_capture = None
    audit = []
    latest_grid = None
    while clock() < deadline:
        if check_hold is not None:
            check_hold()
        response = vision.request('visible_buttons', min_confidence=.50)
        sequence = response.get('frame_sequence')
        if sequence is None or int(sequence) <= last:
            sleep(.01)
            continue
        last = int(sequence)
        search_audit = None
        context = None
        try:
            age = time.time()-float(response.get('capture_time_unix_sec',0))
            if not -.05 <= age <= .35:
                raise ValueError('stale frame')
            detections = [ButtonDetection(0,str(d['label']),float(d['confidence']),
                                          tuple(map(float,d['bbox_xyxy'])))
                          for d in response.get('detections',[])]
            width,height = map(float,response['image_size'])
            qualified = [d for d in detections if math.isfinite(d.confidence)
                         and 0 <= d.confidence <= 1
                         and all(math.isfinite(v) for v in d.bbox_xyxy)
                         and 0 <= d.bbox_xyxy[0] < d.bbox_xyxy[2] <= width
                         and 0 <= d.bbox_xyxy[1] < d.bbox_xyxy[3] <= height
                         and d.bbox_xyxy[2]>d.bbox_xyxy[0]
                         and d.bbox_xyxy[3]>d.bbox_xyxy[1]]
            try:
                target = select_button(qualified,str(target_label),min_confidence)
                center = np.asarray(target.center_pixel)
                mode = 'target_visible'
            except RuntimeError:
                target = None
                region=selected_button_region(response)
                if region_is_inferred(region) and region['label']==str(target_label):
                    box=np.asarray(region['bbox_xyxy'],float)
                    center=(box[:2]+box[2:])/2
                    mode='layout_target_visible'
                    context=('panel_layout_inferred',region['profile_sha256'])
                else:
                    _, prediction, search_audit = numeric_search_evidence(
                        response, target_label, min_confidence)
                    center = np.asarray(prediction.center_xy)
                    mode = 'grid_prediction'
                    if search_audit['source'] == 'numeric_tile_context':
                        context = (search_audit['source'], tuple(sorted(search_audit['labels'])),
                                   search_audit['anchor_label'])
        except (RuntimeError, ValueError, TypeError, KeyError):
            previous_mode, previous_center, streak = None, None, 0
            previous_context = None
            previous_capture = None
            latest_grid = None
            sleep(.01)
            if clock() >= notify_at:
                emit_numeric_progress('numeric_visibility_waiting',target_label=str(target_label),
                                      reason='numeric_detection_unstable',budget_sec=timeout_sec)
                notify_at = float('inf')
            continue
        streak = streak+1 if (mode == previous_mode and context == previous_context
                             and previous_center is not None
                             and (context is None or previous_capture is not None
                                  and float(response['capture_time_unix_sec']) > previous_capture)
                             and np.linalg.norm(center-previous_center) <= 8.) else 1
        previous_mode, previous_center = mode, center
        previous_context = context
        previous_capture = float(response['capture_time_unix_sec'])
        hints = [*response.get('detections',[])]
        enhancement = response.get('detection_enhancement',{})
        for key in ('raw_detections','tile_detections','enhanced_detections'):
            hints.extend(enhancement.get(key,[]))
        weak_target_present = any(str(d.get('label')) == str(target_label)
                                  and float(d.get('confidence',0)) >= .15 for d in hints)
        rejected_by_layout=target_conflicted(response.get('numeric_yolo_consistency'),
            target=target_label,sequence=last)
        if rejected_by_layout:
            weak_target_present=False
        audit.append(dict(frame_sequence=last, mode=mode, consecutive=streak,
                          rejected_by_layout=rejected_by_layout,
                          search_evidence=search_audit,
                          target_scores=[d.get('confidence') for d in hints
                                         if str(d.get('label')) == str(target_label)]))
        if streak >= required_consecutive:
            response = {**response,'numeric_visibility_audit':audit[-32:]}
            if search_audit is not None:
                response['numeric_search_evidence'] = search_audit
            if target is not None:
                emit_numeric_progress('numeric_visibility_recovered',target_label=str(target_label),
                                      frame_sequence=last,confidence=target.confidence)
                return response, True
            if mode=='layout_target_visible':
                return {**response,'visibility_resolution':'qualified_same_frame_layout'},True
            latest_grid = response
            # A truly absent high target can use the existing bounded search
            # immediately. A weak target hint gets a stationary recovery budget.
            if not weak_target_present:
                return response, False
        if clock() >= notify_at:
            emit_numeric_progress('numeric_visibility_waiting',target_label=str(target_label),
                                  reason='weak_numeric_target',budget_sec=timeout_sec)
            notify_at = float('inf')
        sleep(.01)
    if latest_grid is not None and previous_mode == 'grid_prediction' and streak >= required_consecutive:
        age = time.time()-float(latest_grid.get('capture_time_unix_sec',0))
        if -.05 <= age <= .35:
            return latest_grid, False
    emit_numeric_progress('numeric_visibility_failed',target_label=str(target_label),
                          reason='numeric_detection_unstable',audit=audit[-32:])
    raise RuntimeError(f'no stable fresh visibility after target switch within {timeout_sec:.1f}s')


def recover_numeric_target(target, anchor_label, estimate, samples, capture_joints,
                           robot, acquire, *, image_size, min_confidence=.70):
    """Promote only a same-label, fresh, stationary, qualified reference."""
    state = robot.state(max_age=.35)
    if state.raw.get('hardware_ok') != 1 or not np.isfinite(state.joints).all():
        raise RuntimeError('robot hardware is not OK during numeric recovery')
    accepted = [s for s in samples if s.get('accepted')]
    last = accepted[-1] if accepted else {}
    summary = last.get('acquisition_summary',{})
    age = time.time()-float(last.get('capture_time_unix_sec',0))
    def qualified_reference(row):
        if row.get('source')=='panel_layout_inferred':
            digest=summary.get('profile_sha256')
            return (row.get('confidence','missing') is None and row.get('target_detected') is False
                    and 'panel_layout_inferred' in getattr(estimate,'position_sources',())
                    and isinstance(digest,str) and len(digest)==64
                    and row.get('profile_sha256')==digest
                    and row.get('transform_source')=='capture_time_encoder_interpolation'
                    and len(set(row.get('anchors',[])))>=3)
        try:return float(row.get('confidence',0))>=min_confidence
        except (TypeError,ValueError):return False
    reusable = (str(target) == str(anchor_label) and estimate is not None
                and estimate.stable and np.isfinite(estimate.base_to_button).all()
                and len(accepted) >= 4 and len({s['sequence'] for s in accepted[-4:]}) == 4
                and summary.get('target_label') == str(target)
                and summary.get('window_sample_count') == 4
                and -.05 <= age <= .35
                and np.max(np.abs(state.joints-np.asarray(capture_joints))) <= .003
                and all(qualified_reference(s) for s in accepted[-4:])
                and all(complete_button_box(s.get('bbox_xyxy'),image_size) for s in accepted[-2:]))
    if not reusable:
        estimate, samples = acquire()
        state = robot.state(max_age=.35)
        if state.raw.get('hardware_ok') != 1 or not np.isfinite(state.joints).all():
            raise RuntimeError('robot hardware is not OK after numeric recovery')
        fresh = [s for s in samples if s.get('accepted')]
        final = fresh[-1] if fresh else {}
        if (estimate is None or not estimate.stable or len(fresh) < 4
                or not np.isfinite(estimate.base_to_button).all()
                or final.get('acquisition_summary',{}).get('target_label') != str(target)
                or not -.05 <= time.time()-float(final.get('capture_time_unix_sec',0)) <= .35):
            raise RuntimeError('numeric target reacquisition did not provide a fresh stable target')
    emit_numeric_progress('numeric_target_recovered',target_label=str(target),
                          source='same_target_reference' if reusable else 'fresh_target_acquisition')
    return estimate, samples, state.joints.copy(), bool(reusable)
