#!/usr/bin/env python3
"""Offline research only. No camera, RPC, ROS, robot or production fallback.

Hide the target AND all overlapping detections, fit a panel homography from
other labels, then score against separately annotated button centres.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def project(h, points):
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    q = np.column_stack((p, np.ones(len(p)))) @ h.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-8):
        raise ValueError('projective_horizon')
    return q[:, :2] / q[:, 2:]


def infer(template, anchors, target, image_size):
    """Conservative experimental gate; returned pixels are NEVER press targets."""
    result = dict(accepted=False, purpose='offline_2d_research_only')
    try:
        if target not in template:
            raise ValueError('target_not_in_verified_template')
        # The target must not participate in fitting, even if a caller forgets.
        anchors = [a for a in anchors if a['label'] != target]
        labels = [a['label'] for a in anchors]
        if len(labels) != len(set(labels)):
            raise ValueError('duplicate_anchor_label')
        if any(l not in template for l in labels):
            raise ValueError('unknown_anchor')
        result['anchors'] = labels
        if len(anchors) < 5:
            raise ValueError('fewer_than_five_anchors')
        src = np.array([template[l] for l in labels], dtype=float)
        dst = np.array([a['center'] for a in anchors], dtype=float)
        diam = float(np.median([a['diameter'] for a in anchors]))
        if not np.isfinite(src).all() or not np.isfinite(dst).all() or not math.isfinite(diam) or diam < 8:
            raise ValueError('invalid_anchor_geometry')
        if min(sum(src[:, 0] == x) for x in (0, 1)) < 2 or len(set(src[:, 1])) < 3:
            raise ValueError('insufficient_two_column_row_coverage')
        for i in range(len(dst)):
            if any(np.linalg.norm(dst[i]-dst[j]) < .35*diam for j in range(i)):
                raise ValueError('overlapping_anchor_labels')
        # Fixed BEFORE holdout evaluation; not tuned to validation errors.
        limit = min(4.0, max(2.0, .08*diam))
        cv2.setRNGSeed(0)
        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, limit)
        if h is None or mask is None:
            raise ValueError('degenerate_fit')
        result['ransac_outlier_labels'] = [l for l, ok in zip(labels, mask.ravel()) if not ok]
        # Two narrow columns produce many degenerate 4-point RANSAC samples.
        # Also try ALL anchors together; never silently discard a wrong label.
        # Whichever fit is better must satisfy the same ALL-point/LOO gates.
        all_h, _ = cv2.findHomography(src, dst, 0)
        if all_h is not None and max(np.linalg.norm(project(all_h,src)-dst,axis=1)) < max(np.linalg.norm(project(h,src)-dst,axis=1)):
            h = all_h
        residual = np.linalg.norm(project(h, src)-dst, axis=1)
        result.update(max_anchor_error_px=float(max(residual)), rms_anchor_error_px=float(np.sqrt(np.mean(residual**2))))
        if max(residual) > limit:
            raise ValueError('anchor_layout_conflict')
        # Independent geometric stability, not just fitting the same points.
        loo = []
        for i in range(len(src)):
            s, d = np.delete(src, i, 0), np.delete(dst, i, 0)
            # A single point in one column cannot determine a projectivity.
            if min(sum(s[:, 0] == x) for x in (0, 1)) < 2:
                continue
            hi, _ = cv2.findHomography(s, d, 0)
            if hi is None:
                raise ValueError('unstable_leave_one_out_fit')
            loo.append(float(np.linalg.norm(project(hi, [src[i]])[0]-dst[i])))
        result['max_leave_one_out_px'] = max(loo, default=None)
        if not loo or max(loo) > max(4.0, .12*diam):
            raise ValueError('leave_one_out_unstable')
        t = np.array(template[target], dtype=float)
        if t[1] < min(src[:, 1])-1 or t[1] > max(src[:, 1])+1:
            raise ValueError('extrapolation_more_than_one_row')
        p, right, down = project(h, [t, t+[.1, 0], t+[0, .1]])
        if right[0] <= p[0] or down[1] <= p[1]:
            raise ValueError('inverted_panel_orientation')
        w, hh = image_size
        if not (5+diam/2 <= p[0] <= w-5-diam/2 and 5+diam/2 <= p[1] <= hh-5-diam/2):
            raise ValueError('predicted_target_not_fully_visible')
        result.update(accepted=True, center=p.tolist(), estimated_diameter_px=diam,
                      reason='qualified_offline_estimate')
    except (ValueError, cv2.error) as exc:
        result['reason'] = str(exc)
    return result


def hide_target(detections, truth, target):
    """Evaluation-only truth is used SOLELY to remove target leakage.

    A wrong label on the target's physical box must also disappear. Truth is
    never used to clean/relabel remaining YOLO anchors or select a fit.
    """
    t = truth[target]
    return [a for a in detections if a['label'] != target and
            np.linalg.norm(np.asarray(a['center'])-t['center']) > .65*t['diameter']]


def yolo_anchors(detections, template, size):
    w, h = size
    out = []
    for d in detections:
        l, conf = d['label'], float(d['confidence'])
        if l not in template or conf < .70:
            continue
        x1, y1, x2, y2 = d['bbox_xyxy']
        if not np.isfinite([x1,y1,x2,y2,conf]).all() or min(x1,y1,w-x2,h-y2) < 5 or min(x2-x1,y2-y1) < 14:
            continue
        out.append(dict(label=l, center=[(x1+x2)/2,(y1+y2)/2],
                        diameter=(x2-x1+y2-y1)/2, confidence=conf))
    return out


def evaluate(manifest, base, replay=None):
    template = manifest['template']['coordinates']
    references = {f['task'] for f in manifest['frames'] if f['split']=='reference'}
    validations = {f['task'] for f in manifest['frames'] if f['split']=='validation'}
    if references & validations:
        raise ValueError('template_validation_task_leakage')
    rows = []
    for f in manifest['frames']:
        data = (base/f['image']).read_bytes()
        if hashlib.sha256(data).hexdigest() != f['sha256']:
            raise ValueError('image_hash_mismatch')
        if f['split'] != 'validation':
            continue
        truth = f['buttons']
        for mode in ['verified_manual_anchors'] + (['raw_yolo_cpu_jpeg'] if replay is not None else []):
            anchors = [dict(label=l, **v) for l,v in truth.items()] if mode=='verified_manual_anchors' else yolo_anchors(replay[f['id']]['detections'], template, f['size'])
            for target in f['targets']:
                hidden = hide_target(anchors, truth, target)
                started = time.perf_counter()
                result = infer(template, hidden, target, f['size'])
                result.update(frame=f['id'], task=f['task'], target=target, mode=mode,
                              geometry_ms=(time.perf_counter()-started)*1000,
                              manual_center=truth[target]['center'], diameter=truth[target]['diameter'])
                if result['accepted']:
                    error = float(np.linalg.norm(np.asarray(result['center'])-truth[target]['center']))
                    result.update(error_px=error,error_diameters=error/truth[target]['diameter'],
                                  within_provisional_2d_tolerance=error <= min(5, .15*truth[target]['diameter']))
                rows.append(result)
    return rows


def summarize(rows):
    out = {}
    for mode in sorted({r['mode'] for r in rows}):
        rr = [r for r in rows if r['mode']==mode]
        aa = [r for r in rr if r['accepted']]
        reasons = {}
        for r in rr:
            if not r['accepted']:
                reasons[r['reason']] = reasons.get(r['reason'],0)+1
        errors = [r['error_px'] for r in aa]
        times = [r['geometry_ms'] for r in rr]
        out[mode] = dict(cases=len(rr),accepted=len(aa),rejected=len(rr)-len(aa),
            accepted_outside_tolerance=sum(not r['within_provisional_2d_tolerance'] for r in aa),
            mean_error_px=float(np.mean(errors)) if errors else None,
            max_error_px=max(errors,default=None), geometry_median_ms=float(np.median(times)),
            geometry_p95_ms=float(np.percentile(times,95)), rejection_reasons=reasons,
            by_target={t:dict(cases=sum(r['target']==t for r in rr),
                              accepted=sum(r['target']==t for r in aa)) for t in ['1','2','3','4','5']})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay-yolo',action='store_true',help='Offline saved JPEG, CPU only; never opens camera')
    parser.add_argument('--detections',type=Path,help='Reuse recorded offline replay')
    args = parser.parse_args()
    manifest=json.loads(args.manifest.read_text()); base=args.manifest.parent
    args.output.mkdir(parents=True,exist_ok=True)
    replay=json.loads(args.detections.read_text())['frames'] if args.detections else None
    if args.replay_yolo:
        sys.path.insert(0,str(ROOT))
        from piper_handeye.button_pose import YoloButtonDetector
        model=ROOT.parent/'button_recognition/models/elevator_button_yolo26m_best.pt'
        detector=YoloButtonDetector(model,conf=.15,iou=.5,imgsz=640,device='cpu')
        import torch
        torch.set_num_threads(2)
        replay={}
        for f in manifest['frames']:
            if f['split']!='validation': continue
            image=cv2.imread(str(base/f['image']))
            if image is None: raise ValueError(f['image'])
            started=time.perf_counter(); dd=detector.detect(image)
            replay[f['id']]=dict(seconds=time.perf_counter()-started,detections=[dict(label=d.label,confidence=float(d.confidence),bbox_xyxy=list(d.bbox_xyxy)) for d in dd])
            print(f['id'],len(dd),flush=True)
        (args.output/'yolo_replay.json').write_text(json.dumps(dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            scope='Raw-only saved JPEG CPU replay, not historical live output; no CLAHE or local retry',frames=replay),indent=2))
    rows=evaluate(manifest,base,replay)
    report=dict(scope='Offline 2D centres against assistant visual annotations (~2px uncertainty); not physical accuracy, 3D validation, light-state detection or deployment.',
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                template_validation_tasks_disjoint=True,summary=summarize(rows),results=rows)
    (args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
