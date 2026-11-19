#!/usr/bin/env python3
"""Practical offline 2D recovery: retain strong geometry, optional rim refinement.

All outputs remain inferred positions, never YOLO detections/press commands.
No camera, RPC, ROS, service or robot access. Reuses saved raw YOLO results.
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.offline_numeric_panel_layout import infer as infer_v1, project, hide_target, yolo_anchors, summarize
from tools.offline_numeric_panel_layout_v2 import candidate as candidate_v2, verify_rim, local_scale


def sparse_candidate(template,anchors,target,size):
    result=dict(accepted=False,purpose='offline_inferred_2d_only',target_detected=False,method='sparse_geometry')
    try:
        if target not in template:raise ValueError('unknown_target')
        aa=[a for a in anchors if a['label']!=target];labels=[a['label'] for a in aa];result['anchors']=labels
        if len(labels)!=len(set(labels)):raise ValueError('duplicate_anchor_label')
        if any(l not in template for l in labels):raise ValueError('unknown_anchor')
        if len(aa) not in (3,4):raise ValueError('sparse_mode_requires_three_or_four')
        s=np.array([template[l] for l in labels],float);d=np.array([a['center'] for a in aa],float)
        ds=np.array([a['diameter'] for a in aa],float);median=float(np.median(ds));t=np.array(template[target],float)
        if not np.isfinite(s).all() or not np.isfinite(d).all() or not np.isfinite(ds).all() or min(ds)<8:raise ValueError('invalid_geometry')
        if min(sum(s[:,0]==x) for x in (0,1))<1:raise ValueError('single_column_only')
        for i in range(len(d)):
            if any(np.linalg.norm(d[i]-d[j])<.35*median for j in range(i)):raise ValueError('overlapping_anchor_labels')
        affine_s=np.column_stack((s,np.ones(len(s))));affine,_,rank,_=np.linalg.lstsq(affine_s,d,rcond=None)
        if rank<3:raise ValueError('degenerate_affine')
        ap=np.append(t,1)@affine
        if len(aa)==3:
            # E.g. left 1/3 bracket 2, right 9 is the independently known same row.
            own=s[s[:,0]==t[0]];other=s[s[:,0]!=t[0]]
            if len(own)!=2 or len(other)!=1 or not min(own[:,1])<t[1]<max(own[:,1]) or other[0,1]!=t[1]:raise ValueError('three_anchor_target_not_bracketed_with_row_partner')
            if max(own[:,1])-min(own[:,1])>3:raise ValueError('bracketing_gap_too_large')
            if any(a.get('confidence',1)<.85 for a in aa):raise ValueError('three_anchor_confidence_low')
            h=np.vstack((affine.T,[0,0,1]));p=ap;result['method']='three_anchor_bracket_and_row_partner'
        else:
            if min(sum(s[:,0]==x) for x in (0,1))<2:raise ValueError('four_anchors_need_two_per_column')
            h,_=cv2.findHomography(s,d,0)
            if h is None:raise ValueError('degenerate_fit')
            if t[1]<min(s[:,1])-1 or t[1]>max(s[:,1])+1:raise ValueError('extrapolation_more_than_one_row')
            p=project(h,[t])[0]
        scales=np.array([local_scale(h,point) for point in s]);ratios=ds/scales
        if not np.isfinite(scales).all() or min(scales)<1e-6 or not np.isfinite(ratios).all():raise ValueError('invalid_scale')
        mismatch=float(max(abs(ratios/np.median(ratios)-1)));result['anchor_size_scale_mismatch']=mismatch
        if mismatch>.25:raise ValueError('anchor_size_geometry_conflict')
        diameter=float(np.median(ratios)*local_scale(h,t))
        if not .6*median<=diameter<=1.6*median:raise ValueError('target_scale_unstable')
        disagreement=float(np.linalg.norm(ap-p));result['affine_projective_disagreement_px']=disagreement
        if disagreement>max(12,.25*diameter):raise ValueError('model_disagreement')
        pp,right,down=project(h,[t,t+[.1,0],t+[0,.1]])
        if right[0]<=pp[0] or down[1]<=pp[1]:raise ValueError('inverted_panel')
        w,hh=size
        if not (5+diameter/2<=p[0]<=w-5-diameter/2 and 5+diameter/2<=p[1]<=hh-5-diameter/2):raise ValueError('target_not_fully_visible')
        result.update(accepted=True,center=p.tolist(),estimated_diameter_px=diameter,reason='geometry_inferred_not_visual_detection')
    except (ValueError,cv2.error,np.linalg.LinAlgError) as exc:result['reason']=str(exc)
    return result


def infer_image(template,anchors,target,image):
    size=[image.shape[1],image.shape[0]]
    # Retain the previous high-redundancy geometry route; lack of a closed rim
    # must not introduce a new mandatory rejection on that route.
    result=infer_v1(template,anchors,target,size)
    if result['accepted']:result['method']='existing_five_plus_geometry'
    else:
        original_reason=result['reason'];result=candidate_v2(template,anchors,target,size)
        if not result['accepted']:
            result=sparse_candidate(template,anchors,target,size)
        result['previous_reason']=original_reason
    result.update(target_detected=False,purpose='offline_inferred_2d_only',light_state='not_evaluated',requires_live_depth_and_motion_validation=True)
    if not result['accepted']:return result
    result['layout_center']=result['center'];result['rim_verification']=verify_rim(image,result['center'],result['estimated_diameter_px'])
    rim=result['rim_verification']
    if rim['accepted']:
        result.update(center=rim['center'],reason='geometry_with_optional_rim_refinement')
    else:result['reason']='geometry_accepted_without_complete_rim'
    return result


def evaluate(manifest,base,replay):
    ref={f['task'] for f in manifest['frames'] if f['split']=='reference'}
    if ref & {f['task'] for f in manifest['frames'] if f['split']=='validation'}:raise ValueError('task_leakage')
    rows=[];template=manifest['template']['coordinates']
    for f in manifest['frames']:
        path=base/f['image']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('image_hash_mismatch')
        if f['split']!='validation':continue
        image=cv2.imread(str(path));truth=f['buttons']
        for mode in ['verified_manual_anchors','raw_yolo_cpu_jpeg']:
            anchors=[dict(label=l,**v) for l,v in truth.items()] if mode=='verified_manual_anchors' else yolo_anchors(replay[f['id']]['detections'],template,f['size'])
            for target in f['targets']:
                aa=hide_target(anchors,truth,target);start=time.perf_counter();r=infer_image(template,aa,target,image)
                r.update(frame=f['id'],task=f['task'],target=target,mode=mode,geometry_ms=(time.perf_counter()-start)*1000,manual_center=truth[target]['center'],diameter=truth[target]['diameter'])
                if r['accepted']:
                    error=float(np.linalg.norm(np.array(r['center'])-truth[target]['center']))
                    r.update(error_px=error,error_diameters=error/truth[target]['diameter'],within_provisional_2d_tolerance=error<=min(5,.15*truth[target]['diameter']))
                rows.append(r)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--detections',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads(a.manifest.read_text());replay=json.loads(a.detections.read_text())['frames'];rows=evaluate(m,a.manifest.parent,replay)
    report=dict(scope='Development replay of previously inspected images. No new independent accuracy or physical press claims; inferred positions are not detections.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary=summarize(rows),results=rows)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
