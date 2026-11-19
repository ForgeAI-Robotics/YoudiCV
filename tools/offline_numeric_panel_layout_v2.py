#!/usr/bin/env python3
"""Experimental 2D panel inference plus saved-image rim check. NO robot API.

The target numeral is not read. A complete circular/elliptical physical rim
must independently support the location suggested by other digit labels.
This is panel-specific and is not an ON/OFF or pressing implementation.
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
from tools.offline_numeric_panel_layout import project, hide_target, yolo_anchors, summarize


def local_scale(h, point):
    p=np.asarray(point,float)
    v=project(h,[p,p+[.001,0],p+[0,.001]])
    jac=np.column_stack(((v[1]-v[0])/.001,(v[2]-v[0])/.001))
    return math.sqrt(abs(float(np.linalg.det(jac))))


def candidate(template,anchors,target,size):
    result=dict(accepted=False,purpose='offline_2d_research_only',method='layout_plus_complete_rim_v2')
    try:
        if target not in template: raise ValueError('unknown_target')
        aa=[a for a in anchors if a['label']!=target]; labels=[a['label'] for a in aa]
        result['anchors']=labels
        if len(labels)!=len(set(labels)): raise ValueError('duplicate_anchor_label')
        if len(aa)<4: raise ValueError('fewer_than_four_anchors')
        if any(l not in template for l in labels): raise ValueError('unknown_anchor')
        src=np.array([template[l] for l in labels],float);dst=np.array([a['center'] for a in aa],float)
        diameters=np.array([a['diameter'] for a in aa],float);diam=float(np.median(diameters))
        if not np.isfinite(src).all() or not np.isfinite(dst).all() or not np.isfinite(diameters).all() or min(diameters)<8: raise ValueError('invalid_geometry')
        if min(sum(src[:,0]==x) for x in (0,1))<2 or len(set(src[:,1]))<2: raise ValueError('insufficient_two_column_coverage')
        for i in range(len(dst)):
            if any(np.linalg.norm(dst[i]-dst[j])<.35*diam for j in range(i)): raise ValueError('overlapping_anchor_labels')
        h,_=cv2.findHomography(src,dst,0)
        if h is None: raise ValueError('degenerate_fit')
        errors=np.linalg.norm(project(h,src)-dst,axis=1)
        result['max_anchor_error_px']=float(max(errors))
        if max(errors)>min(4,max(2,.08*diam)): raise ValueError('anchor_layout_conflict')
        t=np.array(template[target],float)
        if t[1]<min(src[:,1])-1 or t[1]>max(src[:,1])+1: raise ValueError('extrapolation_more_than_one_row')
        p,right,down=project(h,[t,t+[.1,0],t+[0,.1]])
        if right[0]<=p[0] or down[1]<=p[1]: raise ValueError('inverted_panel')
        scales=np.array([local_scale(h,s) for s in src])
        if not np.isfinite(scales).all() or min(scales)<1e-6: raise ValueError('invalid_local_scale')
        normalized=diameters/scales; median=float(np.median(normalized))
        mismatch=float(max(np.abs(normalized/median-1)))
        result['anchor_size_scale_mismatch']=mismatch
        if mismatch>.25: raise ValueError('anchor_size_geometry_conflict')
        target_diameter=median*local_scale(h,t)
        if not math.isfinite(target_diameter) or not .6*diam<=target_diameter<=1.6*diam: raise ValueError('target_scale_unstable')
        # With four points a homography fits exactly. A simpler affine model
        # and independently observed rim are mandatory, not an optional gate.
        affine_src=np.column_stack((src,np.ones(len(src))))
        affine,_,rank,_=np.linalg.lstsq(affine_src,dst,rcond=None)
        if rank<3: raise ValueError('degenerate_affine')
        pa=np.append(t,1)@affine
        disagreement=float(np.linalg.norm(p-pa)); result['affine_projective_disagreement_px']=disagreement
        if len(aa)==4 and disagreement>max(6,.15*target_diameter): raise ValueError('model_disagreement')
        leave_predictions=[]
        for i in range(len(src)):
            ss,dd=np.delete(src,i,0),np.delete(dst,i,0)
            if len(ss)<4 or min(sum(ss[:,0]==x) for x in (0,1))<2: continue
            hh,_=cv2.findHomography(ss,dd,0)
            if hh is not None: leave_predictions.append(project(hh,[t])[0])
        spread=max((float(np.linalg.norm(x-p)) for x in leave_predictions),default=0.)
        result['leave_one_anchor_target_spread_px']=spread
        if spread>max(6,.15*target_diameter): raise ValueError('target_prediction_unstable')
        w,hh=size
        if not (5+target_diameter/2<=p[0]<=w-5-target_diameter/2 and 5+target_diameter/2<=p[1]<=hh-5-target_diameter/2): raise ValueError('target_not_fully_visible')
        result.update(accepted=True,center=p.tolist(),estimated_diameter_px=target_diameter,reason='candidate_requires_rim_check',four_anchor_mode=len(aa)==4)
    except (ValueError,cv2.error,np.linalg.LinAlgError) as exc: result['reason']=str(exc)
    return result


def verify_rim(image,center,diameter):
    """No digit label/truth/ON state input: only bounded local edge geometry."""
    fail=dict(accepted=False,reason='no_complete_button_rim')
    if image is None or image.ndim not in (2,3): return fail
    p=np.asarray(center,float)
    if not np.isfinite(p).all() or not math.isfinite(diameter) or diameter<8: return fail
    h,w=image.shape[:2];rad=.85*diameter
    x1=max(0,int(math.floor(p[0]-rad)));y1=max(0,int(math.floor(p[1]-rad)))
    x2=min(w,int(math.ceil(p[0]+rad)));y2=min(h,int(math.ceil(p[1]+rad)))
    if x2-x1<8 or y2-y1<8:return fail
    crop=image[y1:y2,x1:x2];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
    edge=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),30,90)
    contours,_=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    # Dark button-body contour is complementary to broken specular rim edges.
    # This segments physical shape only; it does not classify lamp brightness.
    _,dark=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    dark=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    body_contours,_=cv2.findContours(dark,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours=list(contours)+list(body_contours)
    choices=[]
    for contour in contours:
        if len(contour)<16:continue
        (cx,cy),(a,b),angle=cv2.fitEllipse(contour)
        if min(a,b)<.65*diameter or max(a,b)>1.35*diameter or max(a,b)/max(min(a,b),1e-6)>1.7:continue
        global_center=np.array([cx+x1,cy+y1])
        if np.linalg.norm(global_center-p)>min(8,.15*diameter):continue
        if min(global_center[0]-max(a,b)/2,global_center[1]-max(a,b)/2,w-global_center[0]-max(a,b)/2,h-global_center[1]-max(a,b)/2)<5:continue
        theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        points=(contour[:,0,:]-[cx,cy])@rot.T
        normalized=points/[a/2,b/2];rho=np.linalg.norm(normalized,axis=1)
        angles=np.arctan2(normalized[:,1],normalized[:,0]);bins=np.unique(np.floor((angles+np.pi)/(2*np.pi)*36).astype(int)%36)
        gaps=np.diff(np.r_[np.sort(angles),min(angles)+2*np.pi])
        fit=float(np.percentile(abs(rho-1),90));area=cv2.contourArea(contour)/(np.pi*a*b/4)
        if len(bins)<28 or max(gaps)>np.pi/3 or fit>.10 or not .80<=area<=1.15:continue
        choices.append(dict(center=global_center.tolist(),axes=[a,b],angle=angle,fit_p90=fit,angular_bins=len(bins),area_fraction=area))
    if not choices:return fail
    centers=np.array([c['center'] for c in choices]);median=np.median(centers,axis=0)
    if max(np.linalg.norm(centers-median,axis=1))>max(2,.05*diameter):return dict(accepted=False,reason='ambiguous_button_rims',candidates=choices)
    best=min(choices,key=lambda c:c['fit_p90'])
    return dict(accepted=True,reason='complete_rim_verified',center=median.tolist(),matching_contours=len(choices),rim=best,correction_px=float(np.linalg.norm(median-p)))


def infer_image(template,anchors,target,image):
    result=candidate(template,anchors,target,[image.shape[1],image.shape[0]])
    if not result['accepted']:return result
    result['layout_center']=result['center'];verification=verify_rim(image,result['center'],result['estimated_diameter_px'])
    result['rim_verification']=verification;result.update(accepted=verification['accepted'],reason=verification['reason'])
    if verification['accepted']:result['center']=verification['center']
    return result


def evaluate(manifest,base,replay):
    refs={f['task'] for f in manifest['frames'] if f['split']=='reference'}
    if refs & {f['task'] for f in manifest['frames'] if f['split']=='validation'}:raise ValueError('task_leakage')
    rows=[];template=manifest['template']['coordinates']
    for f in manifest['frames']:
        path=base/f['image']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('image_hash_mismatch')
        if f['split']!='validation':continue
        image=cv2.imread(str(path));truth=f['buttons']
        for mode in ['verified_manual_anchors','raw_yolo_cpu_jpeg']:
            anchors=[dict(label=l,**v) for l,v in truth.items()] if mode=='verified_manual_anchors' else yolo_anchors(replay[f['id']]['detections'],template,f['size'])
            for target in f['targets']:
                aa=hide_target(anchors,truth,target);start=time.perf_counter();result=infer_image(template,aa,target,image)
                result.update(frame=f['id'],task=f['task'],target=target,mode=mode,geometry_ms=(time.perf_counter()-start)*1000,manual_center=truth[target]['center'],diameter=truth[target]['diameter'])
                if result['accepted']:
                    error=float(np.linalg.norm(np.array(result['center'])-truth[target]['center']))
                    result.update(error_px=error,error_diameters=error/truth[target]['diameter'],within_provisional_2d_tolerance=error<=min(5,.15*truth[target]['diameter']))
                rows.append(result)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--detections',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    m=json.loads(args.manifest.read_text());replay=json.loads(args.detections.read_text())['frames'];rows=evaluate(m,args.manifest.parent,replay)
    report=dict(scope='Offline only; old dataset is development data for v2, not new independent validation. Centres are assistant visual annotations, not physical truth.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary=summarize(rows),results=rows)
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
