"""Bounded image registration for stationary numeric post-press evidence.

Fit on surrounding scene features, excluding the changing button. This changes
comparison pixels only; original RGB, YOLO boxes and robot targets stay intact.
No image writes, inference, robot access or threshold changes happen here.
"""
import time

import cv2
import numpy as np


POLICY = dict(version=1, scope='stationary_numeric_strong_on_only',
              min_button_size_px=48, max_features=600,
              max_center_shift_fraction=.10, max_log_size_change=.08,
              max_rotation_deg=2., min_inliers=24, min_inlier_fraction=.45,
              max_median_reprojection_px=1.0, max_p90_reprojection_px=1.8,
              max_target_registration_error_fraction=.03,
              min_supported_quadrants=3, max_reference_age_sec=30.)


def _features(path, box, cache=None):
    key=(str(path),tuple(float(x) for x in box))
    if cache is not None and key in cache:
        return cache[key]
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('registration_image_unreadable')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape
    if w>1280 or h>720:
        raise ValueError('registration_image_size_outside_budget')
    x1,y1,x2,y2=box;bw,bh=x2-x1,y2-y1;cx,cy=(x1+x2)/2,(y1+y2)/2
    mask=np.zeros_like(gray)
    # Local surroundings avoid unrelated distant walls. The generous button
    # exclusion removes both its glyph and its changing illuminated bezel.
    lx,rx=max(8,int(cx-4*bw)),min(w-8,int(cx+4*bw))
    ty,by=max(8,int(cy-4*bh)),min(h-8,int(cy+4*bh))
    mask[ty:by,lx:rx]=255
    ex1,ex2=max(0,int(x1-.35*bw)),min(w,int(np.ceil(x2+.35*bw)))
    ey1,ey2=max(0,int(y1-.35*bh)),min(h,int(np.ceil(y2+.35*bh)))
    mask[ey1:ey2,ex1:ex2]=0
    orb=cv2.ORB_create(nfeatures=POLICY['max_features'],edgeThreshold=10,fastThreshold=10)
    keys,descriptors=orb.detectAndCompute(gray,mask)
    points=np.array([k.pt for k in keys],dtype=np.float32).reshape(-1,2)
    value=(gray,points,descriptors)
    if cache is not None:
        cache[key]=value
        while len(cache)>4:
            cache.pop(next(iter(cache)))
    return value


def register_reference(reference, current, *, cache=None):
    """Return a registered 64x64 OFF patch, or None plus rejection evidence.

    The caller authorizes a stationary same-button comparison and its phase.
    A successful fit says nothing about ON/OFF; original lamp rules must run.
    """
    began=time.perf_counter()
    audit=dict(policy=dict(POLICY),accepted=False,
               reference_sequence=reference.sequence,current_sequence=current.sequence,
               reference_box=np.asarray(reference.box).tolist(),
               current_box=np.asarray(current.box).tolist(),
               raw_images_modified=False,fit_region='surroundings_excluding_button')
    def reject(reason):
        audit.update(reason=reason,elapsed_sec=time.perf_counter()-began)
        return None,audit
    try:
        a,b=np.asarray(reference.box,float),np.asarray(current.box,float)
        if (a.shape!=(4,) or b.shape!=(4,) or not np.isfinite([a,b]).all()
                or current.sequence<=reference.sequence
                or not 0<current.captured-reference.captured<=POLICY['max_reference_age_sec']):
            return reject('invalid_geometry_or_pair_time')
        sa,sb=a[2:]-a[:2],b[2:]-b[:2]
        if min(*sa,*sb)<POLICY['min_button_size_px']:
            return reject('button_too_small_for_registration')
        ca,cb=(a[:2]+a[2:])/2,(b[:2]+b[2:])/2
        shift=np.abs(cb-ca)/sa;size=np.abs(np.log(sa/sb))
        audit.update(raw_center_shift_fraction=shift.tolist(),log_size_change=size.tolist())
        if (np.max(shift)>POLICY['max_center_shift_fraction']
                or np.max(size)>POLICY['max_log_size_change']):
            return reject('view_outside_relaxed_numeric_limits')
        ga,pa,da=_features(reference.image,a,cache)
        gb,pb,db=_features(current.image,b)
        if ga.shape!=gb.shape:
            return reject('image_size_changed')
        h,w=ga.shape
        if min(*a[:2],*b[:2],w-a[2],h-a[3],w-b[2],h-b[3])<2:
            return reject('button_not_complete')
        if da is None or db is None or min(len(pa),len(pb))<POLICY['min_inliers']:
            return reject('insufficient_background_features')
        matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        forward=matcher.knnMatch(da,db,k=2)
        reverse={x.queryIdx:x.trainIdx for pair in matcher.knnMatch(db,da,k=2)
                 if len(pair)==2 for x,y in [pair] if x.distance<.72*y.distance}
        matches=[x for pair in forward if len(pair)==2 for x,y in [pair]
                 if x.distance<.72*y.distance and reverse.get(x.trainIdx)==x.queryIdx]
        audit['mutual_matches']=len(matches)
        if len(matches)<POLICY['min_inliers']:
            return reject('insufficient_independent_matches')
        x=np.float32([pa[m.queryIdx] for m in matches]);y=np.float32([pb[m.trainIdx] for m in matches])
        matrix,inliers=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,
            ransacReprojThreshold=1.5,maxIters=1500,confidence=.995)
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            return reject('background_fit_failed')
        selected=inliers.ravel().astype(bool);n=int(selected.sum())
        audit.update(inliers=n,inlier_fraction=n/len(matches))
        if n<POLICY['min_inliers'] or n/len(matches)<POLICY['min_inlier_fraction']:
            return reject('insufficient_background_consensus')
        errors=np.linalg.norm(x@matrix[:,:2].T+matrix[:,2]-y,axis=1)[selected]
        median,p90=float(np.median(errors)),float(np.percentile(errors,90))
        audit.update(median_reprojection_px=median,p90_reprojection_px=p90)
        if median>POLICY['max_median_reprojection_px'] or p90>POLICY['max_p90_reprojection_px']:
            return reject('background_fit_inaccurate')
        support=x[selected]
        counts=[int(np.count_nonzero(horizontal & vertical))
                for horizontal in (support[:,0]<ca[0],support[:,0]>=ca[0])
                for vertical in (support[:,1]<ca[1],support[:,1]>=ca[1])]
        span=np.ptp(support,axis=0)
        audit.update(supported_quadrants=counts,background_span_px=span.tolist())
        if sum(c>=4 for c in counts)<POLICY['min_supported_quadrants'] or np.any(span<1.25*sa):
            return reject('background_support_not_distributed')
        scale=float(np.hypot(matrix[0,0],matrix[1,0]))
        angle=float(np.rad2deg(np.arctan2(matrix[1,0],matrix[0,0])))
        predicted=matrix[:,:2]@ca+matrix[:,2]
        mismatch=np.abs(predicted-cb)/sb
        audit.update(matrix_reference_to_current=matrix.tolist(),scale=scale,
                     rotation_deg=angle,target_fit_error_fraction=mismatch.tolist())
        if (scale<=0 or abs(np.log(scale))>POLICY['max_log_size_change']
                or abs(angle)>POLICY['max_rotation_deg']
                or np.max(mismatch)>POLICY['max_target_registration_error_fraction']
                or np.max(np.abs(predicted-ca)/sa)>POLICY['max_center_shift_fraction']):
            return reject('transform_not_consistent_with_same_button')
        x1,y1,x2,y2=np.rint(b).astype(int)
        # Verify every comparison pixel samples valid original reference data.
        inverse=cv2.invertAffineTransform(matrix)
        corners=np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]],float)
        source=corners@inverse[:,:2].T+inverse[:,2]
        if np.any(source<1) or np.any(source[:,0]>w-2) or np.any(source[:,1]>h-2):
            return reject('comparison_uses_warp_border')
        aligned=cv2.warpAffine(ga,matrix,(w,h),flags=cv2.INTER_LINEAR)
        patch=cv2.resize(aligned[y1:y2,x1:x2],(64,64),interpolation=cv2.INTER_AREA).astype(np.float32)
        audit.update(accepted=True,reason='background_registered_same_button',
                     elapsed_sec=time.perf_counter()-began)
        return patch,audit
    except (ValueError,cv2.error,OSError) as exc:
        audit['error']=f'{type(exc).__name__}: {exc}'
        return reject('registration_unavailable')
