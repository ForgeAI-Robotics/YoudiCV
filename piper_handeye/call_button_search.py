"""Bounded wrist search helpers; an UP anchor never authorizes a DOWN press."""
import numpy as np
from .kinematics import forward_kinematics
from .robot import PIPER_JOINT_LIMITS
from .observation_visibility import complete_button_box


def down_search_anchor(response, min_confidence=.70):
    """A partially clipped UP may seed a guarded search, not lamp/contact IK.

    The caller must subsequently collect its stable RGB-D plane and arm the
    existing depth supervisor. Ambiguous or malformed anchors are rejected.
    """
    try:
        w,h=map(float,response['image_size'])
        candidates=[]
        for item in response.get('detections',[]):
            if item.get('label')!='up' or float(item.get('confidence',0))<min_confidence:
                continue
            x1,y1,x2,y2=map(float,item['bbox_xyxy'])
            if (np.isfinite([w,h,x1,y1,x2,y2]).all() and
                    0<=x1<x2<=w and 0<=y1<y2<=h and
                    14<=x2-x1<=w*.5 and 14<=y2-y1<=h*.5):
                candidates.append(item)
        if len(candidates)==1:
            return candidates[0]
    except (KeyError,TypeError,ValueError):
        pass
    return None


def up_search_anchor(response, min_confidence=.70):
    """Select a safety-plane cue for UP search, never an observation result.

    Prefer the existing complete DOWN reference. If it is missing, one strong
    UP box wholly inside the image but within the observation edge margin can
    seed the same guarded wrist search. Stable RGB-D and collision validation
    remain mandatory before moving; full UP frames are mandatory afterwards.
    """
    try:
        image_size = response['image_size']
        complete_down = []
        for item in response.get('detections', []):
            confidence = float(item.get('confidence', 0))
            if not np.isfinite(confidence) or confidence < min_confidence:
                continue
            if item.get('label') == 'up' and complete_button_box(
                item.get('bbox_xyxy'), image_size
            ):
                return None  # A complete UP belongs to the direct path.
            if item.get('label') == 'down' and complete_button_box(
                item.get('bbox_xyxy'), image_size
            ):
                complete_down.append(item)
        if len(complete_down) == 1:
            return complete_down[0]
        if complete_down:
            return None
        anchor = down_search_anchor(response, min_confidence)
        if anchor is not None and np.isfinite(float(anchor['confidence'])):
            return anchor
    except (KeyError, TypeError, ValueError):
        pass
    return None


def choose_down_search_j5_step(joints, handeye, anchor_base, camera_matrix,
                               calibration_size, image_size):
    """Choose only a 0.05 rad wrist step which improves the panel view.

    Place the known UP above image centre, leaving space beneath for DOWN.
    Candidates still require the caller's full trajectory collision validation.
    """
    current=np.asarray(joints,dtype=float)
    matrix=np.asarray(camera_matrix,dtype=float).copy()
    width,height=map(float,image_size)
    cw,ch=map(float,calibration_size)
    matrix[0,:]*=width/cw;matrix[1,:]*=height/ch
    point=np.r_[np.asarray(anchor_base,dtype=float)[:3],1.]
    def projection(q):
        xyz=(np.linalg.inv(forward_kinematics(q)@handeye)@point)[:3]
        if not np.isfinite(xyz).all() or xyz[2]<=.10:
            raise RuntimeError('search anchor is not safely in front of camera')
        uv=matrix@xyz
        return uv[:2]/uv[2]
    initial=projection(current)
    def cost(uv):return abs(float(uv[1])-height*.35)+.1*abs(float(uv[0])-width*.5)
    candidates=[]
    for delta in (-.05,.05):
        q=current.copy();q[4]+=delta
        if not PIPER_JOINT_LIMITS[4,0]<=q[4]<=PIPER_JOINT_LIMITS[4,1]:continue
        try:uv=projection(q)
        except RuntimeError:continue
        candidates.append((cost(uv),q,uv,delta))
    if not candidates:raise RuntimeError('no legal J5 search candidate')
    best=min(candidates,key=lambda item:item[0])
    if best[0]>=cost(initial)-.5:
        raise RuntimeError('J5 search cannot further improve the panel view')
    return best[1],dict(delta_j5_rad=best[3],anchor_pixel_before=initial.tolist(),
                        predicted_anchor_pixel_after=best[2].tolist(),
                        view_error_before=cost(initial),view_error_after=best[0])
