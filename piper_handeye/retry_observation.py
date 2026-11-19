"""Plan the outward-only change of photo pose after a confirmed OFF."""
import numpy as np

from .kinematics import forward_kinematics, forward_kinematics_batch
from .segmented_button_motion import solve_camera_centered_observation_joints
from .timed_trajectory import generate_timed_joint_trajectory


from .button_light_flow import RETRY_PREPARATION_DISTANCE_M


def checked_retry_distance(joints, panel, requested_distance_m=RETRY_PREPARATION_DISTANCE_M):
    """Keep exact verified joints; allow the original +/-2cm target refinement band."""
    q=np.asarray(joints,float);frame=np.asarray(panel,float)
    if q.shape!=(6,) or frame.shape!=(4,4) or not np.isfinite(q).all() or not np.isfinite(frame).all():
        raise ValueError('invalid retry observation geometry')
    normal=frame[:3,2];normal=normal/np.linalg.norm(normal)
    distance=float((forward_kinematics(q)[:3,3]-frame[:3,3])@normal)
    if not np.isfinite(distance) or abs(distance-float(requested_distance_m))>.02:
        raise RuntimeError('retry holding pose is outside the requested observation band')
    return distance


def plan_retry_observation_transition(current_joints, panel, handeye):
    """Reuse an existing 8cm hold, otherwise aim a new outward 8cm endpoint.

    Caller must check current encoders against the old validated photo pose and
    dry-validate the complete path and a return path before execution.
    """
    current=np.asarray(current_joints,float)
    frame=np.asarray(panel,float).copy()
    if current.shape!=(6,) or frame.shape!=(4,4) or not np.isfinite(current).all() or not np.isfinite(frame).all():
        raise ValueError('invalid retry transition start')
    normal=frame[:3,2]/np.linalg.norm(frame[:3,2])
    start=float((forward_kinematics(current)[:3,3]-frame[:3,3])@normal)
    if not .03<=start<=.10:
        raise RuntimeError('retry transition must start at a verified near observation')
    target,alignment=solve_camera_centered_observation_joints(
        current,frame,handeye,RETRY_PREPARATION_DISTANCE_M)
    points,audit=generate_timed_joint_trajectory(current,target,
        cartesian_speed_m_s=.15,joint_velocity_scale=.75,point_period_sec=.05,duration_margin=1.02)
    distances=(forward_kinematics_batch(np.array([p['positions'] for p in points]))[:,:3,3]
               -frame[:3,3])@normal
    if (np.min(np.diff(distances))<-.002 or np.min(distances)<start-.002
            or np.max(distances)>RETRY_PREPARATION_DISTANCE_M+.002):
        raise RuntimeError('retry transition is not bounded outward retreat')
    return points,target,dict(mode='retry_camera_aimed_8cm',start_panel_distance_m=start,
        requested_panel_distance_m=RETRY_PREPARATION_DISTANCE_M,
        trajectory_duration_sec=audit.duration_sec,**alignment)
