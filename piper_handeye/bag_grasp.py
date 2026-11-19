"""纸袋 RGB-D 定位、观测位和有界触碰规划。无网络或机器人命令。"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .kinematics import forward_kinematics, forward_kinematics_batch, inverse_kinematics
from .robot import PIPER_JOINT_LIMITS
from .timed_trajectory import generate_timed_joint_trajectory, reverse_timed_trajectory, PIPER_MAX_VELOCITIES

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_PATH = ROOT / 'config/bag_observation.json'
PRECONTACT_M = 0.020
GRASP_OVERTRAVEL_M = 0.070
GRASP_LIFT_M = 0.050
CYCLE_PATH_NAMES = ('approach','insertion','lift','carry_out_near','carry_out_far',
                    'carry_back_far','carry_back_near','lower','withdraw_insertion','withdraw_approach')

def joints(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('需要六个有限的关节编码器读数')
    if np.any(q < PIPER_JOINT_LIMITS[:, 0]) or np.any(q > PIPER_JOINT_LIMITS[:, 1]):
        raise ValueError('关节编码器超出既有限位')
    return q


def calibration(config):
    p = Path(config['calibration']['handeye_yaml'])
    if not p.is_absolute():
        p = ROOT / p
    import yaml
    raw = p.read_bytes()
    data = yaml.safe_load(raw)
    K = np.asarray(data['camera_matrix'], dtype=float)
    size = np.asarray(config['calibration'].get('rgb_calibration_size', [1920, 1080]), dtype=float)
    scale = np.array([640., 360.]) / size
    if not np.allclose(scale[0], scale[1]):
        raise ValueError('RGB 标定与采集尺寸宽高比不同')
    K[:2] *= scale[0]
    T = np.asarray(data['gripper_to_camera'], dtype=float)
    d = np.asarray(data['distortion'], dtype=float)
    if T.shape != (4,4) or K.shape != (3,3) or not all(np.isfinite(v).all() for v in [T,K,d]):
        raise ValueError('手眼标定无效')
    return K, d, T, hashlib.sha256(raw).hexdigest()


def save_observation(q, calibration_sha256, *, path=OBSERVATION_PATH, captured_unix=None):
    q = joints(q)
    data = {'schema_version': 1, 'name': '纸袋观测位',
            'joints_rad': q.tolist(), 'base_to_tcp': forward_kinematics(q).tolist(),
            'joint_state_source': 'orange_pi_ros_joint_states',
            'captured_unix_sec': time.time() if captured_unix is None else float(captured_unix),
            'calibration_sha256': calibration_sha256}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    temp.replace(path)
    return data


def load_observation(calibration_sha256, *, path=OBSERVATION_PATH):
    data = json.loads(Path(path).read_text())
    if data.get('schema_version') != 1 or data.get('joint_state_source') != 'orange_pi_ros_joint_states':
        raise ValueError('观测位格式或编码器来源错误')
    if data.get('calibration_sha256') != calibration_sha256:
        raise ValueError('手眼标定已改变，请重新保存观测位')
    return joints(data['joints_rad'])


def fresh(captured_unix, *, now=None, max_age=.35):
    age = (time.time() if now is None else now) - float(captured_unix)
    if not np.isfinite(age) or not -.05 <= age <= max_age:
        raise ValueError('RGB-D 画面已过期或时间戳异常')


def localize(detections, depth_m, K, distortion, q, tcp_to_camera, *, require_full_box=True):
    """只使用本帧检测与本帧注册深度；不寻找附近替代像素。"""
    import cv2
    depth = np.asarray(depth_m, dtype=float)
    if depth.shape != (360,640):
        raise ValueError('RGB 与注册深度必须同为 640×360')
    eligible = [o for o in detections if o.get('grasp_ready_2d') and o.get('status') == 'valid']
    if len(eligible) != 1:
        raise ValueError('需要唯一、明确可用的纸袋提手点')
    o = eligible[0]
    if not np.isfinite(o['confidence']) or o['confidence'] < .65 or o.get('class_id') != 0:
        raise ValueError('纸袋检测置信度不足')
    p = np.asarray(o['handle_center_xy'], dtype=float)
    box = np.asarray(o['bbox_xyxy'], dtype=float)
    if p.shape != (2,) or box.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(box).all():
        raise ValueError('提手点或检测框包含非法坐标')
    if not (0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 360):
        raise ValueError('纸袋框坐标越界')
    if require_full_box and not (2 <= box[0] < box[2] <= 638 and 2 <= box[1] < box[3] <= 358):
        raise ValueError('纸袋框不完整或触及图像边缘')
    if not (box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]):
        raise ValueError('提手点在检测框之外')
    pc = float(o['keypoint_confidence'])
    if not np.isfinite(pc) or pc < .5:
        raise ValueError('提手点置信度不足')
    x,y = np.rint(p).astype(int)
    if not (3 <= x < 637 and 3 <= y < 357):
        raise ValueError('提手点过于靠近图像边缘')
    patch = depth[y-2:y+3,x-2:x+3]
    valid = np.isfinite(patch) & (patch >= .12) & (patch <= 1.)
    if np.mean(valid) < .8 or not np.isfinite(depth[y,x]) or not .12 <= depth[y,x] <= 1.:
        raise ValueError('提手点及邻域深度不足，禁止补点代替')
    values = patch[valid]
    z = float(np.median(values))
    if np.ptp(values) > .010 or abs(float(depth[y,x])-z) > .005:
        raise ValueError('提手邻域存在深度跳变或前后景混合')
    ray = cv2.undistortPoints(p.reshape(1,1,2), K, distortion)[0,0]
    camera = np.r_[ray*z,z]
    base_to_camera = forward_kinematics(joints(q)) @ tcp_to_camera
    base = (base_to_camera @ np.r_[camera,1])[:3]
    return {'pixel':p.tolist(), 'camera_xyz_m':camera.tolist(), 'base_xyz_m':base.tolist(),
            'depth_m':z, 'depth_valid_fraction':float(np.mean(valid)),
            'depth_range_m':float(np.ptp(values)), 'confidence':float(o['confidence'])}


def stable_target(samples, *, now=None):
    if len(samples) < 4:
        raise ValueError('需要四张独立的新鲜 RGB-D 帧')
    if len({s['sequence'] for s in samples}) != len(samples):
        raise ValueError('重复帧不能用于稳定定位')
    stamps = np.array([s['captured_unix'] for s in samples], dtype=float)
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0) or stamps[-1]-stamps[0]>1.5:
        raise ValueError('定位帧时间跨度或顺序异常')
    fresh(stamps[-1], now=now)
    xyz = np.array([s['target']['base_xyz_m'] for s in samples], dtype=float)
    pixels = np.array([s['target']['pixel'] for s in samples], dtype=float)
    if not np.isfinite(xyz).all() or not np.isfinite(pixels).all():
        raise ValueError('定位结果包含非有限数')
    center = np.median(xyz,axis=0)
    if np.max(np.linalg.norm(xyz-center,axis=1))>.003 or np.max(np.linalg.norm(pixels-np.median(pixels,axis=0),axis=1))>3:
        raise ValueError('连续帧提手点不稳定')
    return center


def trajectory(start, end, *, speed=.05):
    start,end = joints(start),joints(end)
    points,audit = generate_timed_joint_trajectory(start,end,cartesian_speed_m_s=speed,
                                                   joint_velocity_scale=.25,point_period_sec=.05)
    if audit.duration_sec > 14.5 or len(points)>500:
        raise ValueError('路径过长，超出单次有界运动预算')
    return points,audit


def vertical_lift_trajectory(start, *, height=GRASP_LIFT_M):
    """Solve Cartesian Z waypoints, then time a smooth continuous joint spline."""
    from scipy.interpolate import CubicSpline
    start=joints(start)
    if not np.isfinite(height) or not 0<height<=.05:
        raise ValueError('垂直抬升高度须在0到5厘米之间')
    pose=forward_kinematics(start)
    fractions=np.linspace(0,1,11);solutions=[start]
    for index,fraction in enumerate(fractions[1:],1):
        desired=pose.copy();desired[2,3]+=height*fraction
        q,pe,ae=inverse_kinematics(desired,solutions[-1],regularization=.001,max_nfev=180)
        if pe>.001 or ae>np.deg2rad(.5) or np.max(np.abs(q-solutions[-1]))>.15:
            raise ValueError(f'垂直抬升第{index}/10点IK不可达或关节不连续：位置误差{pe*1000:.3f}毫米')
        solutions.append(joints(q))
    spline=CubicSpline(fractions,np.array(solutions),axis=0,bc_type='natural')
    def evaluate(r):
        s=10*r**3-15*r**4+6*r**5
        ds=30*r**2-60*r**3+30*r**4
        dds=60*r-180*r**2+120*r**3
        q=spline(s)
        v=spline(s,1)*ds[:,None]
        a=spline(s,2)*ds[:,None]**2+spline(s,1)*dds[:,None]
        return q,v,a
    r=np.linspace(0,1,1001)
    qs,vs,_=evaluate(r)
    if (np.any(qs<PIPER_JOINT_LIMITS[:,0]) or np.any(qs>PIPER_JOINT_LIMITS[:,1])
            or np.max(np.abs(qs-start))>.6):
        raise ValueError('垂直抬升路径超出关节限位或变化预算')
    poses=forward_kinematics_batch(qs)
    delta=poses[:,:3,3]-pose[:3,3]
    drift=float(np.max(np.linalg.norm(delta[:,:2],axis=1)))
    if (drift>.003 or np.any(np.diff(delta[:,2])<-.0001)
            or np.max(delta[:,2])>height+.002 or abs(delta[-1,2]-height)>.001):
        raise ValueError(f'笛卡尔抬升路径不合格：横向偏移{drift*1000:.3f}毫米，升高{delta[-1,2]*1000:.3f}毫米')
    relative=pose[:3,:3].T@poses[:,:3,:3]
    angles=np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))
    if np.max(angles)>np.deg2rad(.5):raise ValueError('垂直抬升中途工具朝向偏差超限')
    tcp_derivative=np.gradient(poses[:,:3,3],r,axis=0)
    duration=1.05*max(1.,float(np.max(np.linalg.norm(tcp_derivative,axis=1)))/.05,
                      float(np.max(np.abs(vs)/(PIPER_MAX_VELOCITIES*.25))))
    count=int(np.ceil(duration/.05))+1
    if duration>14.5 or count>500:raise ValueError('垂直抬升路径超出单段时长预算')
    times=np.linspace(0,duration,count);qs,vs,acc=evaluate(times/duration)
    points=[{'time_from_start_sec':float(t),'positions':q.tolist(),
             'velocities':(v/duration).tolist(),'accelerations':(a/duration**2).tolist()}
            for t,q,v,a in zip(times,qs,vs,acc)]
    return points,qs[-1],{'kind':'cartesian_vertical_ik_spline','height_m':height,
        'max_xy_drift_m':drift,'duration_sec':duration,'waypoint_count':11}


def grasp_cycle_plan(q, target_base):
    """固定本次提手点；沿工具前向在基座XY平面的投影越过70mm。"""
    q=joints(q)
    target=np.asarray(target_base,dtype=float)
    if target.shape!=(3,) or not np.isfinite(target).all():
        raise ValueError('提手三维位置异常')
    tcp=forward_kinematics(q)
    direction=tcp[:3,2].copy();direction[2]=0
    length=np.linalg.norm(direction)
    if length<.5:raise ValueError('工具前向接近竖直，无法确定水平前伸方向')
    direction/=length
    if (target-tcp[:3,3])@direction<=0 or np.linalg.norm(target-tcp[:3,3])>.35:
        raise ValueError('提手方向或距离超出本次抓取范围')
    grasp=target+direction*GRASP_OVERTRAVEL_M
    desired=tcp.copy();desired[:3,3]=target
    qtarget,pe,ae=inverse_kinematics(desired,q,regularization=.001,max_nfev=180)
    if pe>.001 or ae>np.deg2rad(.5) or np.max(np.abs(qtarget-q))>1:
        raise ValueError('提手点IK精度或关节变化不合格')
    desired[:3,3]=grasp
    qgrasp,pe,ae=inverse_kinematics(desired,qtarget,regularization=.001,max_nfev=180)
    if pe>.001 or ae>np.deg2rad(.5) or np.max(np.abs(qgrasp-qtarget))>.6:
        raise ValueError('水平前伸7厘米IK精度或关节变化不合格')
    approach,_=trajectory(q,qtarget)
    insertion,_=trajectory(qtarget,qgrasp)
    poses=forward_kinematics_batch(np.array([p['positions'] for p in insertion]))
    delta=poses[:,:3,3]-poses[0,:3,3]
    along=delta@direction
    lateral=delta-along[:,None]*direction
    if (np.any(np.diff(along)<-.0001) or np.max(along)>.072
            or np.max(np.linalg.norm(lateral,axis=1))>.003):
        raise ValueError('7厘米前伸路径偏离水平方向或行程异常')
    lift,qraised,lift_audit=vertical_lift_trajectory(qgrasp)
    raised=forward_kinematics(qgrasp)[:3,3]+np.array([0,0,GRASP_LIFT_M])
    # Retain the existing per-segment duration budget on the longer carry path.
    # Both halves trace the same joint-space line, with a stationary midpoint.
    qmid=(qraised+q)/2
    carry_near,_=trajectory(qraised,qmid)
    carry_far,_=trajectory(qmid,q)
    return {'target_base_m':target.tolist(),'grasp_base_m':grasp.tolist(),
            'horizontal_direction_base':direction.tolist(),'overtravel_m':GRASP_OVERTRAVEL_M,
            'lift_m':GRASP_LIFT_M,'raised_base_m':raised.tolist(),'raised_joints':qraised.tolist(),
            'lift_audit':lift_audit,
            'carry_mid_joints':qmid.tolist(),
            'observation_joints':q.tolist(),'target_joints':qtarget.tolist(),
            'grasp_joints':qgrasp.tolist(),'approach':approach,'insertion':insertion,
            'lift':lift,'lower':reverse_timed_trajectory(lift),
            'carry_out_near':carry_near,'carry_out_far':carry_far,
            'carry_back_far':reverse_timed_trajectory(carry_far),
            'carry_back_near':reverse_timed_trajectory(carry_near),
            'withdraw_insertion':reverse_timed_trajectory(insertion),
            'withdraw_approach':reverse_timed_trajectory(approach)}


def contact_plan(q, target_base, tcp_to_camera):
    """保持当前工具姿态，先到前方20mm，再用独立受力桥触碰并回撤。"""
    q = joints(q)
    target = np.asarray(target_base,dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('提手三维位置异常')
    tcp = forward_kinematics(q)
    direction = tcp[:3,2]
    if float((target-tcp[:3,3])@direction) < PRECONTACT_M:
        raise ValueError('提手不在工具前方，禁止向后或越过提手运动')
    if np.linalg.norm(target-tcp[:3,3]) > .35:
        raise ValueError('提手距离超出本轮35厘米移动范围')
    pre = tcp.copy();pre[:3,3] = target-direction*PRECONTACT_M
    qpre,pe,ae = inverse_kinematics(pre,q,regularization=.001,max_nfev=180)
    if pe>.001 or ae>np.deg2rad(.5) or np.max(np.abs(qpre-q))>1.0:
        raise ValueError('预接触位 IK 精度或关节变化不合格')
    touch = pre.copy();touch[:3,3]=target
    qt,pe,ae = inverse_kinematics(touch,qpre,regularization=.001,max_nfev=180)
    if pe>.001 or ae>np.deg2rad(.5) or np.max(np.abs(qt-qpre))>.45:
        raise ValueError('接触段 IK 精度或关节变化不合格')
    approach,audit = trajectory(q,qpre)
    forward,fa = trajectory(qpre,qt,speed=.010)
    # Joint interpolation must follow the intended Cartesian contact direction.
    poses=forward_kinematics_batch(np.array([p['positions'] for p in forward]))
    displacement=poses[:,:3,3]-poses[0,:3,3]
    along=displacement@direction
    lateral=displacement-along[:,None]*direction
    if np.any(np.diff(along)<-.0001) or np.max(along)>.022 or np.max(np.linalg.norm(lateral,axis=1))>.0015:
        raise ValueError('受力接触段方向、横向偏移或位移上限异常')
    allposes=forward_kinematics_batch(np.array([p['positions'] for p in approach+forward]))
    camera=allposes@tcp_to_camera
    pcam=(np.linalg.inv(camera)@np.r_[target,1])[:,:3]
    if np.min(pcam[:,2])<.10 or np.any(pcam[:,2]>1.):
        raise ValueError('接近时提手超出相机有效观测深度')
    return {'approach':approach,'forward':forward,'precontact_joints':qpre.tolist(),
            'touch_joints':qt.tolist(),'target_base_m':target.tolist(),
            'normal_base':(-direction).tolist(),'approach_duration_sec':audit.duration_sec,
            'contact_duration_sec':fa.duration_sec,'maximum_contact_travel_m':.022}

