"""Cycle-only odometry guard; startup and feedback-loss gates remain strict."""
from __future__ import annotations

from collections import deque
import math


class ChassisNotStationary(RuntimeError):
    def __init__(self, reason, audit):
        self.audit = dict(audit, decision='stop', reason=reason)
        super().__init__(reason)


def velocity_norms(sample):
    age, velocity = sample.get('age_sec'), sample.get('velocity')
    if (sample.get('available') is not True or sample.get('error')
            or type(age) not in (int, float) or not math.isfinite(age) or not 0 <= age <= .5
            or not isinstance(velocity, (list, tuple)) or len(velocity) != 6
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in velocity)):
        raise ValueError('底盘速度反馈不可用或已过期，测试保持停止')
    return math.sqrt(sum(v*v for v in velocity[:3])), math.sqrt(sum(v*v for v in velocity[3:]))


class StationarityGuard:
    """Only small isolated speed spikes get <=120 ms to clear.

    All six velocity components are checked. Obvious motion, displacement,
    feedback loss and source resets stop immediately. Duplicate cache reads
    never count as new evidence. This class never clears a stop or restarts.
    """
    def __init__(self):
        self.key = None
        self.stamp = None
        self.anchor = None
        self.suspect_since = None
        self.suspect_stamp = None
        self.suspect_count = 0
        self.history = deque(maxlen=12)
        self.last_fresh_at = None

    def observe(self, sample, now):
        audit = dict(sample=sample, recent=list(self.history))
        def stop(reason):
            raise ChassisNotStationary(reason, audit)
        try:
            linear, angular = velocity_norms(sample)
        except ValueError as exc:
            stop(str(exc))
        seq, stamp, pose = sample.get('sequence'), sample.get('source_stamp_sec'), sample.get('pose_xy_yaw')
        if (type(seq) is not int or type(stamp) not in (int, float) or not math.isfinite(stamp)
                or not isinstance(pose, (tuple, list)) or len(pose) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pose)):
            stop('底盘里程计编号或位置无效，测试保持停止')
        if self.key is not None and (seq < self.key or stamp < self.stamp):
            stop('底盘里程计发生重启或时间倒退，测试保持停止')
        if self.anchor is None:
            self.anchor = list(pose)
        displacement = math.hypot(pose[0]-self.anchor[0], pose[1]-self.anchor[1])
        yaw = abs(math.atan2(math.sin(pose[2]-self.anchor[2]), math.cos(pose[2]-self.anchor[2])))
        audit.update(linear_norm_m_s=linear, angular_norm_rad_s=angular,
                     displacement_m=displacement, yaw_change_rad=yaw)
        if linear >= .010 or angular >= .040 or displacement > .002 or yaw > .010:
            stop('底盘运动或位置变化超限，测试保持停止')
        fresh = seq != self.key and (self.stamp is None or stamp > self.stamp)
        if fresh:
            self.key, self.stamp = seq, stamp
            self.last_fresh_at = now
            self.history.append(dict(sequence=seq, source_stamp_sec=stamp, velocity=list(sample['velocity']),
                                     age_sec=sample['age_sec'], pose_xy_yaw=list(pose)))
        elif self.last_fresh_at is not None and now-self.last_fresh_at > .5:
            stop('底盘里程计没有新的独立数据，测试保持停止')
        if self.suspect_since is not None and now-self.suspect_since >= .12:
            stop('底盘轻微速度持续超限或未及时恢复，测试保持停止')
        suspect = linear > .005 or angular > .02
        if suspect:
            if self.suspect_since is None:
                self.suspect_since, self.suspect_stamp, self.suspect_count = now, stamp, 0
            if fresh:
                self.suspect_count += 1
            audit.update(suspect_sec=now-self.suspect_since, distinct_suspect_samples=self.suspect_count)
            if (now-self.suspect_since >= .12 or
                    (self.suspect_count >= 3 and stamp-self.suspect_stamp >= .10)):
                stop('底盘轻微速度持续超限，测试保持停止')
            return dict(audit, decision='transient_pending')
        if self.suspect_since is not None:
            # Only a genuinely new good frame clears a suspect cached frame.
            if not fresh:
                if now-self.suspect_since >= .12:
                    stop('底盘异常后没有新的停稳反馈，测试保持停止')
                return dict(audit, decision='transient_pending')
            audit.update(suspect_sec=now-self.suspect_since)
            self.suspect_since, self.suspect_stamp, self.suspect_count = None, None, 0
            return dict(audit, decision='transient_cleared')
        return dict(audit, decision='stationary')
