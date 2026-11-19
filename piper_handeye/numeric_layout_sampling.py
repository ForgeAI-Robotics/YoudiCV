"""Read-only stationary sampling of layout-inferred RGB-D positions.

This consumes the live vision RPC, but never selects a target, moves a robot,
clears a stop, supplies a lamp ROI, or returns an ordinary motion sample. Target
confidence is absent by design. Spatial stability uses the existing window's
estimate implementation, after independent layout/depth/provenance checks.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from .realtime_planner import RobotTransformHistory, SlidingButtonWindow


def rigid_transform(value, name):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(matrix[:3, :3]) - 1) > 1e-5):
        raise ValueError(f'{name} must be a finite rigid transform')
    return matrix.copy()


@dataclass(frozen=True)
class LayoutTransform:
    sequence: int
    capture_time_unix_sec: float
    base_to_button: np.ndarray
    plane_rms_m: float
    # There is deliberately no target YOLO confidence on inferred positions.


class _LayoutWindow(SlidingButtonWindow):
    """Reuse spatial/normal averaging, not the YOLO-confidence admission path."""

    def add(self, sample, *, now=None):
        current = time.time() if now is None else float(now)
        age = current - sample.capture_time_unix_sec
        if (sample.sequence <= self._last_sequence or not math.isfinite(age)
                or not -.05 <= age <= self.max_input_age_sec
                or not math.isfinite(sample.plane_rms_m)
                or not 0 <= sample.plane_rms_m <= self.max_plane_rms_m):
            return False
        matrix = rigid_transform(sample.base_to_button, 'base_to_button')
        self._samples.append(LayoutTransform(sample.sequence, sample.capture_time_unix_sec,
                                             matrix, sample.plane_rms_m))
        self._last_sequence = sample.sequence
        return True


class LayoutStationarySampler:
    """Feed encoder observations and same-frame shadow RPC envelopes; no I/O."""

    def __init__(self, config, handeye, target_label, *, now_unix, now_monotonic,
                 frame_count=4, min_capture_unix_sec=None):
        if type(frame_count) is not int or frame_count not in (2, 4):
            raise ValueError('stationary frame_count must be 2 or 4')
        settings = config.get('numeric_layout_localization', {})
        self.target = str(target_label)
        if (settings.get('mode') != 'shadow' or self.target not in ('1','2','3','4','5')
                or self.target not in settings.get('target_labels', [])):
            raise ValueError('layout sampling requires configured numeric shadow target 1..5')
        self.profile_sha256 = settings.get('profile_sha256', '')
        if len(self.profile_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.profile_sha256):
            raise ValueError('invalid configured profile SHA256')
        self.handeye = rigid_transform(handeye, 'handeye')
        self.offset = float(now_unix) - float(now_monotonic)
        self.barrier = max(float(now_unix), float(min_capture_unix_sec or now_unix))
        if not math.isfinite(self.offset) or not math.isfinite(self.barrier):
            raise ValueError('invalid acquisition clocks')
        self.frame_count = frame_count
        self.max_plane_rms_m = float(config.get('button_geometry', {}).get('max_plane_rms_m', .006))
        if not 0 < self.max_plane_rms_m <= .006:
            raise ValueError('invalid layout plane RMS limit')
        self.history = RobotTransformHistory(max_gap_sec=.12)
        self._lock = threading.RLock()
        self._stationary_joints = None
        self._last_state_stamp = None
        self._last_sequence = 0
        self._last_capture = None
        self._profile_id = None
        self._audit = deque(maxlen=128)
        self.encoder_resets = 0
        self.window = self._new_window()

    def _new_window(self):
        return _LayoutWindow(size=self.frame_count, max_input_age_sec=.35,
                             max_latest_age_sec=.35, max_span_sec=2.,
                             max_plane_rms_m=self.max_plane_rms_m,
                             max_position_rms_m=.0015, max_position_jump_m=.003,
                             max_normal_rms_deg=1.5)

    def observe_robot(self, state, *, now_unix, now_monotonic):
        joints = np.asarray(state.joints, dtype=float)
        stamp = float(state.timestamp)
        age = float(now_monotonic) - stamp
        if joints.shape != (6,) or not np.isfinite(joints).all() or not -.05 <= age <= .35:
            raise ValueError('invalid or stale encoder observation')
        matrix = rigid_transform(state.base_to_gripper, 'base_to_gripper')
        with self._lock:
            if self._last_state_stamp is not None and stamp < self._last_state_stamp:
                raise ValueError('encoder timestamp moved backwards')
            self.history.add(stamp, matrix)
            self._last_state_stamp = stamp
            if self._stationary_joints is None:
                self._stationary_joints = joints.copy()
            elif float(np.max(np.abs(joints-self._stationary_joints))) > .003:
                self._stationary_joints = joints.copy()
                self.barrier = float(now_unix)
                self.window = self._new_window()
                self.encoder_resets += 1
                self._audit.append(dict(accepted=False, reason='encoder_motion_during_stationary_capture',
                                        barrier_unix_sec=self.barrier))

    def consume(self, envelope, *, now_unix, now_monotonic):
        with self._lock:
            row = dict(accepted=False, source='panel_layout_inferred', motion_eligible=False)
            try:
                record = envelope.get('localization')
                if not envelope.get('ok') or not isinstance(record, dict):
                    raise ValueError('no_layout_evidence')
                sequence = record.get('frame_sequence')
                row['sequence'] = sequence
                if type(sequence) is not int or sequence <= 0:
                    raise ValueError('invalid_frame_sequence')
                if sequence <= self._last_sequence:
                    raise ValueError('not_a_new_frame')
                self._last_sequence = sequence
                if (sequence != record.get('rgbd_bundle_sequence')
                        or sequence != envelope.get('frame_sequence')
                        or record.get('pairing') != 'same_rgbd_bundle'
                        or record.get('target_label') != self.target):
                    raise ValueError('frame_or_target_mismatch')
                if (record.get('source') != 'panel_layout_inferred' or record.get('mode') != 'shadow'
                        or record.get('motion_eligible') is not False
                        or record.get('target_detected') is not False
                        or record.get('yolo_confidence', 'missing') is not None
                        or record.get('light_state') != 'not_evaluated'):
                    raise ValueError('invalid_layout_provenance')
                if (record.get('profile_sha256') != self.profile_sha256 or not record.get('profile_id')
                        or self._profile_id not in (None, record['profile_id'])):
                    raise ValueError('profile_changed_or_mismatched')
                capture = float(record['capture_time_unix_sec'])
                row['capture_time_unix_sec'] = capture
                if envelope.get('capture_time_unix_sec') != capture:
                    raise ValueError('capture_time_mismatch')
                age = float(now_unix) - capture
                if not math.isfinite(age) or not -.05 <= age <= .35:
                    raise ValueError('sample_age_out_of_range')
                if capture < self.barrier:
                    raise ValueError('captured_before_stationary_pose')
                if self._last_capture is not None and capture <= self._last_capture:
                    raise ValueError('capture_time_not_increasing')
                if (envelope.get('valid_3d') is not True or record.get('valid_3d') is not True
                        or record.get('accepted_2d') is not True or envelope.get('fresh') is not True):
                    raise ValueError('layout_depth_not_valid')
                geometry = record.get('geometry', {})
                anchors = geometry.get('anchors', [])
                if (geometry.get('accepted') is not True or len(anchors) < 3
                        or len(set(anchors)) != len(anchors) or self.target in anchors
                        or any(str(a) not in tuple(map(str, range(1,17))) for a in anchors)):
                    raise ValueError('invalid_independent_layout_references')
                pose = record['pose']
                bbox = np.asarray(record['bbox_xyxy'], dtype=float)
                if (bbox.shape != (4,) or not np.isfinite(bbox).all()
                        or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                    raise ValueError('invalid_layout_bbox')
                if int(pose['panel_support_detections']) < 3:
                    raise ValueError('insufficient_depth_supports')
                camera_to_button = rigid_transform(pose['camera_to_button'], 'camera_to_button')
                plane_rms = float(pose['plane_rms_m'])
                if not math.isfinite(plane_rms) or not 0 <= plane_rms <= self.max_plane_rms_m:
                    raise ValueError('plane_rms_above_threshold')
                # Same-host monotonic timestamp comes from the RGB-D bundle.
                # Also cross-check the frozen UNIX/monotonic conversion used by
                # the existing moving collector; a clock step cannot silently
                # pair the picture with a different encoder time.
                capture_mono = float(record['received_monotonic'])
                if (not math.isfinite(capture_mono) or abs(capture-self.offset-capture_mono) > .05
                        or abs(float(now_unix)-float(now_monotonic)-self.offset) > .05):
                    raise ValueError('capture_clock_mismatch')
                if self._last_state_stamp is None or not -.05 <= now_monotonic-self._last_state_stamp <= .35:
                    raise ValueError('no_fresh_encoder_observation')
                base_to_gripper = self.history.transform_at(capture_mono, wait_sec=0)
                base_to_button = base_to_gripper @ self.handeye @ camera_to_button
                sample = LayoutTransform(sequence, capture, base_to_button, plane_rms)
                if not self.window.add(sample, now=now_unix):
                    raise ValueError('window_rejected_sample')
                self._last_capture = capture
                self._profile_id = record['profile_id']
                estimate = self.window.estimate(now=now_unix)
                row.update(accepted=True, reason='stable' if estimate.stable else 'collecting',
                           bbox_xyxy=deepcopy(record['bbox_xyxy']), anchors=list(anchors),
                           profile_sha256=self.profile_sha256, plane_rms_m=plane_rms,
                           capture_monotonic=capture_mono, sample_age_sec=age,
                           transform_source='capture_time_encoder_interpolation',
                           base_to_button=base_to_button.tolist(),
                           window_reasons=list(estimate.reasons))
            except (ValueError, RuntimeError, KeyError, TypeError, OverflowError) as exc:
                row['reason'] = str(exc)
            self._audit.append(row)
            return deepcopy(row)

    def snapshot(self, *, now_unix):
        with self._lock:
            estimate = self.window.estimate(now=now_unix)
            return dict(schema=1, source='panel_layout_inferred', mode='shadow',
                        target_label=self.target, target_detected=False, yolo_confidence=None,
                        motion_eligible=False, light_state='not_evaluated',
                        stable=estimate.stable, reasons=list(estimate.reasons),
                        base_to_button=estimate.base_to_button.tolist() if estimate.stable else None,
                        profile_sha256=self.profile_sha256, profile_id=self._profile_id,
                        sequence_range=estimate.sequence_range, sample_count=estimate.sample_count,
                        latest_age_sec=estimate.latest_age_sec, span_sec=estimate.span_sec,
                        position_rms_m=estimate.position_rms_m, normal_rms_deg=estimate.normal_rms_deg,
                        max_position_jump_m=estimate.max_position_jump_m,
                        frame_count=self.frame_count, encoder_resets=self.encoder_resets,
                        audit=deepcopy(list(self._audit)))

    def consume_detected(self, sample, *, min_confidence, now_unix, now_monotonic):
        """Task adapter only: genuine YOLO confidence, same capture-time history."""
        with self._lock:
            row=dict(accepted=False,source='yolo',target_detected=True)
            try:
                seq=int(sample['sequence']);capture=float(sample['capture_time_unix_sec'])
                row.update(sequence=seq,capture_time_unix_sec=capture)
                if seq<=self._last_sequence:raise ValueError('not_a_new_frame')
                self._last_sequence=seq
                if str(sample['target_label'])!=self.target:raise ValueError('wrong_target_label')
                confidence=float(sample['confidence']);rms=float(sample['plane_rms_m'])
                if not math.isfinite(confidence) or not min_confidence<=confidence<=1:
                    raise ValueError('confidence_below_threshold')
                if not math.isfinite(rms) or not 0<=rms<=self.max_plane_rms_m:
                    raise ValueError('plane_rms_above_threshold')
                if not -.05<=now_unix-capture<=.35:raise ValueError('sample_age_out_of_range')
                if capture<self.barrier:raise ValueError('captured_before_stationary_pose')
                if self._last_capture is not None and capture<=self._last_capture:
                    raise ValueError('capture_time_not_increasing')
                if abs(now_unix-now_monotonic-self.offset)>.05:raise ValueError('capture_clock_mismatch')
                if self._last_state_stamp is None or not -.05<=now_monotonic-self._last_state_stamp<=.35:
                    raise ValueError('no_fresh_encoder_observation')
                box=np.asarray(sample['bbox_xyxy'],float)
                if box.shape!=(4,) or not np.isfinite(box).all() or min(box[2:]-box[:2])<=0:
                    raise ValueError('invalid_detected_bbox')
                camera=rigid_transform(sample['camera_to_button'],'camera_to_button')
                base=self.history.transform_at(capture-self.offset,wait_sec=0)@self.handeye@camera
                if not self.window.add(LayoutTransform(seq,capture,base,rms),now=now_unix):
                    raise ValueError('window_rejected_sample')
                self._last_capture=capture
                row.update(accepted=True,bbox_xyxy=box.tolist(),confidence=confidence,plane_rms_m=rms,
                           base_to_button=base.tolist(),transform_source='capture_time_encoder_interpolation')
            except (ValueError,RuntimeError,KeyError,TypeError,OverflowError) as exc:
                row['reason']=str(exc)
            self._audit.append(row)
            return deepcopy(row)


def collect_stationary_layout_shadow(vision, robot, handeye, target_label, config, *,
                                     frame_count=4, min_capture_unix_sec=None, cancelled=None,
                                     _task_handoff=False):
    """Three-second read-only acquisition against the existing live interfaces.

Caller owns an already-open encoder adapter and vision client. Only robot.state
and layout_localization_snapshot are called. The current perception target must
already match; this function never changes it or opens a robot command socket.
"""
    started = time.monotonic()
    sampler = LayoutStationarySampler(config, handeye, target_label,
                                     now_unix=time.time(), now_monotonic=started,
                                     frame_count=frame_count,
                                     min_capture_unix_sec=min_capture_unix_sec)
    stop = threading.Event()
    failures = []

    def update_encoders():
        try:
            while not stop.is_set():
                state = robot.state(max_age=.35)
                sampler.observe_robot(state, now_unix=time.time(), now_monotonic=time.monotonic())
                stop.wait(.01)
        except Exception as exc:
            failures.append(exc)
            stop.set()

    thread = threading.Thread(target=update_encoders, daemon=True, name='layout-shadow-encoders')
    thread.start()
    sequence = -1
    try:
        while time.monotonic()-started < 3.:
            if cancelled is not None and cancelled():
                raise RuntimeError('layout shadow acquisition cancelled')
            if failures:
                raise RuntimeError(f'layout shadow encoder failure: {failures[0]}')
            envelope = (vision.request('latest',target_label=str(target_label),window_size=6)
                        if _task_handoff else
                        vision.request('layout_localization_snapshot', target_label=str(target_label),
                                       after_sequence=sequence, include_rgb=False))
            if cancelled is not None and cancelled():
                raise RuntimeError('layout shadow acquisition cancelled')
            if time.monotonic()-started >= 3.:
                break
            if _task_handoff:
                from .numeric_layout_handoff import target_batch
                rows=[]
                for candidate in target_batch(envelope,after_sequence=sequence):
                    if candidate.get('position_source')=='panel_layout_inferred':
                        e=candidate['layout_evidence']
                        rows.append(sampler.consume(dict(ok=True,fresh=True,valid_3d=e['valid_3d'],
                            frame_sequence=e['frame_sequence'],capture_time_unix_sec=e['capture_time_unix_sec'],
                            localization=e),now_unix=time.time(),now_monotonic=time.monotonic()))
                    else:
                        rows.append(sampler.consume_detected(candidate,
                            min_confidence=float(config.get('detector',{}).get('target_confidence',.70)),
                            now_unix=time.time(),now_monotonic=time.monotonic()))
                row=max(rows,key=lambda r:r.get('sequence',-1)) if rows else {}
            else:
                row = sampler.consume(envelope, now_unix=time.time(), now_monotonic=time.monotonic())
            if type(row.get('sequence')) is int:
                sequence = max(sequence, row['sequence'])
            result = sampler.snapshot(now_unix=time.time())
            if failures:
                raise RuntimeError(f'layout shadow encoder failure: {failures[0]}')
            if result['stable']:
                result['elapsed_sec'] = time.monotonic()-started
                return result
            stop.wait(.01)
        result = sampler.snapshot(now_unix=time.time())
        error = RuntimeError('no stable fresh layout shadow target')
        error.layout_sampling = result
        raise error
    finally:
        stop.set()
        thread.join(timeout=1.)
        if thread.is_alive():
            raise RuntimeError('layout shadow encoder reader did not stop')
