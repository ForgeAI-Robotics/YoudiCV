"""Task-local retry localization from fresh RGB rays and a verified panel.

No acquisition, model, depth fitting, safety-mode change or actuator command is
implemented here. The collector reads the resident service's existing 2D boxes
and interpolates encoder poses at the image capture time. The caller continues
to own chassis immobilization, stop supervision and full motion validation.
"""
from collections import deque
from dataclasses import dataclass
import threading
import time

import numpy as np

from .numeric_layout_handoff import (
    SourcedWindowEstimate, selected_button_region,
)
from .numeric_layout_sampling import LayoutTransform, _LayoutWindow, rigid_transform
from .numeric_yolo_consistency import target_conflicted
from .observation_visibility import complete_button_box
from .realtime_planner import RobotTransformHistory


POLICY = 'task_panel_rgb_ray_retry_v1'
MAX_REFERENCE_CORRECTION_M = .010


def capture_camera_matrix(full_matrix, config):
    old = np.asarray(config['calibration'].get('rgb_calibration_size', [1920, 1080]), float)
    new = np.asarray(config['depth']['capture_size'], float)
    if (old.shape != (2,) or new.shape != (2,) or not np.isfinite([old, new]).all()
            or min(old) <= 0 or min(new) <= 0 or abs(new[0]/old[0]-new[1]/old[1]) > 1e-9):
        raise ValueError('retry capture aspect ratio differs from calibration')
    matrix = np.asarray(full_matrix, float).copy()
    if matrix.shape != (3, 3):
        raise ValueError('invalid calibrated camera matrix')
    matrix[0] *= new[0]/old[0]
    matrix[1] *= new[1]/old[1]
    return matrix


@dataclass(frozen=True)
class RetryPanelReference:
    target_label: str
    task_id: str
    matrix_values: tuple
    plane_rms_m: float
    sequence_range: tuple

    @property
    def matrix(self):
        return np.asarray(self.matrix_values, dtype=float).copy()

    @classmethod
    def freeze(cls, estimate, audit, *, target_label, task_id):
        if (not estimate.stable or estimate.sample_count < 2 or not task_id
                or estimate.sequence_range is None):
            raise ValueError('retry reference requires a stable current-task target')
        matrix = rigid_transform(estimate.base_to_button, 'reference panel')
        rows = [r for r in audit if r.get('accepted')]
        values = [float(r['plane_rms_m']) for r in rows[-estimate.sample_count:]]
        if (len(values) < estimate.sample_count or not np.isfinite(values).all()
                or min(values) < 0 or max(values) > .006):
            raise ValueError('retry reference has no qualified panel fit evidence')
        return cls(str(target_label), str(task_id), tuple(map(tuple, matrix)),
                   max(values), tuple(estimate.sequence_range))

    def audit(self):
        return dict(policy=POLICY, target_label=self.target_label, task_id=self.task_id,
                    base_to_button=self.matrix.tolist(), sequence_range=list(self.sequence_range),
                    reference_plane_rms_m=self.plane_rms_m,
                    max_reference_correction_m=MAX_REFERENCE_CORRECTION_M,
                    depth_source='frozen_verified_task_panel_not_current_near_depth')


def intersect_rgb_ray(camera, camera_matrix, bbox, reference):
    """Intersect a new pixel ray with the task plane; keep its verified axes."""
    camera = rigid_transform(camera, 'capture-time camera')
    matrix = np.asarray(camera_matrix, dtype=float)
    box = np.asarray(bbox, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-9, rtol=0)):
        raise ValueError('invalid capture-size camera matrix')
    if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]-box[:2]) <= 0:
        raise ValueError('invalid fresh button bbox')
    plane = reference.matrix
    ray = camera[:3, :3] @ np.linalg.solve(matrix, np.r_[(box[:2]+box[2:])/2, 1.])
    ray /= np.linalg.norm(ray)
    normal = plane[:3, 2]
    denominator = float(normal @ ray)
    if abs(denominator) < .10:
        raise ValueError('RGB ray is too oblique to the reference panel')
    distance = float(normal @ (plane[:3, 3]-camera[:3, 3]) / denominator)
    if distance <= 0 or not np.isfinite(distance):
        raise ValueError('reference panel intersection is behind the camera')
    result = plane.copy()
    result[:3, 3] = camera[:3, 3] + distance*ray
    correction = float(np.linalg.norm(result[:3, 3]-plane[:3, 3]))
    if correction > MAX_REFERENCE_CORRECTION_M:
        raise ValueError(f'fresh button moved {correction*1000:.2f}mm from task reference')
    return result, correction


class RetryPanelSampler:
    def __init__(self, reference, handeye, camera_matrix, image_size, expected_joints,
                 *, target_label, task_id, min_confidence, now_unix, now_monotonic):
        if reference.task_id != str(task_id) or reference.target_label != str(target_label):
            raise ValueError('retry reference belongs to a different task or target')
        self.reference = reference
        self.handeye = rigid_transform(handeye, 'handeye')
        self.camera_matrix = np.asarray(camera_matrix, float).copy()
        self.image_size = list(image_size)
        self.expected_joints = np.asarray(expected_joints, float).copy()
        if (self.expected_joints.shape != (6,) or not np.isfinite(self.expected_joints).all()
                or len(self.image_size) != 2 or min(self.image_size) <= 0):
            raise ValueError('retry requires a verified near observation hold')
        self.min_confidence = float(min_confidence)
        self.offset = float(now_unix)-float(now_monotonic)
        self.barrier = float(now_unix)
        self.history = RobotTransformHistory(max_gap_sec=.12)
        self.lock = threading.RLock()
        self.joints = None
        self.last_state_stamp = None
        self.last_sequence = 0
        self.last_capture = None
        self.encoder_resets = 0
        self.rows = deque(maxlen=128)
        self.sources = deque(maxlen=2)
        self.window = self._window()

    @staticmethod
    def _window():
        return _LayoutWindow(size=2, max_input_age_sec=.35, max_latest_age_sec=.35,
            max_span_sec=2., max_plane_rms_m=.006, max_position_rms_m=.0015,
            max_position_jump_m=.003, max_normal_rms_deg=1.5)

    def observe_robot(self, state, *, now_unix, now_monotonic):
        q = np.asarray(state.joints, float)
        age = float(now_monotonic)-float(state.timestamp)
        if (q.shape != (6,) or not np.isfinite(q).all() or not -.05 <= age <= .35
                or state.raw.get('hardware_ok') != 1):
            raise RuntimeError('retry encoder feedback invalid, stale or unhealthy')
        if np.max(np.abs(q-self.expected_joints)) > .03:
            raise RuntimeError('retry arm left its verified observation hold')
        pose = rigid_transform(state.base_to_gripper, 'encoder pose')
        with self.lock:
            if self.last_state_stamp is not None and state.timestamp < self.last_state_stamp:
                raise RuntimeError('retry encoder timestamp moved backwards')
            self.history.add(state.timestamp, pose)
            self.last_state_stamp = state.timestamp
            if self.joints is None:
                self.joints = q.copy()
            elif np.max(np.abs(q-self.joints)) > .003:
                self.joints = q.copy()
                self.barrier = float(now_unix)
                self.window = self._window()
                self.sources.clear()
                self.encoder_resets += 1
                self.rows.append(dict(accepted=False, reason='encoder_motion_during_stationary_capture'))

    def consume(self, response, *, now_unix, now_monotonic):
        with self.lock:
            row = dict(accepted=False, localization_method=POLICY)
            try:
                seq = response.get('frame_sequence')
                row['sequence'] = seq
                if response.get('ok') is not True or type(seq) is not int or seq <= self.last_sequence:
                    raise ValueError('not_a_new_inference_frame')
                self.last_sequence = seq
                stamp = float(response['capture_time_unix_sec'])
                row['capture_time_unix_sec'] = stamp
                if (not np.isfinite(stamp) or not -.05 <= now_unix-stamp <= .35):
                    raise ValueError('sample_age_out_of_range')
                if stamp < self.barrier:
                    raise ValueError('captured_before_stationary_pose')
                if self.last_capture is not None and stamp <= self.last_capture:
                    raise ValueError('capture_time_not_increasing')
                if abs(now_unix-now_monotonic-self.offset) > .05:
                    raise ValueError('capture_clock_mismatch')
                if self.last_state_stamp is None or not -.05 <= now_monotonic-self.last_state_stamp <= .35:
                    raise ValueError('no_fresh_encoder_observation')
                if response.get('image_size') != self.image_size:
                    raise ValueError('capture_resolution_changed')
                if target_conflicted(response.get('numeric_yolo_consistency'),
                                     target=self.reference.target_label, sequence=seq):
                    raise ValueError('numeric_yolo_layout_conflict')
                detections = [d for d in response.get('detections', [])
                              if str(d.get('label')) == self.reference.target_label
                              and np.isfinite(float(d.get('confidence', 0)))
                              and self.min_confidence <= float(d.get('confidence', 0)) <= 1]
                if len(detections) > 1:
                    raise ValueError('ambiguous_target_detections')
                region = detections[0] if detections else selected_button_region(
                    {**response, 'target_detection': None}, now=now_unix,
                    target_label=self.reference.target_label)
                if region is None:
                    raise ValueError('no_qualified_fresh_button_region')
                box = region['bbox_xyxy']
                if not complete_button_box(box, self.image_size):
                    raise ValueError('button_not_fully_visible')
                source = 'yolo' if detections else 'panel_layout_inferred'
                row.update(source=source, target_detected=bool(detections), bbox_xyxy=list(box),
                           confidence=float(region['confidence']) if detections else None)
                base_camera = self.history.transform_at(stamp-self.offset, wait_sec=0) @ self.handeye
                target, correction = intersect_rgb_ray(base_camera, self.camera_matrix, box, self.reference)
                # The RMS belongs to the frozen verified plane, never a claim
                # that the current frame has usable depth. The temporal window
                # still checks the new intersections' positions and timestamps.
                sample = LayoutTransform(seq, stamp, target, self.reference.plane_rms_m)
                if not self.window.add(sample, now=now_unix):
                    raise ValueError('window_rejected_sample')
                self.sources.append(source)
                self.last_capture = stamp
                row.update(accepted=True, base_to_button=target.tolist(),
                           base_to_camera=base_camera.tolist(), reference_correction_m=correction,
                           reference_plane_rms_m=self.reference.plane_rms_m,
                           transform_source='capture_time_encoder_interpolation',
                           depth_source='frozen_verified_task_panel_not_current_near_depth')
            except (ValueError, RuntimeError, KeyError, TypeError, OverflowError) as exc:
                row['reason'] = str(exc)
            self.rows.append(row)
            return dict(row)

    def snapshot(self, *, now_unix):
        with self.lock:
            e = self.window.estimate(now=now_unix)
            sources = tuple(sorted(set(self.sources)))
            estimate = SourcedWindowEstimate(**vars(e), position_sources=sources)
            audit = list(self.rows)
            diagnostics = dict(localization_method=POLICY, target_label=self.reference.target_label,
                position_sources=list(sources), stable=e.stable, reasons=list(e.reasons),
                sequence_range=e.sequence_range, sample_count=e.sample_count,
                latest_age_sec=e.latest_age_sec, span_sec=e.span_sec,
                position_rms_m=e.position_rms_m, normal_rms_deg=e.normal_rms_deg,
                max_position_jump_m=e.max_position_jump_m, frame_count=2,
                encoder_resets=self.encoder_resets, reference=self.reference.audit(), audit=audit)
            return estimate, audit, diagnostics


def collect_retry_panel_target(vision, robot, handeye, reference, config, camera_matrix,
                               expected_joints, *, target_label, task_id, check_hold):
    """Read-only 3s collector using existing completed same-frame 2D results."""
    started = time.monotonic()
    sampler = RetryPanelSampler(reference, handeye, camera_matrix, config['depth']['capture_size'],
        expected_joints, target_label=target_label, task_id=task_id,
        min_confidence=config['detector'].get('target_confidence', .70),
        now_unix=time.time(), now_monotonic=started)
    stop = threading.Event()
    failures = []

    def encoders():
        try:
            while not stop.is_set():
                sampler.observe_robot(robot.state(max_age=.35), now_unix=time.time(),
                                      now_monotonic=time.monotonic())
                stop.wait(.01)
        except Exception as exc:
            failures.append(exc)
            stop.set()

    reader = threading.Thread(target=encoders, daemon=True, name='retry-panel-encoders')
    reader.start()
    try:
        while time.monotonic()-started < 3.:
            check_hold()
            if failures:
                raise RuntimeError(f'retry panel encoder failure: {failures[0]}')
            response = vision.request('visible_buttons', target_label=str(target_label),
                                      min_confidence=sampler.min_confidence)
            check_hold()
            if time.monotonic()-started >= 3.:
                break
            sampler.consume(response, now_unix=time.time(), now_monotonic=time.monotonic())
            estimate, audit, diagnostics = sampler.snapshot(now_unix=time.time())
            if failures:
                raise RuntimeError(f'retry panel encoder failure: {failures[0]}')
            if estimate.stable:
                audit[-1]['acquisition_summary'] = dict(localization_method=POLICY, status='ready',
                    frame_count=2, elapsed_sec=time.monotonic()-started,
                    reference=reference.audit(), position_sources=list(estimate.position_sources))
                return estimate, audit
            stop.wait(.01)
        _, _, diagnostics = sampler.snapshot(now_unix=time.time())
        error = RuntimeError('no stable fresh button target before motion')
        error.stationary_target_failure = diagnostics
        raise error
    finally:
        stop.set()
        reader.join(timeout=1.)
        if reader.is_alive():
            raise RuntimeError('retry panel encoder reader did not stop')
