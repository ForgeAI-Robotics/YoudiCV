from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from copy import deepcopy
import threading
import time

from .button_pose import (
    ButtonCameraPose,
    ButtonDetection,
    estimate_button_camera_pose,
    select_button,
)
from .orbbec_v1_stream import LiveRgbdFrame, OrbbecV1Stream
from .detection_enhancement import detect_with_enhancement_fallback
from .pipeline_timing import mark, interval
from .numeric_yolo_consistency import admitted_detections, target_conflicted


@dataclass(frozen=True)
class LiveButtonSample:
    sequence: int
    target_label: str
    frame: LiveRgbdFrame
    pose: ButtonCameraPose | None
    error: str | None
    processing_sec: float
    detections: tuple[ButtonDetection, ...]
    detection_enhancement: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    numeric_layout_localization: dict = field(default_factory=dict)
    numeric_yolo_consistency: dict = field(default_factory=dict)
    publication_stage: str = 'completed_rgbd'


@dataclass(frozen=True)
class LiveButtonBatch:
    accepted: tuple[LiveButtonSample, ...]
    rejected: tuple[dict, ...]
    attempts: int
    first_sequence: int
    last_sequence: int
    collection_sec: float


class LiveButtonPerception:
    """Continuously run YOLO/depth on the newest frame from a live camera."""

    def __init__(
        self,
        camera: OrbbecV1Stream,
        detector,
        camera_matrix,
        *,
        target_label: str,
        target_confidence: float,
        geometry_options: dict,
        cache_size: int = 32,
        enhancement_fallback_enabled: bool = False,
        activity=None,
        numeric_layout_localizer=None,
        layout_before_local_yolo_enabled: bool = False,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.camera_matrix = camera_matrix
        self.target_label = str(target_label)
        self.target_confidence = float(target_confidence)
        self.geometry_options = dict(geometry_options)
        self.enhancement_fallback_enabled = bool(enhancement_fallback_enabled)
        self.activity = activity
        self.numeric_layout_localizer = numeric_layout_localizer
        self.layout_before_local_yolo_enabled = bool(
            layout_before_local_yolo_enabled and self.enhancement_fallback_enabled
            and callable(getattr(numeric_layout_localizer, 'localize_2d', None)))
        self._samples: deque[LiveButtonSample] = deque(maxlen=int(cache_size))
        self._light_samples: deque[LiveButtonSample] = deque(maxlen=int(cache_size))
        self._target_epoch = 0
        self._light_barrier_sequence = -1
        self._conflict_barriers = {}
        self._condition = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self._fatal_error: BaseException | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        sequence = 0
        try:
            while self._running:
                cycle_started = mark()
                if self.activity is not None and not self.activity.wait_for_turn():
                    break
                activity_timing = interval('vision.activity_wait', 'idle', cycle_started)
                frame_wait_started = mark()
                frame = self.camera.wait_for_sequence(sequence, timeout=3.0)
                frame_wait_timing = interval('vision.frame_wait', 'capture', frame_wait_started)
                sequence = frame.sequence
                with self._condition:
                    target_label = self.target_label
                    target_epoch = self._target_epoch
                started = time.perf_counter()
                pose = None
                error = None
                detections = ()
                enhancement_audit = {}
                layout_audit = {}
                consistency_audit = {}
                early_layout = {}
                task_detections = ()
                spans = [activity_timing, frame_wait_timing]
                try:
                    detection_started = mark()
                    detection_status = 'ok'
                    try:
                        layout_options = {}
                        localize_2d = getattr(self.numeric_layout_localizer, 'localize_2d', None)
                        if self.layout_before_local_yolo_enabled and callable(localize_2d):
                            layout_options['before_numeric_retry'] = (
                                lambda raw: localize_2d(frame, target_label, raw))
                        detections, enhancement_audit = detect_with_enhancement_fallback(
                            self.detector, frame.color, target_label, self.target_confidence,
                            enabled=self.enhancement_fallback_enabled,
                            **layout_options,
                        )
                        if enhancement_audit.get('numeric_retry_skipped_by_layout') is True:
                            early_layout = enhancement_audit['pre_retry_layout_2d']
                    except BaseException:
                        detection_status = 'error'
                        raise
                    finally:
                        spans.append(interval('vision.detection', 'detection', detection_started,
                            status=detection_status,
                            raw_inference_sec=enhancement_audit.get('raw_inference_sec'),
                            retry_inference_sec=enhancement_audit.get('retry_inference_sec'),
                            enhancement_sec=enhancement_audit.get('enhancement_sec'),
                            layout_2d_sec=enhancement_audit.get('layout_2d_sec', 0.),
                            numeric_retry_skipped_by_layout=enhancement_audit.get(
                                'numeric_retry_skipped_by_layout', False),
                            layout_2d_reason=(enhancement_audit.get('pre_retry_layout_2d') or {}).get('reason'),
                            raw_detections=enhancement_audit.get('raw_detections')))
                    selection_started = mark()
                    selection_status = 'ok'
                    try:
                        task_detections=detections
                        if getattr(self.numeric_layout_localizer,'yolo_consistency_enabled',False) is True:
                            check_started=mark()
                            consistency_audit=self.numeric_layout_localizer.inspect_yolo(frame,target_label,detections)
                            spans.append(interval('vision.numeric_yolo_consistency','geometry',check_started,
                                                  decision=consistency_audit.get('decision')))
                            task_detections=admitted_detections(detections,consistency_audit,
                                target=target_label,sequence=sequence)
                        if target_conflicted(consistency_audit,target=target_label,sequence=sequence):
                            raise RuntimeError('numeric_yolo_layout_conflict: '+target_label+
                                               ' at cell '+str(consistency_audit.get('alternative_label')))
                        try:
                            detection = select_button(
                                list(task_detections), target_label, self.target_confidence)
                        except RuntimeError as exc:
                            if not early_layout:
                                raise
                            # An inferred light ROI is not a YOLO detection or
                            # an ordinary motion pose. Full depth stays below.
                            detection = None
                            error = str(exc)
                        panel_supports = tuple(
                            item
                            for item in task_detections
                            if item.confidence >= self.target_confidence
                        )
                    except BaseException:
                        selection_status = 'error'
                        raise
                    finally:
                        spans.append(interval('vision.selection', 'selection', selection_started,
                                              status=selection_status))
                    # This immutable 2-D bundle is the exact YOLO input and
                    # admitted detections. Never expose it in the 3-D cache.
                    light_stage=('layout_2d_before_depth' if detection is None
                                 else 'yolo_2d_before_depth')
                    light_sample=LiveButtonSample(
                        sequence=sequence,target_label=target_label,frame=frame,
                        pose=None,error='depth_pending',
                        processing_sec=time.perf_counter()-started,
                        detections=tuple(detections),
                        detection_enhancement=deepcopy(enhancement_audit),
                        numeric_yolo_consistency=deepcopy(consistency_audit),
                        numeric_layout_localization=(deepcopy(early_layout) if detection is None else {}),
                        publication_stage=light_stage,
                        timing=dict(version=1,spans=list(spans),frame_sequence=sequence,
                            frame_received_unix_sec=frame.received_unix,
                            frame_received_monotonic_sec=frame.received_monotonic,
                            publication_stage=light_stage,
                            published_monotonic_sec=time.monotonic(),
                            processing_start_delay_sec=max(0.,detection_started[0]-frame.received_monotonic),
                            status='2d_ready_depth_pending'))
                    self._publish_light_sample(light_sample,target_epoch)
                    if detection is None:
                        # Use the original complete same-frame layout/depth
                        # path below; never promote this 2-D result to pose.
                        raise RuntimeError(error)
                    depth_started = mark()
                    depth_status = 'ok'
                    depth_failure = {}
                    try:
                        pose = estimate_button_camera_pose(
                            detection,
                            frame.depth_m,
                            self.camera_matrix,
                            panel_support_detections=panel_supports,
                            **self.geometry_options,
                        )
                        plane_audit = getattr(pose, "panel_plane_fit", None)
                        if plane_audit is not None:
                            depth_failure["panel_plane_fit"] = plane_audit
                    except BaseException as exc:
                        depth_status = 'error'
                        # Keep the original exception and rejection semantics.
                        # Existing RPC timing export carries this exact reason
                        # into task traces instead of losing it with the cache.
                        depth_failure = dict(error_type=type(exc).__name__,
                                             error_message=str(exc),
                                             target_label=target_label)
                        raise
                    finally:
                        spans.append(interval('vision.depth_pose', 'depth', depth_started,
                                              status=depth_status, **depth_failure))
                except RuntimeError as exc:
                    error = str(exc)
                    if target_conflicted(consistency_audit,target=target_label,sequence=sequence):
                        # Invalidate old evidence immediately, while the same-
                        # frame geometric fallback can still finish normally.
                        with self._condition:
                            self._conflict_barriers[target_label]=sequence
                            self._condition.notify_all()
                # Shadow evidence is independent of normal target selection.
                # Never insert its inferred ROI into detections or sample.pose.
                if self.numeric_layout_localizer is not None:
                    layout_started = mark()
                    try:
                        layout_audit = self.numeric_layout_localizer.localize(
                            frame, target_label, task_detections)
                    except Exception as exc:
                        layout_audit = dict(mode='shadow',motion_eligible=False,
                            valid_3d=False,reason='layout_shadow_failed',error=str(exc))
                    spans.append(interval('vision.numeric_layout_shadow','geometry',layout_started,
                        status='ok' if layout_audit.get('valid_3d') else 'skipped_or_rejected',
                        reason=layout_audit.get('reason')))
                sample = LiveButtonSample(
                    sequence=sequence,
                    target_label=target_label,
                    frame=frame,
                    pose=pose,
                    error=error,
                    processing_sec=time.perf_counter() - started,
                    detections=detections,
                    detection_enhancement=enhancement_audit,
                    numeric_layout_localization=layout_audit,
                    numeric_yolo_consistency=consistency_audit,
                    timing=dict(version=1, spans=spans, frame_sequence=sequence,
                        frame_received_unix_sec=frame.received_unix,
                        frame_received_monotonic_sec=frame.received_monotonic,
                        processing_start_delay_sec=max(0., detection_started[0]-frame.received_monotonic),
                        status='error' if error else 'ok',
                        scope='host wall time including inference pre/postprocessing; no added CUDA synchronization'),
                )
                with self._condition:
                    self._samples.append(sample)
                    self._condition.notify_all()
                # Targets without an early bundle keep the completed same-frame
                # path. A full depth result never overwrites an early 2-D one.
                self._publish_light_sample(sample,target_epoch)
        except BaseException as exc:
            if self._running:
                with self._condition:
                    self._fatal_error = exc
                    self._condition.notify_all()

    def mark(self) -> int:
        """Return a camera sequence barrier; later batches exclude older frames."""
        return self.camera.latest_sequence

    def set_target(self, target_label: str) -> int:
        """Change the live label and return a barrier excluding old-label work."""
        barrier=self.camera.latest_sequence
        with self._condition:
            self.target_label = str(target_label)
            self._target_epoch += 1
            self._light_barrier_sequence = barrier
            self._light_samples.clear()
            self._condition.notify_all()
        return barrier

    def _publish_light_sample(self, sample, epoch):
        with self._condition:
            if (epoch!=self._target_epoch or sample.target_label!=self.target_label
                    or sample.sequence<=self._light_barrier_sequence):
                return
            if any(s.sequence==sample.sequence for s in self._light_samples):
                return
            self._light_samples.append(sample)
            self._condition.notify_all()

    def _conflict_barrier(self,label):
        return max(getattr(self,'_conflict_barriers',{}).get(label,-1),
            max((s.sequence for s in self._samples if s.target_label==label
                 and target_conflicted(getattr(s,'numeric_yolo_consistency',{}),
                                      target=label,sequence=s.sequence)),default=-1))

    def latest_light_sample(self,target_label=None):
        """Latest same-frame lamp bundle; it carries no motion qualification."""
        label=str(target_label) if target_label is not None else self.target_label
        with self._condition:
            if label!=self.target_label:return None
            barrier=self._conflict_barrier(label)
            return next((s for s in reversed(self._light_samples)
                         if s.target_label==label and s.sequence>=barrier),None)

    def latest_valid(self, target_label: str | None = None) -> LiveButtonSample | None:
        label = str(target_label) if target_label is not None else self.target_label
        with self._condition:
            barrier=self._conflict_barrier(label)
            for sample in reversed(self._samples):
                if sample.sequence<=barrier:continue
                if sample.target_label==label and target_conflicted(getattr(sample,'numeric_yolo_consistency',{}),
                        target=label,sequence=sample.sequence):
                    return None
                if sample.target_label == label and sample.pose is not None:
                    return sample
        return None

    def latest_sample(self, target_label: str | None = None) -> LiveButtonSample | None:
        """Return the newest processed sample, including a failed detection."""
        label = str(target_label) if target_label is not None else self.target_label
        with self._condition:
            for sample in reversed(self._samples):
                if sample.target_label == label:
                    return sample
        return None

    def valid_window(
        self, target_label: str | None = None, size: int = 6
    ) -> tuple[LiveButtonSample, ...]:
        """Return the newest valid, label-consistent samples in time order."""
        label = str(target_label) if target_label is not None else self.target_label
        if size < 1:
            raise ValueError("window size must be positive")
        with self._condition:
            barrier=self._conflict_barrier(label)
            selected = [
                sample
                for sample in self._samples
                if sample.target_label == label and sample.pose is not None and sample.sequence>barrier
            ]
            return tuple(selected[-int(size):])

    def collect_after(
        self,
        sequence_barrier: int,
        *,
        valid_results: int = 6,
        max_attempts: int = 12,
        timeout: float = 8.0,
    ) -> LiveButtonBatch:
        if not 1 <= valid_results <= max_attempts:
            raise ValueError("require 1 <= valid_results <= max_attempts")
        started = time.perf_counter()
        deadline = time.monotonic() + float(timeout)
        seen: set[int] = set()
        accepted: list[LiveButtonSample] = []
        rejected: list[dict] = []
        first_sequence = 0
        last_sequence = 0
        target_label = self.target_label
        with self._condition:
            while self._running:
                if self._fatal_error is not None:
                    raise RuntimeError(
                        f"live button perception failed: {self._fatal_error}"
                    ) from self._fatal_error
                conflict_barrier=self._conflict_barrier(target_label)
                accepted=[s for s in accepted if s.sequence>conflict_barrier]
                fresh = [
                    sample
                    for sample in self._samples
                    if sample.sequence > sequence_barrier
                    and sample.sequence not in seen
                    and sample.target_label == target_label
                ]
                for sample in fresh:
                    seen.add(sample.sequence)
                    if first_sequence == 0:
                        first_sequence = sample.sequence
                    last_sequence = sample.sequence
                    if sample.pose is None:
                        if target_conflicted(getattr(sample,'numeric_yolo_consistency',{}),target=target_label,sequence=sample.sequence):
                            accepted.clear()
                        rejected.append(
                            {"sequence": sample.sequence, "reason": sample.error}
                        )
                    else:
                        if sample.sequence>conflict_barrier:accepted.append(sample)
                    if len(accepted) >= valid_results:
                        return LiveButtonBatch(
                            accepted=tuple(accepted),
                            rejected=tuple(rejected),
                            attempts=len(seen),
                            first_sequence=first_sequence,
                            last_sequence=last_sequence,
                            collection_sec=time.perf_counter() - started,
                        )
                    if len(seen) >= max_attempts:
                        raise RuntimeError(
                            f"only {len(accepted)}/{len(seen)} live frames valid; "
                            f"require {valid_results}; rejected={rejected}"
                        )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"only {len(accepted)} live results after sequence "
                        f"{sequence_barrier} within {timeout:.1f} s"
                    )
                self._condition.wait(min(remaining, 0.05))
        raise RuntimeError("live button perception is closed")

    def close(self) -> None:
        self._running = False
        if self.activity is not None:
            self.activity.close()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
