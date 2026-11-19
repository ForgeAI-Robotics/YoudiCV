"""Bindings for fresh RGB observations and dry-validated segmented motion."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
from pathlib import Path
import time
import threading
import os
from typing import Any

import numpy as np

from .button_light_flow import LightState
from .button_light_runtime import ButtonLightRuntime
from .missing_button_capture import save_missing_button_view
from .segmented_button_motion import SegmentedPressPlan
from .retry_observation import plan_retry_observation_transition, RETRY_PREPARATION_DISTANCE_M
from .near_button_depth_pause import NearButtonDepthPause, NEAR_BUTTON_DEPTH_LEASE_SEC
from .observation_visibility import OBSERVATION_EDGE_MARGIN_PX
from .timed_trajectory import generate_timed_joint_trajectory
from .pipeline_timing import traced, stage
from .pose_timing import PoseTimingTrace
from .retreat_off_monitor import RetreatOffMonitor
from .numeric_layout_handoff import selected_button_region, region_is_inferred
from .observation_pose import (
    OBSERVATION_JOINT_TOLERANCE_RAD, OBSERVATION_CAPTURE_DRIFT_RAD,
    OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD, capture_drift_exceeds_limit,
    OBSERVATION_FEEDBACK_MAX_AGE_SEC, wait_for_stable_observation_pose,
    stationary_reference_failure_audit,
)


POST_PRESS_VISIBILITY_EDGE_MARGIN_PX = 2.0
POST_PRESS_ON_CONFIRMATION_FRAMES = 2
POST_PRESS_MONITOR_FRAME_TIMEOUT_SEC = 0.20
POST_PRESS_RECHECK_BUDGET_SEC = 2.5
POST_PRESS_RECHECK_FRAME_WAIT_SEC = .7
STATIONARY_REFERENCE_TOTAL_SEC = 10.0
STATIONARY_REFERENCE_FRAME_WAIT_SEC = 1.5
POST_PRESS_RECHECK_MAX_FRAMES = 6
POST_PRESS_RECHECK_OFF_FRAMES = 3
POST_PRESS_RECHECK_ON_FRAMES = 1


def target_is_safely_visible(
    snapshot: dict,
    *,
    min_confidence: float = 0.70,
    max_frame_age_sec: float = .35,
    min_box_size_px: float = 14.0,
    edge_margin_px: float = POST_PRESS_VISIBILITY_EDGE_MARGIN_PX,
    max_frame_fraction: float = 0.85,
    target_label: str | None = None,
) -> tuple[bool, dict]:
    """Check whether the target crop is usable at the current robot pose."""
    detection = selected_button_region(snapshot,target_label=target_label,
                                       max_frame_age_sec=max_frame_age_sec) or {}
    image_size = snapshot.get("image_size") or []
    audit = {
        "fresh": bool(snapshot.get("fresh")),
        "frame_sequence": snapshot.get("frame_sequence"),
        "detection_frame_sequence": detection.get("frame_sequence"),
        "confidence": detection.get("confidence"),
        "bbox_xyxy": detection.get("bbox_xyxy"),
        "image_size": image_size,
    }
    try:
        width, height = map(float, image_size)
        x1, y1, x2, y2 = map(float, detection["bbox_xyxy"])
        inferred = region_is_inferred(detection)
        confidence = None if inferred else float(detection["confidence"])
        frame_sequence = int(snapshot["frame_sequence"])
        detection_sequence = int(detection["frame_sequence"])
        box_width, box_height = x2 - x1, y2 - y1
        checks = {
            "fresh_rgb": bool(snapshot.get("fresh")),
            "recent_detection": frame_sequence == detection_sequence,
            **({'layout_region_qualified': True} if inferred else
               {'confidence_ok': confidence >= float(min_confidence)}),
            "finite_geometry": bool(np.all(np.isfinite(
                [width, height, x1, y1, x2, y2]
            ))),
            "box_large_enough": (
                box_width >= float(min_box_size_px)
                and box_height >= float(min_box_size_px)
            ),
            "inside_frame": (
                x1 >= float(edge_margin_px)
                and y1 >= float(edge_margin_px)
                and x2 <= width - float(edge_margin_px)
                and y2 <= height - float(edge_margin_px)
            ),
            "not_camera_filling": (
                box_width <= width * float(max_frame_fraction)
                and box_height <= height * float(max_frame_fraction)
            ),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        checks = {"valid_payload": False}
    audit["checks"] = checks
    audit['region_source'] = detection.get('source','yolo')
    visible = bool(checks) and all(bool(value) for value in checks.values())
    audit["visible"] = visible
    return visible, audit


@dataclass
class FreshLightObserver:
    vision: Any
    recognizer: Any
    target_label: str
    fresh_frame_timeout_sec: float = 1.0
    poll_sec: float = 0.01
    last_sequence: int = -1
    audit: list[dict] = field(default_factory=list)
    event_sink: Any | None = None
    off_brightness_baseline: float | None = None
    near_brightness_rise_streak: int = 0
    brightness_on_delta: float = 40.0
    missing_button_evidence_saved: bool = False
    unsafe_waiting_evidence_saved: bool = False
    paired_light: Any | None = None
    failure_image_archive: Any | None = None
    initial_light_guard: Any | None = None
    initial_view_recheck: Any | None = None
    initial_recovery_active: bool = False
    initial_not_before_unix: float | None = None
    stationary_retry_light_guard: Any | None = None
    stationary_reference_capture_active: bool = field(default=False, init=False)
    stationary_budget_remaining: Any | None = None
    post_press_health_check: Any | None = None

    def _stationary_budget(self, limit):
        remaining = self.stationary_budget_remaining
        if remaining is None:return float(limit)
        value = float(remaining())
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError('stationary near-panel observation budget exhausted')
        return min(float(limit), value)

    def _frame_age_limit(self, snapshot, position):
        guard = self.stationary_retry_light_guard
        admission = guard.admission(snapshot, position) if guard is not None else None
        return (float(admission['max_rgb_age_sec']) if admission else .35), admission

    def finish_initial_observation(self) -> None:
        self.initial_recovery_active = False
        if self.initial_light_guard is not None:
            self.initial_light_guard.finish()

    def initial_observation_reason(self) -> str:
        if not self.audit:
            return "observation_pending"
        last = self.audit[-1]
        result = last.get("recognition") or {}
        if result.get("recognizer") == "skipped_unsafe_target_crop":
            return ("missing_target_box" if not (last.get("target_detection") or last.get('target_region'))
                    else "unqualified_target_view")
        if result.get("initial_light_guard"):
            return "local_highlight_conflict"
        return "light_state_unconfirmed"

    def _append_observation_audit(self, entry):
        self.audit.append(entry)
        if self.failure_image_archive is not None:
            self.failure_image_archive.capture(entry)
        if self.initial_recovery_active and len(self.audit) > 192:
            # Retain the first observation evidence and a rolling recent tail.
            del self.audit[36:-156]

    def recheck_initial_observation(self) -> bool:
        guard = self.initial_light_guard
        if (guard is None or not guard.active or not guard.conflict_seen
                or guard.rechecks >= 2 or self.initial_view_recheck is None):
            return False
        # Exceptions must propagate: communication/collision/stop failures are
        # never treated as another observation opportunity.
        if not self.initial_view_recheck(guard.rechecks + 1):
            return False
        guard.rechecks += 1
        guard.conflict_seen = False
        guard.not_before_unix = time.time()
        return True

    def observe(
        self,
        position: str,
        *,
        timeout_sec: float | None = None,
        emit_event: bool = True,
        visibility_edge_margin_px: float | None = None,
        not_before_unix: float | None = None,
    ) -> str:
        wait_limit = (
            self.fresh_frame_timeout_sec
            if timeout_sec is None else max(0.01, float(timeout_sec))
        )
        guard = self.stationary_retry_light_guard
        if guard is not None:
            guard.poll(position)
        deadline = time.monotonic() + wait_limit
        if (position == "waiting" and self.initial_recovery_active
                and self.initial_not_before_unix is not None):
            not_before_unix = max(not_before_unix or 0., self.initial_not_before_unix)
        if (position == "waiting" and self.initial_light_guard is not None
                and self.initial_light_guard.active):
            barrier = self.initial_light_guard.not_before_unix
            if barrier is not None:
                not_before_unix = max(not_before_unix or 0.0, barrier)
        snapshot = None
        stationary_admission = None
        stale_snapshot = None
        search_sequence = self.last_sequence
        missing_box_deadline = None
        acquisition_started=time.monotonic()
        skipped_stale=[]
        # Never hide an invalid frame between two decisive observations. Once
        # continuity is already broken (or no evidence exists), wait within
        # this call's existing budget instead of rapidly counting old RGBs.
        may_wait_past_stale=not self.audit or self.audit[-1].get('state')=='UNCERTAIN'
        while time.monotonic() < deadline:
            if guard is not None:
                guard.poll(position)
            candidate = self.vision.request(
                "light_snapshot", after_sequence=search_sequence,
                target_label=str(self.target_label),
            )
            if candidate.get("fresh"):
                search_sequence = int(candidate["frame_sequence"])
                if (not_before_unix is not None and
                        float(candidate.get("capture_time_unix_sec", 0)) <= not_before_unix):
                    time.sleep(self.poll_sec)
                    continue
                age=time.time()-float(candidate.get('capture_time_unix_sec',0))
                age_limit, _ = self._frame_age_limit(candidate, position)
                if not -.05<=age<=age_limit:
                    stale_snapshot=candidate
                    if may_wait_past_stale:
                        if not skipped_stale or skipped_stale[-1]['sequence']!=search_sequence:
                            skipped_stale.append(dict(sequence=search_sequence,age_sec=age))
                            skipped_stale=skipped_stale[-8:]
                        if missing_box_deadline is not None and time.monotonic()>=missing_box_deadline:
                            break
                        time.sleep(self.poll_sec)
                        continue
                snapshot = candidate
                if (selected_button_region(candidate,target_label=self.target_label,
                        max_frame_age_sec=min(age_limit,.7)) or {}).get("bbox_xyxy") is not None:
                    break
                if missing_box_deadline is None:
                    missing_box_deadline = min(deadline, time.monotonic() + .20)
                if time.monotonic() >= missing_box_deadline:
                    break
            if missing_box_deadline is not None and time.monotonic() >= missing_box_deadline:
                break
            time.sleep(self.poll_sec)
        if snapshot is None:
            snapshot=stale_snapshot
        if snapshot is None:
            raise TimeoutError("no fresh RGB frame for button-light observation")
        acquisition=dict(elapsed_sec=time.monotonic()-acquisition_started,
                         skipped_stale_frames=skipped_stale,
                         waited_after_broken_continuity=may_wait_past_stale,
                         publication_stage=snapshot.get('light_publication_stage','completed_rgbd'))
        sequence = int(snapshot["frame_sequence"])
        if sequence <= self.last_sequence:
            raise RuntimeError("vision service returned a stale light snapshot")
        self.last_sequence = sequence
        age_limit, stationary_admission = self._frame_age_limit(snapshot, position)
        region = selected_button_region(snapshot,target_label=self.target_label,
                                        max_frame_age_sec=min(age_limit,.7))
        if visibility_edge_margin_px is None:
            visibility_edge_margin_px=(OBSERVATION_EDGE_MARGIN_PX if position.startswith('waiting')
                                       else POST_PRESS_VISIBILITY_EDGE_MARGIN_PX)
        if (region or {}).get("bbox_xyxy") is None:
            if not self.missing_button_evidence_saved:
                evidence = save_missing_button_view(
                    self.vision, self.target_label,
                    f"missing_target_box_at_{position}", snapshot=snapshot,
                )
                self.missing_button_evidence_saved = bool(evidence.get("image"))
                if emit_event and self.event_sink is not None:
                    self.event_sink("missing_button_capture", evidence)
        elif not self.initial_recovery_active:
            self.missing_button_evidence_saved = False
        if visibility_edge_margin_px is not None:
            visible, visibility = target_is_safely_visible(
                snapshot, edge_margin_px=float(visibility_edge_margin_px),target_label=self.target_label,
                max_frame_age_sec=min(age_limit,.7)
            )
            age=time.time()-float(snapshot.get('capture_time_unix_sec',0))
            detection_sequence=(region or {}).get('frame_sequence',-100)
            age_limit, stationary_admission = self._frame_age_limit(snapshot, position)
            visibility['checks']['recent_rgb_time']=-.05<=age<=age_limit
            visibility['rgb_age_limit_sec']=age_limit
            visibility['stationary_retry_rgb_admission']=stationary_admission
            visibility['checks']['recent_detection_for_lamp']=sequence==int(detection_sequence)
            visible=visible and all(visibility['checks'].values())
            visibility['visible']=visible
            if not visible:
                self.near_brightness_rise_streak=0
                if self.paired_light is not None:
                    self.paired_light.reset_pending()
                if position.startswith('waiting') and not self.unsafe_waiting_evidence_saved:
                    evidence=save_missing_button_view(self.vision,self.target_label,'unsafe_waiting_button_crop',snapshot=snapshot)
                    self.unsafe_waiting_evidence_saved=True
                    if self.event_sink is not None:self.event_sink('unsafe_waiting_view',evidence)
                entry = {
                    "position": str(position),
                    "frame_sequence": sequence,
                    "capture_time_unix_sec": snapshot.get("capture_time_unix_sec"),
                    "state": LightState.UNCERTAIN.value,
                    "image": snapshot.get("path"),
                    "target_detection": snapshot.get("target_detection"),
                    "target_region": region,
                    "recognition": {
                        "state": LightState.UNCERTAIN.value,
                        "recognizer": "skipped_unsafe_target_crop",
                        "visibility": visibility,
                    },
                    "frame_acquisition": acquisition,
                }
                self._append_observation_audit(entry)
                if emit_event and self.event_sink is not None:
                    self.event_sink("light_observation", entry)
                return LightState.UNCERTAIN.value
        result = self.recognizer.observe(
            Path(snapshot["path"]), str(self.target_label),
            bbox_xyxy=(region or {}).get("bbox_xyxy"),
        )
        if region_is_inferred(region):
            result={**result,'region_source':'panel_layout_inferred',
                    'region_profile_sha256':region['profile_sha256'],'target_detected':False}
        if position == "waiting" and self.initial_light_guard is not None:
            result = self.initial_light_guard.filter(result)
        state = LightState(result["state"])
        brightness = float(result.get("brightness_feature", float("nan")))
        # A post-press OFF result is never sufficient to authorize another
        # press when two fresh frames show the strong illumination rise from a
        # known OFF waiting reference.  This is an independent, recorded
        # safeguard against a one-frame false-OFF classification.
        rise = (
            self.off_brightness_baseline is not None
            and np.isfinite(brightness)
            and brightness >= self.off_brightness_baseline + self.brightness_on_delta
        )
        post_press_positions = {
            "near", "near_confirmation", "prepress", "prepress_confirmation",
            "press_monitor", "near_recheck",
        }
        if str(position) in post_press_positions:
            # Preserve the actual single-image result BEFORE comparison can
            # replace its state or raise. Never invent ON from an absent result.
            absolute_age = time.time()-float(snapshot.get('capture_time_unix_sec', 0))
            result = {**result, 'absolute_state': state.value,
                'absolute_input_valid_before_comparison': bool(
                    state is LightState.ON and result.get('recognizer') == 'opencv_self_contrast'
                    and visibility.get('visible') and snapshot.get('fresh') is True
                    and -.05 <= absolute_age <= age_limit),
                'absolute_input_age_sec': absolute_age}
        if self.paired_light is None and str(position) in post_press_positions and state is LightState.OFF and rise:
            self.near_brightness_rise_streak += 1
            if self.near_brightness_rise_streak >= 2:
                state = LightState.ON
                result = {**result, "state": state.value,
                          "brightness_confirmation": "two_fresh_frames_above_off_baseline"}
            else:
                state = LightState.UNCERTAIN
                result = {**result, "state": state.value,
                          "disagreement_guard": "post_press_brightness_rise"}
        elif str(position) in post_press_positions:
            self.near_brightness_rise_streak = 0
        # Do not start a second physical press from a one-frame OFF result when
        # the lightweight state model sees a possible illumination transition.
        if (
            self.paired_light is None
            and str(position) in post_press_positions
            and state is LightState.OFF
            and float(result.get("fast_on_probability", 0.0)) >= 0.25
        ):
            state = LightState.UNCERTAIN
            result = {**result, "state": state.value,
                      "disagreement_guard": "fast_classifier_possible_on"}
        if str(position) == "waiting" and state is LightState.OFF and np.isfinite(brightness):
            self.off_brightness_baseline = brightness
        if self.paired_light is not None:
            try:
                if str(position) == "before_press":
                    reference_options = ({'current_freshness_check':
                        lambda: self._stationary_retry_frame_current(snapshot, position)}
                        if stationary_admission is not None else {})
                    self.paired_light.collect(snapshot, state.value,
                                              stationary=self.stationary_reference_capture_active,
                                              **reference_options)
                    if self.paired_light.last_reference_comparison is not None:
                        result = {**result, "off_reference_comparison":
                                  dict(self.paired_light.last_reference_comparison)}
                elif str(position) in post_press_positions:
                    # Opt in only for numeric buttons whose whole-center and
                    # local-stroke classifiers agree on a substantial ON.
                    # Initial/absolute observations and arrows are unchanged.
                    strong_numeric_on = (
                        str(self.target_label).isdigit()
                        and 1 <= int(self.target_label) <= 16
                        and result.get('legacy_state') == 'ON'
                        and result.get('state') == 'ON'
                        and any(float(region.get('contrast', 0)) >=
                                float(result.get('local_on_contrast_threshold', 85.))
                                and float(region.get('fraction', 0)) >= .08
                                and int(region.get('area_px', 0)) >= 12
                                for region in result.get('local_regions', [])))
                    strong_on_options = {'strong_on': True} if strong_numeric_on else {}
                    if position in {'near', 'near_confirmation', 'near_recheck'}:
                        strong_on_options['allow_post_press_off_alignment'] = True
                    if stationary_admission is not None:
                        strong_on_options['current_freshness_check'] = (
                            lambda: self._stationary_retry_frame_current(snapshot, position))
                    if position == "near_recheck":
                        # The bounded recheck needs durable evidence even when
                        # the motion monitor already used the ordinary quota.
                        paired = self.paired_light.classify(
                            snapshot, state.value, preserve_evidence=True, stationary=True,
                            **strong_on_options,
                        )
                    else:
                        paired = self.paired_light.classify(snapshot, state.value,
                            stationary=str(position) in {
                                "near", "near_confirmation", "prepress", "prepress_confirmation"},
                            **strong_on_options)
                    state = LightState(paired["state"])
                    result = {**result, "absolute_state": result["state"],
                              "state": state.value, "paired_comparison": paired}
            except Exception as exc:
                # The new exception fallback belongs only to post-press image
                # comparison. Keep the pre-press collection's original errors.
                if (str(position) not in post_press_positions
                        and not isinstance(exc, (ValueError, OSError, RuntimeError))):
                    raise
                self.paired_light.reset_pending()
                state = LightState.UNCERTAIN
                result = {**result, "state": state.value,
                          "paired_comparison": {"state": state.value, "reason": str(exc),
                                                "exception_type": type(exc).__name__}}
        if stationary_admission is not None and not self._stationary_retry_frame_current(snapshot, position):
            state = LightState.UNCERTAIN
            if self.paired_light is not None:
                self.paired_light.reset_pending()
                # A delayed comparison must not leave its newly added OFF
                # reference behind after its encoder/image admission expires.
                if position == 'before_press' and not self.paired_light.frozen:
                    while (self.paired_light.references
                           and self.paired_light.references[-1].sequence >= sequence):
                        self.paired_light.references.pop()
                    self.paired_light.stationary_reference_ready = False
            visibility['checks']['recent_rgb_time'] = False
            visibility['visible'] = False
            result = {**result, 'state': state.value,
                      'freshness_rejection': 'stationary_retry_rgb_no_longer_current'}
        saved_waiting_image = None
        if position == "waiting" and self.initial_light_guard is not None:
            saved_waiting_image = self.initial_light_guard.save(snapshot, result)
        entry = {
                "position": str(position),
                "frame_sequence": sequence,
                "capture_time_unix_sec": snapshot.get("capture_time_unix_sec"),
                "state": state.value,
                "image": saved_waiting_image or (result.get("paired_comparison") or {}).get("current_image", snapshot["path"]),
                "recognition": result,
                "target_detection": snapshot.get("target_detection"),
                "target_region": region,
                "visibility": visibility,
                "frame_acquisition": acquisition,
                "stationary_retry_rgb_snapshot": ({
                    k: snapshot[k] for k in ('fresh', 'frame_sequence', 'capture_time_unix_sec',
                        'target_detection', 'perception_timing','layout_target_region',
                        'stationary_light_layout_candidate','layout_requested_target_label') if k in snapshot
                } if stationary_admission is not None else None),
            }
        priority = self._opencv_on_priority(entry, position)
        if priority is not None:
            state = LightState.ON
            result = {**result, 'state': state.value, 'opencv_on_priority': priority}
            entry.update(state=state.value, recognition=result)
        self._append_observation_audit(entry)
        if emit_event and self.event_sink is not None:
            self.event_sink("light_observation", entry)
        return state.value

    def _stationary_retry_frame_current(self, snapshot, position):
        guard = self.stationary_retry_light_guard
        admission = guard.admission(snapshot, position) if guard is not None else None
        return bool(admission and -.05 <= time.time()-float(snapshot['capture_time_unix_sec'])
                    <= admission['max_rgb_age_sec'])

    def _entry_frame_current(self, entry, position):
        stamp = entry.get('capture_time_unix_sec')
        if type(stamp) not in (int, float):
            return False
        if -.05 <= time.time()-stamp <= .35:
            return True
        snapshot = entry.get('stationary_retry_rgb_snapshot')
        return bool(snapshot and self._stationary_retry_frame_current(snapshot, position))

    def _opencv_on_priority(self, entry, position):
        """Adopt an actual post-press OpenCV ON unless comparison says OFF.

        Only the single-image input must have passed its own original checks.
        Pairing/registration/geometry/saturation/photometric comparison failures
        cannot veto that ON. Independent robot feedback and stop checks still
        run outside the image-comparison exception handler and may raise.
        """
        if (self.paired_light is None or position not in {
                'near', 'near_confirmation', 'near_recheck',
                'prepress', 'prepress_confirmation', 'press_monitor'}):
            return None
        recognition = entry.get('recognition') or {}
        comparison = recognition.get('paired_comparison') or {}
        if (entry.get('position') != position
                or recognition.get('recognizer') != 'opencv_self_contrast'
                or recognition.get('absolute_state') != 'ON'
                or recognition.get('absolute_input_valid_before_comparison') is not True
                or comparison.get('state') == 'OFF'
                or (comparison.get('state') == 'ON' and entry.get('state') == 'ON')):
            return None
        region = entry.get('target_region') or entry.get('target_detection') or {}
        sequence = entry.get('frame_sequence')
        if (type(sequence) is not int or sequence != self.last_sequence
                or sequence != region.get('frame_sequence')
                or str(region.get('label')) != str(self.target_label)):
            return None
        if not callable(self.post_press_health_check):
            return None
        # A health/stop exception must propagate; it is NOT a comparison error.
        health = self.post_press_health_check()
        if not isinstance(health, dict) or health.get('healthy') is not True:
            raise RuntimeError('independent post-press motion health unavailable')
        return dict(state='ON', policy='opencv_on_priority_unless_paired_off_v1',
            source='opencv_immediate_on_without_paired_off',
            frame_sequence=sequence, capture_time_unix_sec=entry.get('capture_time_unix_sec'),
            target_label=str(self.target_label), image=entry.get('image'),
            comparison_state=comparison.get('state'), comparison_reason=comparison.get('reason'),
            comparison_exception=comparison.get('exception_type'),
            absolute_input_valid_before_comparison=True,
            absolute_input_age_sec=recognition.get('absolute_input_age_sec'),
            image_age_at_adoption_sec=time.time()-float(entry['capture_time_unix_sec']),
            independent_motion_health=health)

    def post_press_opencv_result(self, position: str) -> dict | None:
        """Immediate real single-image ON, plus legacy bounded OFF fallback.

        ON ignores comparison-only errors after its own input was valid, but
        independent motion health must pass. The older OFF fallback still needs
        fresh same-frame evidence and the verified fixed observation pose.
        """
        if self.audit:
            priority = self._opencv_on_priority(self.audit[-1], position)
            if priority is not None:
                return priority
        if position not in {'near', 'near_confirmation', 'near_recheck'} or not self.audit:
            return None
        paired = self.paired_light
        if (paired is None or not paired.frozen or not paired.require_stationary_reference
                or not paired.stationary_reference_ready or not callable(paired.stationary_check)):
            return None
        entry = self.audit[-1]
        if entry.get('position') != position or not (entry.get('visibility') or {}).get('visible'):
            return None
        recognition = entry.get('recognition') or {}
        comparison = recognition.get('paired_comparison') or {}
        state = recognition.get('absolute_state')
        if (state not in {'ON', 'OFF'} or recognition.get('recognizer') != 'opencv_self_contrast'
                or comparison.get('state') != 'UNCERTAIN'
                or comparison.get('reason') in {None, 'image_predates_press',
                    'no_comparable_off_reference', 'view_geometry_changed', 'saturated_button'}
                or not comparison.get('reference_count')):
            return None
        sequence = entry.get('frame_sequence')
        captured = entry.get('capture_time_unix_sec')
        region = entry.get('target_region') or entry.get('target_detection') or {}
        if (sequence != self.last_sequence or sequence != region.get('frame_sequence')
                or type(sequence) is not int or sequence <= paired.last_reference_sequence
                or type(captured) not in (int, float) or captured <= paired.press_started
                or not self._entry_frame_current(entry, position)):
            return None
        if not paired.stationary_check() or not self._entry_frame_current(entry, position):
            return None
        return {'state': state, 'source': 'current_same_frame_opencv',
                'frame_sequence': sequence, 'capture_time_unix_sec': captured,
                'image': entry.get('image'), 'comparison_state': 'UNCERTAIN',
                'comparison_reason': comparison.get('reason'),
                'stationary_observation_verified': True,
                'target_label': str(self.target_label)}

    def reobserve_post_press(self, position: str) -> dict | None:
        """One bounded recheck at the already verified fixed observation pose.

        One qualified paired ON is enough here. Borderline paired OFF needs three
        additional fresh OFF-supporting images. Every image still compares to
        the frozen pre-press bank; motion-monitor images cannot use this path.
        """
        paired = self.paired_light
        if (position != "near" or paired is None or not paired.frozen
                or not paired.require_stationary_reference
                or not paired.stationary_reference_ready):
            return None
        deadline = time.monotonic() + self._stationary_budget(POST_PRESS_RECHECK_BUDGET_SEC)
        not_before = time.time()
        previous = None
        streak = []
        checked = 0
        on_seen = False
        for _ in range(POST_PRESS_RECHECK_MAX_FRAMES):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                state = self.observe(
                    "near_recheck", timeout_sec=min(POST_PRESS_RECHECK_FRAME_WAIT_SEC, remaining,
                        self._stationary_budget(remaining)), 
                    not_before_unix=not_before,
                )
            except TimeoutError:
                previous, streak = None, []
                continue
            if time.monotonic() > deadline:
                break
            checked += 1
            entry = self.audit[-1]
            if state != 'OFF':
                priority = self.post_press_opencv_result('near_recheck')
                if isinstance(priority, dict) and priority.get('state') == 'ON':
                    return dict(confirmed_state='ON', samples=checked,
                        source='opencv_immediate_on_without_paired_off',
                        opencv_adopted=True, evidence=priority)
            if state == 'ON':
                captured = entry.get('capture_time_unix_sec')
                check_pose = paired.stationary_check
                if (type(captured) not in (int, float)
                        or not self._entry_frame_current(entry, "near_recheck")
                        or not callable(check_pose) or not check_pose()
                        or not self._entry_frame_current(entry, "near_recheck")
                        or time.monotonic() > deadline):
                    previous, streak = None, []
                    continue
            on_seen = on_seen or state == "ON"
            recognition = entry.get("recognition") or {}
            comparison = recognition.get("paired_comparison") or {}
            if state == "ON":
                supported = "ON"
            elif state == "OFF" or (
                    comparison.get("off_recheck_candidate") is True
                    and recognition.get("absolute_state") == "OFF"
            ):
                supported = "OFF"
            else:
                supported = None
            if supported is None:
                previous, streak = None, []
                continue
            if supported != previous:
                streak = []
            previous = supported
            streak.append({
                "frame_sequence": entry["frame_sequence"],
                "capture_time_unix_sec": entry["capture_time_unix_sec"],
                "image": entry["image"],
                "paired_state": state,
                "absolute_state": recognition.get("absolute_state"),
                "off_recheck_candidate": comparison.get("off_recheck_candidate", False),
                "center_error": comparison.get("center_error"),
            })
            required = (POST_PRESS_RECHECK_OFF_FRAMES if supported == "OFF"
                        else POST_PRESS_RECHECK_ON_FRAMES)
            if len(streak) >= required and (supported == "ON" or not on_seen):
                return {
                    "confirmed_state": supported,
                    "source": ('single_paired_on_stationary_recheck_v2' if supported == 'ON'
                               else 'bounded_stationary_post_press_recheck_v1'),
                    "samples": checked, "frames": streak,
                }
        # No timed-out/older image can become a fallback. The helper rechecks
        # the newest audit, its original image age and the current encoders.
        absolute = self.post_press_opencv_result('near_recheck')
        if absolute is not None:
            return {'confirmed_state': absolute['state'], 'samples': checked,
                    'source': 'opencv_after_bounded_stationary_recheck_v1',
                    'opencv_adopted': True, 'evidence': absolute}
        return {"confirmed_state": None, "samples": checked,
                "source": "bounded_stationary_post_press_recheck_v1",
                "reason": ("on_seen_without_two_confirmations" if on_seen
                           else "insufficient_consecutive_valid_frames")}

    @traced('light.off_reference_collection', 'reference')
    def capture_stationary_off_reference(self, timeout_sec: float = STATIONARY_REFERENCE_TOTAL_SEC) -> dict:
        """Called only after the executor verifies its fixed 5 cm pose.

        No concurrent approach collector is allowed for this mode. The time
        barrier excludes inference frames captured before the arm settled.
        """
        paired = self.paired_light
        if paired is None or not paired.require_stationary_reference:
            raise RuntimeError("stationary reference mode is not enabled")
        if paired.frozen:
            raise RuntimeError("cannot capture reference after press begins")
        if paired.references:
            raise RuntimeError("stationary reference must start with an empty bank")
        paired.reset_pending()
        started = time.time()
        deadline = time.monotonic() + self._stationary_budget(timeout_sec)
        self.stationary_reference_capture_active = True
        try:
            while time.monotonic() < deadline:
                try:
                    self.observe("before_press", timeout_sec=min(STATIONARY_REFERENCE_FRAME_WAIT_SEC,
                        deadline-time.monotonic(), self._stationary_budget(deadline-time.monotonic())),
                                 emit_event=False, not_before_unix=started)
                except TimeoutError:
                    paired.reset_pending()
                    continue
                if len(paired.references) >= 2:
                    paired.stationary_reference_ready = True
                    return {"mode": "stationary_5cm_off_reference",
                            "reference_count": len(paired.references),
                            "sequences": [p.sequence for p in paired.references],
                            "capture_barrier_unix": started,
                            "reference_comparison": paired.last_reference_comparison}
        finally:
            self.stationary_reference_capture_active = False
        raise RuntimeError("cannot confirm OFF at the fixed 5 cm observation pose; press withheld")

    def observe_during_motion(self) -> str:
        """Sample one safely cropped frame without blocking motion shutdown."""
        return self.observe(
            "press_monitor",
            timeout_sec=POST_PRESS_MONITOR_FRAME_TIMEOUT_SEC,
            emit_event=False,
            visibility_edge_margin_px=POST_PRESS_VISIBILITY_EDGE_MARGIN_PX,
        )

    def wait_for_next_frame(self) -> None:
        # observe() enforces a strict sequence increase; no fixed sleep is needed.
        return None


class DryValidatedSegmentExecutor:
    """Validate all commands before optionally allowing segmented execution."""

    def __init__(
        self,
        bridge: Any,
        robot: Any,
        token: str,
        *,
        execute_enabled: bool = False,
        joint_tolerance_rad: float = 0.03,
        motion_safety: Any | None = None,
        event_sink: Any | None = None,
        guarded_bridge: Any | None = None,
        guarded_press_config: dict | None = None,
    ):
        self.bridge = bridge
        self.robot = robot
        self.token = str(token)
        self.execute_enabled = bool(execute_enabled)
        self.joint_tolerance_rad = float(joint_tolerance_rad)
        self.motion_safety = motion_safety
        self.event_sink = event_sink
        self.guarded_bridge = guarded_bridge
        self.guarded_press_config = guarded_press_config
        self.plan: SegmentedPressPlan | None = None
        self.validations: dict[str, dict] = {}
        self.plan_validation_scope = "full"
        self.executions: list[dict] = []
        self.visual_safety_disabled_for_return = False
        self.before_guarded_press_callback: Any | None = None

    def set_before_guarded_press_callback(self, callback: Any | None) -> None:
        """Install a zero-motion hook at the exact guarded-press boundary."""
        if callback is not None and not callable(callback):
            raise TypeError("guarded press callback must be callable")
        self.before_guarded_press_callback = callback

    @traced('trajectory.validate_plan', 'validation')
    def load_plan(self, plan: SegmentedPressPlan, *, already_at_prepress=False,
                  already_at_observation=False) -> None:
        began=time.monotonic()
        self.plan=None
        self.validations={}
        if already_at_prepress and already_at_observation:
            raise ValueError('approach endpoint cannot be both prepress and observation')
        self.plan_validation_scope=('observation' if already_at_observation else
                                    'prepress' if already_at_prepress else 'full')
        if already_at_observation:
            state = self.robot.state(max_age=.25)
            error = float(np.max(np.abs(state.joints - plan.observation_joints)))
            if state.raw.get('hardware_ok') != 1 or not np.isfinite(error) or error > OBSERVATION_JOINT_TOLERANCE_RAD:
                raise RuntimeError('post-approach preparation is not at observation: %.4f rad' % error)
        if already_at_prepress:
            state=self.robot.state(max_age=2.0)
            target=np.asarray(plan.observation_to_prepress[-1]['positions'],dtype=float)
            error=float(np.max(np.abs(state.joints-target)))
            if state.raw.get('hardware_ok')!=1 or error>self.joint_tolerance_rad:
                raise RuntimeError('post-approach preparation is not at prepress: %.4f rad'%error)
        segments = {
            "approach_to_observation": plan.approach,
            "observation_to_prepress": plan.observation_to_prepress,
            "press_cycle": plan.press_cycle,
            "retreat_to_observation": plan.retreat_to_observation,
            "observation_to_waiting": plan.observation_to_waiting,
        }
        if plan.prepress_to_waiting is not None:
            segments["prepress_to_waiting"] = plan.prepress_to_waiting
        skipped=[]
        if already_at_observation:
            segments.pop('approach_to_observation')
            skipped.append('approach_to_observation')
        if already_at_prepress:
            for name in ('approach_to_observation','observation_to_prepress'):
                segments.pop(name);skipped.append(name)
        validated = {}
        validation_seconds={}
        for name, points in segments.items():
            segment_started=time.monotonic()
            with stage('trajectory.validate.'+name, 'validation', scope='remote roundtrip, not isolated MoveIt time'):
                validated[name] = self.bridge.execute(
                    points,
                    token=self.token,
                    task_id=f"button-light-{name}-validate-{time.time_ns()}",
                    dry_run=True,
                )
            validation_seconds[name]=time.monotonic()-segment_started
        if self.guarded_bridge is not None:
            segment_started=time.monotonic()
            if self.guarded_press_config is None:
                raise RuntimeError("guarded press bridge requires guarded press config")
            outbound_count = (len(plan.press_cycle) + 1) // 2
            validated["guarded_press_forward"] = self.guarded_bridge.execute(
                plan.press_cycle[:outbound_count], token=self.token,
                max_forward_travel_m=float(self.guarded_press_config["max_forward_travel_m"]),
                current_guard=self.guarded_press_config["current_guard"],
                contact_normal_base=(
                    None if plan.base_to_button is None
                    else np.asarray(plan.base_to_button, dtype=float)[:3, 2].tolist()
                ),
                contact_panel_origin_base=(
                    None if plan.base_to_button is None
                    else np.asarray(plan.base_to_button, dtype=float)[:3, 3].tolist()
                ),
                retract_target_joints=plan.observation_joints.tolist(),
                retreat_to_observation=plan.retreat_to_observation,
                task_id=f"button-light-guarded-validate-{time.time_ns()}",
                dry_run=True,
            )
            validation_seconds['guarded_press_forward']=time.monotonic()-segment_started
        self.plan = plan
        self.validations = validated
        if self.event_sink is not None:
            self.event_sink(
                "motion_plan_validated",
                {"segments": list(validated), "observation_distance_m": plan.observation_distance_m,
                 "validation_scope":self.plan_validation_scope,"skipped_unused_segments":skipped,
                 "validation_seconds":validation_seconds,"load_plan_sec":time.monotonic()-began},
            )
        if self.motion_safety is not None:
            self.motion_safety.set_target(plan.base_to_button)

    def _require_plan(self) -> SegmentedPressPlan:
        required={'press_cycle','retreat_to_observation','observation_to_waiting'}
        if self.plan_validation_scope=='full':
            required.update(['approach_to_observation','observation_to_prepress'])
        if self.plan_validation_scope=='observation':
            required.add('observation_to_prepress')
        if self.plan is not None and self.plan.prepress_to_waiting is not None:
            required.add('prepress_to_waiting')
        if self.guarded_bridge is not None:required.add('guarded_press_forward')
        if self.plan is None or not required.issubset(self.validations):
            raise RuntimeError('segmented press plan has not passed all dry validations')
        return self.plan

    def recheck_observation_view(self, target_joints, origin_joints) -> dict:
        """Execute only a bounded, independently validated initial wrist look."""
        if self.plan is not None or self.visual_safety_disabled_for_return:
            raise RuntimeError("initial view recheck is unavailable after approach")
        if not self.execute_enabled or self.motion_safety is None:
            raise RuntimeError("initial view recheck requires active motion safety")
        state = self.robot.state(max_age=.25)
        target = np.asarray(target_joints, dtype=float)
        delta = target - state.joints
        if (target.shape != (6,) or not np.isfinite(target).all()
                or state.raw.get("hardware_ok") != 1
                or abs(delta[4]) > .028001
                or np.max(np.abs(np.delete(delta, 4))) > .003):
            raise RuntimeError("initial view recheck exceeds healthy J5-only bounds")
        # FK selection and this read can differ by encoder quantization. Hold
        # every other joint at its newest value, never command those residuals;
        # clamp J5 to the original per-move bound rather than enlarging it.
        target = state.joints.copy()
        target[4] += float(np.clip(delta[4], -.025, .025))
        points, _ = generate_timed_joint_trajectory(
            state.joints, target, cartesian_speed_m_s=.03,
            joint_velocity_scale=.10, point_period_sec=.05, duration_margin=1.05)
        return_points, _ = generate_timed_joint_trajectory(
            target, origin_joints, cartesian_speed_m_s=.15,
            joint_velocity_scale=.75, point_period_sec=.10, duration_margin=1.02)
        name = "initial_light_view_recheck"
        self.validations.pop(name, None)
        validation = self.bridge.execute(points, token=self.token,
            task_id=f"initial-view-validate-{time.time_ns()}", dry_run=True)
        return_validation = self.bridge.execute(return_points, token=self.token,
            task_id=f"initial-view-origin-validate-{time.time_ns()}", dry_run=True)
        fresh = self.robot.state(max_age=.25)
        if fresh.raw.get("hardware_ok") != 1 or np.max(np.abs(fresh.joints - state.joints)) > .003:
            raise RuntimeError("arm moved during initial view validation")
        self.validations[name] = validation
        event = self._execute(name, points, target, endpoint_tolerance_rad=.005)
        actual = self.robot.state(max_age=.25)
        if (actual.raw.get("hardware_ok") != 1
                or np.max(np.abs(actual.joints - target)) > .005):
            raise RuntimeError("initial view recheck did not reach its wrist target")
        return {"execution": event, "origin_return_validation": return_validation,
                "commanded_joints_rad": target.tolist(),
                "joints_rad": actual.joints.tolist()}

    @traced('motion.segment', 'motion')
    def _execute(self, name: str, points: list[dict], target_joints: np.ndarray,
                 *, endpoint_tolerance_rad: float | None = None) -> dict:
        if name not in self.validations:
            raise RuntimeError('motion segment has not been validated: '+name)
        if not self.execute_enabled:
            raise RuntimeError("physical segmented execution is disabled")
        if self.motion_safety is None:
            raise RuntimeError("physical segmented execution requires motion safety")
        use_visual_safety = not self.visual_safety_disabled_for_return
        if use_visual_safety:
            self.motion_safety.before_segment(name)
        task_id = f"button-light-{name}-execute-{time.time_ns()}"
        if self.event_sink is not None:
            self.event_sink("motion_segment_started", {
                "segment": name, "task_id": task_id,
            })
        primary_error: BaseException | None = None
        try:
            result = self.bridge.execute(
                points,
                token=self.token,
                task_id=task_id,
            )
        except BaseException as exc:
            primary_error = exc
            if self.event_sink is not None:
                self.event_sink(
                    "motion_segment_failed",
                    {"segment": name, "task_id": task_id,
                     "error": f"{type(exc).__name__}: {exc}"},
                )
            raise
        finally:
            if use_visual_safety:
                try:
                    self.motion_safety.after_segment(name)
                except Exception as cleanup_exc:
                    if self.event_sink is not None:
                        self.event_sink("motion_segment_cleanup_failed", {
                            "segment": name,
                            "task_id": task_id,
                            "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                            "primary_error_present": primary_error is not None,
                        })
                        if primary_error is None:
                            self.event_sink("motion_segment_failed", {
                                "segment": name,
                                "task_id": task_id,
                                "phase": "safety_cleanup",
                                "error": (
                                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                                ),
                            })
                    # A safety latch during cleanup is important state, but must
                    # not erase the bridge failure that caused this segment to end.
                    if primary_error is None:
                        raise
        try:
            state = self.robot.state(max_age=2.0)
            error = float(np.max(np.abs(state.joints - target_joints)))
            tolerance = (self.joint_tolerance_rad if endpoint_tolerance_rad is None
                         else min(self.joint_tolerance_rad, float(endpoint_tolerance_rad)))
            if state.raw.get("hardware_ok") != 1 or error > tolerance:
                raise RuntimeError(f"{name} encoder target error is {error:.4f} rad")
        except BaseException as exc:
            if self.event_sink is not None:
                self.event_sink("motion_segment_failed", {
                    "segment": name,
                    "task_id": task_id,
                    "phase": "terminal_encoder_verification",
                    "bridge_result": result,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            raise
        event = {
            "segment": name, "task_id": task_id,
            "result": result, "encoder_error_rad": error,
        }
        self.executions.append(event)
        if self.event_sink is not None:
            self.event_sink("motion_segment_completed", event)
        return event

    def disable_visual_safety_for_return(self) -> None:
        """Disarm depth/RGB supervision for validated retreat-only segments."""
        if self.motion_safety is None:
            raise RuntimeError("return safety cannot be disabled without a supervisor")
        self.motion_safety.disable_for_return()
        self.visual_safety_disabled_for_return = True

    def prepare_origin_return(self) -> None:
        """Commit the outward safety state only after its path was validated."""
        if self.motion_safety is None:
            raise RuntimeError("return safety requires an initialized supervisor")
        finish = getattr(self.motion_safety, "finish_near_observation", None)
        held = bool(finish()) if callable(finish) else False
        if held or not self.visual_safety_disabled_for_return:
            self.disable_visual_safety_for_return()

    def begin_press_attempt(self) -> None:
        """Restore outbound guarding after the preceding attempt's retreat."""
        if self.motion_safety is None:
            raise RuntimeError("press attempt requires motion safety")
        self.motion_safety.begin_outbound_cycle()
        self.visual_safety_disabled_for_return = False

    @traced('motion.guarded_press_and_retract', 'motion')
    def _execute_guarded_press(self, plan: SegmentedPressPlan, prepress: np.ndarray) -> dict:
        if self.guarded_bridge is None:
            raise RuntimeError(
                "physical button press requires the current-feedback guarded bridge"
            )
        if not self.execute_enabled or self.motion_safety is None:
            raise RuntimeError("physical guarded press requires execution and motion safety")
        name = "guarded_press"
        self.motion_safety.before_segment("press_cycle")
        if self.before_guarded_press_callback is not None:
            self.before_guarded_press_callback()
        task_id = f"button-light-guarded-execute-{time.time_ns()}"
        if self.event_sink is not None:
            self.event_sink("motion_segment_started", {
                "segment": name, "task_id": task_id,
            })
        primary_error = None
        try:
            outbound_count = (len(plan.press_cycle) + 1) // 2
            result = self.guarded_bridge.execute(
                plan.press_cycle[:outbound_count], token=self.token,
                max_forward_travel_m=float(self.guarded_press_config["max_forward_travel_m"]),
                current_guard=self.guarded_press_config["current_guard"],
                contact_normal_base=(
                    None if plan.base_to_button is None
                    else np.asarray(plan.base_to_button, dtype=float)[:3, 2].tolist()
                ),
                contact_panel_origin_base=(
                    None if plan.base_to_button is None
                    else np.asarray(plan.base_to_button, dtype=float)[:3, 3].tolist()
                ),
                retract_target_joints=plan.observation_joints.tolist(),
                retreat_to_observation=plan.retreat_to_observation,
                task_id=task_id,
            )
        except BaseException as exc:
            primary_error = exc
            if self.event_sink is not None:
                self.event_sink("motion_segment_failed", {
                    "segment": name, "task_id": task_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "bridge_failure": getattr(exc, 'response', None),
                })
            raise
        finally:
            try:
                self.motion_safety.after_segment("press_cycle")
            except Exception as cleanup_exc:
                if self.event_sink is not None:
                    self.event_sink("motion_segment_cleanup_failed", {
                        "segment": name,
                        "task_id": task_id,
                        "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                        "primary_error_present": primary_error is not None,
                    })
                    if primary_error is None:
                        self.event_sink("motion_segment_failed", {
                            "segment": name,
                            "task_id": task_id,
                            "phase": "safety_cleanup",
                            "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                        })
                if primary_error is None:
                    raise
        try:
            state = self.robot.state(max_age=2.0)
            retract_destination = result.get("retract_destination", "prepress")
            expected_joints = (
                np.asarray(plan.observation_joints, dtype=float)
                if retract_destination == "observation"
                else prepress
            )
            error = float(np.max(np.abs(state.joints - expected_joints)))
            if state.raw.get("hardware_ok") != 1 or error > self.joint_tolerance_rad:
                raise RuntimeError(f"guarded press retract error is {error:.4f} rad")
        except BaseException as exc:
            if self.event_sink is not None:
                self.event_sink("motion_segment_failed", {
                    "segment": name,
                    "task_id": task_id,
                    "phase": "terminal_retract_verification",
                    "bridge_result": result,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            raise
        event = {
            "segment": name, "task_id": task_id,
            "result": result, "encoder_error_rad": error,
        }
        self.executions.append(event)
        feedback_confirmed = (
            result.get("status") == "contact_retracted"
            and result.get("feedback_contact_confirmed") is True
        )
        if not feedback_confirmed:
            if self.event_sink is not None:
                self.event_sink("motion_segment_feedback_unconfirmed", event)
            # Reaching the bounded displacement without sustained force is a
            # completed, safely retracted attempt. Let the state machine check
            # the lamp: ON proves the button actuated despite weak force,
            # while OFF authorizes a fresh-target retry. A current/feedback
            # integrity limit remains an exceptional abort.
            recoverable_unconfirmed_attempt = (
                result.get("status") == "maximum_travel_retracted"
                or (
                    result.get("status") == "limit_retracted"
                    and result.get("reason") == "stale_current_sample"
                )
            )
            if recoverable_unconfirmed_attempt:
                return event
            raise RuntimeError(
                "guarded press retracted without feedback-confirmed contact: "
                f"status={result.get('status')}, reason={result.get('reason')}"
            )
        if self.event_sink is not None:
            self.event_sink("motion_segment_completed", event)
        return event

    def approach_and_press(self) -> list[dict]:
        plan = self._require_plan()
        observation = np.asarray(plan.approach[-1]["positions"], dtype=float)
        prepress = np.asarray(plan.observation_to_prepress[-1]["positions"], dtype=float)
        return [
            self._execute("approach_to_observation", plan.approach, observation),
            self._execute("observation_to_prepress", plan.observation_to_prepress, prepress),
            self._execute_guarded_press(plan, prepress),
        ]

    def press_from_prepress(self) -> dict:
        """Execute the guarded press after an online planner reached prepress.

        This entry point deliberately sends no reaching trajectory.  It lets
        the light state machine use the receding-IK/splice path for approach
        while retaining the same validated current-feedback press and the
        same post-press retreat plan.
        """
        plan = self._require_plan()
        prepress = np.asarray(plan.observation_to_prepress[-1]["positions"], dtype=float)
        state = self.robot.state(max_age=2.0)
        error = float(np.max(np.abs(state.joints - prepress)))
        if state.raw.get("hardware_ok") != 1 or error > self.joint_tolerance_rad:
            raise RuntimeError(
                f"online approach did not reach validated prepress: {error:.4f} rad"
            )
        return self._execute_guarded_press(plan, prepress)

    def press_from_observation(self) -> list[dict]:
        """Make a retry directly from its freshly verified near observation."""
        plan = self._require_plan()
        observation = np.asarray(plan.approach[-1]["positions"], dtype=float)
        prepress = np.asarray(plan.observation_to_prepress[-1]["positions"], dtype=float)
        state = self.robot.state(max_age=2.0)
        error = float(np.max(np.abs(state.joints - observation)))
        if state.raw.get("hardware_ok") != 1 or error > self.joint_tolerance_rad:
            raise RuntimeError(
                f"retry did not start at validated observation: {error:.4f} rad"
            )
        return [
            self._execute("observation_to_prepress", plan.observation_to_prepress, prepress),
            self._execute_guarded_press(plan, prepress),
        ]

    def press_with_stationary_reference(self, capture_reference: Any, *,
                                        already_at_observation: bool,
                                        keep_approach_lease: bool = False) -> list[dict]:
        """Acquire the before image at the exact pose used by guarded retract.

        Production fixed-reference approaches and retries arrive directly at the
        photo pose. The legacy prepress entry remains guarded and validated.
        Direct first approaches preserve their bounded near-panel lease across
        the photo stop, without extending it or clearing any safety latch.
        """
        plan = self._require_plan()
        if keep_approach_lease and not already_at_observation:
            raise ValueError('direct reference handoff requires the observation endpoint')
        if "observation_to_prepress" not in self.validations:
            self.validations["observation_to_prepress"] = self.bridge.execute(
                plan.observation_to_prepress, token=self.token,
                task_id=f"button-light-reference-inward-validate-{time.time_ns()}",
                dry_run=True,
            )
        motion_safety = getattr(self, "motion_safety", None)
        transition = (motion_safety.stationary_reference_transition()
                      if keep_approach_lease or getattr(motion_safety, "stationary_reference_handoff", False)
                      else nullcontext())
        with transition:
            if not already_at_observation:
                self.retreat_to_observation(plan.observation_distance_m)
            def check_pose_safety():
                if motion_safety is not None:
                    motion_safety._assert_clear()
            try:
                state, pose_audit = wait_for_stable_observation_pose(
                    plan.observation_joints,
                    lambda: self.robot.latest_state(max_age=OBSERVATION_FEEDBACK_MAX_AGE_SEC),
                    check_pose_safety)
            except BaseException as exc:
                if self.event_sink is not None:
                    self.event_sink("stationary_reference_pose_confirmation", {
                        "stage": "before_capture", "status": "failed",
                        "audit": getattr(exc, "audit", getattr(exc, "observation_pose_audit", {})),
                        "error": str(exc)})
                raise
            if self.event_sink is not None:
                self.event_sink("stationary_reference_pose_confirmation", {
                    "stage": "before_capture", "status": "stable", "audit": pose_audit})
            capture_start_joints = np.asarray(state.joints, dtype=float).copy()
            capture_start_state = state
            capture_started = time.monotonic()
            reference = capture_reference()
            reference_ready = time.monotonic()
            after_capture_state = None
            after_feedback_read = None
            final_error = capture_drift = None
            failed_check_stage = "feedback_read"
            try:
                # Keep the original read, calculations, safety call and guard.
                state = self.robot.latest_state(max_age=OBSERVATION_FEEDBACK_MAX_AGE_SEC)
                after_capture_state = state
                after_feedback_read = time.monotonic()
                failed_check_stage = "error_calculation"
                final_error = float(np.max(np.abs(state.joints - plan.observation_joints)))
                capture_drift = float(np.max(np.abs(state.joints - capture_start_joints)))
                failed_check_stage = "safety_check"
                check_pose_safety()
                failed_check_stage = "pose_guard"
                if (state.raw.get("hardware_ok") != 1 or not np.isfinite(final_error)
                        or final_error > OBSERVATION_JOINT_TOLERANCE_RAD
                        or not np.isfinite(capture_drift) or capture_drift_exceeds_limit(capture_drift)):
                    raise RuntimeError("arm moved while collecting the stationary reference")
            except Exception as exc:
                # Build/persist only after rejection. Reporting must never
                # replace the original error, authorize motion, or reread state.
                try:
                    audit = stationary_reference_failure_audit(
                        capture_start_state, after_capture_state, plan.observation_joints,
                        capture_started=capture_started, reference_ready=reference_ready,
                        feedback_checked=(time.monotonic() if after_feedback_read is None else after_feedback_read),
                        failed_check_stage=failed_check_stage,
                        final_error=final_error, capture_drift=capture_drift,
                        reference=reference, exception=exc)
                    exc.stationary_reference_pose_audit = audit
                    if self.event_sink is not None:
                        self.event_sink("stationary_reference_pose_confirmation", {
                            "stage": "after_capture", "status": "failed",
                            "audit": audit, "error": str(exc)})
                except Exception as reporting_exc:
                    exc.stationary_reference_pose_diagnostics_error = (
                        f"{type(reporting_exc).__name__}: {reporting_exc}")
                raise
            if self.event_sink is not None:
                self.event_sink("stationary_press_reference_confirmed", {
                    **reference, "observation_joints": plan.observation_joints.tolist(),
                    "encoder_error_rad": final_error,
                    "observation_joint_tolerance_rad": OBSERVATION_JOINT_TOLERANCE_RAD,
                    "capture_drift_rad": capture_drift,
                    "capture_drift_tolerance_rad": OBSERVATION_CAPTURE_DRIFT_RAD,
                    "capture_drift_numeric_epsilon_rad": OBSERVATION_CAPTURE_DRIFT_NUMERIC_EPS_RAD,
                    "already_at_observation": bool(already_at_observation),
                    "direct_approach_reference": bool(keep_approach_lease),
                })
            # Retreat disarms visual supervision; explicitly restore outbound
            # supervision before any inward motion. Collision/force limits persist.
            self.begin_press_attempt()
            return self.press_from_observation()

    def post_press_motion_health(self):
        """Check independent stops and fresh healthy joint feedback; no motion."""
        if self.motion_safety is None:
            raise RuntimeError('post-press motion safety unavailable')
        self.motion_safety._assert_clear()
        state = self.robot.latest_state(max_age=OBSERVATION_FEEDBACK_MAX_AGE_SEC)
        joints = np.asarray(state.joints, dtype=float)
        age = time.monotonic()-float(state.timestamp)
        if (state.raw.get('hardware_ok') != 1 or joints.shape != (6,)
                or not np.isfinite(joints).all() or not np.isfinite(age)
                or not 0 <= age <= OBSERVATION_FEEDBACK_MAX_AGE_SEC):
            raise RuntimeError('post-press joint feedback unhealthy or stale')
        return dict(healthy=True, hardware_ok=1, feedback_age_sec=age)

    def stationary_observation_verified(self) -> bool:
        """Read fresh encoders only; called lazily for clipped light comparisons.

        The caller must also be in a completed stationary observation phase.
        Movement monitors cannot use this endpoint proximity as a stop signal.
        """
        plan = self._require_plan()
        state = self.robot.state(max_age=.25)
        error = float(np.max(np.abs(state.joints - plan.observation_joints)))
        return bool(state.raw.get('hardware_ok') == 1 and np.isfinite(error) and error <= OBSERVATION_JOINT_TOLERANCE_RAD)

    def approach_to_observation(self) -> dict:
        """Execute only the validated reaching-path observation leg."""
        plan = self._require_plan()
        observation = np.asarray(plan.approach[-1]["positions"], dtype=float)
        return self._execute("approach_to_observation", plan.approach, observation)

    def retreat_to_observation(self, requested_distance_m: float) -> dict:
        plan = self._require_plan()
        # The state machine requests the nominal reaching-path standoff. A retry
        # deliberately preserves the current, vision-verified hold, whose
        # distance changes by millimetres when the panel plane is refreshed.
        if abs(float(requested_distance_m) - plan.observation_distance_m) > 0.02:
            raise ValueError("requested observation distance differs from validated plan")
        if not self.visual_safety_disabled_for_return:
            self.disable_visual_safety_for_return()
        state = self.robot.state(max_age=2.0)
        observation_error = float(np.max(
            np.abs(state.joints - np.asarray(plan.observation_joints, dtype=float))
        ))
        if state.raw.get("hardware_ok") == 1 and observation_error <= OBSERVATION_JOINT_TOLERANCE_RAD:
            event = {
                "segment": "retreat_to_observation",
                "result": {
                    "status": "already_at_target",
                    "point_count": 0,
                    "blended_guarded_retract": True,
                },
                "encoder_error_rad": observation_error,
            }
            self.executions.append(event)
            if self.event_sink is not None:
                self.event_sink("motion_segment_completed", event)
            return event
        return self._execute(
            "retreat_to_observation",
            plan.retreat_to_observation,
            plan.observation_joints,
        )

    def return_to_waiting(self) -> dict:
        plan = self._require_plan()
        current = self.robot.state(max_age=2.0)
        if current.raw.get("hardware_ok") != 1:
            raise RuntimeError("robot hardware is not OK before return")
        observation = np.asarray(plan.observation_joints, dtype=float)
        prepress = np.asarray(
            plan.observation_to_prepress[-1]["positions"], dtype=float
        )
        waiting = np.asarray(
            plan.observation_to_waiting[-1]["positions"], dtype=float
        )
        waiting_error = float(np.max(np.abs(current.joints - waiting)))
        if waiting_error <= self.joint_tolerance_rad:
            self.prepare_origin_return()
            event = {
                "segment": "observation_to_waiting",
                "result": {"status": "already_at_target", "point_count": 0},
                "encoder_error_rad": waiting_error,
            }
            self.executions.append(event)
            if self.event_sink is not None:
                self.event_sink("motion_segment_completed", event)
            return event
        start_error = float(np.max(np.abs(current.joints - observation)))
        prepress_error = float(np.max(np.abs(current.joints - prepress)))
        if (
            prepress_error <= self.joint_tolerance_rad
            and plan.prepress_to_waiting is not None
        ):
            points = plan.prepress_to_waiting
            segment_name = "prepress_to_waiting"
        elif start_error <= self.joint_tolerance_rad:
            points = plan.observation_to_waiting
            segment_name = "observation_to_waiting"
        else:
            # Post-press light confirmation may deliberately remain at the
            # prepress pose. Build the return from measured encoders instead
            # of reusing a trajectory whose first point is the 5 cm hold.
            points, audit = generate_timed_joint_trajectory(
                current.joints,
                waiting,
                cartesian_speed_m_s=0.20,
                cartesian_speed_cap_m_s=0.25,
                joint_velocity_scale=0.75,
                point_period_sec=0.10,
                duration_margin=1.02,
            )
            validation = self.bridge.execute(
                points,
                token=self.token,
                task_id=f"button-light-current_to_waiting-validate-{time.time_ns()}",
                dry_run=True,
            )
            self.validations["current_to_waiting"] = validation
            if self.event_sink is not None:
                self.event_sink("dynamic_return_validated", {
                    "start_error_from_standoff_rad": start_error,
                    "trajectory_audit": {
                        "duration_sec": audit.duration_sec,
                        "point_count": audit.point_count,
                        "cartesian_path_m": audit.cartesian_path_m,
                        "cartesian_peak_speed_m_s": audit.cartesian_peak_speed_m_s,
                        "joint_peak_velocity_rad_s": (
                            audit.joint_peak_velocity_rad_s.tolist()
                        ),
                        "joint_velocity_fraction": (
                            audit.joint_velocity_fraction.tolist()
                        ),
                    },
                    "validation": validation,
                })
            segment_name = "current_to_waiting"
        self.prepare_origin_return()
        return self._execute(segment_name, points, waiting)

    def prepare_retry_observation(self, handeye) -> dict:
        """After OFF, reach the 8cm photo pose before refreshing/retrying.

        Does not replace the press plan. If refresh fails afterwards, the
        existing measured-encoder return remains available from this new pose.
        """
        plan=self._require_plan()
        safety=self.motion_safety
        if not self.execute_enabled or safety is None:
            raise RuntimeError('retry observation requires active validated motion')
        safety._assert_clear()
        state=self.robot.state(max_age=.25)
        error=float(np.max(np.abs(state.joints-plan.observation_joints)))
        if state.raw.get('hardware_ok')!=1 or not np.isfinite(error) or error>OBSERVATION_JOINT_TOLERANCE_RAD:
            raise RuntimeError('retry transition did not start at the verified photo pose')
        cached=getattr(self,'_retry_preparation_joints',None)
        if (cached is not None and np.allclose(plan.observation_joints,cached,atol=1e-9,rtol=0)):
            return dict(observation_joints=plan.observation_joints.tolist(),
                requested_panel_distance_m=RETRY_PREPARATION_DISTANCE_M,
                panel_distance_m=plan.observation_distance_m,mode='retry_reused_8cm')
        points,target,alignment=plan_retry_observation_transition(
            state.joints,plan.base_to_button,handeye)
        name='retry_observation_retreat'
        self.validations.pop(name,None)
        validation=self.bridge.execute(points,token=self.token,
            task_id=f'retry-observation-validate-{time.time_ns()}',dry_run=True)
        return_points,_=generate_timed_joint_trajectory(target,
            np.asarray(plan.observation_to_waiting[-1]['positions'],float),
            cartesian_speed_m_s=.20,cartesian_speed_cap_m_s=.25,joint_velocity_scale=.75,
            point_period_sec=.10,duration_margin=1.02)
        return_validation=self.bridge.execute(return_points,token=self.token,
            task_id=f'retry-observation-return-validate-{time.time_ns()}',dry_run=True)
        fresh=self.robot.state(max_age=.25)
        if fresh.raw.get('hardware_ok')!=1 or np.max(np.abs(fresh.joints-state.joints))>.003:
            raise RuntimeError('arm moved during retry observation validation')
        safety._assert_clear()
        self.validations[name]=validation
        self.disable_visual_safety_for_return()
        execution=self._execute(name,points,target,endpoint_tolerance_rad=OBSERVATION_JOINT_TOLERANCE_RAD)
        self._retry_preparation_joints=target.copy()
        return dict(observation_joints=target.tolist(),execution=execution,
                    return_validation=return_validation,**alignment)


class VisualSegmentMotionSafety:
    """Keep RGB-D supervision and bounded depth pause across each segment."""

    def __init__(self, supervisor: Any, vision: Any, robot: Any, handeye: Any, *,
                 pause_depth_near_button: bool = False,
                 depth_degraded_entry_distance_m: float = 0.10,
                 stationary_reference_handoff: bool = False):
        self.supervisor = supervisor
        self.vision = vision
        self.robot = robot
        self.handeye = np.asarray(handeye, dtype=float)
        self.pause_depth_near_button = bool(pause_depth_near_button)
        self.stationary_reference_handoff = bool(stationary_reference_handoff)
        self._reference_hold = False
        self._reference_observation_hold = False
        # Arm before the observed 5.4 cm matrix-coverage drop.  This only
        # permits the specifically named coverage-loss status; a depth
        # residual, collision, stale encoder, or RPC failure still stops.
        self.near_pause = NearButtonDepthPause(
            entry_distance_m=float(depth_degraded_entry_distance_m)
        )
        self.depth_degraded_commit_active = False
        self.depth_degraded_commit_events = []
        self._depth_degraded_commit_lock = threading.Lock()
        self.target = None
        self.closed = threading.Event()
        self.publisher_error = None
        self._thread = None
        self._pose_timing = PoseTimingTrace()

    def set_target(self, target: Any) -> None:
        if target is None:
            raise RuntimeError("visual motion safety requires a live button target")
        value = np.asarray(target, dtype=float)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError("visual motion safety target must be a finite 4x4 transform")
        self.target = value.copy()

    def begin_outbound_cycle(self) -> None:
        """Reset per-attempt near-field state before any new inward motion."""
        if self.supervisor.error:
            raise RuntimeError(
                f"visual safety supervisor failed: {self.supervisor.error}"
            )
        if self.supervisor.triggered.is_set() or self.supervisor.stop_active.is_set():
            raise RuntimeError("visual safety stop is active")
        self.supervisor.depth_pause_until = 0.0
        if self._reference_hold:
            # This is the same validated near-panel cycle, including the short
            # reference retreat. Do not open a gap in its existing authorization.
            self._check_reference_pose()
            return
        if self._reference_observation_hold:
            # A new, fully validated retry starts at the previous photo pose.
            # Keep the near mode armed while creating its new bounded attempt;
            # only this explicit attempt boundary can renew the lease.
            sample = self._check_reference_pose()
            self.near_pause.begin_cycle()
            lease = self.near_pause.update(sample.base_to_gripper, self.target,
                sample.timestamp, time.monotonic(), enabled=True,
                hardware_ok=sample.raw.get("hardware_ok") == 1)
            if not lease:
                raise RuntimeError("stationary retry near-panel lease unavailable")
            self.depth_degraded_commit_events.append(dict(active=True,
                reason="validated_stationary_retry_cycle", monotonic=time.monotonic(),
                lease_max_duration_sec=NEAR_BUTTON_DEPTH_LEASE_SEC))
            return
        self.near_pause.begin_cycle()
        self._deactivate_depth_degraded_commit("new_press_attempt")

    def stationary_observation_remaining(self):
        """Read the original attempt deadline; observations never renew it."""
        self._assert_clear()
        if self.near_pause.started is None:
            return NEAR_BUTTON_DEPTH_LEASE_SEC
        return max(0., self.near_pause.started + NEAR_BUTTON_DEPTH_LEASE_SEC - time.monotonic())

    def _read_reference_pose(self):
        self._assert_clear()
        sample = self.robot.state(max_age=0.25)
        now = time.monotonic()
        probe = NearButtonDepthPause(self.near_pause.entry_distance_m)
        if (not self.pause_depth_near_button or self.target is None or
                not probe.update(sample.base_to_gripper, self.target,
                                 sample.timestamp, now, enabled=True,
                                 hardware_ok=sample.raw.get("hardware_ok") == 1)):
            raise RuntimeError("stationary reference requires fresh healthy near-panel encoders")
        if (self.near_pause.finished or (self.near_pause.started is not None and
                now >= self.near_pause.started + NEAR_BUTTON_DEPTH_LEASE_SEC)):
            raise RuntimeError("stationary reference near-panel lease expired")
        return sample

    def _check_reference_pose(self):
        sample = self._read_reference_pose()
        self.vision.request("safety_camera_pose", timestamp_monotonic=float(sample.timestamp),
                            base_to_camera=(sample.base_to_gripper @ self.handeye).tolist())
        # A healthy sample before a slow RPC is not necessarily fresh afterwards.
        # Re-read real feedback; never relabel an old timestamp or extend a lease.
        return self._read_reference_pose()

    def _fresh_reference_lease(self, timeout_sec=1.0):
        deadline = time.monotonic() + min(float(timeout_sec), self.stationary_observation_remaining())
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            sample = self._check_reference_pose()
            if time.monotonic() >= deadline:break
            lease = self.near_pause.update(sample.base_to_gripper, self.target,
                sample.timestamp, time.monotonic(), enabled=True,
                hardware_ok=sample.raw.get("hardware_ok") == 1)
            if lease:
                self.depth_degraded_commit_events.append(dict(active=True,
                    reason='stationary_reference_fresh_encoder_lease', attempts=attempts,
                    encoder_age_sec=time.monotonic()-float(sample.timestamp),
                    remaining_attempt_sec=self.stationary_observation_remaining()))
                return lease
            # Only an unaccepted transient lease is rechecked. Stops, expired
            # attempts, unhealthy feedback and out-of-range geometry still raise.
            self._assert_clear()
            if self.near_pause.finished or not self.stationary_observation_remaining():
                raise RuntimeError("stationary reference near-panel lease expired")
            time.sleep(min(.02,max(0.,deadline-time.monotonic())))
        raise RuntimeError("stationary reference lease unavailable after bounded recheck")

    def _wait_reference_status(self, *, require_near, after_sequence, timeout_sec=2.0):
        deadline = time.monotonic() + timeout_sec
        last = {}
        while time.monotonic() < deadline:
            self._check_reference_pose()
            last = self.vision.request("safety")["visual_safety"]
            # Do not reset either service or independent stop latches. Ordinary
            # stationary clear frames may naturally retire a coverage latch.
            if (int(last.get("frame_sequence", -1)) > after_sequence and
                    last.get("ready") and not last.get("blocked") and
                    last.get("monitoring_available", False) and
                    (not require_near or
                     (last.get("depth_degraded_commit_armed") and
                      last.get("depth_monitoring_suppressed")))):
                return last
            time.sleep(.02)
        raise RuntimeError("stationary reference safety handoff not ready: " +
                           str(last.get("reason", "no fresh safety frame")))

    @contextmanager
    def stationary_reference_transition(self):
        """Keep the bounded near policy across reference retreat and re-entry.

        Called only after collision validation. No latch reset, no new motion,
        and no lease extension. Other labels retain their existing policy.
        """
        self._check_reference_pose()
        self.supervisor.disarm()
        self.vision.request("safety_motion_state", paused=True)
        # Do not demand ordinary depth coverage in its known near blind zone.
        # Enable the already bounded near policy first, then require a NEW
        # acknowledged near-mode frame. Existing stop/residual latches are
        # neither reset nor ignored by the service or supervisor.
        try:
            with self._depth_degraded_commit_lock:
                self._reference_hold = True
                self._fresh_reference_lease()
                acknowledgement = self.vision.request("safety_set_depth_degraded_commit", enabled=True)
                self.depth_degraded_commit_active = True
            # The barrier is sampled from the enable acknowledgement; do not
            # open a raw-depth processing gap just to obtain a frame number.
            status = (acknowledgement or {}).get("visual_safety") or self.vision.request("safety")["visual_safety"]
            status = self._wait_reference_status(require_near=True,
                        after_sequence=int(status["frame_sequence"]))
            self.depth_degraded_commit_events.append({"active":True,
                "reason":"stationary_reference_continuous_handoff",
                "frame_sequence":status["frame_sequence"],"monotonic":time.monotonic()})
            yield
            # The guarded press has returned to the verified photo pose. Keep
            # this lease across light observation and an optional short retry.
            # It still expires; no timer is renewed by observation or polling.
            self._check_reference_pose()
            self._reference_observation_hold = True
        finally:
            self._reference_hold = False

    def finish_near_observation(self):
        """End the photo/retry allowance before a final, outward return."""
        if self._reference_hold:
            raise RuntimeError("cannot finalize return inside an active press/reference transition")
        held = self._reference_observation_hold
        self._reference_observation_hold = False
        return held

    def prepare(self) -> dict:
        """Run once before any trajectory is submitted.

        preflight resets the independent stop latch, so callers must invoke this
        only after confirming no trajectory is queued or active.
        """
        state = self.robot.state(max_age=2.0)
        self.vision.request(
            "safety_camera_pose",
            timestamp_monotonic=float(state.timestamp),
            base_to_camera=(state.base_to_gripper @ self.handeye).tolist(),
        )
        self.vision.request("safety_motion_state", paused=True)
        self.vision.request("safety_set_mode", mode="predicted_depth_matrix")
        # Building a multi-frame matrix reference outlives the 250 ms camera
        # pose lease. Keep publishing before either preflight or capture.
        self._thread = threading.Thread(target=self._publish_pose, daemon=True)
        self._thread.start()
        first = self.supervisor.preflight(3.0)
        reference = self.vision.request(
            "safety_capture_matrix_reference", frame_count=25
        )
        final = self.supervisor.preflight(3.0)
        self.supervisor.start()
        return {"ring_preflight": first, "matrix_reference": reference, "final_preflight": final}

    def _publish_pose(self) -> None:
        previous_start = None
        while not self.closed.is_set():
            began = time.monotonic()
            evidence = dict(kind='publish', start_monotonic=began,
                            iteration_interval_sec=None if previous_start is None else began-previous_start)
            previous_start = began
            try:
                sample = self.robot.state(max_age=0.25)
                ready = time.monotonic()
                evidence.update(feedback_sample_monotonic=float(sample.timestamp),
                                state_read_sec=ready-began,
                                feedback_age_at_send_sec=ready-float(sample.timestamp),
                                rpc_started_monotonic=ready)
                response = self.vision.request(
                    "safety_camera_pose",
                    timestamp_monotonic=float(sample.timestamp),
                    base_to_camera=(sample.base_to_gripper @ self.handeye).tolist(),
                )
                sent = time.monotonic()
                evidence.update(rpc_completed_monotonic=sent, rpc_sec=sent-ready,
                                service=response.get('pose_timing') if isinstance(response, dict) else None)
                if isinstance(response, dict):
                    evidence['server_dispatch_sec'] = (response.get('server_request_timing') or {}).get('elapsed_sec')
                if self.pause_depth_near_button and self.target is not None:
                    lease_sample = self.robot.state(max_age=0.25)
                    evidence['feedback_age_at_lease_sec'] = time.monotonic()-float(lease_sample.timestamp)
                    pause_lease = self.near_pause.update(
                        lease_sample.base_to_gripper,
                        self.target,
                        lease_sample.timestamp,
                        time.monotonic(),
                        enabled=(self.supervisor.armed.is_set() or self._reference_hold
                                 or self._reference_observation_hold),
                        hardware_ok=lease_sample.raw.get("hardware_ok") == 1,
                    )
                    # Cover the short RPC transition, then resume supervisor
                    # polling. The service ignores matrix coverage loss only;
                    # a real residual or collision remains armed.
                    if pause_lease and not self.depth_degraded_commit_active:
                        self.supervisor.depth_pause_until = pause_lease
                        with self._depth_degraded_commit_lock:
                            if not self.depth_degraded_commit_active:
                                response = self.vision.request(
                                    "safety_set_depth_degraded_commit", enabled=True
                                )
                                self.depth_degraded_commit_active = True
                                self.depth_degraded_commit_events.append(
                                    {"active": True, "response": response,
                                     "monotonic": time.monotonic()}
                                )
                    # Once entered, retain the coverage-only exception through
                    # the complete retreat to the observation pose. It is
                    # disabled by after_segment only after motion is disarmed.
                    if (
                        self.depth_degraded_commit_active
                        and self.near_pause.started is not None
                        and time.monotonic() >= self.near_pause.started + NEAR_BUTTON_DEPTH_LEASE_SEC
                    ):
                        self._deactivate_depth_degraded_commit("lease_expired")
                    self.supervisor.depth_pause_until = 0.0
            except Exception as exc:
                self.publisher_error = f"{type(exc).__name__}: {exc}"
                self.supervisor.depth_pause_until = 0.0
                evidence['error_type'] = type(exc).__name__
            finally:
                evidence['iteration_sec'] = time.monotonic()-began
                self._pose_timing.record(evidence, slow=(
                    bool(evidence.get('error_type')) or evidence['iteration_sec'] > .1
                    or (evidence['iteration_interval_sec'] or 0.) > .1))
            self.closed.wait(0.02)

    def _assert_clear(self) -> None:
        if self.publisher_error:
            raise RuntimeError(f"safety pose publisher failed: {self.publisher_error}")
        if self.supervisor.triggered.is_set() or self.supervisor.stop_active.is_set():
            raise RuntimeError("visual safety stop is active")
        if self.supervisor.error:
            raise RuntimeError(f"visual safety supervisor failed: {self.supervisor.error}")

    def before_segment(self, name: str) -> None:
        self._assert_clear()
        if self._reference_hold:
            status = self.vision.request("safety")["visual_safety"]
            confirmed = self._wait_reference_status(require_near=True,
                            after_sequence=int(status.get("frame_sequence", -1)))
            self.depth_degraded_commit_events.append({"active": True,
                "reason": "near_reentry_confirmed_before_arm", "segment": name,
                "frame_sequence": confirmed["frame_sequence"], "monotonic": time.monotonic()})
        response = self.vision.request("safety_motion_state", paused=False)
        self.supervisor.arm()
        try:
            status = (response or {}).get('visual_safety') or {}
            self._pose_timing.record(dict(kind='segment_armed', segment=name,
                start_monotonic=time.monotonic(), safety_frame_sequence=status.get('frame_sequence'),
                safety_ready=status.get('ready'), safety_blocked=status.get('blocked'),
                safety_reason=status.get('reason'), pose_timing=status.get('pose_timing')))
        except Exception:
            pass

    def disable_for_return(self) -> None:
        """Disable visual cancellation before a prevalidated outward return."""
        self.supervisor.depth_pause_until = 0.0
        self.supervisor.disarm()
        self.vision.request("safety_motion_state", paused=True)
        if (getattr(self, "_reference_hold", False)
                or getattr(self, "_reference_observation_hold", False)):
            self._check_reference_pose()
            return
        self.near_pause.finished = True
        self.near_pause.active = False
        self._deactivate_depth_degraded_commit("return_visual_safety_disabled")

    def after_segment(self, name: str) -> None:
        self._assert_clear()
        if name in {"retreat_to_observation", "observation_to_waiting", "initial_light_view_recheck"}:
            self.supervisor.depth_pause_until = 0.0
            self.supervisor.disarm()
            self.vision.request("safety_motion_state", paused=True)
            if name == "retreat_to_observation":
                # The near policy belongs to one approach/press/retreat
                # cycle. Encoder noise around the chosen standoff must not re-arm
                # it during the subsequent waiting-position return.
                self.near_pause.finished = True
                self.near_pause.active = False
                self._deactivate_depth_degraded_commit("observation_reached")

    def _deactivate_depth_degraded_commit(self, reason: str) -> None:
        with self._depth_degraded_commit_lock:
            if not self.depth_degraded_commit_active:
                return
            response = self.vision.request(
                "safety_set_depth_degraded_commit", enabled=False
            )
            self.depth_degraded_commit_active = False
            self.depth_degraded_commit_events.append(
                {"active": False, "reason": str(reason), "response": response,
                 "monotonic": time.monotonic()}
            )

    def close(self) -> None:
        self.supervisor.depth_pause_until = 0.0
        self.supervisor.disarm()
        self.closed.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.supervisor.close()
        try:
            self._deactivate_depth_degraded_commit("close")
            self.vision.request("safety_motion_state", paused=True)
            keep_reference=(os.environ.get('YOUXIAOMEI_CYCLE_AUTO_RECOVERY')=='1'
                and (self.supervisor.triggered.is_set() or self.supervisor.stop_active.is_set()
                     or self.supervisor.error))
            if keep_reference:
                self.depth_degraded_commit_events.append(dict(active=False,
                    reason='matrix_reference_retained_for_cycle_recovery',monotonic=time.monotonic()))
            else:
                self.vision.request("safety_disable_matrix_reference")
        except Exception:
            # Shutdown is already fail-closed; do not mask the primary result.
            pass

def assemble_button_light_runtime(
    observer: FreshLightObserver,
    executor: DryValidatedSegmentExecutor,
    refreshed_plan: Any,
    move_origin: Any,
    *,
    waiting_position_is_origin: bool,
    press_attempt: Any | None = None,
    hold_at_prepress_if_visible: Any | None = None,
    retreat_off_evidence_directory: Path | None = None,
) -> ButtonLightRuntime:
    """Join the tested state machine to fresh vision and segmented motion.

    ``refreshed_plan`` acquires a new RGB-D target. It normally returns a newly
    generated ``SegmentedPressPlan``, which is dry-validated before motion. It
    may return ``None`` when the first long approach has its own independently
    validated online trajectory; retries must still return a complete plan.
    """

    def refresh() -> None:
        plan = refreshed_plan()
        if plan is not None:
            executor.load_plan(plan)

    post_press_choice_available = False
    monitor_thread: threading.Thread | None = None
    monitor_stop = threading.Event()
    monitor_lock = threading.Lock()
    monitor_result: dict[str, Any] = {}
    monitor_started_by_executor = False

    reference_thread: threading.Thread | None = None
    reference_stop = threading.Event()
    reference_errors: list[str] = []

    def stop_reference_collection() -> None:
        nonlocal reference_thread
        reference_stop.set()
        if reference_thread is not None:
            reference_thread.join(timeout=2.5)
            if reference_thread.is_alive():
                raise RuntimeError("before-press reference collector did not stop")
            reference_thread = None

    def start_reference_collection() -> None:
        nonlocal reference_thread
        paired = getattr(observer, "paired_light", None)
        if paired is None:
            return
        paired.begin_attempt()
        if paired.require_stationary_reference:
            return
        reference_stop.clear()
        reference_errors.clear()

        def collect() -> None:
            while not reference_stop.is_set():
                try:
                    observer.observe("before_press", timeout_sec=.20, emit_event=False)
                except Exception as exc:
                    paired.reset_pending()
                    reference_errors.append(f"{type(exc).__name__}: {exc}")
                    del reference_errors[:-3]
                    reference_stop.wait(.01)
        reference_thread = threading.Thread(target=collect, name="pre-press-off-reference", daemon=True)
        reference_thread.start()

    def start_press_monitor() -> None:
        nonlocal monitor_thread, monitor_result
        stop_reference_collection()
        paired = getattr(observer, "paired_light", None)
        if paired is not None:
            # Normally collected concurrently with approach/plan validation.
            # A very short approach may need two fresh stationary samples.
            if paired.require_stationary_reference and not paired.stationary_reference_ready:
                raise RuntimeError("fixed observation OFF reference missing; press withheld")
            if not paired.references:
                for _ in range(2):
                    observer.observe("before_press", timeout_sec=.20, emit_event=False)
            if not paired.references:
                raise RuntimeError("no confirmed OFF reference before press; inspect button view")
            paired.freeze()
            if observer.event_sink is not None:
                observer.event_sink("press_off_reference_frozen", {
                    "reference_count": len(paired.references),
                    "last_sequence": paired.last_reference_sequence,
                    "references": [{"sequence": p.sequence, "captured": p.captured,
                                    "region_source": p.region_source,
                                    "bbox": p.box.tolist(), "image": p.image,
                                    "crop": p.evidence} for p in paired.references],
                    "collection_errors": list(reference_errors),
                })
        sampler = getattr(observer, "observe_during_motion", None)
        monitor_stop.clear()
        with monitor_lock:
            monitor_result = {
                "supported": callable(sampler),
                "samples": 0,
                "on_streak": 0,
                "confirmed_state": None,
                "evidence": [],
                "errors": [],
            }
        if not callable(sampler):
            monitor_thread = None
            return

        def monitor() -> None:
            while not monitor_stop.is_set():
                try:
                    state = LightState(sampler())
                    entry = observer.audit[-1] if observer.audit else {}
                except TimeoutError:
                    with monitor_lock:
                        monitor_result["on_streak"] = 0
                        monitor_result["evidence"] = []
                    continue
                except Exception as exc:
                    with monitor_lock:
                        monitor_result["on_streak"] = 0
                        monitor_result["evidence"] = []
                        monitor_result["errors"].append(
                            f"{type(exc).__name__}: {exc}"
                        )
                        monitor_result["errors"] = monitor_result["errors"][-3:]
                    time.sleep(0.01)
                    continue
                with monitor_lock:
                    monitor_result["samples"] += 1
                    if state is LightState.ON:
                        monitor_result["on_streak"] += 1
                        monitor_result["evidence"].append({
                            "frame_sequence": entry.get("frame_sequence"),
                            "capture_time_unix_sec": entry.get(
                                "capture_time_unix_sec"
                            ),
                            "state": state.value,
                            "image": entry.get("image"),
                            "opencv_on_priority": (entry.get('recognition') or {}).get('opencv_on_priority'),
                        })
                        monitor_result["evidence"] = monitor_result["evidence"][-2:]
                        if (
                            monitor_result["on_streak"]
                            >= POST_PRESS_ON_CONFIRMATION_FRAMES
                        ):
                            monitor_result["confirmed_state"] = LightState.ON.value
                            return
                    else:
                        monitor_result["on_streak"] = 0
                        monitor_result["evidence"] = []

        monitor_thread = threading.Thread(
            target=monitor, name="post-press-light-monitor", daemon=True
        )
        monitor_thread.start()

    def stop_press_monitor() -> dict[str, Any]:
        nonlocal monitor_thread
        monitor_stop.set()
        if monitor_thread is not None:
            # The RPC client itself has a two-second transport timeout. Normal
            # shutdown takes less than one camera period; allow the transport
            # bound so a failed request cannot leave a concurrent consumer of
            # ``last_sequence`` behind.
            monitor_thread.join(timeout=2.5)
            if monitor_thread.is_alive():
                raise RuntimeError("post-press light monitor did not stop")
        monitor_thread = None
        with monitor_lock:
            return {
                **monitor_result,
                "evidence": list(monitor_result.get("evidence", [])),
                "errors": list(monitor_result.get("errors", [])),
            }

    def confirmed_monitor_result(result: dict[str, Any]) -> dict | None:
        if result.get("confirmed_state") != LightState.ON.value:
            return None
        return {
            "position": "press_monitor",
            "confirmed_state": LightState.ON.value,
            "evidence": {
                "source": "two_consecutive_fresh_frames_during_press_or_retreat",
                "samples": result.get("samples"),
                "frames": result.get("evidence"),
                "errors": result.get("errors"),
                "opencv_adopted": any(frame.get('opencv_on_priority')
                    for frame in result.get('evidence', [])),
            },
        }

    set_press_hook = getattr(executor, "set_before_guarded_press_callback", None)
    if callable(set_press_hook):
        set_press_hook(start_press_monitor)
        monitor_started_by_executor = True

    def press(attempt: int) -> Any:
        nonlocal post_press_choice_available
        callback = (
            (lambda: press_attempt(attempt))
            if press_attempt is not None
            else executor.approach_and_press
        )
        start_reference_collection()
        try:
            if not monitor_started_by_executor:
                # Test/dry adapters without a guarded boundary.
                start_press_monitor()
            result = callback()
        except BaseException:
            stop_reference_collection()
            stop_press_monitor()
            raise
        post_press_choice_available = True
        return result

    def move_near(distance_m: float) -> str | dict:
        nonlocal post_press_choice_available
        if post_press_choice_available:
            post_press_choice_available = False
            with monitor_lock:
                early = confirmed_monitor_result(dict(monitor_result))
            if early is not None:
                stop_press_monitor()
                return early
            visible_at_prepress = (
                hold_at_prepress_if_visible is not None
                and bool(hold_at_prepress_if_visible())
            )
            # The second frame may have arrived while the visibility RPC was
            # in flight. Recheck before authorizing any avoidable retreat.
            with monitor_lock:
                early = confirmed_monitor_result(dict(monitor_result))
            if early is not None:
                stop_press_monitor()
                return early
            if visible_at_prepress:
                result = stop_press_monitor()
                early = confirmed_monitor_result(result)
                if early is not None:
                    return early
                return "prepress"
            # Keep the monitor active while following the already validated
            # retreat. It can prove OFF->ON before the endpoint is reached.
            executor.retreat_to_observation(distance_m)
            result = stop_press_monitor()
            early = confirmed_monitor_result(result)
            if early is not None:
                return early
            return "near"
        executor.retreat_to_observation(distance_m)
        return "near"

    def move_waiting():
        if retreat_off_evidence_directory is None:
            return executor.return_to_waiting()
        # This callback is reached only after the flow confirmed press ON.
        # The previous press-ON collector has been joined by move_near().
        monitor = RetreatOffMonitor(observer, retreat_off_evidence_directory)
        monitor.start()
        try:
            verification = executor.return_to_waiting()
        except BaseException:
            # Preserve the motion/stop error; a visual OFF can never override it.
            try:
                monitor.finish()
            except Exception:
                pass
            raise
        result = monitor.finish()
        if observer.event_sink is not None:
            observer.event_sink("retreat_off_monitor_finished", result)
        return {**(verification or {}), "retreat_off_confirmation": result}

    return ButtonLightRuntime(
        observe=observer.observe,
        post_press_recheck=getattr(observer, "reobserve_post_press", None),
        post_press_opencv=getattr(observer, "post_press_opencv_result", None),
        next_frame=observer.wait_for_next_frame,
        refresh_target=refresh,
        press=press,
        move_near=move_near,
        move_waiting=move_waiting,
        move_origin=move_origin,
        waiting_position_is_origin=bool(waiting_position_is_origin),
        initial_recheck=getattr(observer, "recheck_initial_observation", None),
        initial_finished=getattr(observer, "finish_initial_observation", None),
    )
