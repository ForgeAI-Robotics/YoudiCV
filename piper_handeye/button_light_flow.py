"""Button press verification state machine independent of robot and camera I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Protocol
from .retreat_off_monitor import confirmed_retreat_off


class LightState(str, Enum):
    ON = "ON"
    OFF = "OFF"
    UNCERTAIN = "UNCERTAIN"


PRODUCTION_RETRY_STANDOFF_M = 0.05
RETRY_PREPARATION_DISTANCE_M = 0.08
CALL_OFF_TIMEOUT_FLOW = "external_call_off_timeout_after_verified_press"
NUMERIC_CYCLE_ON_TIMEOUT_SEC = 10.0
NUMERIC_CYCLE_ON_TIMEOUT_FLOW = "numeric_cycle_on_timeout_after_verified_press"
NUMERIC_CYCLE_INITIAL_ON_TIMEOUT_FLOW = "numeric_cycle_initial_on_timeout_skip"
NUMERIC_CYCLE_TIMEOUT_FLOWS = frozenset({NUMERIC_CYCLE_ON_TIMEOUT_FLOW,
                                       NUMERIC_CYCLE_INITIAL_ON_TIMEOUT_FLOW})


class ButtonFlowHardware(Protocol):
    """Small production boundary implemented by the robot/vision runtime."""

    waiting_position_is_origin: bool

    def observe_light(self, position: str) -> LightState: ...
    def wait_for_next_observation(self) -> None: ...
    def refresh_button_target(self) -> None: ...
    def press_button(self, attempt: int) -> Any: ...
    def move_to_near_observation(self, panel_distance_m: float) -> str | dict | None: ...
    def move_to_waiting_observation(self) -> dict | None: ...
    def move_to_origin(self) -> dict | None: ...


@dataclass(frozen=True)
class ButtonFlowConfig:
    near_observation_distance_m: float = PRODUCTION_RETRY_STANDOFF_M
    max_press_attempts: int = 3
    max_uncertain_observations: int = 3
    waiting_heartbeat_sec: float = 10.0
    initial_confirmation_frames: int = 3
    off_confirmation_frames: int = 3
    initial_max_frames: int = 12
    target_label: str | None = None
    external_call_off_wait_timeout_sec: float | None = None
    cycle_numeric_on_timeout_sec: float | None = None
    retry_observation_distance_m: float | None = None

    def __post_init__(self) -> None:
        if not 0.04 <= self.near_observation_distance_m <= 0.20:
            raise ValueError("near observation distance must be in 0.04..0.20 m")
        if self.retry_observation_distance_m is not None and (
                not math.isfinite(self.retry_observation_distance_m)
                or not .04<=self.retry_observation_distance_m<=.20):
            raise ValueError('retry observation distance must be in 0.04..0.20 m')
        if not 1 <= self.max_press_attempts <= 3:
            raise ValueError("max press attempts must be in 1..3")
        if self.max_uncertain_observations < 1:
            raise ValueError("max uncertain observations must be positive")
        if self.waiting_heartbeat_sec <= 0:
            raise ValueError("waiting heartbeat interval must be positive")
        if self.initial_confirmation_frames < 2 or self.off_confirmation_frames < 2:
            raise ValueError("initial and arrival confirmation require multiple frames")
        if self.initial_max_frames < self.initial_confirmation_frames:
            raise ValueError("initial frame budget is too small")
        timeout = self.external_call_off_wait_timeout_sec
        if timeout is not None and (
            self.target_label not in {"up", "down"}
            or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("external call timeout requires up/down and positive finite seconds")
        numeric_timeout = self.cycle_numeric_on_timeout_sec
        if numeric_timeout is not None and (
            self.target_label not in {str(i) for i in range(1, 17)}
            or type(numeric_timeout) not in (int, float)
            or not math.isfinite(numeric_timeout) or numeric_timeout != NUMERIC_CYCLE_ON_TIMEOUT_SEC
        ):
            raise ValueError("numeric cycle ON timeout requires button 1..16 and exactly 10 seconds")


def production_flow_config(target_label, config, *, max_press_attempts=3, cycle_numeric_wait=False):
    """Keep mission rules; only explicit numeric cycle context adds the ON skip."""
    label = str(target_label)
    timeout = ((config.get("external_call_wait") or {}).get("post_press_off_timeout_sec")
               if label in {"up", "down"} else None)
    return ButtonFlowConfig(max_press_attempts=max_press_attempts, target_label=label,
                           retry_observation_distance_m=RETRY_PREPARATION_DISTANCE_M,
                           external_call_off_wait_timeout_sec=timeout,
                           cycle_numeric_on_timeout_sec=(NUMERIC_CYCLE_ON_TIMEOUT_SEC
                               if cycle_numeric_wait is True and label in {str(i) for i in range(1,17)} else None))


@dataclass
class ButtonFlowResult:
    status: str
    press_attempts: int
    events: list[dict] = field(default_factory=list)


class ButtonLightFlow:
    def __init__(self, hardware: ButtonFlowHardware, config: ButtonFlowConfig | None = None,
                 event_sink=None):
        self.hardware = hardware
        self.config = config or ButtonFlowConfig()
        self.events: list[dict] = []
        self.event_sink = event_sink

    def _record(self, event: str, **values) -> None:
        value = {"event": event, **values}
        self.events.append(value)
        if self.event_sink is not None:
            self.event_sink("flow_event", value)

    @staticmethod
    def _press_outcome(value: Any) -> dict:
        """Reduce an adapter result to stable facts suitable for the flow log."""
        candidates = value if isinstance(value, list) else [value]
        event = next(
            (
                item for item in reversed(candidates)
                if isinstance(item, dict) and item.get("segment") == "guarded_press"
            ),
            value if isinstance(value, dict) else None,
        )
        if not isinstance(event, dict):
            return {"outcome_reported": False}
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        return {
            "outcome_reported": True,
            "bridge_status": result.get("status"),
            "bridge_reason": result.get("reason"),
            "feedback_contact_confirmed": result.get("feedback_contact_confirmed"),
            "encoder_error_rad": event.get("encoder_error_rad"),
        }

    def _observe_decisive(self, position: str) -> LightState | None:
        uncertain = 0
        while uncertain < self.config.max_uncertain_observations:
            state = LightState(self.hardware.observe_light(position))
            self._record("light_observed", position=position, state=state.value)
            if state is not LightState.UNCERTAIN:
                return state
            uncertain += 1
            if uncertain < self.config.max_uncertain_observations:
                self.hardware.wait_for_next_observation()
        return None

    def _observe_post_press_decisive(self, position: str):
        """Adopt current OpenCV ON immediately unless comparison says OFF.

        The observer owns single-image provenance and independent robot health.
        Initial observation, arrival OFF and the moving two-frame count remain
        on their original paths. No former ON can replace a newer invalid frame.
        """
        callback = getattr(self.hardware, 'post_press_opencv_result', None)
        for index in range(self.config.max_uncertain_observations):
            state = LightState(self.hardware.observe_light(position))
            self._record('light_observed', position=position, state=state.value)
            if state is LightState.OFF:
                return state, None
            evidence = callback(position) if callable(callback) else None
            if isinstance(evidence, dict) and evidence.get('state') == 'ON':
                return LightState.ON, {**evidence,
                    'policy': 'opencv_on_priority_unless_paired_off_v1',
                    'uncertain_frames': index+1 if state is LightState.UNCERTAIN else 0,
                    'source': 'opencv_immediate_on_without_paired_off'}
            if state is LightState.ON:
                return state, None
            if index+1 < self.config.max_uncertain_observations:
                self.hardware.wait_for_next_observation()
        return None, None

    def _return_origin(self) -> None:
        verification = self.hardware.move_to_origin()
        values = {"verification": verification} if isinstance(verification, dict) else {}
        self._record("at_origin", **values)

    def _observe_initial(self):
        previous=None;streak=0;uncertain=0
        for index in range(self.config.initial_max_frames):
            if index:self.hardware.wait_for_next_observation()
            state=LightState(self.hardware.observe_light("waiting"))
            self._record("light_observed",position="waiting",state=state.value)
            if state is LightState.UNCERTAIN:
                previous=None;streak=0;uncertain+=1
                if uncertain>=self.config.max_uncertain_observations:return None
                continue
            uncertain=0;streak=streak+1 if state is previous else 1;previous=state
            if streak>=self.config.initial_confirmation_frames:
                self._record("initial_state_confirmed",state=state.value,frames=streak)
                return state
        self._record("initial_state_unstable",frames=self.config.initial_max_frames)
        return None

    def _wait_until_off(self, position: str, *, verified_press_on_at=None) -> bool:
        """Observe a known-active button until a decisive OFF is measured.

        Once ON has been confirmed, UNCERTAIN is not evidence that the lift
        arrived. By default wait indefinitely. The configured call-button
        demonstration exception requires this task's verified press, and
        reports timeout release separately from measured OFF. Numeric cycle
        tests may additionally skip a continuously ON button after ten seconds;
        uncertain/OFF observations break that timer and it never fabricates OFF.
        """
        started = time.monotonic()
        last_heartbeat = started
        off_streak = 0
        # The entry state has already been confirmed ON. Missing/uncertain or
        # OFF observations reset this continuous-ON timer; they are not ON.
        on_since = started
        while True:
            self.hardware.wait_for_next_observation()
            state = LightState(self.hardware.observe_light(position))
            self._record("light_observed",position=position,state=state.value)
            if state is LightState.OFF:
                off_streak += 1
                if off_streak >= self.config.off_confirmation_frames:
                    self._record("button_became_off", position=position, frames=off_streak)
                    return True
            else:
                off_streak = 0
            now = time.monotonic()
            if state is LightState.ON:
                if on_since is None:
                    on_since = now
                numeric_timeout = self.config.cycle_numeric_on_timeout_sec
                if numeric_timeout is not None and now-on_since >= numeric_timeout:
                    self._record("numeric_cycle_on_wait_timeout",
                        target_label=self.config.target_label, timeout_sec=numeric_timeout,
                        elapsed_continuous_on_sec=now-on_since,
                        last_observed_state=state.value, off_confirmed=False,
                        after_verified_press=verified_press_on_at is not None,
                        release_policy="cycle_numeric_on_timeout_skip")
                    return False
            else:
                on_since = None
            timeout = self.config.external_call_off_wait_timeout_sec
            if (timeout is not None and verified_press_on_at is not None
                    and time.monotonic() - verified_press_on_at >= timeout):
                self._record(
                    "external_call_off_wait_timeout", target_label=self.config.target_label,
                    timeout_sec=timeout, elapsed_sec=time.monotonic()-verified_press_on_at,
                    last_observed_state=state.value, consecutive_off_frames=off_streak,
                    off_confirmed=False, release_policy="mock_scene_timeout_release",
                )
                return False
            if state is LightState.UNCERTAIN:
                self._record("waiting_observation_uncertain", position=position)
                now = time.monotonic()
                if now - last_heartbeat >= self.config.waiting_heartbeat_sec:
                    self._record(
                        "waiting_button_still_on",
                        position=position,
                        elapsed_sec=now - started,
                        observation_uncertain=True,
                    )
                    last_heartbeat = now
                continue
            now = time.monotonic()
            if now - last_heartbeat >= self.config.waiting_heartbeat_sec:
                self._record(
                    "waiting_button_still_on",
                    position=position,
                    elapsed_sec=now - started,
                )
                last_heartbeat = now

    def run(self) -> ButtonFlowResult:
        try:
            recovery = getattr(self.hardware, "initial_observation", None)
            waiting_state = (LightState(recovery(self._record)) if callable(recovery)
                             else self._observe_initial())
            for recheck in range(1, 3) if not callable(recovery) else ():
                if waiting_state is not None:
                    break
                callback = getattr(self.hardware, "recheck_initial_observation", None)
                if not callable(callback) or not callback():
                    break
                self._record("initial_light_recheck", attempt=recheck)
                # A new view starts a new streak; never mix confirmation frames
                # from two camera poses or carry an ON across the movement.
                waiting_state = self._observe_initial()
        finally:
            finished = getattr(self.hardware, "finish_initial_observation", None)
            if callable(finished):
                finished()
        if waiting_state is None:
            self._record("initial_light_unresolved", reason="灯态无法可靠确认，未执行按压")
            self._return_origin()
            return ButtonFlowResult("initial_observation_uncertain", 0, self.events)

        if waiting_state is LightState.ON:
            self._record("button_already_on",frames=self.config.initial_confirmation_frames,press_performed=False)
            arrived = self._wait_until_off("waiting")
            self._return_origin()
            status = ("arrived_after_initial_on" if arrived else
                      NUMERIC_CYCLE_INITIAL_ON_TIMEOUT_FLOW if self.config.cycle_numeric_on_timeout_sec is not None
                      else "arrival_observation_uncertain")
            return ButtonFlowResult(status, 0, self.events)

        for attempt in range(1, self.config.max_press_attempts + 1):
            observation_distance_m = (self.config.retry_observation_distance_m
                if attempt > 1 and self.config.retry_observation_distance_m is not None
                else self.config.near_observation_distance_m)
            # Refresh even on attempt one: the waiting observation can finish while
            # the robot is moving, and every retry must correct an inaccurate aim.
            try:
                self.hardware.refresh_button_target()
            except Exception as exc:
                self._record(
                    "button_target_refresh_failed", attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
                try:
                    self._return_origin()
                except Exception as return_exc:
                    raise RuntimeError(
                        "button target refresh failed and origin recovery failed: "
                        f"{type(return_exc).__name__}: {return_exc}"
                    ) from exc
                raise
            self._record("button_target_refreshed", attempt=attempt)
            press_outcome = self.hardware.press_button(attempt)
            press_feedback = self._press_outcome(press_outcome)
            self._record(
                "press_attempt_finished", attempt=attempt,
                **press_feedback,
            )
            if press_feedback.get("bridge_reason") == "sustained_cartesian_tangential_force_kalman":
                self._record("press_tangential_protection", attempt=attempt,
                             light_confirmation_required=True)
            observation_move = self.hardware.move_to_near_observation(
                observation_distance_m
            )
            early_state = None
            early_evidence = None
            if isinstance(observation_move, dict):
                observation_position = str(observation_move.get("position") or "near")
                early_state_value = observation_move.get("confirmed_state")
                if early_state_value is not None:
                    early_state = LightState(early_state_value)
                early_evidence = observation_move.get("evidence")
            else:
                observation_position = observation_move or "near"
            self._record(
                "at_post_press_observation",
                attempt=attempt,
                position=observation_position,
                requested_panel_distance_m=(
                    observation_distance_m
                    if observation_position == "near" else None
                ),
            )
            confirmation_on_seen = False
            confirmation_source = 'paired_two_effective_observations'

            def confirmed_stationary_state(position: str):
                nonlocal confirmation_on_seen, confirmation_source
                first, adopted = self._observe_post_press_decisive(position)
                confirmation_on_seen = confirmation_on_seen or first is LightState.ON
                if adopted is not None:
                    confirmation_source = adopted.get('source', 'opencv_immediate_on_without_paired_off')
                    self._record('post_press_opencv_adopted', attempt=attempt,
                        position=position, state='ON', evidence=adopted)
                    return LightState.ON, None
                if first is LightState.ON:
                    # move_to_near_observation has completed its endpoint check
                    # and joined the motion monitor. A qualified paired ON at
                    # this stationary pose is sufficient; only OFF needs a pair
                    # before another press may be attempted.
                    confirmation_source = 'paired_single_stationary_observation'
                    return LightState.ON, None
                if first is None:
                    return None, "uncertain"
                self.hardware.wait_for_next_observation()
                second, adopted = self._observe_post_press_decisive(position + "_confirmation")
                confirmation_on_seen = confirmation_on_seen or second is LightState.ON
                if adopted is not None:
                    confirmation_source = adopted.get('source', 'opencv_immediate_on_without_paired_off')
                    self._record('post_press_opencv_adopted', attempt=attempt,
                        position=position+'_confirmation', state='ON', evidence=adopted)
                    return LightState.ON, None
                if second is LightState.ON:
                    confirmation_source = 'paired_single_stationary_observation'
                    return LightState.ON, None
                if second is None:
                    return None, "uncertain"
                if second is not first:
                    return None, "disagreement"
                return second, None

            if early_state is LightState.ON:
                state, observation_error = LightState.ON, None
                confirmation_source = 'paired_two_frame_motion_monitor'
                if isinstance(early_evidence, dict) and early_evidence.get('opencv_adopted'):
                    confirmation_source = 'opencv_on_priority_two_frame_motion_monitor'
                    self._record('post_press_opencv_adopted', attempt=attempt,
                        position=observation_position, state='ON', evidence=early_evidence)
                self._record(
                    "post_press_on_captured_early",
                    attempt=attempt,
                    position=observation_position,
                    evidence=early_evidence,
                )
            else:
                state, observation_error = confirmed_stationary_state(observation_position)
            if observation_error is not None and observation_position != "near":
                self._record(
                    "post_press_observation_fallback",
                    attempt=attempt,
                    from_position=observation_position,
                    reason=observation_error,
                )
                observation_position = self.hardware.move_to_near_observation(
                    observation_distance_m
                ) or "near"
                self._record(
                    "at_post_press_observation",
                    attempt=attempt,
                    position=observation_position,
                    requested_panel_distance_m=observation_distance_m,
                )
                state, observation_error = confirmed_stationary_state(observation_position)
            if (observation_error == "uncertain" and observation_position == "near"
                    and not confirmation_on_seen):
                recheck = getattr(self.hardware, "reobserve_post_press", None)
                if callable(recheck):
                    self._record("post_press_recheck_started", attempt=attempt,
                                 position=observation_position)
                    recovery = recheck(observation_position)
                    self._record("post_press_recheck_finished", attempt=attempt,
                                 position=observation_position, evidence=recovery)
                    if (isinstance(recovery, dict)
                            and recovery.get("confirmed_state") in {"ON", "OFF"}):
                        state = LightState(recovery["confirmed_state"])
                        observation_error = None
                        confirmation_source = recovery.get('source', 'stationary_recheck')
                        if recovery.get('opencv_adopted') is True:
                            self._record('post_press_opencv_adopted', attempt=attempt,
                                position=observation_position, state=state.value, evidence=recovery)
            if observation_error is not None:
                self._record(
                    "post_press_confirmation_" + observation_error,
                    attempt=attempt,
                    position=observation_position,
                )
                self._return_origin()
                return ButtonFlowResult(
                    "post_press_confirmation_" + observation_error,
                    attempt,
                    self.events,
                )
            self._record(
                "post_press_state_confirmed", attempt=attempt,
                position=observation_position, state=state.value,
                confirmation_source=confirmation_source,
            )
            if state is LightState.OFF:
                if attempt < self.config.max_press_attempts:
                    if observation_position != "near":
                        observation_position = self.hardware.move_to_near_observation(
                            observation_distance_m
                        ) or "near"
                        self._record(
                            "retry_moved_to_near_observation",
                            attempt=attempt,
                            position=observation_position,
                        )
                    # The next refresh prepares its configured retry photo pose
                    # before localization; the current attempt is still held here.
                    self._record(
                        "retry_held_at_near_observation", attempt=attempt,
                        requested_panel_distance_m=observation_distance_m,
                    )
                    self._record("retry_required", attempt=attempt,
                        next_retry_panel_distance_m=(self.config.retry_observation_distance_m
                            or observation_distance_m))
                    continue
                self._return_origin()
                return ButtonFlowResult("button_not_lit_after_max_attempts", attempt, self.events)

            verified_press_on_at = time.monotonic()
            self._record("press_verified_on", attempt=attempt,
                         confirmation_source=confirmation_source)
            # Leave the post-press confirmation pose before the potentially
            # long ON->OFF wait. For a target visible at origin this moves to
            # origin; otherwise it moves to the raised waiting observation
            # pose.  The callback verifies the commanded endpoint before the
            # state machine records that it is there.
            waiting_verification = self.hardware.move_to_waiting_observation()
            waiting_values = (
                {"motion_verification": waiting_verification}
                if isinstance(waiting_verification, dict) else {}
            )
            self._record("at_waiting_observation", **waiting_values)
            retreat_off = (waiting_verification or {}).get("retreat_off_confirmation")
            if confirmed_retreat_off(retreat_off, self.config.target_label,
                                     frames=self.config.off_confirmation_frames):
                self._record("button_became_off", position="waiting_retreat",
                             frames=self.config.off_confirmation_frames, evidence=retreat_off)
                arrived = True
            else:
                arrived = self._wait_until_off("waiting", verified_press_on_at=verified_press_on_at)
            self._return_origin()
            status = ("arrived_after_verified_press" if arrived else
                      NUMERIC_CYCLE_ON_TIMEOUT_FLOW if self.config.cycle_numeric_on_timeout_sec is not None
                      else CALL_OFF_TIMEOUT_FLOW)
            return ButtonFlowResult(status, attempt, self.events)

        raise AssertionError("unreachable")
