"""Recover initial call-button observations without guessing a lamp state.

Only the initial stationary observation uses this policy. Contact, retry,
post-press confirmation and arrival confirmation keep their own policies.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .button_light_flow import LightState
from .kinematics import forward_kinematics
from .robot import PIPER_JOINT_LIMITS


@dataclass(frozen=True)
class InitialRecoveryPolicy:
    view_budget_sec: float = 6.0
    max_view_attempts: int = 2
    pending_trigger_frames: int = 3
    confirmation_frames: int = 3
    hold_interval_sec: float = 1.0
    heartbeat_sec: float = 10.0

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in
                    (self.view_budget_sec, self.hold_interval_sec, self.heartbeat_sec))
                or not 0 <= self.max_view_attempts <= 2
                or self.pending_trigger_frames < 1 or self.confirmation_frames < 2):
            raise ValueError("invalid initial observation recovery policy")


class InitialObservationRecovery:
    """Time-based observation, at most two wrist looks, then stationary wait.

    observe() must enforce fresh, same-frame, qualified target crops. A missing
    RGB stream, stop, robot error or failed motion is an exception, not another
    UNCERTAIN vote. Every view change and UNCERTAIN resets confirmation.
    """
    def __init__(self, observe, check_hold, recheck_view, reason, *, confirm_target=None, policy=None,
                 clock=time.monotonic, sleep=time.sleep):
        self.observe = observe
        self.check_hold = check_hold
        self.recheck_view = recheck_view
        self.reason = reason
        self.confirm_target = confirm_target or (lambda: True)
        self.policy = policy or InitialRecoveryPolicy()
        self.clock, self.sleep = clock, sleep

    def run(self, record):
        policy = self.policy
        deadline = self.clock() + policy.view_budget_sec
        attempts = 0
        previous, streak, uncertain = None, 0, 0
        holding = False
        announced_pending = False
        last_heartbeat = self.clock()
        samples = 0
        last_reported_state = None
        while True:
            self.check_hold()
            state = LightState(self.observe())
            samples += 1
            if not holding or state != last_reported_state:
                record("light_observed", position="waiting", state=state.value)
                last_reported_state = state
            if state is LightState.UNCERTAIN:
                previous, streak = None, 0
                uncertain += 1
                if uncertain >= policy.pending_trigger_frames and not announced_pending:
                    record("initial_observation_reacquiring", reason=self.reason(),
                           uncertain_frames=uncertain, view_budget_sec=policy.view_budget_sec)
                    announced_pending = True
            else:
                uncertain = 0
                streak = streak + 1 if state is previous else 1
                previous = state
                if streak >= policy.confirmation_frames:
                    confirmed = self.confirm_target()
                    if confirmed is not True:
                        previous, streak = None, 0
                        if confirmed == "resample":
                            # Fresh RGB-D reacquisition finished at this same
                            # pose. Confirm the lamp again after its time barrier.
                            deadline = self.clock() + policy.view_budget_sec
                            continue
                    else:
                        if holding or attempts or announced_pending:
                            record("initial_observation_recovered", state=state.value,
                                   view_attempts=attempts, samples=samples)
                        record("initial_state_confirmed", state=state.value, frames=streak)
                        return state
            now = self.clock()
            if not holding and now >= deadline:
                previous, streak, uncertain = None, 0, 0
                if attempts < policy.max_view_attempts:
                    attempts += 1  # A rejected/unavailable look also consumes its attempt.
                    record("initial_view_search_started", attempt=attempts, reason=self.reason())
                    moved = bool(self.recheck_view(attempts))
                    record("initial_view_search_finished", attempt=attempts, moved=moved)
                    self.check_hold()
                    deadline = self.clock() + policy.view_budget_sec
                    announced_pending = False
                    continue
                holding = True
                last_heartbeat = now
                record("initial_observation_waiting", reason=self.reason(),
                       interval_sec=policy.hold_interval_sec, view_attempts=attempts)
            elif holding and now - last_heartbeat >= policy.heartbeat_sec:
                record("initial_observation_waiting", reason=self.reason(),
                       interval_sec=policy.hold_interval_sec, view_attempts=attempts,
                       samples=samples)
                last_heartbeat = now
            # Once a valid state reappears, collect the remaining confirmation
            # frames immediately. Never stitch an earlier ON across a gap.
            if holding and (state is LightState.UNCERTAIN or streak == 0):
                self.sleep(policy.hold_interval_sec)


def choose_initial_call_search_joints(current, original, handeye, panel,
                                      matrix, calibration_size, image_size,
                                      *, target_label, anchor_label):
    """Select one J5-only <=0.025 rad look, total offset <=0.05 rad.

    A fresh detected anchor supplies the safety plane, never the missing
    target's position. Geometric checks keep the far observation far; callers
    still must validate both motion and origin return with MoveIt.
    """
    q, start = np.asarray(current, float), np.asarray(original, float)
    handeye, panel = np.asarray(handeye, float), np.asarray(panel, float)
    k = np.asarray(matrix, float).copy()
    width, height = map(float, image_size)
    cw, ch = map(float, calibration_size)
    if (target_label not in {"up", "down"} or anchor_label not in {"up", "down"}
            or q.shape != (6,) or start.shape != (6,) or panel.shape != (4, 4)
            or handeye.shape != (4, 4) or k.shape != (3, 3)
            or min(width, height, cw, ch) <= 0
            or not np.isfinite(np.r_[q, start, panel.ravel(), handeye.ravel(),
                                     k.ravel(), width, height, cw, ch]).all()):
        raise ValueError("invalid initial call search geometry")
    if np.max(np.abs(np.delete(q-start, 4))) > .003:
        raise RuntimeError("initial search moved outside its saved wrist-only pose")
    k[0] *= width/cw
    k[1] *= height/ch
    point, normal = np.r_[panel[:3, 3], 1.], panel[:3, 2]
    if not .99 <= np.linalg.norm(normal) <= 1.01:
        raise ValueError("invalid search anchor panel normal")
    tcp = forward_kinematics(q)[:3, 3]
    side = float((tcp-point[:3]) @ normal)
    if abs(side) < .15:
        return None, {"reason": "initial_search_requires_15cm_panel_clearance"}

    def project(joints):
        xyz = (np.linalg.inv(forward_kinematics(joints) @ handeye) @ point)[:3]
        if not np.isfinite(xyz).all() or xyz[2] <= .15:
            return None
        uv = k @ xyz
        return uv[:2]/uv[2]

    initial = project(q)
    if initial is None:
        return None, {"reason": "search_anchor_not_in_front_of_camera"}
    desired_y = height * (.65 if target_label == "up" and anchor_label == "down" else .35)

    def cost(uv):
        return abs(float(uv[1])-desired_y) + .1*abs(float(uv[0])-width*.5)

    candidates = []
    for delta in (-.025, .025):
        goal = q.copy(); goal[4] += delta
        if (abs(goal[4]-start[4]) > .050001
                or abs(goal[4]-start[4]) <= abs(q[4]-start[4]) + .001):
            continue
        valid = True
        for t in np.linspace(0, 1, 11):
            sample = q + t*(goal-q)
            if (np.any(sample < PIPER_JOINT_LIMITS[:, 0])
                    or np.any(sample > PIPER_JOINT_LIMITS[:, 1])):
                valid = False; break
            position = forward_kinematics(sample)[:3, 3]
            distance = float((position-point[:3]) @ normal) * np.sign(side)
            pixel = project(sample)
            if (distance < max(.15, abs(side)-.002)
                    or np.linalg.norm(position-tcp) > .01 or pixel is None
                    or not (0 <= pixel[0] <= width and 0 <= pixel[1] <= height)):
                valid = False; break
        after = project(goal)
        if valid and cost(after) < cost(initial)-.5:
            candidates.append((cost(after), goal, delta, after))
    if not candidates:
        return None, {"reason": "no_safe_improving_initial_call_view"}
    _, goal, delta, after = min(candidates, key=lambda item: item[0])
    return goal, {"delta_j5_rad": delta,
                  "total_j5_offset_rad": float(goal[4]-start[4]),
                  "anchor_label": anchor_label, "anchor_is_search_reference_only": True,
                  "anchor_pixel_before": initial.tolist(),
                  "predicted_anchor_pixel_after": after.tolist(),
                  "initial_panel_clearance_m": abs(side)}


def recheck_initial_call_view(*, vision, robot, handeye, config, matrix,
                              executor, safety, origin, original_watch,
                              target_label, reason, attempt, collect_target,
                              wait_clear, check_hold, commit_view, checkpoint):
    """Use only fresh, real detections as the pre-motion depth reference.

    Failure to collect a stable reference consumes a look without moving.
    Communication, stop, collision and encoder failures always propagate.
    """
    from .call_button_search import down_search_anchor, up_search_anchor
    from .initial_light_recheck import choose_initial_recheck_joints
    from .observation_visibility import complete_button_box

    if target_label not in {"up", "down"} or attempt not in (1, 2):
        return False
    check_hold()
    response = vision.request("visible_buttons", min_confidence=.50)
    threshold = float(config['detector'].get('target_confidence', .70))
    age = time.time() - float(response.get('capture_time_unix_sec', 0))
    if not -.05 <= age <= .35:
        checkpoint("initial_light_view_unavailable", {"reason": "stale_search_reference"})
        return False
    targets = [d for d in response.get('detections', [])
               if d.get('label') == target_label
               and math.isfinite(float(d.get('confidence', 0)))
               and float(d.get('confidence', 0)) >= threshold
               and complete_button_box(d.get('bbox_xyxy'), response.get('image_size'))]
    if len(targets) > 1:
        checkpoint("initial_light_view_unavailable", {"reason": "ambiguous_target"})
        return False
    anchor = (targets[0] if targets else
              down_search_anchor(response, threshold) if target_label == 'down'
              else up_search_anchor(response, threshold))
    if anchor is None or not math.isfinite(float(anchor.get('confidence', 0))):
        checkpoint("initial_light_view_unavailable", {"reason": "no_qualified_search_reference"})
        return False
    label = str(anchor['label'])
    # set_target installs the real detected reference for depth safety. Restore
    # perception to the requested button even if stable reference capture fails.
    try:
        vision.request("set_target", target_label=label)
        try:
            reference, audit = collect_target(
                vision, robot, handeye, label, config,
                min_capture_unix_sec=time.time())
        except RuntimeError as exc:
            if not hasattr(exc, 'stationary_target_failure'):
                raise
            checkpoint("initial_light_view_unavailable", {
                "reason": "search_reference_localization_pending", "anchor_label": label,
                "sampling": exc.stationary_target_failure})
            return False
    finally:
        vision.request("set_perception_target", target_label=target_label)
    safety.set_target(reference.base_to_button)
    clear = wait_clear(vision, timeout_sec=2.0, required_clear_frames=3)
    checkpoint("initial_light_view_safety_ready", {"attempt": attempt, "status": clear})
    current = robot.state(max_age=.25)
    size = config['depth']['capture_size']
    calibration_size = config['calibration'].get('rgb_calibration_size', [1920, 1080])
    if label == target_label and target_label == 'down' and reason == 'local_highlight_conflict':
        goal, view_audit = choose_initial_recheck_joints(
            current.joints, original_watch, handeye, reference.base_to_button,
            matrix, calibration_size, size, audit[-1]['bbox_xyxy'])
    else:
        goal, view_audit = choose_initial_call_search_joints(
            current.joints, original_watch, handeye, reference.base_to_button,
            matrix, calibration_size, size, target_label=target_label, anchor_label=label)
    if goal is None:
        checkpoint("initial_light_view_unavailable", view_audit)
        return False
    checkpoint("initial_light_view_requested", {"attempt": attempt, **view_audit,
                                                "reference_audit": audit})
    check_hold()
    execution = executor.recheck_observation_view(goal, origin)
    # Commit actual encoders before any subsequent observation can fail. Origin
    # recovery must never think the arm is still at its previous watch pose.
    commit_view(execution)
    checkpoint("initial_light_view_moved", {"attempt": attempt, **view_audit, **execution})
    return True
