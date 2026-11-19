from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RollingSpeedConfig:
    max_speed_m_s: float = 0.15
    desired_schedule_margin_sec: float = 0.60
    hard_schedule_margin_sec: float = 0.30
    planning_tail_sec: float = 0.45
    minimum_valid_rate_hz: float = 3.0
    contact_deceleration_m_s2: float = 0.35
    normal_acceleration_m_s2: float = 0.25
    normal_deceleration_m_s2: float = 0.35
    max_jerk_m_s3: float = 1.50
    depth_uncertainty_m: float = 0.005
    coarse_target_uncertainty_m: float = 0.020
    endpoint_margin_m: float = 0.003
    control_latency_sec: float = 0.12
    max_target_age_sec: float = 0.20


@dataclass(frozen=True)
class RollingSpeedInput:
    distance_to_target_m: float
    dt_sec: float
    target_ready: bool
    accepted_samples: int = 0
    required_samples: int = 6
    valid_frame_rate_hz: float = 10.0
    latest_target_age_sec: float = 0.0
    predicted_ready_sec: float | None = None
    obstacle_speed_cap_m_s: float | None = None


@dataclass(frozen=True)
class RollingSpeedOutput:
    commanded_speed_m_s: float
    commanded_acceleration_m_s2: float
    raw_speed_cap_m_s: float
    schedule_speed_cap_m_s: float
    contact_speed_cap_m_s: float
    predicted_ready_sec: float
    estimated_schedule_margin_sec: float
    dynamic_guard_distance_m: float
    mode: str
    reason: str


@dataclass(frozen=True)
class RecedingApproachPlan:
    """One stable approach followed by at most one visual trajectory splice."""

    speed_cap_m_s: float
    predicted_ready_sec: float
    expected_guard_arrival_sec: float
    dynamic_guard_distance_m: float
    brake_entry_distance_m: float
    braking_distance_m: float
    latency_distance_m: float
    usable_approach_distance_m: float
    splice_count_limit: int = 1


def jerk_limited_stop_distance(
    speed_m_s: float,
    *,
    deceleration_m_s2: float,
    max_jerk_m_s3: float,
) -> float:
    """Conservative stop distance from zero acceleration."""
    speed = max(0.0, float(speed_m_s))
    deceleration = float(deceleration_m_s2)
    jerk = float(max_jerk_m_s3)
    if not math.isfinite(speed):
        raise ValueError("speed_m_s must be finite")
    if not math.isfinite(deceleration) or deceleration <= 0.0:
        raise ValueError("deceleration_m_s2 must be finite and positive")
    if not math.isfinite(jerk) or jerk <= 0.0:
        raise ValueError("max_jerk_m_s3 must be finite and positive")
    ramp_time = deceleration / jerk
    ramp_velocity_loss = 0.5 * jerk * ramp_time**2
    if speed <= ramp_velocity_loss:
        stop_time = math.sqrt(2.0 * speed / jerk)
        return max(0.0, speed * stop_time - jerk * stop_time**3 / 6.0)
    ramp_distance = speed * ramp_time - jerk * ramp_time**3 / 6.0
    remaining_speed = speed - ramp_velocity_loss
    return ramp_distance + remaining_speed**2 / (2.0 * deceleration)


def plan_receding_approach(
    distance_to_target_m: float,
    predicted_ready_sec: float,
    config: RollingSpeedConfig = RollingSpeedConfig(),
    *,
    obstacle_speed_cap_m_s: float | None = None,
) -> RecedingApproachPlan:
    """Plan the fastest stable approach that still leaves computation margin.

    This does not stream replacement goals.  It produces one speed cap and one
    dynamic fallback distance for the active trajectory.  Once vision and IK
    are ready, the caller may replace the future tail exactly once.
    """
    distance = float(distance_to_target_m)
    ready = float(predicted_ready_sec)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance_to_target_m must be finite and non-negative")
    if not math.isfinite(ready) or ready < 0.0:
        raise ValueError("predicted_ready_sec must be finite and non-negative")
    maximum_speed = float(config.max_speed_m_s)
    if obstacle_speed_cap_m_s is not None:
        obstacle_cap = float(obstacle_speed_cap_m_s)
        if not math.isfinite(obstacle_cap) or obstacle_cap < 0.0:
            raise ValueError("obstacle_speed_cap_m_s must be finite and non-negative")
        maximum_speed = min(maximum_speed, obstacle_cap)

    # Guard distance and selected speed depend on each other.  A short fixed
    # point iteration converges monotonically for this one-dimensional bound.
    speed = maximum_speed
    guard = distance
    braking = 0.0
    latency_distance = 0.0
    usable = 0.0
    schedule_time = ready + float(config.desired_schedule_margin_sec)
    for _ in range(20):
        braking = jerk_limited_stop_distance(
            speed,
            deceleration_m_s2=config.normal_deceleration_m_s2,
            max_jerk_m_s3=config.max_jerk_m_s3,
        )
        latency_distance = speed * float(config.control_latency_sec)
        guard = min(
            distance,
            braking
            + latency_distance
            + float(config.coarse_target_uncertainty_m)
            + float(config.endpoint_margin_m),
        )
        usable = max(0.0, distance - guard)
        scheduled_speed = usable / max(schedule_time, 1e-6)
        updated = min(maximum_speed, scheduled_speed)
        if abs(updated - speed) <= 1e-9:
            speed = updated
            break
        speed = updated

    arrival = math.inf if speed <= 1e-9 else usable / speed
    return RecedingApproachPlan(
        speed_cap_m_s=speed,
        predicted_ready_sec=ready,
        expected_guard_arrival_sec=arrival,
        dynamic_guard_distance_m=guard,
        brake_entry_distance_m=min(
            distance, guard + braking + latency_distance
        ),
        braking_distance_m=braking,
        latency_distance_m=latency_distance,
        usable_approach_distance_m=usable,
    )


class RollingSpeedScheduler:
    """Jerk-limited rolling speed supervisor with no fixed 10 cm boundary."""

    def __init__(self, config: RollingSpeedConfig = RollingSpeedConfig()):
        self.config = config
        self.speed_m_s = 0.0
        self.acceleration_m_s2 = 0.0

    def reset(self, speed_m_s: float = 0.0) -> None:
        self.speed_m_s = max(0.0, float(speed_m_s))
        self.acceleration_m_s2 = 0.0

    def _jerk_limited_stop_distance(self, speed_m_s: float) -> float:
        """Conservative stop distance from zero acceleration with jerk limiting."""
        return jerk_limited_stop_distance(
            speed_m_s,
            deceleration_m_s2=self.config.normal_deceleration_m_s2,
            max_jerk_m_s3=self.config.max_jerk_m_s3,
        )

    def _contact_speed_cap(self, available_distance_m: float) -> float:
        cfg = self.config
        available = max(0.0, float(available_distance_m))
        low, high = 0.0, cfg.max_speed_m_s
        for _ in range(50):
            middle = 0.5 * (low + high)
            if self._jerk_limited_stop_distance(middle) <= available:
                low = middle
            else:
                high = middle
        return low

    def _predict_ready(self, value: RollingSpeedInput) -> float:
        if value.predicted_ready_sec is not None:
            return max(0.0, float(value.predicted_ready_sec))
        if value.target_ready:
            return 0.0
        remaining = max(0, int(value.required_samples) - int(value.accepted_samples))
        observed_rate = float(value.valid_frame_rate_hz)
        valid_rate = (
            observed_rate
            if math.isfinite(observed_rate) and observed_rate > 0.0
            else float(self.config.minimum_valid_rate_hz)
        )
        return remaining / valid_rate + float(self.config.planning_tail_sec)

    def update(self, value: RollingSpeedInput) -> RollingSpeedOutput:
        cfg = self.config
        dt = float(value.dt_sec)
        if not math.isfinite(dt) or not 0.01 <= dt <= 0.50:
            raise ValueError("dt_sec must be finite and in 0.01..0.50")
        distance = float(value.distance_to_target_m)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("distance_to_target_m must be finite and non-negative")

        ready_sec = self._predict_ready(value)
        sensor_fresh = (
            math.isfinite(value.latest_target_age_sec)
            and 0.0 <= value.latest_target_age_sec <= cfg.max_target_age_sec
        )
        braking_distance = self._jerk_limited_stop_distance(self.speed_m_s)
        latency_distance = self.speed_m_s * cfg.control_latency_sec
        dynamic_guard = (
            braking_distance
            + latency_distance
            + cfg.depth_uncertainty_m
            + cfg.endpoint_margin_m
        )
        usable_distance = max(0.0, distance - dynamic_guard)

        denominator = ready_sec + cfg.desired_schedule_margin_sec
        schedule_cap = (
            cfg.max_speed_m_s
            if value.target_ready
            else min(cfg.max_speed_m_s, usable_distance / max(denominator, 1e-6))
        )
        contact_cap = self._contact_speed_cap(
            max(
                0.0,
                distance - cfg.depth_uncertainty_m - cfg.endpoint_margin_m,
            )
        )
        obstacle_cap = (
            cfg.max_speed_m_s
            if value.obstacle_speed_cap_m_s is None
            else max(0.0, min(cfg.max_speed_m_s, value.obstacle_speed_cap_m_s))
        )
        raw_cap = min(schedule_cap, contact_cap, obstacle_cap)

        if not sensor_fresh:
            raw_cap = 0.0
            mode = "decelerate_unknown"
            reason = "target data is stale or invalid"
        elif value.target_ready:
            mode = "target_ready"
            reason = "target ready; contact braking envelope is active"
        elif raw_cap <= 1e-6:
            mode = "dynamic_stop"
            reason = "dynamic braking guard reached before target became ready"
        elif schedule_cap < cfg.max_speed_m_s - 1e-9:
            mode = "schedule_limited"
            reason = "speed reduced to preserve predicted scheduling margin"
        else:
            mode = "fast_approach"
            reason = "predicted scheduling margin permits maximum speed"

        desired_acceleration = (raw_cap - self.speed_m_s) / dt
        desired_acceleration = max(
            -cfg.normal_deceleration_m_s2,
            min(cfg.normal_acceleration_m_s2, desired_acceleration),
        )
        acceleration_delta = cfg.max_jerk_m_s3 * dt
        acceleration = max(
            self.acceleration_m_s2 - acceleration_delta,
            min(self.acceleration_m_s2 + acceleration_delta, desired_acceleration),
        )
        speed = max(0.0, self.speed_m_s + acceleration * dt)
        if acceleration >= 0.0:
            speed = min(speed, raw_cap)
        elif speed < raw_cap:
            speed = raw_cap

        time_available = usable_distance / max(self.speed_m_s, 0.005)
        estimated_margin = time_available - ready_sec
        if estimated_margin < cfg.hard_schedule_margin_sec and not value.target_ready:
            mode = "margin_recovery"
            reason = "predicted margin below hard threshold; controlled deceleration"

        self.speed_m_s = speed
        self.acceleration_m_s2 = acceleration
        return RollingSpeedOutput(
            commanded_speed_m_s=speed,
            commanded_acceleration_m_s2=acceleration,
            raw_speed_cap_m_s=raw_cap,
            schedule_speed_cap_m_s=schedule_cap,
            contact_speed_cap_m_s=contact_cap,
            predicted_ready_sec=ready_sec,
            estimated_schedule_margin_sec=estimated_margin,
            dynamic_guard_distance_m=dynamic_guard,
            mode=mode,
            reason=reason,
        )
