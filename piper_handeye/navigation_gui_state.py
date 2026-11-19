"""Deterministic first-leg gate; stopping invalidates every pending dispatch."""
INITIAL_NAVIGATION_DELAY_SEC = 3


class NavigationState:
    def __init__(self):
        self.enabled = False
        self.floor = None
        self.deadline = None
        self.sent = False
        self.stopped = False
        self.arrived = False
        self.goal_name = 'liftout'
        self.goal_pose = (-.525359, 3.050205, -.019123, .999817)

    def enable(self, goal_name='liftout', goal_pose=None):
        if goal_pose is None:
            if goal_name != 'liftout':
                raise ValueError('selected goal needs known coordinates')
            goal_pose = (-.525359, 3.050205, -.019123, .999817)
        import math
        if len(goal_pose) != 4 or not all(math.isfinite(float(v)) for v in goal_pose):
            raise ValueError('invalid goal coordinates')
        self.goal_name = str(goal_name)
        self.goal_pose = tuple(float(v) for v in goal_pose)
        self.enabled = True
        self.floor = None
        self.deadline = None
        self.sent = False
        self.stopped = False
        self.arrived = False

    def accept(self, floor, now):
        if not self.enabled or self.stopped:
            return 'monitor'
        if self.floor is not None:
            return 'duplicate' if self.floor == floor else 'busy'
        self.floor = floor
        self.deadline = now + INITIAL_NAVIGATION_DELAY_SEC
        return 'accepted'

    def due(self, now):
        if self.enabled and not self.stopped and not self.sent and self.deadline is not None and now >= self.deadline:
            self.sent = True
            return True
        return False

    def stop(self):
        self.enabled = False
        self.stopped = True
        self.deadline = None


def telemetry_fresh(sample, now, timeout=1.0):
    """Use receipt monotonic time and source-local ages, never mixed wall clocks."""
    elapsed = now - sample.get('received_monotonic', float('-inf'))
    return (0 <= elapsed < timeout and all(
        key in sample and 0 <= sample.get(key + '_age', float('inf')) < timeout
        for key in ('pose', 'velocity')))


def resume_allowed(state, sample, baseline_revision, now):
    import math
    goal = sample.get('goal')
    return bool(state.sent and not state.stopped and telemetry_fresh(sample, now)
                and sample.get('goal_revision', 0) > baseline_revision and goal
                and math.hypot(goal[0] - state.goal_pose[0], goal[1] - state.goal_pose[1]) < .02
                and abs(math.atan2(math.sin(2*math.atan2(goal[2],goal[3]) - 2*math.atan2(state.goal_pose[2],state.goal_pose[3])),
                                   math.cos(2*math.atan2(goal[2],goal[3]) - 2*math.atan2(state.goal_pose[2],state.goal_pose[3])))) < .1)


def charge_resume_allowed(state, sample, request_baseline, service_baseline, acknowledged, now, dispatched_source_time):
    """Require this dispatch's ACK and fresh command-9 reception/service success."""
    return bool(state.goal_name == 'back' and state.sent and not state.stopped
                and acknowledged and dispatched_source_time is not None and telemetry_fresh(sample, now)
                and sample.get('charge_request_time', 0) >= dispatched_source_time
                and sample.get('charge_service_time', 0) >= dispatched_source_time
                and sample.get('charge_error_time', 0) < dispatched_source_time
                and sample.get('charge_request_revision', 0) > request_baseline
                and sample.get('charge_service_revision', 0) > service_baseline
                and 0 <= sample.get('charge_request_age', float('inf')) < 10
                and 0 <= sample.get('charge_service_age', float('inf')) < 10)
