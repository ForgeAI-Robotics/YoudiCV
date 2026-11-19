"""Bounded, best-effort pose timing evidence; no I/O or control decisions."""
from collections import deque
import math
import threading

from .pipeline_timing import CLOCK_DOMAIN


class PoseTimingTrace:
    """Keep startup, recent and slow samples without growing during ON waits."""

    def __init__(self, first_limit=256, recent_limit=32, slow_limit=64):
        self._lock = threading.Lock()
        self._first = []
        self._first_limit = first_limit
        self._recent = deque(maxlen=recent_limit)
        self._slow = deque(maxlen=slow_limit)
        self._count = 0
        self._errors = 0
        self._stats = {}

    def record(self, row, *, slow=False):
        try:
            row = dict(row)
            with self._lock:
                self._count += 1
                row['diagnostic_sequence'] = self._count
                if len(self._first) < self._first_limit:
                    self._first.append(row)
                self._recent.append(row)
                if slow:
                    self._slow.append(row)
                for key, value in row.items():
                    if not key.endswith('_sec') or type(value) not in (int, float):
                        continue
                    if not math.isfinite(value):
                        continue
                    s = self._stats.setdefault(key, dict(count=0, total=0., maximum=None))
                    s['count'] += 1
                    s['total'] += value
                    s['maximum'] = value if s['maximum'] is None else max(s['maximum'], value)
        except Exception:
            self._errors += 1

    def snapshot(self):
        with self._lock:
            return dict(clock_domain=CLOCK_DOMAIN, count=self._count,
                        diagnostic_errors=self._errors,
                        first=list(self._first), recent=list(self._recent),
                        slow=list(self._slow),
                        metrics={k: dict(count=v['count'], mean=v['total']/v['count'],
                                         maximum=v['maximum']) for k, v in self._stats.items()},
                        scope='host monotonic wall time; diagnostic only; buffers may overlap')


def collect_pose_timing(safety, robot):
    """Called after existing cleanup; optional evidence cannot fail a mission."""
    result = dict(version=1, clock_domain=CLOCK_DOMAIN,
                  scope='AGX receive timestamps; not controller hardware acquisition times')
    for name, obj in [('publisher', safety), ('robot_feedback', robot)]:
        try:
            trace = getattr(obj, '_pose_timing', None)
            if trace is not None:
                result[name] = trace.snapshot()
        except Exception as exc:
            result[name] = dict(diagnostic_error=type(exc).__name__)
    if safety is None:
        result['service'] = dict(available=False, reason='safety_not_initialized')
        return result
    try:
        # A separate short-timeout connection cannot change the live client's
        # control timeout, and no diagnostic RPC is added to motion loops.
        from .vision_rpc import VisionRpcClient
        vision = safety.vision
        probe = VisionRpcClient(vision.host, vision.port, vision.token, timeout=.5)
        result['service'] = probe.request('safety_pose_timing')['evidence']
    except Exception as exc:
        result['service'] = dict(diagnostic_error=type(exc).__name__)
    return result
