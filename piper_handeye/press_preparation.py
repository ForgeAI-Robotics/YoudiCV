"""Overlap CPU-only planning with approach; this module never commands motion."""
from concurrent.futures import ThreadPoolExecutor
import time
from .pipeline_timing import traced


class BackgroundPressPreparation:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='press-plan')
        self.future=None
        self.build_interval=None

    def start(self,builder):
        if self.future is not None:raise RuntimeError('press plan preparation already started')
        @traced('press.background_plan_build', 'planning')
        def prepare():
            began=time.monotonic()
            plan=builder()
            ended=time.monotonic()
            self.build_interval=dict(start_monotonic_sec=began,end_monotonic_sec=ended)
            return plan,ended-began
        self.future=self.pool.submit(prepare)

    @traced('press.wait_for_background_plan', 'planning_wait')
    def finish(self,timeout_sec=10.):
        if self.future is None:raise RuntimeError('press plan preparation was not scheduled')
        began=time.monotonic()
        plan,build_sec=self.future.result(timeout=timeout_sec)
        build_interval=dict(self.build_interval)
        ended=time.monotonic()
        return plan,dict(build_sec=build_sec,post_approach_wait_sec=ended-began,
                         build_interval=build_interval,
                         wait_interval=dict(start_monotonic_sec=began,end_monotonic_sec=ended),
                         build_completed_before_finish=build_interval['end_monotonic_sec']<=began)

    def close(self):
        if self.future is not None:self.future.cancel()
        self.pool.shutdown(wait=True)

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
