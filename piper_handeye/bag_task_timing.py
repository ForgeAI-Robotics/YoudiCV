"""Wall-clock stage timings for the paper bag task's existing report."""
from contextlib import contextmanager
import time


@contextmanager
def timed_stage(report, stage, **details):
    started=time.perf_counter()
    entry={'stage':stage,'started_unix_sec':time.time(),**details}
    try:
        yield entry
    except BaseException:
        entry['outcome']='failed'
        raise
    else:
        entry['outcome']='complete'
    finally:
        entry['duration_sec']=time.perf_counter()-started
        report.setdefault('timings',[]).append(entry)
