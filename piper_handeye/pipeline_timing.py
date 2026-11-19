"""Bounded, opt-in timing evidence. Never log command payloads or alter results.

Task processes buffer spans in memory and append them only at finalization.
Monotonic times may only be compared inside the same host/boot clock domain.
"""
from __future__ import annotations

import atexit
from collections import deque
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid


def _clock_domain():
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        boot = 'unknown-boot'
    return socket.gethostname() + ':' + boot


CLOCK_DOMAIN = _clock_domain()
_recorder = None
_local = threading.local()


def mark():
    return time.monotonic(), time.time()


def interval(name, category, began, *, status='ok', **metadata):
    end = time.monotonic()
    return dict(name=name, category=category, start_monotonic_sec=began[0],
                end_monotonic_sec=end, start_unix_sec=began[1],
                elapsed_sec=max(0., end-began[0]), clock_domain=CLOCK_DOMAIN,
                source_pid=os.getpid(), source_thread_id=threading.get_ident(),
                status=status, **metadata)


class TimingRecorder:
    def __init__(self, path, task_id='', max_spans=20000):
        self.path = Path(path)
        self.task_id = str(task_id)
        self.spans = deque(maxlen=max_spans)
        self.dropped = 0
        self.errors = 0
        self.lock = threading.Lock()
        self.seen = set()
        self.finished = False
        self.started = mark()

    def add(self, span, *, source_key=None):
        try:
            with self.lock:
                if len(self.spans) == self.spans.maxlen:
                    # Retain startup/pre-press evidence during intentionally
                    # unlimited ON waiting. Dropping later diagnostics must
                    # never change the task's wait or control policy.
                    self.dropped += 1
                    return
                if source_key is not None:
                    if source_key in self.seen:
                        return
                    # Bound deduplication storage as well as event storage.
                    if len(self.seen) >= self.spans.maxlen:
                        self.seen.clear()
                    self.seen.add(source_key)
                self.spans.append(dict(schema_version=1, kind='span',
                    pid=os.getpid(), thread_id=threading.get_ident(),
                    task_id=self.task_id, **span))
        except Exception:
            self.errors += 1

    def flush(self):
        try:
            with self.lock:
                if self.finished:
                    return self.summary()
                rows = list(self.spans)
                rows.append(dict(kind='session', pid=os.getpid(), task_id=self.task_id,
                    clock_domain=CLOCK_DOMAIN, start_monotonic_sec=self.started[0],
                    end_monotonic_sec=time.monotonic(), dropped_spans=self.dropped,
                    diagnostic_errors=self.errors))
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # One append per process; no file I/O in camera/IK/motion loops.
                data = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'
                               for row in rows).encode()
                with self.path.open('ab', buffering=0) as stream:
                    view = memoryview(data)
                    while view:
                        count = stream.write(view)
                        if not count:
                            raise OSError('timing evidence write incomplete')
                        view = view[count:]
                self.finished = True
        except Exception:
            self.errors += 1
        return self.summary()

    def summary(self):
        return dict(path=str(self.path), buffered_spans=len(self.spans),
                    dropped_spans=self.dropped, diagnostic_errors=self.errors,
                    clock_domain=CLOCK_DOMAIN, written=self.finished,
                    scope='wall time, not CPU/GPU kernel time; overlapping spans must not be summed')


def configure_timing(path, task_id=''):
    global _recorder
    _recorder = TimingRecorder(path, task_id)
    os.environ['YOUXIAOMEI_TIMING_FILE'] = str(Path(path).resolve())
    os.environ['YOUXIAOMEI_TIMING_TASK'] = str(task_id)
    return _recorder


def get_recorder():
    global _recorder
    if _recorder is None and os.environ.get('YOUXIAOMEI_TIMING_FILE'):
        try:
            _recorder = TimingRecorder(os.environ['YOUXIAOMEI_TIMING_FILE'],
                                       os.environ.get('YOUXIAOMEI_TIMING_TASK', ''))
        except Exception:
            return None
    return _recorder


@contextmanager
def stage(name, category, **metadata):
    recorder = get_recorder()
    if recorder is None:
        yield
        return
    began = mark()
    identifier = uuid.uuid4().hex
    stack = getattr(_local, 'stack', [])
    parent = stack[-1] if stack else None
    _local.stack = [*stack, identifier]
    status = 'ok'
    try:
        yield
    except BaseException:
        status = 'error'
        raise
    finally:
        _local.stack = stack
        # Instrumentation failure must not replace a control exception.
        try:
            recorder.add(interval(name, category, began, status=status,
                                  span_id=identifier, parent_span_id=parent, **metadata))
        except Exception:
            recorder.errors += 1


def traced(name, category):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with stage(name, category):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def record_interval(name, category, began, **metadata):
    recorder = get_recorder()
    if recorder is not None:
        try:
            recorder.add(interval(name, category, began, **metadata))
        except Exception:
            recorder.errors += 1


def record_vision_response(response):
    """Copy only optional timings, without changing a response or sample ages."""
    recorder = get_recorder()
    if recorder is None:
        return
    try:
        samples = [response.get('sample'), response.get('latest_processed_timing')] + list(response.get('samples', []))
        if 'perception_timing' in response:
            samples.append(dict(sequence=response.get('frame_sequence'),
                                timing=response['perception_timing']))
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            timing = sample.get('timing')
            if not isinstance(timing, dict):
                continue
            for span in timing.get('spans', []):
                if not isinstance(span, dict):
                    continue
                key = (span.get('clock_domain'), span.get('start_monotonic_sec'),
                       sample.get('sequence'), span.get('name'))
                recorder.add(dict(span, source='vision_sample',
                                  frame_sequence=sample.get('sequence')), source_key=key)
        remote = response.get('server_request_timing')
        if isinstance(remote, dict):
            recorder.add(dict(remote, source='vision_service'))
    except Exception:
        recorder.errors += 1


def finish_timing():
    recorder = get_recorder()
    return recorder.flush() if recorder is not None else None


def finish_timing_with_analysis():
    result = finish_timing()
    if result is not None and result.get('written'):
        try:
            from tools.analyze_pipeline_timing import write_analysis
            output = Path(result['path']).parent/'pipeline_timing_analysis'
            write_analysis(result['path'], output)
            result['analysis_directory'] = str(output)
        except Exception as exc:
            # Analysis is diagnostic only and never a mission-success gate.
            result['analysis_error'] = type(exc).__name__
    return result


atexit.register(finish_timing)
