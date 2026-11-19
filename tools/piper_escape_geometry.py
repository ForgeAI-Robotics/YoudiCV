"""Resident, request-driven reader for the read-only full-mesh metric."""
import json
import math
from pathlib import Path
import queue
import subprocess
import threading


class EscapeGeometry:
    PREFIX = 'ESCAPE_GEOMETRY '

    def __init__(self, executable=None):
        self.executable = str(executable or Path(__file__).with_suffix(''))
        self._process = None
        self._lock = threading.Lock()

    def _read(self, timeout):
        try:
            result = self._responses.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError('escape geometry response timeout')
        if result is None:
            self.close()
            raise RuntimeError('escape geometry worker stopped')
        if isinstance(result, Exception):
            self.close()
            raise RuntimeError('invalid escape geometry response') from result
        return result

    def _start(self):
        self._responses = queue.Queue()
        process = subprocess.Popen([self.executable], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True)
        self._process = process
        responses = self._responses
        def reader():
            try:
                for line in process.stdout:
                    if line.startswith(self.PREFIX):
                        try:
                            responses.put(json.loads(line[len(self.PREFIX):]))
                        except ValueError as exc:
                            responses.put(exc)
            finally:
                responses.put(None)
        threading.Thread(target=reader, daemon=True).start()
        if self._read(6.0) != {'ready': True}:
            self.close()
            raise RuntimeError('escape geometry worker not ready')

    def measure(self, joints):
        if len(joints) != 6 or not all(math.isfinite(x) for x in joints):
            raise ValueError('six finite joints required')
        with self._lock:
            if self._process is None:
                self._start()
            try:
                self._process.stdin.write(' '.join(format(x, '.17g') for x in joints)+'\n')
                self._process.stdin.flush()
            except (OSError, ValueError) as exc:
                self.close()
                raise RuntimeError('escape geometry worker write failed') from exc
            result = self._read(2.0)
            if not result.get('ok'):
                raise RuntimeError('escape geometry failed: '+str(result.get('error')))
            for key in ('upward_exit_m', 'clearance_m', 'panel_padding_m'):
                if not math.isfinite(float(result[key])) or result[key] < 0:
                    raise RuntimeError('invalid escape geometry metric')
            return result

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.)
            if process.stdout:
                process.stdout.close()
