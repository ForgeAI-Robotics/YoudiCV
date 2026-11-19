from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess
import threading
import time

import cv2
import numpy as np


_HEADER = struct.Struct("<4sIIfQQQII")
_MAGIC = b"ORBD"


@dataclass(frozen=True)
class LiveRgbdFrame:
    sequence: int
    received_monotonic: float
    received_unix: float
    color_timestamp: int
    depth_timestamp: int
    color: np.ndarray
    depth_m: np.ndarray


class OrbbecV1Stream:
    """Persistent SDK-v1 camera process with a bounded in-memory RGB-D cache."""

    def __init__(
        self,
        executable: str | Path,
        sdk_work_directory: str | Path,
        *,
        cache_size: int = 32,
    ) -> None:
        self.executable = Path(executable).expanduser().resolve()
        self.sdk_work_directory = Path(sdk_work_directory).expanduser().resolve()
        if not self.executable.is_file():
            raise FileNotFoundError(f"Orbbec stream helper not found: {self.executable}")
        if not self.sdk_work_directory.is_dir():
            raise FileNotFoundError(
                f"Orbbec SDK work directory not found: {self.sdk_work_directory}"
            )
        self._frames: deque[LiveRgbdFrame] = deque(maxlen=int(cache_size))
        self._stderr: deque[str] = deque(maxlen=40)
        self._condition = threading.Condition()
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._running = False
        self._error: BaseException | None = None

    def start(self, timeout: float = 8.0) -> None:
        if self._running:
            return
        self._process = subprocess.Popen(
            [str(self.executable)],
            cwd=self.sdk_work_directory,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self._reader.start()
        self._stderr_reader.start()
        self.wait_for_sequence(0, timeout=timeout)

    @staticmethod
    def _read_exact(stream, size: int) -> bytes:
        chunks = bytearray(size)
        view = memoryview(chunks)
        offset = 0
        while offset < size:
            count = stream.readinto(view[offset:])
            if not count:
                raise EOFError("Orbbec stream closed while reading a frame")
            offset += count
        return bytes(chunks)

    @classmethod
    def _read_header(cls, stream) -> bytes:
        """Synchronize after SDK startup text written to the binary stdout pipe."""
        window = bytearray()
        skipped = 0
        while True:
            value = cls._read_exact(stream, 1)
            window += value
            if len(window) > len(_MAGIC):
                del window[0]
                skipped += 1
            if bytes(window) == _MAGIC:
                return _MAGIC + cls._read_exact(stream, _HEADER.size - len(_MAGIC))
            if skipped > 1024 * 1024:
                raise RuntimeError("Orbbec stream magic not found within 1 MiB")

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while self._running:
                header = self._read_header(self._process.stdout)
                (
                    magic,
                    width,
                    height,
                    depth_scale_mm,
                    color_timestamp,
                    depth_timestamp,
                    sequence,
                    color_bytes,
                    depth_bytes,
                ) = _HEADER.unpack(header)
                if magic != _MAGIC:
                    raise RuntimeError(f"invalid Orbbec stream magic: {magic!r}")
                expected_color = width * height * 3
                expected_depth = width * height * 2
                if color_bytes != expected_color or depth_bytes != expected_depth:
                    raise RuntimeError(
                        f"invalid Orbbec payload sizes {color_bytes}/{depth_bytes}"
                    )
                rgb = np.frombuffer(
                    self._read_exact(self._process.stdout, color_bytes), dtype=np.uint8
                ).reshape(height, width, 3)
                depth_raw = np.frombuffer(
                    self._read_exact(self._process.stdout, depth_bytes), dtype="<u2"
                ).reshape(height, width)
                received_monotonic = time.monotonic()
                received_unix = time.time()
                frame = LiveRgbdFrame(
                    sequence=int(sequence),
                    received_monotonic=received_monotonic,
                    received_unix=received_unix,
                    color_timestamp=int(color_timestamp),
                    depth_timestamp=int(depth_timestamp),
                    color=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    depth_m=depth_raw.astype(np.float32)
                    * float(depth_scale_mm)
                    / 1000.0,
                )
                with self._condition:
                    self._frames.append(frame)
                    self._condition.notify_all()
        except BaseException as exc:
            if self._running:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()

    def _stderr_loop(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for raw in iter(self._process.stderr.readline, b""):
            self._stderr.append(raw.decode("utf-8", errors="replace").rstrip())

    @property
    def latest_sequence(self) -> int:
        with self._condition:
            return self._frames[-1].sequence if self._frames else 0

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return the bounded diagnostic output emitted by the SDK helper."""
        with self._condition:
            return tuple(self._stderr)

    def latest_frame(self) -> LiveRgbdFrame:
        """Return the newest immutable frame object without waiting."""
        with self._condition:
            if not self._frames:
                raise RuntimeError("Orbbec stream has no frame")
            return self._frames[-1]

    def recent_frames(self, count: int) -> tuple[LiveRgbdFrame, ...]:
        with self._condition:
            value = tuple(self._frames)[-int(count) :]
            if len(value) < int(count):
                raise RuntimeError(f"only {len(value)} recent RGB-D frames are available")
            return value

    def wait_for_sequence(self, after_sequence: int, timeout: float = 3.0) -> LiveRgbdFrame:
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while self._running:
                if self._error is not None:
                    raise RuntimeError(
                        f"Orbbec stream failed: {self._error}; stderr={list(self._stderr)}"
                    ) from self._error
                if self._frames and self._frames[-1].sequence > after_sequence:
                    return self._frames[-1]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"no Orbbec frame newer than sequence {after_sequence}; "
                        f"stderr={list(self._stderr)}"
                    )
                self._condition.wait(min(remaining, 0.1))
        raise RuntimeError("Orbbec stream is closed")

    def close(self) -> None:
        self._running = False
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        with self._condition:
            self._condition.notify_all()
        for thread in (self._reader, self._stderr_reader):
            if thread is not None:
                thread.join(timeout=1.0)

    def __enter__(self) -> "OrbbecV1Stream":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
