"""Control-environment client for the OpenCV button-light worker."""

from __future__ import annotations

import json
from pathlib import Path
import select
import subprocess
import sys
import time
from .pipeline_timing import traced


PREFIX = "BUTTON_LIGHT_RESPONSE "
OPENCV_COMMAND = [
    sys.executable,
    str(Path(__file__).resolve().parents[1] / "tools" / "opencv_light_worker.py"),
]


class ButtonLightProcess:
    def __init__(self, command, startup_timeout_sec: float = 30.0):
        self.process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout
        self._request_id = 0
        self.ready = self._read_response(startup_timeout_sec)
        self.ready["transport"] = "subprocess"
        if not self.ready.get("ready"):
            raise RuntimeError(f"button light worker did not become ready: {self.ready}")

    def _read_response(self, timeout_sec: float) -> dict:
        # The production worker flushes every protocol line. Model libraries may
        # print unrelated startup logs, which are deliberately skipped.
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"button light worker exited with {self.process.returncode}"
                )
            remaining = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select(
                [self.stdout], [], [], remaining
            )
            if not readable:
                break
            line = self.stdout.readline()
            if line.startswith(PREFIX):
                return json.loads(line[len(PREFIX) :])
        raise TimeoutError("button light worker response timed out")

    @traced('light.worker_roundtrip', 'light')
    def observe(self, image_path: str | Path, target: str, bbox_xyxy=None) -> dict:
        self._request_id += 1
        request = {
            "id": self._request_id,
            "command": "observe",
            "image": str(image_path),
            "target": str(target),
        }
        if bbox_xyxy is not None:
            request["bbox_xyxy"] = [float(value) for value in bbox_xyxy]
        self.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.stdin.flush()
        response = self._read_response(30.0)
        if response.get("id") != self._request_id:
            raise RuntimeError("button light worker response id mismatch")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "button light observation failed"))
        return response

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.stdin.write('{"command":"shutdown"}\n')
            self.stdin.flush()
            self.process.wait(timeout=5.0)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            self.process.wait(timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class OpenCvButtonLightProcess(ButtonLightProcess):
    """Use within-button pixel contrast for the light state."""
    def __init__(self, algorithm="local_regions_v1"):
        if algorithm not in ("legacy", "local_regions_v1"):
            raise ValueError("unknown OpenCV light algorithm")
        super().__init__(command=OPENCV_COMMAND + ["--algorithm", algorithm])
