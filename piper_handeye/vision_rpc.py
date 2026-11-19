from __future__ import annotations

import json
import socket
from typing import Any
from contextlib import nullcontext
from .pipeline_timing import stage, record_vision_response


class VisionRpcClient:
    def __init__(self, host: str, port: int, token: str, timeout: float = 12.0):
        self.host = str(host)
        self.port = int(port)
        self.token = str(token)
        self.timeout = float(timeout)

    def request(self, command: str, **options: Any) -> dict[str, Any]:
        # Do not add per-poll tracing to high-frequency safety/pose supervision.
        profile = command in {'latest', 'light_snapshot', 'visible_buttons', 'collect',
            'set_target', 'set_perception_target', 'safety_capture_matrix_reference'}
        with stage('vision.rpc', 'communication') if profile else nullcontext():
            return self._request(command, options, profile)

    def _request(self, command, options, profile):
        measure = stage if profile else lambda *_a, **_kw: nullcontext()
        payload = {"token": self.token, "command": str(command), **options}
        with measure('vision.connect', 'communication'):
            connection = socket.create_connection((self.host, self.port), timeout=self.timeout)
        with connection as sock:
            # Reference construction is a stationary preparation operation.
            # Do not lengthen safety polling or pose-update timeouts with it.
            sock.settimeout(max(self.timeout, 8.0)
                            if command == "safety_capture_matrix_reference"
                            else self.timeout)
            # Buffered readline avoids one socket read per JSON byte. Flush
            # the request before waiting; close the file as well as the socket.
            with sock.makefile("rwb") as stream:
                with measure('vision.send', 'communication'):
                    stream.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
                    stream.flush()
                with measure('vision.response_wait', 'communication',
                           scope='includes server dispatch, encoding, transfer and scheduling; not pure network latency'):
                    raw = stream.readline(4 * 1024 * 1024)
        if not raw:
            raise RuntimeError("vision service closed without a response")
        with measure('vision.decode', 'communication'):
            response = json.loads(raw)
        if profile:
            record_vision_response(response)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "vision service request failed"))
        return response
