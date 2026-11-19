from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave


SAMPLE_RATE = 16_000
SAMPLE_WIDTH_BYTES = 2
DEFAULT_FRAME_MS = 20


def lowpass_pcm16(pcm: bytes, cutoff_hz: float = 4000.0) -> bytes:
    """Second-order Butterworth low-pass for 16 kHz mono s16le utterances.

    Zero disables filtering. State starts fresh for each complete utterance;
    original recordings are never modified.
    """
    if not math.isfinite(cutoff_hz) or not 0 <= cutoff_hz < SAMPLE_RATE / 2:
        raise ValueError("low-pass cutoff must be zero or below 8000 Hz")
    if len(pcm) % 2:
        raise ValueError("PCM must contain complete int16 samples")
    if cutoff_hz == 0:
        return pcm
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    omega = 2 * math.pi * cutoff_hz / SAMPLE_RATE
    cosine = math.cos(omega)
    alpha = math.sin(omega) / math.sqrt(2)
    a0 = 1 + alpha
    b0 = (1 - cosine) / (2 * a0)
    b1 = (1 - cosine) / a0
    a1 = -2 * cosine / a0
    a2 = (1 - alpha) / a0
    # Use the same coefficients and zero initial state in compiled code.
    # Keep the standard-library path for minimal voice-only environments.
    try:
        import numpy as np
        from scipy.signal import lfilter
    except ImportError:
        pass
    else:
        if not pcm:
            return b""
        values = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        filtered = lfilter([b0, b1, b0], [1.0, a1, a2], values)
        return np.clip(np.rint(filtered), -32768, 32767).astype("<i2").tobytes()
    x1 = x2 = y1 = y2 = 0.0
    result = array("h")
    for x in samples:
        y = b0 * x + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2
        result.append(max(-32768, min(32767, round(y))))
        x2, x1, y2, y1 = x1, x, y1, y
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def pcm16_rms(data: bytes) -> float:
    """Return RMS for little-endian signed 16-bit mono PCM."""
    usable = len(data) - len(data) % SAMPLE_WIDTH_BYTES
    if usable <= 0:
        return 0.0
    samples = array("h")
    samples.frombytes(data[:usable])
    if sys.byteorder != "little":
        samples.byteswap()
    return math.sqrt(sum(int(value) * int(value) for value in samples) / len(samples))


@dataclass(frozen=True)
class CapturedUtterance:
    pcm: bytes
    peak_rms: float
    threshold_rms: float
    duration_sec: float


class AdaptiveEnergyVad:
    """Small fail-quiet VAD for command phrases; it does not classify content."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        calibration_ms: int = 1_000,
        pre_roll_ms: int = 300,
        min_speech_ms: int = 220,
        end_silence_ms: int = 650,
        max_utterance_ms: int = 5_000,
        minimum_rms: float = 260.0,
        noise_ratio: float = 2.8,
        start_frames: int = 3,
    ):
        if frame_ms <= 0 or sample_rate <= 0:
            raise ValueError("sample rate and frame duration must be positive")
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_bytes = (
            self.sample_rate * self.frame_ms // 1000 * SAMPLE_WIDTH_BYTES
        )
        self.calibration_frames = max(1, calibration_ms // frame_ms)
        self.pre_roll_frames = max(0, pre_roll_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.end_silence_frames = max(1, end_silence_ms // frame_ms)
        self.max_utterance_frames = max(1, max_utterance_ms // frame_ms)
        self.minimum_rms = float(minimum_rms)
        self.noise_ratio = float(noise_ratio)
        self.start_frames = max(1, int(start_frames))

        self._calibration: list[float] = []
        self._noise_rms = 0.0
        self._pre_roll: deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self._candidate: list[bytes] = []
        self._candidate_voiced = 0
        self._recording: list[bytes] | None = None
        self._recording_voiced = 0
        self._silence_frames = 0
        self._peak_rms = 0.0

    @property
    def calibrated(self) -> bool:
        return len(self._calibration) >= self.calibration_frames

    @property
    def threshold_rms(self) -> float:
        return max(self.minimum_rms, self._noise_rms * self.noise_ratio)

    def _reset_phrase(self) -> None:
        self._candidate = []
        self._candidate_voiced = 0
        self._recording = None
        self._recording_voiced = 0
        self._silence_frames = 0
        self._peak_rms = 0.0

    def feed(self, frame: bytes) -> CapturedUtterance | None:
        if len(frame) != self.frame_bytes:
            raise ValueError(f"expected {self.frame_bytes} PCM bytes, got {len(frame)}")
        rms = pcm16_rms(frame)
        if not self.calibrated:
            self._calibration.append(rms)
            ordered = sorted(self._calibration)
            # A lower percentile avoids a cough during calibration making the
            # detector deaf for the rest of the process.
            self._noise_rms = ordered[max(0, len(ordered) // 3)]
            return None

        threshold = self.threshold_rms
        voiced = rms >= threshold
        if self._recording is None:
            if voiced:
                self._candidate.append(frame)
                self._candidate_voiced += 1
                self._peak_rms = max(self._peak_rms, rms)
                if self._candidate_voiced >= self.start_frames:
                    self._recording = list(self._pre_roll) + self._candidate
                    self._recording_voiced = self._candidate_voiced
                    self._candidate = []
            else:
                self._candidate = []
                self._candidate_voiced = 0
                self._noise_rms = 0.98 * self._noise_rms + 0.02 * rms
            self._pre_roll.append(frame)
            return None

        self._recording.append(frame)
        self._peak_rms = max(self._peak_rms, rms)
        if voiced:
            self._recording_voiced += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        complete = (
            self._silence_frames >= self.end_silence_frames
            or len(self._recording) >= self.max_utterance_frames
        )
        if not complete:
            return None
        frames = self._recording
        enough_speech = self._recording_voiced >= self.min_speech_frames
        peak = self._peak_rms
        current_threshold = self.threshold_rms
        self._reset_phrase()
        self._pre_roll.clear()
        if not enough_speech:
            return None
        pcm = b"".join(frames)
        return CapturedUtterance(
            pcm=pcm,
            peak_rms=peak,
            threshold_rms=current_threshold,
            duration_sec=len(pcm) / (self.sample_rate * SAMPLE_WIDTH_BYTES),
        )


class MicrophoneCapture:
    """Continuously consume PulseAudio so ASR cannot block microphone capture."""

    def __init__(
        self,
        source: str,
        *,
        vad: AdaptiveEnergyVad | None = None,
        queue_size: int = 8,
    ):
        self.source = source
        self.vad = vad or AdaptiveEnergyVad()
        self.utterances: queue.Queue[CapturedUtterance | Exception] = queue.Queue(
            maxsize=queue_size
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("microphone capture is already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _put(self, value: CapturedUtterance | Exception) -> None:
        try:
            self.utterances.put_nowait(value)
        except queue.Full:
            # Keep older commands in order. The main loop reports this as an
            # error instead of silently replacing a pending confirmation.
            pass

    def _run(self) -> None:
        argv = [
            "parec",
            "--raw",
            f"--device={self.source}",
            "--format=s16le",
            f"--rate={self.vad.sample_rate}",
            "--channels=1",
            "--latency-msec=40",
            f"--process-time-msec={self.vad.frame_ms}",
        ]
        try:
            self._process = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
            assert self._process.stdout is not None
            pending = bytearray()
            while not self._stop.is_set():
                chunk = self._process.stdout.read(self.vad.frame_bytes)
                if not chunk:
                    stderr = b""
                    if self._process.stderr is not None:
                        stderr = self._process.stderr.read()
                    raise RuntimeError(
                        "parec stopped: " + stderr.decode("utf-8", errors="replace").strip()
                    )
                pending.extend(chunk)
                while len(pending) >= self.vad.frame_bytes:
                    frame = bytes(pending[: self.vad.frame_bytes])
                    del pending[: self.vad.frame_bytes]
                    utterance = self.vad.feed(frame)
                    if utterance is not None:
                        self._put(utterance)
        except Exception as exc:
            if not self._stop.is_set():
                self._put(exc)
        finally:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()

    def get(self, timeout: float | None = None) -> CapturedUtterance:
        value = self.utterances.get(timeout=timeout)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self) -> None:
        self._stop.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class PcmFrameCapture:
    """Continuously expose fixed-size PulseAudio frames for streaming ASR."""

    def __init__(
        self,
        source: str,
        *,
        sample_rate: int = SAMPLE_RATE,
        frame_samples: int = 512,
        queue_size: int = 64,
    ):
        if sample_rate <= 0 or frame_samples <= 0 or queue_size <= 0:
            raise ValueError("streaming audio dimensions must be positive")
        self.source = source
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(frame_samples)
        self.frame_bytes = self.frame_samples * SAMPLE_WIDTH_BYTES
        self.frames: queue.Queue[bytes | Exception] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("microphone capture is already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _put_latest(self, value: bytes | Exception) -> None:
        try:
            self.frames.put_nowait(value)
            return
        except queue.Full:
            pass
        # Audio must remain close to real time. If inference ever falls behind,
        # discard the oldest frame rather than recognizing stale commands later.
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frames.put_nowait(value)
        except queue.Full:
            pass

    def _run(self) -> None:
        process_ms = max(1, round(self.frame_samples * 1000 / self.sample_rate))
        argv = [
            "parec", "--raw", f"--device={self.source}", "--format=s16le",
            f"--rate={self.sample_rate}", "--channels=1", "--latency-msec=40",
            f"--process-time-msec={process_ms}",
        ]
        try:
            self._process = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
            assert self._process.stdout is not None
            pending = bytearray()
            while not self._stop.is_set():
                chunk = self._process.stdout.read(self.frame_bytes - len(pending))
                if not chunk:
                    stderr = b""
                    if self._process.stderr is not None:
                        stderr = self._process.stderr.read()
                    raise RuntimeError(
                        "parec stopped: "
                        + stderr.decode("utf-8", errors="replace").strip()
                    )
                pending.extend(chunk)
                if len(pending) == self.frame_bytes:
                    self._put_latest(bytes(pending))
                    pending.clear()
        except Exception as exc:
            if not self._stop.is_set():
                self._put_latest(exc)
        finally:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()

    def get(self, timeout: float | None = None) -> bytes:
        value = self.frames.get(timeout=timeout)
        if isinstance(value, Exception):
            raise value
        return value

    def discard_pending(self) -> int:
        count = 0
        while True:
            try:
                self.frames.get_nowait()
                count += 1
            except queue.Empty:
                return count

    def close(self) -> None:
        self._stop.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(SAMPLE_WIDTH_BYTES)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def parse_verbose_whisper_json(payload: str | bytes) -> tuple[str, float]:
    data = json.loads(payload)
    text = str(data.get("text", "")).strip()
    probabilities: list[float] = []
    no_speech: list[float] = []
    for segment in data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        try:
            no_speech.append(float(segment.get("no_speech_prob", 0.0)))
        except (TypeError, ValueError):
            pass
        for word in segment.get("words", []):
            if not isinstance(word, dict) or not str(word.get("word", "")).strip():
                continue
            try:
                probability = float(word["probability"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0.0 <= probability <= 1.0:
                probabilities.append(probability)
    confidence = sum(probabilities) / len(probabilities) if probabilities else 0.0
    if no_speech:
        confidence *= 1.0 - max(0.0, min(1.0, max(no_speech)))
    return text, max(0.0, min(1.0, confidence))


class WhisperServer:
    """Own a loopback-only whisper.cpp server with one resident model."""

    def __init__(
        self,
        binary: Path,
        model: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8178,
        threads: int = 4,
        prompt: str = (
            "优小妹电梯任务：送外卖到一至十六楼，"
            "按一至十六号按钮，返回原点，取消，停止。"
        ),
        use_gpu: bool = True,
        lowpass_hz: float = 4000.0,
        vad_model: Path | None = None,
        vad_threshold: float = 0.55,
        vad_min_speech_ms: int = 250,
        vad_speech_pad_ms: int = 100,
    ):
        self.binary = binary.resolve()
        self.model = model.resolve()
        self.host = host
        self.port = int(port)
        self.threads = int(threads)
        self.prompt = prompt
        self.use_gpu = bool(use_gpu)
        lowpass_pcm16(b"", lowpass_hz)  # Validate before starting hardware.
        self.lowpass_hz = float(lowpass_hz)
        self.vad_model = None if vad_model is None else Path(vad_model).resolve()
        self.vad_threshold = float(vad_threshold)
        self.vad_min_speech_ms = int(vad_min_speech_ms)
        self.vad_speech_pad_ms = int(vad_speech_pad_ms)
        if not 0.0 < self.vad_threshold < 1.0:
            raise ValueError("VAD threshold must be between 0 and 1")
        if self.vad_min_speech_ms < 100 or self.vad_speech_pad_ms < 0:
            raise ValueError("invalid VAD duration settings")
        self.process: subprocess.Popen | None = None
        self._stderr_file = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _server_argv(self) -> list[str]:
        argv = [
            str(self.binary),
            "-m", str(self.model),
            "-l", "zh",
            "-t", str(self.threads),
            "-bo", "2",
            "-nf",
            "-fa",
            "--host", self.host,
            "--port", str(self.port),
            "--prompt", self.prompt,
            "--suppress-nst",
        ]
        if self.vad_model is not None:
            argv.extend((
                "--vad",
                "--vad-model", str(self.vad_model),
                "--vad-threshold", str(self.vad_threshold),
                "--vad-min-speech-duration-ms", str(self.vad_min_speech_ms),
                "--vad-speech-pad-ms", str(self.vad_speech_pad_ms),
            ))
        if not self.use_gpu:
            argv.append("-ng")
        return argv

    def start(self, timeout_sec: float = 90.0) -> None:
        if not self.binary.is_file():
            raise FileNotFoundError(f"whisper server not found: {self.binary}")
        if not self.model.is_file():
            raise FileNotFoundError(f"whisper model not found: {self.model}")
        if self.vad_model is not None and not self.vad_model.is_file():
            raise FileNotFoundError(f"VAD model not found: {self.vad_model}")
        self._stderr_file = tempfile.TemporaryFile(mode="w+b")
        argv = self._server_argv()
        self.process = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(self._startup_error())
            try:
                with urllib.request.urlopen(self.base_url, timeout=0.5):
                    return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.2)
        self.close()
        raise TimeoutError("timed out while loading the Whisper model")

    def _startup_error(self) -> str:
        if self._stderr_file is None:
            return "whisper-server exited during startup"
        self._stderr_file.seek(0)
        output = self._stderr_file.read().decode("utf-8", errors="replace")
        return "whisper-server exited during startup: " + output[-4000:]

    @staticmethod
    def _multipart(
        path: Path,
        prompt: str,
        *,
        best_of: int = 2,
        beam_size: int = -1,
    ) -> tuple[bytes, str]:
        boundary = "----piper-voice-" + uuid.uuid4().hex
        parts: list[bytes] = []

        def field(name: str, value: str) -> None:
            parts.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode("utf-8"),
                    b"\r\n",
                ]
            )

        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="file"; filename="command.wav"\r\n',
                b"Content-Type: audio/wav\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
            ]
        )
        field("temperature", "0.0")
        field("temperature_inc", "0.0")
        field("prompt", prompt)
        field("best_of", str(int(best_of)))
        field("beam_size", str(int(beam_size)))
        field("response_format", "verbose_json")
        field("no_language_probabilities", "true")
        parts.append(f"--{boundary}--\r\n".encode())
        return b"".join(parts), f"multipart/form-data; boundary={boundary}"

    def _transcribe_path(
        self,
        path: Path,
        *,
        prompt: str,
        best_of: int,
        beam_size: int,
        timeout_sec: float,
    ) -> tuple[str, float]:
        body, content_type = self._multipart(
            path, prompt, best_of=best_of, beam_size=beam_size
        )
        request = urllib.request.Request(
            self.base_url + "/inference",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = response.read()
        return parse_verbose_whisper_json(payload)

    def transcribe(
        self,
        pcm: bytes,
        timeout_sec: float = 30.0,
        *,
        prompt: str | None = None,
        best_of: int = 2,
        beam_size: int = -1,
    ) -> tuple[str, float]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("whisper-server is not running")
        fd, name = tempfile.mkstemp(prefix="piper_voice_", suffix=".wav")
        os.close(fd)
        path = Path(name)
        try:
            write_pcm_wav(path, lowpass_pcm16(pcm, self.lowpass_hz))
            return self._transcribe_path(
                path,
                prompt=self.prompt if prompt is None else prompt,
                best_of=best_of,
                beam_size=beam_size,
                timeout_sec=timeout_sec,
            )
        finally:
            path.unlink(missing_ok=True)

    def transcribe_consensus(
        self, pcm: bytes, timeout_sec: float = 30.0
    ) -> tuple[tuple[str, float], tuple[str, float]]:
        """Decode one recording twice with intentionally different priors."""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("whisper-server is not running")
        fd, name = tempfile.mkstemp(prefix="piper_voice_", suffix=".wav")
        os.close(fd)
        path = Path(name)
        try:
            write_pcm_wav(path, lowpass_pcm16(pcm, self.lowpass_hz))
            primary = self._transcribe_path(
                path,
                prompt=self.prompt,
                best_of=2,
                beam_size=-1,
                timeout_sec=timeout_sec,
            )
            verifier = self._transcribe_path(
                path,
                prompt="普通话短句转写，可能包含一至十六的数字。",
                # A second prompt gives an independent semantic check without
                # the 10+ second latency of a five-wide beam on Jetson.
                best_of=2,
                beam_size=-1,
                timeout_sec=timeout_sec,
            )
            return primary, verifier
        finally:
            path.unlink(missing_ok=True)

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
