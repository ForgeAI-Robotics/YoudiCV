from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class StreamingAsrEvent:
    kind: str
    text: str
    audio_sec: float
    processing_sec: float


class SherpaSenseVoiceAsr:
    """Silero-segmented SenseVoiceSmall for low-latency command recognition."""

    def __init__(self, model_dir: Path, vad_model: Path, *, sample_rate: int = 16_000,
                 threads: int = 4, vad_threshold: float = 0.55,
                 min_silence_sec: float = 1.0, min_speech_sec: float = 0.25,
                 max_speech_sec: float = 5.0):
        import sherpa_onnx
        self.sample_rate = int(sample_rate)
        self.model_dir = Path(model_dir).resolve()
        self.vad_model = Path(vad_model).resolve()
        required = (self.model_dir / "tokens.txt", self.model_dir / "model.int8.onnx",
                    self.vad_model)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing SenseVoice model files: " + ", ".join(missing))
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(self.vad_model)
        config.silero_vad.threshold = float(vad_threshold)
        config.silero_vad.min_silence_duration = float(min_silence_sec)
        config.silero_vad.min_speech_duration = float(min_speech_sec)
        config.silero_vad.max_speech_duration = float(max_speech_sec)
        config.sample_rate = self.sample_rate
        config.num_threads = 1
        if not config.validate():
            raise ValueError("invalid Silero VAD configuration")
        self.window_size = int(config.silero_vad.window_size)
        self.vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=max(10, int(max_speech_sec) + 2)
        )
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.model_dir / "model.int8.onnx"),
            tokens=str(self.model_dir / "tokens.txt"), num_threads=int(threads),
            sample_rate=self.sample_rate, feature_dim=80,
            decoding_method="greedy_search", language="zh", use_itn=False,
        )
        self._pending = np.empty(0, dtype=np.float32)

    def reset(self) -> None:
        self.vad.reset()
        self._pending = np.empty(0, dtype=np.float32)

    def _decode_segment(self, samples: np.ndarray) -> StreamingAsrEvent | None:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        started = time.perf_counter()
        self.recognizer.decode_stream(stream)
        elapsed = time.perf_counter() - started
        text = stream.result.text.strip()
        if not text:
            return None
        return StreamingAsrEvent("final", text, len(samples) / self.sample_rate, elapsed)

    def accept_pcm16(self, pcm: bytes) -> list[StreamingAsrEvent]:
        if len(pcm) % 2:
            raise ValueError("PCM frame has an incomplete int16 sample")
        incoming = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        samples = np.concatenate((self._pending, incoming))
        events: list[StreamingAsrEvent] = []
        offset = 0
        while len(samples) - offset >= self.window_size:
            self.vad.accept_waveform(samples[offset:offset + self.window_size])
            offset += self.window_size
            while not self.vad.empty():
                segment = self.vad.front
                event = self._decode_segment(np.asarray(segment.samples, dtype=np.float32))
                self.vad.pop()
                if event is not None:
                    events.append(event)
        self._pending = samples[offset:].copy()
        return events
