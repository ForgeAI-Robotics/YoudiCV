from array import array
import json
import math
import struct

import pytest

from piper_handeye.voice_audio import (
    AdaptiveEnergyVad,
    WhisperServer,
    parse_verbose_whisper_json,
    pcm16_rms,
)
from piper_handeye.voice_audio import lowpass_pcm16


def test_lowpass_preserves_speech_band_and_attenuates_high_frequency():
    def ratio(hz):
        pcm = b"".join(struct.pack("<h", round(10000 * math.sin(2 * math.pi * hz * i / 16000)))
                       for i in range(16000))
        filtered = lowpass_pcm16(pcm)
        assert len(filtered) == len(pcm)
        return pcm16_rms(filtered[2000:]) / pcm16_rms(pcm[2000:])
    assert 0.95 < ratio(1000) < 1.01
    assert ratio(7000) < 0.05


def test_lowpass_bypass_and_invalid_input():
    assert lowpass_pcm16(b"\x01\x00", 0) == b"\x01\x00"
    assert lowpass_pcm16(bytes(640)) == bytes(640)
    for cutoff in (-1, 8000, float("nan")):
        with pytest.raises(ValueError):
            lowpass_pcm16(b"", cutoff)


def frame(value: int, samples: int = 320) -> bytes:
    return array("h", [value] * samples).tobytes()


def test_pcm16_rms_constant_signal():
    assert pcm16_rms(frame(500)) == pytest.approx(500.0)


def test_adaptive_vad_emits_one_phrase():
    vad = AdaptiveEnergyVad(
        calibration_ms=100,
        pre_roll_ms=40,
        min_speech_ms=60,
        end_silence_ms=60,
        minimum_rms=200,
        noise_ratio=2,
        start_frames=2,
    )
    for _ in range(5):
        assert vad.feed(frame(50)) is None
    for _ in range(5):
        assert vad.feed(frame(800)) is None
    result = None
    for _ in range(3):
        result = vad.feed(frame(40))
    assert result is not None
    assert result.peak_rms == pytest.approx(800.0)
    assert result.duration_sec > 0.1


def test_short_noise_is_rejected():
    vad = AdaptiveEnergyVad(
        calibration_ms=40,
        min_speech_ms=100,
        end_silence_ms=40,
        minimum_rms=200,
        start_frames=1,
    )
    vad.feed(frame(50))
    vad.feed(frame(50))
    vad.feed(frame(900))
    assert vad.feed(frame(30)) is None
    assert vad.feed(frame(30)) is None


def test_parse_verbose_json_uses_token_and_no_speech_probability():
    payload = {
        "text": " 十五 ",
        "segments": [
            {
                "no_speech_prob": 0.1,
                "words": [
                    {"word": " 十", "probability": 0.9},
                    {"word": "五", "probability": 0.8},
                ],
            }
        ],
    }
    text, confidence = parse_verbose_whisper_json(json.dumps(payload))
    assert text == "十五"
    assert confidence == pytest.approx(0.765)


def test_parse_verbose_json_without_probabilities_fails_closed():
    assert parse_verbose_whisper_json('{"text":"十五"}') == ("十五", 0.0)


def test_whisper_vad_configuration_is_validated(tmp_path):
    binary = tmp_path / "server"
    model = tmp_path / "small.bin"
    vad_model = tmp_path / "vad.bin"
    for path in (binary, model, vad_model):
        path.touch()
    configured = WhisperServer(binary, model, vad_model=vad_model, vad_threshold=0.55)
    assert configured.vad_model == vad_model.resolve()
    assert configured.vad_threshold == pytest.approx(0.55)
    argv = configured._server_argv()
    assert "--suppress-nst" in argv
    assert argv[argv.index("--vad-model") + 1] == str(vad_model.resolve())
    assert argv[argv.index("--vad-threshold") + 1] == "0.55"
    with pytest.raises(ValueError):
        WhisperServer(binary, model, vad_model=vad_model, vad_threshold=1.0)
