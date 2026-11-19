from array import array
from pathlib import Path
import queue

import pytest
from unittest.mock import patch

from piper_handeye.voice_audio import PcmFrameCapture
from piper_handeye.voice_command import (
    DecisionStatus,
    StreamingNamedAssistantSession,
)
from run_voice_streaming import present_final


def test_streaming_named_session_requires_wake_and_strict_task_phrase():
    session = StreamingNamedAssistantSession(wake_window_sec=8)
    rejected = session.submit("送外卖到十五楼", now=0)
    assert rejected.decision.status is DecisionStatus.REJECTED

    wake = session.submit("优小妹", now=1)
    assert wake.awakened and wake.wake_message == "在的"

    malformed = session.submit("十五楼", now=2)
    assert malformed.decision.status is DecisionStatus.REJECTED
    accepted = session.submit("送外卖到十五楼", now=3)
    assert accepted.decision.status is DecisionStatus.EXECUTE
    assert accepted.decision.command.button == 15


def test_streaming_named_session_supports_same_utterance_and_fail_safe_stop():
    session = StreamingNamedAssistantSession()
    accepted = session.submit("优小妹送外卖到8楼", now=1)
    assert accepted.awakened
    assert accepted.decision.status is DecisionStatus.EXECUTE
    assert accepted.decision.command.button == 8
    stopped = session.submit("立即停止", now=2)
    assert stopped.decision.status is DecisionStatus.STOP


def test_wake_word_requires_exact_chinese_phrase():
    session = StreamingNamedAssistantSession()
    wake = session.submit("123", now=1)
    assert not wake.awakened
    assert not session.submit("一二", now=20).awakened
    assert not session.submit("一二三", now=21).awakened
    assert session.submit("优小妹", now=22).awakened


def test_youxiaomei_then_go_to_first_floor_is_accepted():
    session = StreamingNamedAssistantSession()
    wake = session.submit("优小妹", now=1)
    assert wake.awakened and wake.wake_message == "在的"
    command = session.submit("去一楼", now=2)
    assert command.decision.status is DecisionStatus.EXECUTE
    assert command.decision.command.button == 1


def test_strict_phrase_rejects_asr_substitutions_extra_words_and_multiple_floors():
    for text in (
        "送外卖到十五",
        "动外卖到十五楼",
        "动物外卖到十五楼",
        "动物外卖到十五楼是",
        "我不想送外卖到十五楼",
        "动物外卖到八楼和十五楼",
    ):
        session = StreamingNamedAssistantSession()
        session.submit("优小妹", now=0)
        assert session.submit(text, now=1).decision.status is DecisionStatus.REJECTED


def test_streaming_wake_persists_until_valid_command_and_suppresses_duplicate():
    session = StreamingNamedAssistantSession(wake_window_sec=3, duplicate_window_sec=3)
    session.submit("优小妹", now=0)
    unclear = session.submit("九楼", now=400)
    assert unclear.decision.status is DecisionStatus.REJECTED
    first = session.submit("送外卖到九楼", now=500)
    assert first.decision.status is DecisionStatus.EXECUTE
    duplicate = session.submit("优小妹送外卖到九楼", now=501)
    assert duplicate.decision.status is DecisionStatus.REJECTED


def test_meishile_cancels_only_the_armed_wake_state():
    session = StreamingNamedAssistantSession()
    session.submit("优小妹", now=0)
    cancelled = session.submit("没事了", now=100)
    assert cancelled.wake_cancelled
    assert cancelled.decision is None
    command = session.submit("送外卖到九楼", now=101)
    assert command.decision.status is DecisionStatus.REJECTED


def test_wake_prompts_after_ten_seconds_and_cancels_after_twenty():
    session = StreamingNamedAssistantSession()
    session.submit("优小妹", now=100)
    assert session.poll_wake(now=109.9) is None
    assert session.poll_wake(now=110.0) == "prompt"
    assert session.poll_wake(now=119.9) is None
    assert session.poll_wake(now=120.0) == "timeout"
    assert session.poll_wake(now=130.0) is None
    command = session.submit("送外卖到九楼", now=131)
    assert command.decision.status is DecisionStatus.REJECTED


def test_any_heard_words_restart_wake_silence_timer():
    session = StreamingNamedAssistantSession()
    session.submit("优小妹", now=0)
    assert session.poll_wake(now=10) == "prompt"
    unclear = session.submit("九楼", now=15)
    assert unclear.decision.status is DecisionStatus.REJECTED
    assert session.poll_wake(now=24.9) is None
    assert session.poll_wake(now=25.0) == "prompt"
    accepted = session.submit("送外卖到九楼", now=26)
    assert accepted.decision.status is DecisionStatus.EXECUTE
    assert session.poll_wake(now=100) is None


def test_default_vad_allows_one_second_between_words():
    import inspect
    from piper_handeye.sherpa_streaming import SherpaSenseVoiceAsr

    default = inspect.signature(SherpaSenseVoiceAsr).parameters[
        "min_silence_sec"
    ].default
    assert default == 1.0


def test_accepted_voice_command_requests_microphone_shutdown():
    class Dispatcher:
        active = None
        trip_stage = None

        @staticmethod
        def dispatch_motion(command):
            return {"event": "motion_started", "button": command.button}

        @staticmethod
        def start_trip(command):
            return {"event": "motion_started", "button": "down"}

    with patch("run_voice_streaming.emit") as emit:
        accepted = present_final(
            "优小妹送外卖到八楼",
            audio_sec=2.0,
            processing_sec=0.1,
            assistant_name="优小妹",
            wake_word="优小妹",
            session=StreamingNamedAssistantSession(),
            dispatcher=Dispatcher(),
        )
    assert accepted
    assert any(
        call.args[0].get("event") == "motion_started"
        for call in emit.call_args_list
    )


def test_meishile_emits_wake_cancel_without_motion():
    class Dispatcher:
        active = None

    session = StreamingNamedAssistantSession()
    session.submit("优小妹", now=0)
    with patch("run_voice_streaming.emit") as emit:
        accepted = present_final(
            "没事了",
            audio_sec=1.0,
            processing_sec=0.1,
            assistant_name="优小妹",
            wake_word="优小妹",
            session=session,
            dispatcher=Dispatcher(),
        )
    assert not accepted
    assert any(
        call.args[0].get("event") == "assistant_wake_cancelled"
        for call in emit.call_args_list
    )


def test_pcm_frame_capture_discards_oldest_audio_when_full():
    capture = PcmFrameCapture("unused", frame_samples=2, queue_size=2)
    capture._put_latest(array("h", [1, 1]).tobytes())
    capture._put_latest(array("h", [2, 2]).tobytes())
    capture._put_latest(array("h", [3, 3]).tobytes())
    assert array("h", capture.get(timeout=0.01)).tolist() == [2, 2]
    assert array("h", capture.get(timeout=0.01)).tolist() == [3, 3]
    with pytest.raises(queue.Empty):
        capture.get(timeout=0.001)


def test_silero_sensevoice_recognizes_recorded_floor_audio_without_repair():
    sherpa_onnx = pytest.importorskip("sherpa_onnx")
    del sherpa_onnx
    from piper_handeye.sherpa_streaming import SherpaSenseVoiceAsr

    root = Path(__file__).resolve().parents[1]
    model_dir = root / "third_party/sherpa-onnx-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
    recordings = (
        (root / "voice_diagnostics/20260907_173524_v_scapyz/mono_16k.wav",
         ["十五楼", "十五楼"]),
        (root / "voice_diagnostics/20260907_180650_nbq3rilo/mono_16k.wav",
         ["送外卖到十五楼"]),
    )
    if not model_dir.is_dir() or any(not item[0].is_file() for item in recordings):
        pytest.skip("local streaming model or reference recording is unavailable")
    import wave

    recognizer = SherpaSenseVoiceAsr(
        model_dir, root / "third_party/sherpa-onnx-models/silero_vad.onnx"
    )
    for recording, expected in recordings:
        recognizer.reset()
        with wave.open(str(recording), "rb") as wav:
            assert wav.getframerate() == 16_000 and wav.getnchannels() == 1
            pcm = wav.readframes(wav.getnframes())
        finals = []
        frame_bytes = recognizer.window_size * 2
        for offset in range(0, len(pcm), frame_bytes):
            for event in recognizer.accept_pcm16(pcm[offset:offset + frame_bytes]):
                if event.kind == "final":
                    finals.append(event)
        assert [event.text for event in finals] == expected
        assert all(event.processing_sec < event.audio_sec for event in finals)
