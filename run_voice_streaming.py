#!/usr/bin/env python3
"""Run the local Silero + SenseVoiceSmall voice task backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import signal
import sys
import time

from piper_handeye.sherpa_streaming import SherpaSenseVoiceAsr
from piper_handeye.voice_audio import PcmFrameCapture
from piper_handeye.voice_command import (
    DecisionStatus,
    StreamingNamedAssistantSession,
    VoiceIntent,
    parse_voice_command,
)
from piper_handeye.voice_runtime import (
    JsonControlInput,
    dialogue_relevant_transcript,
    emit,
    handle_control_message,
)
from run_voice_command_gateway import CommandDispatcher, DEFAULT_HOLD_REPORT, ROOT


DEFAULT_SOURCE = "alsa_input.usb-CF-IC_DOV_2025-0711-1800-00.analog-stereo"
DEFAULT_MODEL_DIR = (
    ROOT / "third_party/sherpa-onnx-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
)
DEFAULT_VAD_MODEL = ROOT / "third_party/sherpa-onnx-models/silero_vad.onnx"


def present_final(
    text: str,
    *,
    audio_sec: float,
    processing_sec: float,
    assistant_name: str,
    wake_word: str,
    session: StreamingNamedAssistantSession,
    dispatcher: CommandDispatcher,
) -> bool:
    turn = session.submit(text)
    relevant = (
        dialogue_relevant_transcript(text, wake_word=wake_word)
        or turn.awakened
        or turn.decision is not None
        or turn.wake_cancelled
    )
    transcript = {
        "event": "transcript" if relevant else "transcript_filtered",
        "text": text,
        "confidence": None,
        "verifier_text": None,
        "verifier_confidence": None,
        "audio_sec": round(audio_sec, 3),
        "asr_sec": round(processing_sec, 3),
        "asr_engine": "sherpa-onnx-sense-voice-small-int8",
    }
    if not relevant:
        transcript["reason"] = "no_wake_word_or_supported_intent"
    emit(transcript)
    if turn.wake_message is not None:
        emit({
            "event": "assistant_wake",
            "assistant_name": assistant_name,
            "message": turn.wake_message,
        })
    if turn.wake_cancelled:
        emit({"event": "assistant_wake_cancelled"})
        return False
    decision = turn.decision
    if decision is None:
        return False
    incoming = decision.command or parse_voice_command(text)
    if (
        (dispatcher.active is not None or dispatcher.trip_stage is not None)
        and decision.status not in (DecisionStatus.STOP, DecisionStatus.CANCELLED)
    ):
        emit({
            "event": "decision",
            "status": DecisionStatus.REJECTED.value,
            "message": "机械臂任务正在执行；当前只接受停止或取消",
            "intent": incoming.intent.value,
            "button": incoming.button,
        })
        return False
    emit({
        "event": "decision",
        "status": decision.status.value,
        "message": decision.message,
        "intent": decision.command.intent.value if decision.command else None,
        "button": decision.command.button if decision.command else None,
    })
    if decision.status is DecisionStatus.STOP:
        emit(dispatcher.stop())
    elif decision.status is DecisionStatus.CANCELLED and dispatcher.active:
        emit(dispatcher.stop(reason="voice_cancel_active_motion"))
    elif decision.status is DecisionStatus.EXECUTE and decision.command:
        try:
            if decision.command.intent is VoiceIntent.PRESS_BUTTON and isinstance(
                decision.command.button, int
            ):
                dispatched = dispatcher.start_trip(decision.command)
            else:
                dispatched = dispatcher.dispatch_motion(decision.command)
        except Exception as exc:
            emit({
                "event": "decision",
                "status": DecisionStatus.REJECTED.value,
                "message": str(exc),
                "intent": decision.command.intent.value,
                "button": decision.command.button,
            })
            return False
        emit(dispatched)
        return dispatched.get("event") in {"motion_started", "dry_run"}
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--vad-model", type=Path, default=DEFAULT_VAD_MODEL)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-silence-sec", type=float, default=1.0)
    parser.add_argument("--vad-min-speech-sec", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--assistant-name", default="优小妹")
    parser.add_argument("--wake-word", default="优小妹")
    parser.add_argument("--wake-window-sec", type=float, default=12.0)
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--hold-report", type=Path, default=DEFAULT_HOLD_REPORT)
    parser.add_argument("--control-stdin", action="store_true")
    parser.add_argument("--start-voice-disabled", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-physical-motion", action="store_true")
    args = parser.parse_args()
    if args.execute != args.acknowledge_physical_motion:
        raise RuntimeError(
            "physical execution requires both --execute and --acknowledge-physical-motion"
        )

    dispatcher = CommandDispatcher(
        execute=args.execute, config_path=args.config, hold_report=args.hold_report
    )
    session = StreamingNamedAssistantSession(
        args.wake_word, wake_window_sec=args.wake_window_sec
    )
    controls = JsonControlInput(sys.stdin) if args.control_stdin else None
    microphone = None
    voice_enabled = not args.start_voice_disabled
    recognizer = None
    interrupted = False

    def request_stop(_signum, _frame) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    emit({
        "event": "loading_model",
        "model": str(args.model_dir),
        "asr_engine": "sherpa-onnx-sense-voice-small-int8",
        "whisper_fallback": False,
    })
    try:
        started = time.monotonic()
        recognizer = SherpaSenseVoiceAsr(
            args.model_dir,
            args.vad_model,
            threads=args.threads,
            vad_threshold=args.vad_threshold,
            min_silence_sec=args.vad_min_silence_sec,
            min_speech_sec=args.vad_min_speech_sec,
        )
        microphone = PcmFrameCapture(
            args.source, frame_samples=recognizer.window_size
        )
        emit({
            "event": "model_ready",
            "model_load_sec": round(time.monotonic() - started, 3),
            "asr_engine": "sherpa-onnx-sense-voice-small-int8",
            "whisper_fallback": False,
        })
        microphone.start()
        if controls is not None:
            controls.start()
        emit({
            "event": "ready",
            "mode": "execute" if args.execute else "dry_run",
            "source": args.source,
            "voice_enabled": voice_enabled,
            "asr_engine": "sherpa-onnx-sense-voice-small-int8",
            "whisper_fallback": False,
            "instruction": (
                f"先说：{args.wake_word}。听到回应后再说：送外卖到十五楼。"
                "停止或取消无需唤醒。"
            ),
        })

        def drain_controls() -> bool:
            nonlocal voice_enabled
            manual_seen = False
            if controls is None:
                return False
            while True:
                try:
                    message = controls.messages.get_nowait()
                except queue.Empty:
                    break
                previous = voice_enabled
                voice_enabled, events = handle_control_message(
                    message, dispatcher, voice_enabled=voice_enabled
                )
                if message.get("command") in {"manual_floor", "manual_button"}:
                    manual_seen = True
                if previous != voice_enabled:
                    recognizer.reset()
                    microphone.discard_pending()
                    session.reset_wake()
                for event in events:
                    emit(event)
            return manual_seen

        while not interrupted:
            completed = dispatcher.refresh()
            if completed is not None:
                emit(completed)
            drain_controls()
            if voice_enabled:
                wake_timer = session.poll_wake()
                if wake_timer == "prompt":
                    emit({"event": "assistant_wake_prompt"})
                elif wake_timer == "timeout":
                    emit({"event": "assistant_wake_timeout"})
            try:
                pcm = microphone.get(timeout=0.05)
            except queue.Empty:
                continue
            except RuntimeError:
                if interrupted:
                    break
                raise
            if not voice_enabled:
                continue
            for event in recognizer.accept_pcm16(pcm):
                manual_seen = drain_controls()
                if manual_seen or not voice_enabled:
                    recognizer.reset()
                    emit({
                        "event": "transcript_discarded",
                        "reason": (
                            "manual_command_priority" if manual_seen else "voice_disabled"
                        ),
                    })
                    break
                if event.kind == "partial":
                    if dialogue_relevant_transcript(
                        event.text, wake_word=args.wake_word
                    ):
                        emit({
                            "event": "partial_transcript",
                            "text": event.text,
                            "audio_sec": round(event.audio_sec, 3),
                            "asr_sec": round(event.processing_sec, 3),
                        })
                    continue
                command_accepted = present_final(
                    event.text,
                    audio_sec=event.audio_sec,
                    processing_sec=event.processing_sec,
                    assistant_name=args.assistant_name,
                    wake_word=args.wake_word,
                    session=session,
                    dispatcher=dispatcher,
                )
                if command_accepted:
                    voice_enabled = False
                    recognizer.reset()
                    microphone.discard_pending()
                    session.reset_wake()
                    emit({
                        "event": "voice_state",
                        "enabled": False,
                        "reason": "command_execution_started",
                    })
                    break
        if dispatcher.active is not None:
            emit(dispatcher.stop(reason="voice_streaming_interrupted"))
        return 130 if interrupted else 0
    except Exception as exc:
        emit({"event": "backend_error", "error": f"{type(exc).__name__}: {exc}"})
        if dispatcher.active is not None:
            emit(dispatcher.stop(reason="voice_streaming_backend_error"))
        return 1
    finally:
        if microphone is not None:
            microphone.close()


if __name__ == "__main__":
    raise SystemExit(main())
