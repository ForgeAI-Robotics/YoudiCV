#!/usr/bin/env python3
"""Listen to the DOV microphone and feed local Whisper text to the safe gateway."""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import sys
import time

from piper_handeye.voice_audio import AdaptiveEnergyVad, MicrophoneCapture, WhisperServer
from piper_handeye.voice_command import (
    DecisionStatus,
    NamedAssistantSession,
    OneShotConsensusSession,
    VoiceCommandSession,
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
DEFAULT_WHISPER = ROOT / "third_party/whisper.cpp/build/bin/whisper-server"
DEFAULT_MODEL = ROOT / "third_party/whisper.cpp/models/ggml-small.bin"
DEFAULT_VAD_MODEL = (
    ROOT / "third_party/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--whisper-server", type=Path, default=DEFAULT_WHISPER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vad-model", type=Path, default=DEFAULT_VAD_MODEL)
    parser.add_argument("--vad-threshold", type=float, default=0.55)
    parser.add_argument("--config", type=Path, default=ROOT / "piper_button_pose.yaml")
    parser.add_argument("--hold-report", type=Path, default=DEFAULT_HOLD_REPORT)
    parser.add_argument("--port", type=int, default=8178)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minimum-rms", type=float, default=260.0)
    parser.add_argument("--noise-ratio", type=float, default=2.8)
    parser.add_argument("--lowpass-hz", type=float, default=4000.0,
                        help="ASR low-pass cutoff in Hz; 0 disables filtering")
    parser.add_argument("--confirmation-timeout-sec", type=float, default=8.0)
    parser.add_argument(
        "--require-spoken-confirmation",
        action="store_true",
        help="use the legacy two-utterance confirmation mode",
    )
    parser.add_argument("--once", action="store_true", help="process one utterance and exit")
    parser.add_argument(
        "--assistant-name",
        help="enable a two-turn wake-name window, for example 优小妹",
    )
    parser.add_argument("--wake-window-sec", type=float, default=8.0)
    parser.add_argument(
        "--control-stdin",
        action="store_true",
        help="accept newline-delimited GUI control messages from stdin",
    )
    parser.add_argument(
        "--start-voice-disabled",
        action="store_true",
        help="load the microphone backend but wait for the GUI to enable ASR",
    )
    parser.add_argument(
        "--listen-delay-sec",
        type=float,
        default=0.0,
        help="delay microphone start after model load (useful for supervised tests)",
    )
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-physical-motion", action="store_true")
    args = parser.parse_args()
    if args.execute != args.acknowledge_physical_motion:
        raise RuntimeError(
            "physical execution requires both --execute and --acknowledge-physical-motion"
        )

    dispatcher = CommandDispatcher(
        execute=args.execute,
        config_path=args.config,
        hold_report=args.hold_report,
    )
    session = VoiceCommandSession(
        confirmation_timeout_sec=args.confirmation_timeout_sec, require_wake_word=True
    )
    consensus_session = OneShotConsensusSession(require_wake_word=True)
    named_session = (
        NamedAssistantSession(
            args.assistant_name, wake_window_sec=args.wake_window_sec
        )
        if args.assistant_name else None
    )
    vad = AdaptiveEnergyVad(minimum_rms=args.minimum_rms, noise_ratio=args.noise_ratio)
    microphone = MicrophoneCapture(args.source, vad=vad)
    whisper = WhisperServer(
        args.whisper_server,
        args.model,
        port=args.port,
        threads=args.threads,
        use_gpu=not args.no_gpu,
        lowpass_hz=args.lowpass_hz,
        vad_model=args.vad_model,
        vad_threshold=args.vad_threshold,
    )
    controls = JsonControlInput(sys.stdin) if args.control_stdin else None
    voice_enabled = not args.start_voice_disabled

    emit({"event": "loading_model", "model": str(args.model), "gpu": not args.no_gpu})
    try:
        whisper.start()
        if args.listen_delay_sec < 0.0 or args.listen_delay_sec > 60.0:
            raise ValueError("--listen-delay-sec must be between 0 and 60")
        emit(
            {
                "event": "model_ready",
                "listen_delay_sec": args.listen_delay_sec,
                "instruction": "倒计时后先保持安静一秒，再说一次命令",
            }
        )
        if args.listen_delay_sec:
            time.sleep(args.listen_delay_sec)
        microphone.start()
        if controls is not None:
            controls.start()
        emit(
            {
                "event": "ready",
                "mode": "execute" if args.execute else "dry_run",
                "source": args.source,
                "voice_enabled": voice_enabled,
                "instruction": (
                    f"保持安静一秒完成底噪校准，然后先说：{args.assistant_name}。"
                    "听到回应后再说：送外卖到十五楼。停止或取消无需唤醒。"
                    if args.assistant_name else
                    "保持安静一秒完成底噪校准，然后连贯说：送外卖到十五楼。停止或取消无需前缀。"
                ),
            }
        )
        count = 0

        def drain_controls() -> bool:
            """Process pending GUI controls; report manual command priority."""
            nonlocal voice_enabled
            manual_command_seen = False
            if controls is None:
                return False
            while True:
                try:
                    control_message = controls.messages.get_nowait()
                except queue.Empty:
                    break
                prior_voice_state = voice_enabled
                voice_enabled, control_events = handle_control_message(
                    control_message,
                    dispatcher,
                    voice_enabled=voice_enabled,
                )
                if control_message.get("command") in {"manual_floor", "manual_button"}:
                    manual_command_seen = True
                if prior_voice_state and not voice_enabled and named_session is not None:
                    named_session.awake_until = None
                for control_event in control_events:
                    emit(control_event)
            return manual_command_seen

        while True:
            completed = dispatcher.refresh()
            if completed is not None:
                emit(completed)
            drain_controls()
            try:
                utterance = microphone.get(timeout=0.25)
            except queue.Empty:
                continue
            if not voice_enabled:
                # Discard frames while disabled so reopening voice recognition
                # cannot execute an utterance captured during the quiet period.
                continue
            started = time.monotonic()
            if args.require_spoken_confirmation:
                text, confidence = whisper.transcribe(utterance.pcm)
                verifier_text = None
                verifier_confidence = None
            else:
                (text, confidence), (verifier_text, verifier_confidence) = (
                    whisper.transcribe_consensus(utterance.pcm)
                )
            elapsed = time.monotonic() - started
            # A toggle may arrive while Whisper is decoding. Honor it before
            # presenting or dispatching this transcript.
            manual_command_seen = drain_controls()
            if manual_command_seen:
                emit({
                    "event": "transcript_discarded",
                    "reason": "manual_command_priority",
                })
                continue
            if not voice_enabled:
                emit({"event": "transcript_discarded", "reason": "voice_disabled"})
                continue
            if named_session is not None and not args.require_spoken_confirmation:
                turn = named_session.submit(
                    text,
                    confidence,
                    verifier_text or "",
                    verifier_confidence or 0.0,
                )
                decision = turn.decision
            elif args.require_spoken_confirmation:
                turn = None
                decision = session.submit(text, confidence=confidence)
            else:
                turn = None
                decision = consensus_session.submit(
                    text,
                    confidence,
                    verifier_text or "",
                    verifier_confidence or 0.0,
                )
            transcript_event = {
                "event": "transcript",
                "text": text,
                "confidence": round(confidence, 4),
                "verifier_text": verifier_text,
                "verifier_confidence": (
                    round(verifier_confidence, 4)
                    if verifier_confidence is not None
                    else None
                ),
                "audio_sec": round(utterance.duration_sec, 3),
                "asr_sec": round(elapsed, 3),
                "peak_rms": round(utterance.peak_rms, 1),
                "threshold_rms": round(utterance.threshold_rms, 1),
            }
            relevant = dialogue_relevant_transcript(
                text, verifier_text or "", args.assistant_name
            ) or (turn is not None and turn.awakened)
            if relevant:
                emit(transcript_event)
            else:
                transcript_event.update({
                    "event": "transcript_filtered",
                    "reason": "no_wake_word_or_supported_intent",
                })
                emit(transcript_event)
            if turn is not None and turn.wake_message is not None:
                emit({
                    "event": "assistant_wake",
                    "assistant_name": args.assistant_name,
                    "message": turn.wake_message,
                })
            if decision is None or not relevant:
                count += 1
                if args.once and count >= 1:
                    return 0
                continue
            incoming = decision.command or parse_voice_command(text)
            if (
                dispatcher.active is not None
                and decision.status not in (DecisionStatus.STOP, DecisionStatus.CANCELLED)
            ):
                emit(
                    {
                        "event": "decision",
                        "status": DecisionStatus.REJECTED.value,
                        "message": "机械臂任务正在执行；当前只接受停止或取消",
                        "intent": incoming.intent.value,
                        "button": incoming.button,
                    }
                )
            else:
                emit(
                    {
                        "event": "decision",
                        "status": decision.status.value,
                        "message": decision.message,
                        "intent": decision.command.intent.value if decision.command else None,
                        "button": decision.command.button if decision.command else None,
                    }
                )
                if decision.status is DecisionStatus.STOP:
                    emit(dispatcher.stop())
                elif decision.status is DecisionStatus.CANCELLED and dispatcher.active:
                    emit(dispatcher.stop(reason="voice_cancel_active_motion"))
                elif decision.status is DecisionStatus.EXECUTE and decision.command:
                    emit(dispatcher.dispatch_motion(decision.command))
            count += 1
            if args.once and count >= 1:
                return 0
    except KeyboardInterrupt:
        if dispatcher.active is not None:
            emit(dispatcher.stop(reason="voice_microphone_interrupted"))
        return 130
    finally:
        microphone.close()
        whisper.close()


if __name__ == "__main__":
    raise SystemExit(main())
