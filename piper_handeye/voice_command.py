from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import time
import unicodedata


class VoiceIntent(str, Enum):
    PRESS_BUTTON = "press_button"
    RETURN_ORIGIN = "return_origin"
    STOP = "stop"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    UNKNOWN = "unknown"


class DecisionStatus(str, Enum):
    CONFIRMATION_REQUIRED = "confirmation_required"
    EXECUTE = "execute"
    STOP = "stop"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class VoiceCommand:
    intent: VoiceIntent
    original_text: str
    normalized_text: str
    # Numeric floor labels and the two elevator call labels are both valid
    # targets for the same guarded pressing pipeline.
    button: int | str | None = None
    bare_number: bool = False
    reason: str | None = None

    @property
    def signature(self) -> tuple[str, int | str | None]:
        return self.intent.value, self.button


@dataclass(frozen=True)
class VoiceDecision:
    status: DecisionStatus
    message: str
    command: VoiceCommand | None = None


@dataclass(frozen=True)
class NamedAssistantTurn:
    """Result of one two-pass utterance in a wake-name conversation."""

    awakened: bool
    wake_message: str | None = None
    decision: VoiceDecision | None = None
    wake_cancelled: bool = False


_DIGITS = "一二三四五六七八九"
_FINANCIAL_TRANSLATION = str.maketrans(
    {
        "壹": "一",
        "贰": "二",
        "貳": "二",
        "叁": "三",
        "參": "三",
        "肆": "四",
        "伍": "五",
        "陆": "六",
        "陸": "六",
        "優": "优",
        "柒": "七",
        "捌": "八",
        "玖": "九",
        "拾": "十",
        "幺": "一",
        "號": "号",
        "樓": "楼",
        "層": "层",
    }
)
_PUNCTUATION_RE = re.compile(r"[，。！？、；：,.!?;:'\"`]")
_ARABIC_RE = re.compile(r"(?<!\d)(\d{1,3})(?!\d)")
_CHINESE_RE = re.compile(r"一?十[一二三四五六]?|[一二三四五六七八九]")
_BUTTON_CONTEXT = (
    "按",
    "按钮",
    "楼",
    "层",
    "去",
    "前往",
    "选择",
    "button",
    "floor",
    "level",
    "press",
    "go to",
    "select",
)
_STOP_PHRASES = (
    "急停",
    "紧急停止",
    "停止机械臂",
    "立即停止",
    "停下",
    "stop",
    "emergency stop",
)
_CANCEL_PHRASES = (
    "取消",
    "算了",
    "不要",
    "别执行",
    "不执行",
    "cancel",
    "never mind",
)
_RETURN_PHRASES = (
    "返回原点",
    "回到原点",
    "回原点",
    "返回初始位置",
    "回到初始位置",
    "return to origin",
    "return home",
    "go home",
)
_CONFIRM_PHRASES = {
    "确认",
    "执行",
    "开始",
    "是",
    "对",
    "好",
    "好的",
    "可以",
    "yes",
    "confirm",
    "execute",
    "do it",
    "ok",
    "okay",
}


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.translate(_FINANCIAL_TRANSLATION)
    normalized = _PUNCTUATION_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def assistant_name_variants(name: str) -> tuple[str, ...]:
    """Return the exact normalized assistant name used by the ASR gate."""
    normalized = normalize_transcript(name).replace(" ", "")
    return (normalized,)


def _chinese_number(token: str) -> int | None:
    token = token.translate(_FINANCIAL_TRANSLATION)
    if token.startswith("一十"):
        token = token[1:]
    if token == "十":
        return 10
    if token.startswith("十") and len(token) == 2 and token[1] in _DIGITS:
        return 10 + _DIGITS.index(token[1]) + 1
    if len(token) == 1 and token in _DIGITS:
        return _DIGITS.index(token) + 1
    return None


def _number_mentions(text: str) -> list[int]:
    mentions = [int(value) for value in _ARABIC_RE.findall(text)]
    mentions.extend(
        value
        for value in (_chinese_number(item) for item in _CHINESE_RE.findall(text))
        if value is not None
    )
    return mentions


def parse_voice_command(
    text: str, *, require_wake_word: bool = False, wake_confirmed: bool = False,
) -> VoiceCommand:
    """Gate microphone motion on a same-utterance address, not speaker identity."""
    if not require_wake_word:
        return _parse_voice_command(text)
    normalized = normalize_transcript(text).replace("賣", "卖")
    body = re.sub(r"^送\s*外\s*卖\s*", "", normalized)
    if wake_confirmed:
        command = _parse_voice_command(body)
        strict_body = body.replace(" ", "")
        number = r"(?:\d{1,3}|一?十[一二三四五六]?|[一二三四五六七八九])"
        task_phrase = rf"(?:(?:送外卖)?到|去|前往)(?:第)?{number}(?:楼|层)"
        allowed = (
            command.intent is VoiceIntent.PRESS_BUTTON
            and re.fullmatch(task_phrase, strict_body) is not None
        ) or (
            command.intent is VoiceIntent.RETURN_ORIGIN
            and body in _RETURN_PHRASES
        )
        if not allowed:
            return VoiceCommand(
                VoiceIntent.UNKNOWN, str(text), normalized,
                reason="unsupported_confirmed_wake_command",
            )
        return command
    addressed = body != normalized
    command = _parse_voice_command(body)
    # Stop/cancel remain available without addressing the robot. Confirmation
    # can only act on a pending, already addressed command in the legacy flow.
    if command.intent in (VoiceIntent.STOP, VoiceIntent.CANCEL, VoiceIntent.CONFIRM):
        return command
    number = r"(?:\d{1,3}|一?十[一二三四五六]?|[一二三四五六七八九])"
    button_phrase = rf"(?:到|去|前往)(?:第)?{number}(?:楼|层)"
    strict_body = body.replace(" ", "")
    allowed = (
        command.intent is VoiceIntent.PRESS_BUTTON
        and re.fullmatch(button_phrase, strict_body) is not None
    ) or (
        command.intent is VoiceIntent.RETURN_ORIGIN
        and body in _RETURN_PHRASES
    )
    if not addressed or not allowed:
        return VoiceCommand(
            VoiceIntent.UNKNOWN, str(text), normalized,
            reason="wake_word_missing" if not addressed else "unsupported_addressed_command",
        )
    return VoiceCommand(
        command.intent, str(text), normalized, button=command.button,
        bare_number=command.bare_number,
    )


def _parse_voice_command(text: str) -> VoiceCommand:
    original = str(text)
    normalized = normalize_transcript(original)
    compact = normalized.replace(" ", "")
    if not normalized:
        return VoiceCommand(
            VoiceIntent.UNKNOWN, original, normalized, reason="empty_transcript"
        )

    # Emergency stop is deliberately checked before every other intent.
    if normalized == "停止" or any(item in normalized for item in _STOP_PHRASES):
        return VoiceCommand(VoiceIntent.STOP, original, normalized)
    if any(item in normalized for item in _CANCEL_PHRASES):
        return VoiceCommand(VoiceIntent.CANCEL, original, normalized)
    if normalized in _CONFIRM_PHRASES:
        return VoiceCommand(VoiceIntent.CONFIRM, original, normalized)
    if any(item in normalized for item in _RETURN_PHRASES):
        return VoiceCommand(VoiceIntent.RETURN_ORIGIN, original, normalized)

    bare_value = None
    if re.fullmatch(r"\d{1,3}", compact):
        bare_value = int(compact)
    else:
        bare_value = _chinese_number(compact)
    bare_number = bare_value is not None

    if bare_number:
        mentions = [bare_value]
    elif any(context in normalized for context in _BUTTON_CONTEXT):
        mentions = _number_mentions(normalized)
    else:
        return VoiceCommand(
            VoiceIntent.UNKNOWN, original, normalized, reason="no_supported_intent"
        )

    distinct = sorted(set(mentions))
    if not distinct:
        return VoiceCommand(
            VoiceIntent.UNKNOWN, original, normalized, reason="button_number_missing"
        )
    if len(distinct) != 1:
        return VoiceCommand(
            VoiceIntent.UNKNOWN, original, normalized, reason="ambiguous_button_numbers"
        )
    button = distinct[0]
    if not 1 <= button <= 16:
        return VoiceCommand(
            VoiceIntent.UNKNOWN,
            original,
            normalized,
            button=button,
            bare_number=bare_number,
            reason="button_out_of_range",
        )
    return VoiceCommand(
        VoiceIntent.PRESS_BUTTON,
        original,
        normalized,
        button=button,
        bare_number=bare_number,
    )


class VoiceCommandSession:
    """Fail-closed confirmation and duplicate-suppression state machine."""

    def __init__(
        self,
        *,
        confirmation_timeout_sec: float = 5.0,
        duplicate_window_sec: float = 3.0,
        minimum_confidence: float = 0.75,
        bare_number_minimum_confidence: float = 0.85,
        require_wake_word: bool = False,
    ):
        self.require_wake_word = require_wake_word
        self.confirmation_timeout_sec = float(confirmation_timeout_sec)
        self.duplicate_window_sec = float(duplicate_window_sec)
        self.minimum_confidence = float(minimum_confidence)
        self.bare_number_minimum_confidence = float(bare_number_minimum_confidence)
        self.pending: VoiceCommand | None = None
        self.pending_since: float | None = None
        self.last_executed_signature: tuple[str, int | None] | None = None
        self.last_executed_at: float | None = None

    def _expire_pending(self, now: float) -> bool:
        if self.pending is None or self.pending_since is None:
            return False
        if now - self.pending_since <= self.confirmation_timeout_sec:
            return False
        self.pending = None
        self.pending_since = None
        return True

    def submit(
        self,
        text: str,
        *,
        confidence: float = 1.0,
        now: float | None = None,
    ) -> VoiceDecision:
        current = time.monotonic() if now is None else float(now)
        expired = self._expire_pending(current)
        command = parse_voice_command(text, require_wake_word=self.require_wake_word)

        if command.intent is VoiceIntent.STOP:
            self.pending = None
            self.pending_since = None
            return VoiceDecision(DecisionStatus.STOP, "紧急停止", command)

        if not 0.0 <= float(confidence) <= 1.0:
            return VoiceDecision(DecisionStatus.REJECTED, "置信度必须在 0 到 1 之间")
        threshold = (
            self.bare_number_minimum_confidence
            if command.bare_number
            else self.minimum_confidence
        )
        if confidence < threshold:
            return VoiceDecision(
                DecisionStatus.REJECTED,
                f"语音置信度过低: {confidence:.2f} < {threshold:.2f}",
                command,
            )

        if command.intent is VoiceIntent.CANCEL:
            self.pending = None
            self.pending_since = None
            return VoiceDecision(DecisionStatus.CANCELLED, "已取消待确认命令", command)

        if command.intent is VoiceIntent.CONFIRM:
            if expired:
                return VoiceDecision(DecisionStatus.REJECTED, "确认已超时，请重新说出命令")
            if self.pending is None:
                return VoiceDecision(DecisionStatus.REJECTED, "当前没有待确认命令")
            selected = self.pending
            self.pending = None
            self.pending_since = None
            self.last_executed_signature = selected.signature
            self.last_executed_at = current
            return VoiceDecision(DecisionStatus.EXECUTE, "确认执行", selected)

        if command.intent not in (VoiceIntent.PRESS_BUTTON, VoiceIntent.RETURN_ORIGIN):
            return VoiceDecision(
                DecisionStatus.REJECTED,
                f"无法识别安全命令: {command.reason or 'unknown'}",
                command,
            )

        if (
            self.last_executed_signature == command.signature
            and self.last_executed_at is not None
            and current - self.last_executed_at <= self.duplicate_window_sec
        ):
            return VoiceDecision(
                DecisionStatus.REJECTED, "检测到重复命令，已忽略", command
            )

        if self.pending is not None:
            if self.pending.signature == command.signature:
                return VoiceDecision(
                    DecisionStatus.CONFIRMATION_REQUIRED,
                    self._confirmation_message(self.pending),
                    self.pending,
                )
            return VoiceDecision(
                DecisionStatus.REJECTED,
                "已有不同命令等待确认，请先说取消或确认",
                command,
            )

        self.pending = command
        self.pending_since = current
        return VoiceDecision(
            DecisionStatus.CONFIRMATION_REQUIRED,
            self._confirmation_message(command),
            command,
        )

    @staticmethod
    def _confirmation_message(command: VoiceCommand) -> str:
        if command.intent is VoiceIntent.PRESS_BUTTON:
            return f"确认按下 {command.button} 号按钮？请说确认或取消"
        return "确认返回原点？请说确认或取消"


class OneShotConsensusSession:
    """Execute one utterance only when two internal decodes agree.

    The two hypotheses must come from the same audio with different decoding
    conditions.  A STOP result remains fail-safe: either decoder may request
    it, while motion commands require agreement from both decoders.
    """

    def __init__(
        self,
        *,
        minimum_each_confidence: float = 0.25,
        minimum_geometric_confidence: float = 0.35,
        duplicate_window_sec: float = 3.0,
        require_wake_word: bool = False,
    ):
        self.require_wake_word = require_wake_word
        self.minimum_each_confidence = float(minimum_each_confidence)
        self.minimum_geometric_confidence = float(minimum_geometric_confidence)
        self.duplicate_window_sec = float(duplicate_window_sec)
        self.last_executed_signature: tuple[str, int | None] | None = None
        self.last_executed_at: float | None = None

    def submit(
        self,
        primary_text: str,
        primary_confidence: float,
        verifier_text: str,
        verifier_confidence: float,
        *,
        now: float | None = None,
    ) -> VoiceDecision:
        confidences = (float(primary_confidence), float(verifier_confidence))
        if any(not 0.0 <= value <= 1.0 for value in confidences):
            return VoiceDecision(DecisionStatus.REJECTED, "置信度必须在 0 到 1 之间")
        primary = parse_voice_command(primary_text, require_wake_word=self.require_wake_word)
        verifier = parse_voice_command(verifier_text, require_wake_word=self.require_wake_word)

        # Stopping on either hypothesis is conservative; starting motion is not.
        if VoiceIntent.STOP in (primary.intent, verifier.intent):
            command = primary if primary.intent is VoiceIntent.STOP else verifier
            return VoiceDecision(DecisionStatus.STOP, "内部识别触发紧急停止", command)
        if VoiceIntent.CANCEL in (primary.intent, verifier.intent):
            command = primary if primary.intent is VoiceIntent.CANCEL else verifier
            return VoiceDecision(DecisionStatus.CANCELLED, "内部识别触发取消", command)

        allowed = (VoiceIntent.PRESS_BUTTON, VoiceIntent.RETURN_ORIGIN)
        if primary.intent not in allowed or verifier.intent not in allowed:
            return VoiceDecision(
                DecisionStatus.REJECTED,
                "两路内部识别未同时得到有效运动命令",
                primary,
            )
        if primary.signature != verifier.signature:
            return VoiceDecision(
                DecisionStatus.REJECTED,
                "两路内部识别结果不一致",
                primary,
            )

        geometric = (confidences[0] * confidences[1]) ** 0.5
        if min(confidences) < self.minimum_each_confidence:
            return VoiceDecision(
                DecisionStatus.REJECTED,
                "至少一路内部识别置信度过低: "
                f"{confidences[0]:.2f}/{confidences[1]:.2f}",
                primary,
            )
        if geometric < self.minimum_geometric_confidence:
            return VoiceDecision(
                DecisionStatus.REJECTED,
                f"内部联合置信度过低: {geometric:.2f}",
                primary,
            )

        current = time.monotonic() if now is None else float(now)
        if (
            self.last_executed_signature == primary.signature
            and self.last_executed_at is not None
            and current - self.last_executed_at <= self.duplicate_window_sec
        ):
            return VoiceDecision(DecisionStatus.REJECTED, "检测到重复命令，已忽略", primary)
        self.last_executed_signature = primary.signature
        self.last_executed_at = current
        return VoiceDecision(DecisionStatus.EXECUTE, "单次语音内部复核通过", primary)


class NamedAssistantSession:
    """Open a short command window after hearing an explicit assistant name."""

    def __init__(
        self,
        assistant_name: str = "优小妹",
        *,
        wake_window_sec: float = 8.0,
        consensus: OneShotConsensusSession | None = None,
    ):
        self.assistant_name = normalize_transcript(assistant_name).replace(" ", "")
        if not self.assistant_name:
            raise ValueError("assistant name must not be empty")
        if not 1.0 <= float(wake_window_sec) <= 30.0:
            raise ValueError("wake window must be in 1..30 seconds")
        self.wake_window_sec = float(wake_window_sec)
        self.consensus = consensus or OneShotConsensusSession(
            require_wake_word=False
        )
        self.awake_until: float | None = None

    def _strip_name(self, text: str) -> tuple[bool, str]:
        normalized = normalize_transcript(text)
        compact = normalized.replace(" ", "")
        index = compact.find(self.assistant_name)
        if index < 0:
            return False, normalized
        return True, compact[:index] + compact[index + len(self.assistant_name):]

    @staticmethod
    def _wake_only(body: str) -> bool:
        return body in {"", "在吗", "你好", "你在吗"}

    def submit(
        self,
        primary_text: str,
        primary_confidence: float,
        verifier_text: str,
        verifier_confidence: float,
        *,
        now: float | None = None,
    ) -> NamedAssistantTurn:
        current = time.monotonic() if now is None else float(now)

        # Emergency stop and cancel remain available while asleep. This uses
        # the same conservative rule: either internal decode may stop motion.
        primary_fail_safe = parse_voice_command(primary_text)
        verifier_fail_safe = parse_voice_command(verifier_text)
        intents = (primary_fail_safe.intent, verifier_fail_safe.intent)
        if VoiceIntent.STOP in intents:
            command = (
                primary_fail_safe
                if primary_fail_safe.intent is VoiceIntent.STOP
                else verifier_fail_safe
            )
            self.awake_until = None
            return NamedAssistantTurn(False, decision=VoiceDecision(
                DecisionStatus.STOP, "内部识别触发紧急停止", command
            ))
        if VoiceIntent.CANCEL in intents:
            command = (
                primary_fail_safe
                if primary_fail_safe.intent is VoiceIntent.CANCEL
                else verifier_fail_safe
            )
            self.awake_until = None
            return NamedAssistantTurn(False, decision=VoiceDecision(
                DecisionStatus.CANCELLED, "内部识别触发取消", command
            ))

        primary_named, primary_body = self._strip_name(primary_text)
        verifier_named, verifier_body = self._strip_name(verifier_text)
        awakened = primary_named or verifier_named
        if awakened:
            self.awake_until = current + self.wake_window_sec
            if self._wake_only(primary_body) and self._wake_only(verifier_body):
                return NamedAssistantTurn(True, wake_message="在的")
            # A same-utterance motion command still requires both decoders to
            # hear the name, preventing one weak hypothesis from opening motion.
            if not (primary_named and verifier_named):
                decision = VoiceDecision(
                    DecisionStatus.REJECTED,
                    "两路内部识别未同时听到唤醒词",
                )
                return NamedAssistantTurn(True, "在的", decision)
        else:
            if self.awake_until is None or current > self.awake_until:
                self.awake_until = None
                allowed = (VoiceIntent.PRESS_BUTTON, VoiceIntent.RETURN_ORIGIN)
                if primary_fail_safe.intent in allowed or verifier_fail_safe.intent in allowed:
                    command = (
                        primary_fail_safe
                        if primary_fail_safe.intent in allowed
                        else verifier_fail_safe
                    )
                    return NamedAssistantTurn(False, decision=VoiceDecision(
                        DecisionStatus.REJECTED,
                        f"请先说{self.assistant_name}，听到“在的”后再说任务指令",
                        command,
                    ))
                return NamedAssistantTurn(False)
            primary_body, verifier_body = primary_text, verifier_text

        decision = self.consensus.submit(
            primary_body,
            primary_confidence,
            verifier_body,
            verifier_confidence,
            now=current,
        )
        if decision.status in (
            DecisionStatus.EXECUTE, DecisionStatus.STOP, DecisionStatus.CANCELLED
        ):
            self.awake_until = None
        return NamedAssistantTurn(
            awakened, "在的" if awakened else None, decision
        )


class StreamingNamedAssistantSession:
    """Fail-closed single-recognizer dialogue for local streaming ASR.

    SenseVoice does not expose calibrated token confidence through sherpa-onnx,
    so motion is gated by an exact wake name and a deliberately narrow command
    grammar instead of pretending that a score is available.
    """

    def __init__(
        self,
        wake_word: str = "优小妹",
        *,
        wake_window_sec: float = 12.0,
        duplicate_window_sec: float = 3.0,
    ):
        self.wake_word = normalize_transcript(wake_word).replace(" ", "")
        if not self.wake_word:
            raise ValueError("wake word must not be empty")
        if not 1.0 <= float(wake_window_sec) <= 30.0:
            raise ValueError("wake window must be in 1..30 seconds")
        # Retained as an accepted API parameter for older launchers. Streaming
        # wake now uses fixed 10/20-second inactivity stages instead.
        self.wake_window_sec = float(wake_window_sec)
        self.duplicate_window_sec = float(duplicate_window_sec)
        self.wake_word_variants = assistant_name_variants(wake_word)
        self.awake_until: float | None = None
        self.last_heard_at: float | None = None
        self.silence_prompt_sent = False
        self.last_executed_signature: tuple[str, int | None] | None = None
        self.last_executed_at: float | None = None

    def reset_wake(self) -> None:
        self.awake_until = None
        self.last_heard_at = None
        self.silence_prompt_sent = False

    def poll_wake(self, *, now: float | None = None) -> str | None:
        """Return one inactivity event while an armed wake waits for speech."""
        if self.awake_until is None or self.last_heard_at is None:
            return None
        current = time.monotonic() if now is None else float(now)
        silent_sec = current - self.last_heard_at
        if silent_sec >= 20.0:
            self.reset_wake()
            return "timeout"
        if silent_sec >= 10.0 and not self.silence_prompt_sent:
            self.silence_prompt_sent = True
            return "prompt"
        return None

    def _strip_name(self, text: str) -> tuple[bool, str]:
        compact = normalize_transcript(text).replace(" ", "")
        matches = (
            (compact.find(variant), variant)
            for variant in self.wake_word_variants
        )
        found = [(index, variant) for index, variant in matches if index >= 0]
        if not found:
            return False, compact
        index, variant = min(found, key=lambda item: item[0])
        return True, compact[:index] + compact[index + len(variant):]

    @staticmethod
    def _wake_only(body: str) -> bool:
        return body in {"", "在吗", "你好", "你在吗"}

    def submit(self, text: str, *, now: float | None = None) -> NamedAssistantTurn:
        current = time.monotonic() if now is None else float(now)
        compact = normalize_transcript(text).replace(" ", "")
        if compact == "没事了":
            if self.awake_until is None:
                return NamedAssistantTurn(False)
            self.reset_wake()
            return NamedAssistantTurn(False, wake_cancelled=True)
        fail_safe = parse_voice_command(text)
        if fail_safe.intent is VoiceIntent.STOP:
            self.reset_wake()
            return NamedAssistantTurn(False, decision=VoiceDecision(
                DecisionStatus.STOP, "流式识别触发紧急停止", fail_safe
            ))
        if fail_safe.intent is VoiceIntent.CANCEL:
            self.reset_wake()
            return NamedAssistantTurn(False, decision=VoiceDecision(
                DecisionStatus.CANCELLED, "流式识别触发取消", fail_safe
            ))

        named, body = self._strip_name(text)
        if named:
            self.awake_until = float("inf")
            self.last_heard_at = current
            self.silence_prompt_sent = False
            if self._wake_only(body):
                return NamedAssistantTurn(True, "在的")
            candidate_text = body
        else:
            if self.awake_until is None:
                if fail_safe.intent in (
                    VoiceIntent.PRESS_BUTTON, VoiceIntent.RETURN_ORIGIN
                ):
                    return NamedAssistantTurn(False, decision=VoiceDecision(
                        DecisionStatus.REJECTED,
                        f"请先说{self.wake_word}，听到“在的”后再说任务指令",
                        fail_safe,
                    ))
                return NamedAssistantTurn(False)
            self.last_heard_at = current
            self.silence_prompt_sent = False
            candidate_text = text

        command = parse_voice_command(
            candidate_text, require_wake_word=True, wake_confirmed=True
        )
        if command.intent not in (VoiceIntent.PRESS_BUTTON, VoiceIntent.RETURN_ORIGIN):
            return NamedAssistantTurn(named, "在的" if named else None,
                                      VoiceDecision(
                                          DecisionStatus.REJECTED,
                                          "未识别到完整任务指令，请说“送外卖到十五楼”",
                                          command,
                                      ))
        if (
            self.last_executed_signature == command.signature
            and self.last_executed_at is not None
            and current - self.last_executed_at <= self.duplicate_window_sec
        ):
            return NamedAssistantTurn(named, "在的" if named else None,
                                      VoiceDecision(
                                          DecisionStatus.REJECTED,
                                          "检测到重复命令，已忽略",
                                          command,
                                      ))
        self.last_executed_signature = command.signature
        self.last_executed_at = current
        self.reset_wake()
        return NamedAssistantTurn(named, "在的" if named else None,
                                  VoiceDecision(
                                      DecisionStatus.EXECUTE,
                                      "流式严格语法校验通过",
                                      command,
                                  ))
