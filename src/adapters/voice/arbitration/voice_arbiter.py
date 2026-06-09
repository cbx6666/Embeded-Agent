from __future__ import annotations

"""VoiceInteractionArbiter：统一回答 TTS 能否入队/能否播放/唤醒如何抢占。"""

from dataclasses import dataclass
from enum import Enum

from src.adapters.voice.arbitration.reminder_buffer import ReminderBuffer
from src.adapters.voice.arbitration.session_probe import VoiceSessionProbe
from src.adapters.voice.arbitration.tts_job_policy import TTSJobSpec, is_autonomous_spec, resolve_job_spec


class ArbiterAction(str, Enum):
    PLAY = "play"
    BUFFER = "buffer"
    DROP = "drop"
    REQUEUE = "requeue"


@dataclass(frozen=True)
class ArbiterDecision:
    action: ArbiterAction
    reason: str


class VoiceInteractionArbiter:
    """音频与用户会话仲裁中心。"""

    def __init__(
        self,
        *,
        probe: VoiceSessionProbe | None = None,
        reminder_buffer: ReminderBuffer | None = None,
    ) -> None:
        self._probe = probe or VoiceSessionProbe.global_probe()
        self._buffer = reminder_buffer or ReminderBuffer()

    @property
    def reminder_buffer(self) -> ReminderBuffer:
        return self._buffer

    def classify(self, *, source: str, reason: str = "", kind: str = "") -> TTSJobSpec:
        return resolve_job_spec(source=source, reason=reason, kind=kind)

    def decide_enqueue(
        self,
        spec: TTSJobSpec,
        *,
        text: str,
        source: str,
        reason: str,
        priority: int,
        payload: dict,
    ) -> ArbiterDecision:
        """Agent speak 进入 TTS 前的第一层仲裁。"""
        if spec.kind.value == "wake_ack":
            return ArbiterDecision(ArbiterAction.PLAY, "wake_ack_always")

        if not is_autonomous_spec(spec):
            if self._probe.is_media_playing() and not spec.allow_during_media:
                return ArbiterDecision(ArbiterAction.REQUEUE, "media_busy_user_reply_wait")
            return ArbiterDecision(ArbiterAction.PLAY, "user_interaction")

        if self._probe.should_defer_autonomous_event():
            reason = "user_session_or_media_defer"
        elif self._probe.is_media_playing() and not spec.allow_during_media:
            reason = "media_playing_buffer"
        else:
            return ArbiterDecision(ArbiterAction.PLAY, "autonomous_ok")

        self._buffer.offer(
            text=text,
            source=source,
            reason=reason,
            priority=priority,
            payload=payload,
            spec=spec,
        )
        return ArbiterDecision(ArbiterAction.BUFFER, reason)

    def decide_pre_play(self, spec: TTSJobSpec, *, created_at: float, now: float) -> ArbiterDecision:
        """TTS worker 真正开播前的二次检查。"""
        if spec.kind.value == "wake_ack":
            return ArbiterDecision(ArbiterAction.PLAY, "wake_ack_preempt")

        if not is_autonomous_spec(spec):
            if self._probe.is_user_voice_session_active() and spec.user_session_protected:
                return ArbiterDecision(ArbiterAction.REQUEUE, "user_session_active_requeue")
            if self._probe.is_media_playing() and not spec.allow_during_media:
                return ArbiterDecision(ArbiterAction.REQUEUE, "media_busy_requeue")
            return ArbiterDecision(ArbiterAction.PLAY, "user_reply_ok")

        if self._probe.is_user_voice_session_active():
            return ArbiterDecision(ArbiterAction.BUFFER, "user_session_buffer")

        if self._probe.is_media_playing():
            return ArbiterDecision(ArbiterAction.BUFFER, "media_playing_buffer")

        if spec.expire_seconds is not None and (now - created_at) > spec.expire_seconds:
            return ArbiterDecision(ArbiterAction.DROP, "expired")

        if self._buffer.in_post_interaction_grace(now=now):
            return ArbiterDecision(ArbiterAction.BUFFER, "post_interaction_grace")

        return ArbiterDecision(ArbiterAction.PLAY, "autonomous_ok")

    def on_wake_preempt(self) -> None:
        """唤醒命中：清空缓冲中的自主提醒，由 TTS 层取消可打断任务。"""
        self._buffer.clear_autonomous()

    def try_pop_one_buffered_reminder(self):
        """会话稳定 idle 后最多取一条缓冲提醒用于播放。"""
        from src.adapters.voice.arbitration.reminder_buffer import BufferedReminder

        if self._probe.should_defer_autonomous_event():
            return None
        if self._buffer.in_post_interaction_grace():
            return None
        return self._buffer.pop_best()
