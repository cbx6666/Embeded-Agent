from __future__ import annotations

"""语音会话状态机：统一替代 _recording_active / _tts_running 等分散标志。"""

import threading
import time
from enum import Enum

from src.adapters.voice.runtime.logger import voice_log


class VoiceState(str, Enum):
    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    ACK_PLAYING = "ack_playing"
    LISTENING = "listening"
    ASR_RUNNING = "asr_running"
    AGENT_THINKING = "agent_thinking"
    SPEAKING = "speaking"


# 用户语音会话保护区：自主提醒不得播放。
_USER_VOICE_PROTECTED = frozenset(
    {
        VoiceState.WAKE_DETECTED,
        VoiceState.ACK_PLAYING,
        VoiceState.LISTENING,
        VoiceState.ASR_RUNNING,
        VoiceState.AGENT_THINKING,
    }
)

# 入队时延后自主提醒（与保护区一致；不含 speaking，由 user_speak_active 标记补充）。
_DEFER_AUTONOMOUS_STATES = _USER_VOICE_PROTECTED


class VoiceSessionStateMachine:
    """线程安全的语音会话状态机。"""

    def __init__(self, *, post_session_grace_sec: float = 2.0) -> None:
        self._lock = threading.RLock()
        self._state = VoiceState.IDLE
        self._reason = "init"
        self._post_session_grace_sec = post_session_grace_sec
        self._last_protected_exit_at = 0.0

    @property
    def state(self) -> VoiceState:
        with self._lock:
            return self._state

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def state_value(self) -> str:
        return self.state.value

    def transition(self, to_state: VoiceState, reason: str) -> bool:
        with self._lock:
            prev = self._state
            if prev == to_state:
                return False
            if prev in _USER_VOICE_PROTECTED and to_state == VoiceState.IDLE:
                self._last_protected_exit_at = time.time()
            self._state = to_state
            self._reason = reason
            voice_log(f"状态迁移：{prev.value} → {to_state.value}（{reason}）")
            return True

    def is_idle(self) -> bool:
        return self.state == VoiceState.IDLE

    def is_speaking(self) -> bool:
        return self.state == VoiceState.SPEAKING

    def is_listening(self) -> bool:
        return self.state == VoiceState.LISTENING

    def in_post_session_grace(self) -> bool:
        with self._lock:
            if self._last_protected_exit_at <= 0:
                return False
            return (time.time() - self._last_protected_exit_at) < self._post_session_grace_sec

    def is_user_voice_protected(self) -> bool:
        """是否处于用户语音保护区（含会话结束后的短暂 grace）。"""
        if self.state in _USER_VOICE_PROTECTED:
            return True
        return self.in_post_session_grace()

    def defer_autonomous_playback(self) -> bool:
        """自主提醒是否应进入缓冲，而非立即播报。"""
        if self.state in _DEFER_AUTONOMOUS_STATES:
            return True
        return self.in_post_session_grace()

    def in_user_session(self) -> bool:
        """是否处于唤醒后的用户语音会话（含应答/录音/识别/思考）。"""
        return self.state not in {VoiceState.IDLE, VoiceState.SPEAKING}

    def allows_wake_barge_in(self) -> bool:
        """播放中检测到唤醒词时是否应打断当前 TTS（媒体播放由 MediaController 单独处理）。"""
        return self.state == VoiceState.SPEAKING
