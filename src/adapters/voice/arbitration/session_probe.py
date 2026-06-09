from __future__ import annotations

"""VoiceRuntime 状态探针：供 Agent handler 查询真实语音会话保护区。"""

import threading
import time
from typing import Protocol


class VoiceStateView(Protocol):
    def is_user_voice_protected(self) -> bool: ...
    def defer_autonomous_playback(self) -> bool: ...
    def is_idle(self) -> bool: ...
    def is_listening(self) -> bool: ...
    def state_value(self) -> str: ...


class VoiceSessionProbe:
    """VoiceRuntime 注册后成为语音会话可信源。"""

    _instance: VoiceSessionProbe | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state_view: VoiceStateView | None = None
        self._media_playing_fn: callable | None = None
        self._user_speak_active = False
        self._last_session_end = 0.0
        self.post_session_grace_sec = 2.0

    @classmethod
    def global_probe(cls) -> VoiceSessionProbe:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = VoiceSessionProbe()
            return cls._instance

    def bind_state_view(self, view: VoiceStateView | None) -> None:
        with self._lock:
            self._state_view = view

    def bind_media_playing(self, fn: callable | None) -> None:
        with self._lock:
            self._media_playing_fn = fn

    def set_user_speak_active(self, active: bool) -> None:
        with self._lock:
            self._user_speak_active = active

    def mark_session_ended(self, *, now: float | None = None) -> None:
        with self._lock:
            self._last_session_end = now if now is not None else time.time()

    def is_media_playing(self) -> bool:
        with self._lock:
            fn = self._media_playing_fn
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    def is_user_voice_session_active(self) -> bool:
        with self._lock:
            view = self._state_view
            if view is not None and view.is_user_voice_protected():
                return True
            if self._user_speak_active:
                return True
            if self._last_session_end > 0:
                if (time.time() - self._last_session_end) < self.post_session_grace_sec:
                    return True
        return False

    def is_user_reply_active(self) -> bool:
        with self._lock:
            return self._user_speak_active

    def should_defer_autonomous_event(self) -> bool:
        """VoiceRuntime 入队/播放仲裁：含保护区、用户回复、放歌、grace。"""
        with self._lock:
            view = self._state_view
            if view is not None and view.defer_autonomous_playback():
                return True
            if self._user_speak_active:
                return True
            if self.is_media_playing():
                return True
            if self._last_session_end > 0:
                if (time.time() - self._last_session_end) < self.post_session_grace_sec:
                    return True
        return False

    def reset_for_tests(self) -> None:
        """单元测试隔离：清空探针运行时状态。"""
        with self._lock:
            self._state_view = None
            self._media_playing_fn = None
            self._user_speak_active = False
            self._last_session_end = 0.0


def should_defer_autonomous_speak(*, dialogue_state: str) -> bool:
    """Handler 统一入口：dialogue_state + 放歌 + VoiceRuntime 保护区 + 用户回复 TTS。"""
    if dialogue_state in {"listening", "thinking"}:
        return True
    probe = VoiceSessionProbe.global_probe()
    if probe.is_media_playing():
        return True
    if dialogue_state == "speaking" and probe.is_user_reply_active():
        return True
    with probe._lock:  # noqa: SLF001
        view = probe._state_view
        if view is not None and view.is_user_voice_protected():
            return True
    return False
