from __future__ import annotations

"""媒体播放 Agent 状态机。"""

import logging
import threading

from src.agent.media.media_models import MediaAgentState

logger = logging.getLogger(__name__)


class MediaAgentStateMachine:
    """线程安全的媒体播放状态流转。

    IDLE -> PLAYING_MEDIA -> INTERRUPTING_MEDIA -> LISTENING_USER_COMMAND -> IDLE/PLAYING_MEDIA
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = MediaAgentState.IDLE
        self._reason = "init"

    @property
    def state(self) -> MediaAgentState:
        with self._lock:
            return self._state

    def transition(self, to_state: MediaAgentState, reason: str) -> bool:
        with self._lock:
            prev = self._state
            if prev == to_state:
                return False
            self._state = to_state
            self._reason = reason
            logger.info("[媒体状态机] %s -> %s（%s）", prev.value, to_state.value, reason)
            return True

    def is_playing_media(self) -> bool:
        return self.state == MediaAgentState.PLAYING_MEDIA

    def occupies_audio_output(self) -> bool:
        """播放状态下独占音频输出通道。"""
        return self.state in {
            MediaAgentState.PLAYING_MEDIA,
            MediaAgentState.INTERRUPTING_MEDIA,
        }
