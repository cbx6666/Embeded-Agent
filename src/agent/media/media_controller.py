from __future__ import annotations

"""媒体对外统一入口：Agent 决策层、语音意图层、唤醒词层均通过此模块操作播放。"""

import logging
import time
from pathlib import Path
from typing import Callable

from src.agent.media.media_library import scan_media_library
from src.agent.media.media_models import (
    MediaAgentState,
    MediaLibraryIndex,
    MediaPlaybackState,
    MediaRequest,
    MediaSelectionContext,
    MediaSource,
    MediaTrack,
)
from src.agent.media.media_player import LocalMediaPlayer, MediaPlayer
from src.agent.media.media_policy import MediaCarePolicy
from src.agent.media.media_selector import MediaSelector
from src.agent.media.media_state_machine import MediaAgentStateMachine
from src.agent.policy_config import MediaPolicy

logger = logging.getLogger(__name__)


class MediaController:
    """统一管理媒体库、选择、策略、播放与状态机。"""

    def __init__(
        self,
        *,
        music_root: str | Path = "data/music",
        policy: MediaCarePolicy | MediaPolicy | None = None,
        player: MediaPlayer | None = None,
        mock_play_sec: float | None = None,
        alsa_device: str | None = None,
        prefer_capture_device: str | None = None,
        on_state_changed: Callable[[MediaPlaybackState], None] | None = None,
    ) -> None:
        self._music_root = Path(music_root)
        self._index = scan_media_library(self._music_root)
        self._selector = MediaSelector(self._index)
        resolved_policy = (
            MediaCarePolicy(policy)
            if isinstance(policy, MediaPolicy)
            else (policy or MediaCarePolicy())
        )
        self._policy = resolved_policy
        backend = LocalMediaPlayer(
            alsa_device=alsa_device,
            prefer_capture_device=prefer_capture_device,
            mock_duration_sec=mock_play_sec,
        )
        self._player = player or MediaPlayer(backend)
        self._player.set_finished_callback(self._on_playback_finished)
        self._state_machine = MediaAgentStateMachine()
        self._on_state_changed = on_state_changed
        self._last_context = MediaSelectionContext()

    @property
    def selector(self) -> MediaSelector:
        return self._selector

    @property
    def policy(self) -> MediaCarePolicy:
        return self._policy

    @property
    def library(self) -> MediaLibraryIndex:
        return self._index

    def configure_playback(
        self,
        *,
        alsa_playback_device: str | None = None,
        prefer_capture_device: str | None = None,
    ) -> None:
        """与 TTS 共用扬声器设备解析逻辑（main 启动语音后注入）。"""
        backend = getattr(self._player, "_backend", None)
        configure = getattr(backend, "configure_devices", None)
        if callable(configure):
            configure(
                alsa_device=alsa_playback_device,
                prefer_capture_device=prefer_capture_device,
            )
            logger.info(
                "[媒体控制] 播放设备已配置 playback=%s capture_hint=%s",
                alsa_playback_device,
                prefer_capture_device,
            )

    def rescan_library(self) -> MediaLibraryIndex:
        self._index = scan_media_library(self._music_root)
        self._selector.refresh(self._index)
        return self._index

    def get_playback_state(self) -> MediaPlaybackState:
        state = self._player.playback_state
        state.agent_state = self._state_machine.state
        state.is_playing = self.is_playing()
        return state

    def is_playing(self) -> bool:
        return self._state_machine.is_playing_media() and self._player.is_playing()

    def get_agent_media_state(self) -> MediaAgentState:
        return self._state_machine.state

    def occupies_audio_output(self) -> bool:
        """播放状态独占音频输出（含打断过渡期）。"""
        return self._state_machine.occupies_audio_output()

    def build_selection_context(
        self,
        *,
        agent_state: object | None = None,
        user_context: dict | None = None,
        care_focus: str | None = None,
    ) -> MediaSelectionContext:
        user_ctx = user_context or {}
        prefs = user_ctx.get("preferences", {}) if isinstance(user_ctx, dict) else {}
        memories = user_ctx.get("memories", {}) if isinstance(user_ctx, dict) else {}
        music_styles = list(prefs.get("favorite_music_styles") or []) if isinstance(prefs, dict) else []
        music_styles.extend(_music_styles_from_user_context(user_ctx))
        playback = self.get_playback_state()

        fatigue = emotion = None
        focus_active = False
        study_sec = 0
        if agent_state is not None:
            user = getattr(agent_state, "user", None)
            focus = getattr(agent_state, "focus", None)
            if user is not None:
                fatigue = getattr(user, "fatigue_level", None)
                emotion = getattr(user, "emotion", None)
            if focus is not None:
                focus_active = bool(getattr(focus, "active", False))
                study_sec = int(getattr(focus, "elapsed_sec", 0) or 0)

        return MediaSelectionContext(
            fatigue_level=fatigue,
            emotion=emotion,
            focus_active=focus_active,
            study_duration_sec=study_sec,
            favorite_music_styles=_dedupe(music_styles),
            favorite_content_types=list(prefs.get("favorite_content_types") or []) if isinstance(prefs, dict) else [],
            disliked_topics=list(prefs.get("disliked_topics") or []) if isinstance(prefs, dict) else [],
            memories=memories if isinstance(memories, dict) else {},
            recent_played_ids=list(playback.recent_played_ids),
            media_suggestion_reject_count=playback.media_suggestion_reject_count,
            care_focus=care_focus,
        )

    def get_track_by_id(self, track_id: str) -> MediaTrack | None:
        """按 LLM 返回的 track_id 查找曲目。"""
        if not track_id:
            return None
        if self._index.count == 0:
            self.rescan_library()
        for track in self._index.tracks:
            if track.id == track_id:
                return track
        return None

    def select_track_for_request(
        self,
        request: MediaRequest,
        context: MediaSelectionContext,
        *,
        exclude_ids: list[str] | None = None,
    ) -> MediaTrack | None:
        """仅选择 track，不启动播放（供决策层构造 play_media Action）。"""
        if self._index.count == 0:
            self.rescan_library()
        if request.track_id:
            track = self.get_track_by_id(request.track_id)
            if track is not None:
                return track
        return self._selector.select(request, context, exclude_ids=exclude_ids)

    def select_track(
        self,
        request: MediaRequest,
        context: MediaSelectionContext,
        *,
        exclude_ids: list[str] | None = None,
    ) -> MediaTrack | None:
        self._last_context = context
        return self.select_track_for_request(request, context, exclude_ids=exclude_ids)

    def handle_media_request(
        self,
        request: MediaRequest,
        context: MediaSelectionContext,
        *,
        timestamp: int | None = None,
    ) -> MediaTrack | None:
        logger.info(
            "[媒体控制] 请求 action=%s type=%s cat=%s source=%s",
            request.action,
            request.media_type,
            request.category,
            request.source.value if isinstance(request.source, MediaSource) else request.source,
        )
        action = request.action
        if action == "stop_media":
            self.stop_by_user()
            return None
        if action == "pause_media":
            self.pause()
            return None
        if action == "resume_media":
            self.resume()
            return None
        if action == "next_media":
            return self.next_track(context)
        if action == "play_media":
            return self.play_selected_media(
                media_type=request.media_type,
                category=request.category,
                context=context,
                source=request.source if isinstance(request.source, MediaSource) else MediaSource.USER_EXPLICIT,
                timestamp=timestamp,
            )
        return None

    def play_selected_media(
        self,
        *,
        media_type: str | None = None,
        category: str | None = None,
        context: MediaSelectionContext | None = None,
        track: MediaTrack | None = None,
        source: MediaSource | str = MediaSource.USER_EXPLICIT,
        timestamp: int | None = None,
    ) -> MediaTrack | None:
        """选择并播放；也可直接传入已选定的 track（Action 落地路径）。"""
        ctx = context or self._last_context
        self._last_context = ctx
        resolved_source = source if isinstance(source, MediaSource) else MediaSource(str(source))
        if track is None:
            request = MediaRequest(
                action="play_media",
                media_type=media_type,
                category=category,
                source=resolved_source,
            )
            track = self._selector.select(request, ctx)
        if track is None:
            logger.warning("[媒体控制] 未找到可播放 track")
            return None
        return self._start_play(track, source=resolved_source, timestamp=timestamp)

    def play_track(
        self,
        track: MediaTrack,
        *,
        source: MediaSource | str = MediaSource.USER_EXPLICIT,
        timestamp: int | None = None,
    ) -> MediaTrack:
        resolved = source if isinstance(source, MediaSource) else MediaSource(str(source))
        return self._start_play(track, source=resolved, timestamp=timestamp)

    def next_track(self, context: MediaSelectionContext | None = None) -> MediaTrack | None:
        ctx = context or self._last_context
        current_id = self.get_playback_state().current_track_id
        playback = self.get_playback_state()
        request = MediaRequest(
            action="next_media",
            media_type=playback.current_media_type,
            category=playback.current_category,
            source=MediaSource.USER_EXPLICIT,
        )
        track = self._selector.select(request, ctx, exclude_ids=[current_id] if current_id else None)
        if track is None:
            logger.warning("[媒体控制] 无可切换 track")
            return None
        return self._start_play(track, source=MediaSource.USER_EXPLICIT)

    def pause(self) -> None:
        self._player.pause()

    def resume(self) -> None:
        self._player.resume()

    def wait_until_stopped(self, *, timeout: float = 2.0) -> bool:
        """等待播放线程与 ALSA 设备释放（唤醒应答前必须完成）。"""
        deadline = time.time() + max(0.1, timeout)
        while time.time() < deadline:
            if not self.is_playing():
                return True
            time.sleep(0.05)
        return not self.is_playing()

    def stop_for_wake_word(self) -> None:
        if not self._state_machine.is_playing_media() and not self._player.is_playing():
            return
        logger.info("[媒体控制] 唤醒词打断播放")
        self._state_machine.transition(MediaAgentState.INTERRUPTING_MEDIA, "wake_word")
        self._player.stop(reason="wake_word")
        self.wait_until_stopped(timeout=2.0)
        self._policy.record_media_finished(self._player.playback_state)
        self._state_machine.transition(MediaAgentState.LISTENING_USER_COMMAND, "wake_word_handled")
        self._notify_state()

    def stop_by_user(self) -> None:
        logger.info("[媒体控制] 用户停止播放")
        self._player.stop(reason="user")
        self._policy.record_media_finished(self._player.playback_state)
        self._state_machine.transition(MediaAgentState.IDLE, "user_stop")
        self._notify_state()

    def on_user_command_finished(self) -> None:
        if self._state_machine.state == MediaAgentState.LISTENING_USER_COMMAND and not self._player.is_playing():
            self._state_machine.transition(MediaAgentState.IDLE, "command_done")

    def record_suggestion(self, *, media_type: str, category: str, timestamp: int | None = None) -> None:
        del timestamp
        playback = self._player.playback_state
        playback.pending_suggestion = {"media_type": media_type, "category": category}
        self._notify_state()

    def play_pending_suggestion(
        self,
        context: MediaSelectionContext,
        *,
        timestamp: int | None = None,
    ) -> MediaTrack | None:
        pending = self._player.playback_state.pending_suggestion
        if not pending:
            return None
        self._player.playback_state.pending_suggestion = None
        return self.play_selected_media(
            media_type=pending.get("media_type"),
            category=pending.get("category"),
            context=context,
            source=MediaSource.AGENT_SUGGESTION,
            timestamp=timestamp,
        )

    def reject_pending_suggestion(self) -> None:
        self._policy.record_suggestion_rejected(self._player.playback_state)

    def _start_play(
        self,
        track: MediaTrack,
        *,
        source: MediaSource,
        timestamp: int | None = None,
    ) -> MediaTrack:
        self._policy.record_media_started(
            self._player.playback_state,
            track_id=track.id,
            media_type=track.media_type,
            category=track.category,
            timestamp=timestamp,
            user_explicit=source == MediaSource.USER_EXPLICIT,
        )
        self._state_machine.transition(MediaAgentState.PLAYING_MEDIA, f"play:{track.id}")
        self._player.play(track)
        self._notify_state()
        return track

    def _on_playback_finished(self) -> None:
        logger.info("[媒体控制] 播放结束")
        self._policy.record_media_finished(self._player.playback_state)
        self._state_machine.transition(MediaAgentState.IDLE, "finished")
        self._notify_state()

    def _notify_state(self) -> None:
        if self._on_state_changed:
            self._on_state_changed(self.get_playback_state())


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(item).strip())
    return out


def _music_styles_from_user_context(user_ctx: dict) -> list[str]:
    """从记忆检索结果与用户原话推断音乐风格偏好，供选曲打分。"""
    from src.agent.media.media_selector import _detect_music_preferences

    parts: list[str] = []
    memories = user_ctx.get("memories") if isinstance(user_ctx, dict) else None
    if isinstance(memories, dict):
        for items in memories.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    parts.append(str(item.get("content", "")))
    hints = user_ctx.get("memory_usage_hints") if isinstance(user_ctx, dict) else None
    if isinstance(hints, dict):
        parts.append(str(hints.get("recommended_content") or ""))
    return _detect_music_preferences(" ".join(parts).lower())
