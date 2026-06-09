from __future__ import annotations

"""媒体相关数据结构。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MediaAgentState(str, Enum):
    """Agent 层媒体播放状态机。"""

    IDLE = "idle"
    PLAYING_MEDIA = "playing_media"
    INTERRUPTING_MEDIA = "interrupting_media"
    LISTENING_USER_COMMAND = "listening_user_command"


class MediaSource(str, Enum):
    """媒体请求来源：用户主动 vs Agent 主动推荐后确认。"""

    USER_EXPLICIT = "user_explicit"
    AGENT_SUGGESTION = "agent_suggestion"


@dataclass
class MediaTrack:
    """本地媒体库中的一条音频。"""

    id: str
    title: str
    path: str
    media_type: str  # music / xiangsheng / opera / unknown
    category: str  # light / study / relaxing / pop / jingju / short / ...
    tags: list[str] = field(default_factory=list)
    duration: float | None = None


@dataclass
class MediaLibraryIndex:
    """扫描 data/music 后的索引。"""

    root: str
    tracks: list[MediaTrack] = field(default_factory=list)
    media_types: list[str] = field(default_factory=list)
    categories_by_type: dict[str, list[str]] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.tracks)


@dataclass
class MediaPlaybackState:
    """当前播放状态快照。"""

    agent_state: MediaAgentState = MediaAgentState.IDLE
    is_playing: bool = False
    current_track_id: str | None = None
    current_media_type: str | None = None
    current_category: str | None = None
    started_at: int | None = None
    last_user_requested_at: int | None = None
    last_suggested_at: int | None = None
    last_finished_at: int | None = None
    interrupted_by_wake_word: bool = False
    recent_played_ids: list[str] = field(default_factory=list)
    media_suggestion_reject_count: int = 0
    pending_suggestion: dict[str, Any] | None = None


@dataclass
class MediaRequest:
    """播放/控制请求。"""

    action: str  # play_media / stop_media / pause_media / resume_media / next_media
    media_type: str | None = None
    category: str | None = None
    source: MediaSource = MediaSource.USER_EXPLICIT
    track_id: str | None = None
    raw_text: str = ""


@dataclass
class MediaSelectionContext:
    """媒体选择上下文。"""

    fatigue_level: str | None = None
    emotion: str | None = None
    focus_active: bool = False
    study_duration_sec: int = 0
    favorite_music_styles: list[str] = field(default_factory=list)
    favorite_content_types: list[str] = field(default_factory=list)
    disliked_topics: list[str] = field(default_factory=list)
    memories: dict[str, Any] = field(default_factory=dict)
    recent_played_ids: list[str] = field(default_factory=list)
    media_suggestion_reject_count: int = 0
    care_focus: str | None = None  # fatigue / emotion / posture / study


@dataclass
class MediaSuggestionResult:
    """主动关怀媒体建议结果。"""

    can_suggest: bool
    media_type: str | None = None
    category: str | None = None
    suggestion_text: str = ""
    cooldown_reason: str = ""


@dataclass
class CareStrategyChoice:
    """关怀策略选择结果。"""

    strategy: str  # media_suggestion / wellness_*
    intent_type: str
    reason_key: str
    default_text_attr: str
    media_type: str | None = None
    category: str | None = None
    suggestion_text: str = ""
