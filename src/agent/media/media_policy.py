from __future__ import annotations

"""媒体主动建议策略：首次问放歌，之后每间隔 2 次纯 wellness 关怀才可再问。"""

import logging
import time

from src.agent.media.media_models import (
    CareStrategyChoice,
    MediaLibraryIndex,
    MediaPlaybackState,
    MediaSuggestionResult,
)
from src.agent.policy_config import MediaPolicy

logger = logging.getLogger(__name__)

# care_focus -> 优先尝试的 (media_type, category)，须曲库中真实存在才建议。
_FOCUS_PREFERRED_CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "fatigue": [("music", "light"), ("music", "relaxing")],
    "emotion": [("music", "relaxing"), ("music", "light")],
    "posture": [("music", "light")],
}

# 距上次询问放歌后，需要累计的纯 wellness 关怀播报次数。
WELLNESS_CARES_BETWEEN_MEDIA_ASK = 2


def _library_has_category(library: MediaLibraryIndex, media_type: str, category: str) -> bool:
    return any(
        t.media_type == media_type and t.category == category for t in library.tracks
    )


def _pick_from_library(
    library: MediaLibraryIndex,
    care_focus: str,
) -> tuple[str, str] | None:
    preferred = _FOCUS_PREFERRED_CATEGORIES.get(care_focus)
    if not preferred:
        logger.info("[媒体策略] media_suggestion_skipped: unknown_focus %s", care_focus)
        return None
    for media_type, category in preferred:
        if _library_has_category(library, media_type, category):
            return media_type, category
    for media_type, categories in library.categories_by_type.items():
        for category in categories:
            if _library_has_category(library, media_type, category):
                return media_type, category
    logger.info("[媒体策略] media_suggestion_skipped: no_library_match focus=%s", care_focus)
    return None


class MediaCarePolicy:
    """主动关怀中的媒体建议：计数式冷却，其余轮次走 wellness LLM 纯关怀。"""

    def __init__(self, policy: MediaPolicy | None = None) -> None:
        self.policy = policy or MediaPolicy()

    def can_suggest_media(
        self,
        *,
        media_suggestion_ever_asked: bool = False,
        wellness_cares_since_media_ask: int = 0,
    ) -> MediaSuggestionResult:
        """是否可在本轮 wellness 中询问放歌（不含用户主动点播）。"""

        if not media_suggestion_ever_asked:
            return MediaSuggestionResult(can_suggest=True)
        if int(wellness_cares_since_media_ask) >= WELLNESS_CARES_BETWEEN_MEDIA_ASK:
            return MediaSuggestionResult(can_suggest=True)
        reason = (
            f"wellness_cares_since_media_ask "
            f"({int(wellness_cares_since_media_ask)}/{WELLNESS_CARES_BETWEEN_MEDIA_ASK})"
        )
        logger.info("[媒体策略] 本轮不询问放歌（%s），wellness 走 LLM 关怀", reason)
        return MediaSuggestionResult(can_suggest=False, cooldown_reason=reason)

    def try_media_suggestion(
        self,
        *,
        care_focus: str,
        media_suggestion_ever_asked: bool = False,
        wellness_cares_since_media_ask: int = 0,
        library: MediaLibraryIndex | None = None,
    ) -> CareStrategyChoice | None:
        """若计数允许且曲库有匹配类别则返回媒体询问策略；否则 None。"""

        check = self.can_suggest_media(
            media_suggestion_ever_asked=media_suggestion_ever_asked,
            wellness_cares_since_media_ask=wellness_cares_since_media_ask,
        )
        if not check.can_suggest:
            return None

        if library is None or library.count == 0:
            logger.info("[媒体策略] media_suggestion_skipped: no_library_match (empty index)")
            return None

        picked = _pick_from_library(library, care_focus)
        if picked is None:
            return None

        media_type, category = picked
        logger.info(
            "[媒体策略] 本轮可询问放歌：focus=%s type=%s cat=%s",
            care_focus,
            media_type,
            category,
        )
        return CareStrategyChoice(
            strategy="media_suggestion",
            intent_type="suggest_media",
            reason_key="media_suggestion",
            default_text_attr="media_suggestion_text",
            media_type=media_type,
            category=category,
            suggestion_text="",
        )

    def record_media_started(
        self,
        playback: MediaPlaybackState,
        *,
        track_id: str,
        media_type: str,
        category: str,
        timestamp: int | None = None,
        user_explicit: bool = True,
    ) -> None:
        now = int(timestamp or time.time())
        if user_explicit:
            playback.last_user_requested_at = now
        playback.started_at = now
        playback.current_track_id = track_id
        playback.current_media_type = media_type
        playback.current_category = category
        playback.is_playing = True
        playback.interrupted_by_wake_word = False
        if track_id not in playback.recent_played_ids:
            playback.recent_played_ids = [track_id, *playback.recent_played_ids[:19]]

    def record_media_finished(self, playback: MediaPlaybackState, *, timestamp: int | None = None) -> None:
        playback.last_finished_at = int(timestamp or time.time())
        playback.is_playing = False
        playback.current_track_id = None

    def record_suggestion_rejected(self, playback: MediaPlaybackState) -> None:
        playback.media_suggestion_reject_count += 1
        playback.pending_suggestion = None
