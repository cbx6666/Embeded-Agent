from __future__ import annotations

"""根据请求与上下文选择具体播放 track（可解释打分，非纯随机）。"""

import logging
import random
from dataclasses import dataclass

from src.agent.media.media_models import MediaLibraryIndex, MediaRequest, MediaSelectionContext, MediaTrack

logger = logging.getLogger(__name__)

# LLM / 用户口语 category 别名 -> 库内规范 category
_CATEGORY_ALIASES: dict[str, str] = {
    "外语": "foreign",
    "外国": "foreign",
    "外文": "foreign",
    "英文": "foreign",
    "英文歌": "foreign",
    "外国歌曲": "foreign",
    "外语歌曲": "foreign",
    "抒情": "relaxing",
    "抒情歌": "relaxing",
    "轻音乐": "light",
    "相声": "short",
}

# 状态 -> 优先类别权重
_FATIGUE_CATEGORY_WEIGHTS = {
    "relaxing": 3.0,
    "light": 2.5,
    "soft": 2.5,
    "study": 1.5,
}
_NEGATIVE_EMOTION_WEIGHTS = {
    "relaxing": 3.0,
    "light": 2.0,
    "short": 2.5,  # xiangsheng short
}
_STUDY_WEIGHTS = {
    "light": 3.0,
    "study": 2.5,
    "relaxing": 2.0,
}


@dataclass
class ScoredTrack:
    track: MediaTrack
    score: float
    reasons: list[str]


class MediaSelector:
    def __init__(self, index: MediaLibraryIndex) -> None:
        self._index = index

    def refresh(self, index: MediaLibraryIndex) -> None:
        self._index = index

    @property
    def index(self) -> MediaLibraryIndex:
        return self._index

    def select(
        self,
        request: MediaRequest,
        context: MediaSelectionContext,
        *,
        exclude_ids: list[str] | None = None,
    ) -> MediaTrack | None:
        candidates = self._filter_candidates(request)
        if not candidates:
            logger.warning(
                "[媒体选择] 无候选 track：media_type=%s category=%s",
                request.media_type,
                request.category,
            )
            return None

        exclude = set(exclude_ids or [])
        if request.track_id:
            for track in candidates:
                if track.id == request.track_id:
                    return track

        scored = [self._score_track(track, request, context, exclude) for track in candidates]
        scored.sort(key=lambda s: s.score, reverse=True)
        for item in scored[:5]:
            logger.info(
                "[媒体选择] 候选 %s score=%.2f reasons=%s",
                item.track.id,
                item.score,
                item.reasons,
            )

        if not scored:
            return None

        # 在最高分候选中轻微随机，避免完全固定
        top_score = scored[0].score
        top_group = [s for s in scored if s.score >= top_score - 0.5]
        chosen = random.choice(top_group)
        logger.info("[媒体选择] 最终选择：%s (%.2f)", chosen.track.id, chosen.score)
        return chosen.track

    def _filter_candidates(self, request: MediaRequest) -> list[MediaTrack]:
        tracks = list(self._index.tracks)
        if not tracks:
            return []

        if request.media_type:
            tracks = [t for t in tracks if t.media_type == request.media_type]
        if request.category:
            canonical = _canonical_category(request.category)
            narrowed = _match_category(tracks, canonical)
            if narrowed:
                tracks = narrowed
            elif request.media_type:
                # 有 type 但 category 无精确匹配时，保留同 type 下全部（如仅指定 music）
                logger.info(
                    "[媒体选择] category=%s 无精确匹配，放宽为 media_type=%s 下全部",
                    request.category,
                    request.media_type,
                )

        # 指定了 type+category 仍为空时，按 tags 模糊匹配一次
        if not tracks and request.media_type and request.category:
            cat = request.category.lower()
            tracks = [
                t
                for t in self._index.tracks
                if t.media_type == request.media_type
                and (t.category == cat or cat in [x.lower() for x in t.tags])
            ]
        return tracks

    def _score_track(
        self,
        track: MediaTrack,
        request: MediaRequest,
        context: MediaSelectionContext,
        exclude: set[str],
    ) -> ScoredTrack:
        score = 1.0
        reasons: list[str] = ["base"]

        # 用户明确指定优先满足
        if request.media_type and track.media_type == request.media_type:
            score += 5.0
            reasons.append("user_media_type")
        req_cat = _canonical_category(request.category) if request.category else ""
        if req_cat and (track.category == req_cat or req_cat in [x.lower() for x in track.tags]):
            score += 4.0
            reasons.append("user_category")

        # 当前状态权重（未指定时生效）
        if not request.category:
            state_weights = self._state_category_weights(context)
            cat_w = state_weights.get(track.category, 0.0)
            if cat_w:
                score += cat_w
                reasons.append(f"state_{track.category}")

        # 用户画像偏好
        for style in context.favorite_music_styles:
            if style and style.lower() in {track.category, track.media_type, *track.tags}:
                score += 2.0
                reasons.append(f"pref_music_{style}")

        for ctype in context.favorite_content_types:
            if ctype and ctype.lower() in {track.media_type, track.category, *track.tags}:
                score += 2.0
                reasons.append(f"pref_content_{ctype}")

        # 长期记忆与用户口语偏好加权（含目录 tag，如「外语」文件夹）
        memory_text = _flatten_memories(context.memories)
        utterance_prefs = _detect_music_preferences(memory_text)
        for pref in utterance_prefs:
            if _track_matches_preference(track, pref):
                score += 3.5
                reasons.append(f"memory_pref_{pref}")

        for keyword in ("相声", "xiangsheng", "京剧", "jingju", "轻音乐", "light", "流行", "pop"):
            if keyword in memory_text:
                if keyword in {track.media_type, track.category} or keyword in track.title:
                    score += 1.5
                    reasons.append(f"memory_{keyword}")

        # 最近播放降权
        recent = set(context.recent_played_ids)
        if track.id in recent:
            score -= 3.0
            reasons.append("recent_played_penalty")
        if track.id in exclude:
            score -= 5.0
            reasons.append("exclude_current")

        # 长时间学习：不优先长相声
        if context.focus_active or context.study_duration_sec > 1800:
            if track.media_type == "xiangsheng" and track.category not in {"short"}:
                score -= 2.0
                reasons.append("study_no_long_xiangsheng")

        # 用户多次拒绝媒体建议 -> 降权主动推荐类（选择时仍可用，由 policy 控制推荐频率）
        if context.media_suggestion_reject_count >= 2 and track.media_type == "xiangsheng":
            score -= 0.5
            reasons.append("reject_penalty")

        return ScoredTrack(track=track, score=score, reasons=reasons)

    def _state_category_weights(self, context: MediaSelectionContext) -> dict[str, float]:
        fatigue = str(context.fatigue_level or "").lower()
        emotion = str(context.emotion or "").lower()
        negative = emotion in {
            "sad", "angry", "anxious", "fear", "stressed", "worried", "frustrated", "upset",
        }
        tired = fatigue in {"high", "severe", "exhausted", "moderate", "fatigued", "weary", "sleepy"}

        if context.care_focus == "fatigue" or tired:
            return _FATIGUE_CATEGORY_WEIGHTS
        if context.care_focus == "emotion" or negative:
            if context.favorite_content_types and "xiangsheng" in [
                c.lower() for c in context.favorite_content_types
            ]:
                return {**_NEGATIVE_EMOTION_WEIGHTS, "short": 3.0}
            return _NEGATIVE_EMOTION_WEIGHTS
        if context.focus_active or context.study_duration_sec > 1200:
            return _STUDY_WEIGHTS
        return {"light": 1.5, "relaxing": 1.5, "short": 1.0}

    def suggest_category_for_care(self, context: MediaSelectionContext) -> tuple[str, str]:
        """为主动关怀推荐推断 media_type 与 category。"""

        fatigue = str(context.fatigue_level or "").lower()
        emotion = str(context.emotion or "").lower()
        negative = emotion in {
            "sad", "angry", "anxious", "fear", "stressed", "worried", "frustrated",
        }
        tired = fatigue in {"high", "severe", "exhausted", "moderate", "fatigued"}

        if context.care_focus == "emotion" or negative:
            if "xiangsheng" in [c.lower() for c in context.favorite_content_types]:
                return "xiangsheng", "short"
            return "music", "relaxing"
        if context.care_focus == "fatigue" or tired:
            return "music", "light"
        if context.focus_active:
            return "music", "light"
        return "music", "relaxing"


def _canonical_category(category: str | None) -> str:
    raw = str(category or "").strip().lower()
    if not raw:
        return ""
    return _CATEGORY_ALIASES.get(raw, _CATEGORY_ALIASES.get(str(category).strip(), raw))


def _match_category(tracks: list[MediaTrack], category: str) -> list[MediaTrack]:
    if not category:
        return tracks
    matched = [
        t
        for t in tracks
        if t.category == category or category in [x.lower() for x in t.tags]
    ]
    return matched


def _detect_music_preferences(text: str) -> list[str]:
    prefs: list[str] = []
    lowered = text.lower()
    if any(k in lowered for k in ("外语", "外国", "外文", "英文歌", "foreign")):
        prefs.append("foreign")
    if any(k in lowered for k in ("抒情", "抒情歌")):
        prefs.append("relaxing")
    if any(k in lowered for k in ("轻音乐", "轻松", "舒缓")):
        prefs.append("light")
    return prefs


def _track_matches_preference(track: MediaTrack, pref: str) -> bool:
    tags_lower = [x.lower() for x in track.tags]
    if pref == "foreign":
        return track.category == "foreign" or "外语" in track.tags or "foreign" in tags_lower
    return track.category == pref or pref in tags_lower


def _flatten_memories(memories: dict) -> str:
    if not isinstance(memories, dict):
        return ""
    parts: list[str] = []
    for items in memories.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
    return " ".join(parts).lower()
