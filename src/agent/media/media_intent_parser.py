from __future__ import annotations

"""规则层媒体意图解析（不依赖 LLM）。"""

import logging
import re
from dataclasses import dataclass

from src.agent.media.media_models import MediaRequest, MediaSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaIntentResult:
    matched: bool
    request: MediaRequest | None = None


# 播放类关键词 -> (media_type, category)
_PLAY_PATTERNS: list[tuple[re.Pattern[str], str | None, str | None]] = [
    (re.compile(r"轻音乐|放点轻|来点轻|放一首轻|来首轻"), "music", "light"),
    (re.compile(r"学习音乐|学习背景"), "music", "study"),
    (re.compile(r"流行音乐|流行歌"), "music", "pop"),
    (re.compile(r"放松|舒缓|轻松"), "music", "relaxing"),
    (re.compile(r"相声|来段笑|段子"), "xiangsheng", "short"),
    (re.compile(r"京剧"), "opera", "jingju"),
    (re.compile(r"音乐|放歌|放首|放一首歌|歌曲|听歌|播放音乐|放点音乐|来首歌"), "music", None),
]

_STOP_RE = re.compile(
    r"停一下|别放了|别放歌|不要放歌|不要放了|关掉音乐|停止播放|不听了|不放了|暂停一下|暂停播放"
)
_RESUME_RE = re.compile(r"继续放|接着播放|继续播放|继续听")
_NEXT_RE = re.compile(r"换一首|换一个|下一首|这个不好听|换个|再来一个")
_CONFIRM_RE = re.compile(r"^(好[的吧呀]?|行|可以|要|来吧|嗯|播吧|放吧|听吧|好的)$")


def _normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.strip().lower())


def parse_media_intent(text: str, *, media_playing: bool = False) -> MediaIntentResult:
    """解析用户语音中的媒体控制意图。"""

    raw = text.strip()
    if not raw:
        return MediaIntentResult(False)

    normalized = _normalize(raw)

    if _STOP_RE.search(raw):
        req = MediaRequest(action="stop_media", source=MediaSource.USER_EXPLICIT, raw_text=raw)
        logger.info("[媒体意图] stop_media: %s", raw)
        return MediaIntentResult(True, req)

    if _NEXT_RE.search(raw):
        req = MediaRequest(action="next_media", source=MediaSource.USER_EXPLICIT, raw_text=raw)
        logger.info("[媒体意图] next_media: %s", raw)
        return MediaIntentResult(True, req)

    if _RESUME_RE.search(raw):
        req = MediaRequest(action="resume_media", source=MediaSource.USER_EXPLICIT, raw_text=raw)
        logger.info("[媒体意图] resume_media: %s", raw)
        return MediaIntentResult(True, req)

    # 播放状态下短句优先媒体控制
    if media_playing and len(normalized) <= 8:
        if "停" in raw or "别" in raw:
            return MediaIntentResult(
                True,
                MediaRequest(action="stop_media", source=MediaSource.USER_EXPLICIT, raw_text=raw),
            )
        if "换" in raw:
            return MediaIntentResult(
                True,
                MediaRequest(action="next_media", source=MediaSource.USER_EXPLICIT, raw_text=raw),
            )
        if "继续" in raw:
            return MediaIntentResult(
                True,
                MediaRequest(action="resume_media", source=MediaSource.USER_EXPLICIT, raw_text=raw),
            )

    for pattern, media_type, category in _PLAY_PATTERNS:
        if pattern.search(raw):
            req = MediaRequest(
                action="play_media",
                media_type=media_type,
                category=category,
                source=MediaSource.USER_EXPLICIT,
                raw_text=raw,
            )
            logger.info("[媒体意图] play_media type=%s category=%s: %s", media_type, category, raw)
            return MediaIntentResult(True, req)

    return MediaIntentResult(False)


def is_media_confirmation(text: str) -> bool:
    """用户确认主动媒体建议。"""
    raw = text.strip()
    if not raw:
        return False
    if _CONFIRM_RE.match(raw):
        return True
    return raw in {"好啊", "可以啊", "要听", "听吧", "放吧", "来吧"}


def is_media_rejection(text: str) -> bool:
    """用户拒绝主动媒体建议。"""
    raw = text.strip()
    reject_patterns = ("不要", "不用", "算了", "不想", "别放", "不听", "下次吧", "不用了")
    return any(p in raw for p in reject_patterns)
