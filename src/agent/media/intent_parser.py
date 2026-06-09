from __future__ import annotations

"""媒体控制意图解析：LLM 结构为主；规则层仅作 speech LLM 异常时的兜底。"""

from dataclasses import dataclass

from src.agent.media.media_intent_parser import parse_media_intent
from src.agent.media.media_models import MediaSource


@dataclass(frozen=True)
class MediaControlIntent:
    intent: str = "media_control"
    action: str = ""
    media_type: str | None = None
    category: str | None = None
    track_id: str | None = None
    source: str = MediaSource.USER_EXPLICIT.value
    reply: str = ""


def parse_media_control(text: str, *, is_playing: bool = False) -> MediaControlIntent | None:
    """规则层媒体意图（仅 LLM 失败时由 SpeechLLMHandler 兜底调用）。"""

    raw = text.strip()
    if not raw:
        return None

    result = parse_media_intent(raw, media_playing=is_playing)
    if not result.matched or result.request is None:
        return None

    req = result.request
    reply = ""
    if req.action == "play_media":
        reply = "好，给你放一段。"
    elif req.action == "stop_media":
        reply = "好的，先不放了。"
    elif req.action == "next_media":
        reply = "好，给你换一个。"
    elif req.action == "resume_media":
        reply = "好，继续放。"

    return MediaControlIntent(
        action=req.action,
        media_type=req.media_type,
        category=req.category,
        source=MediaSource.USER_EXPLICIT.value,
        reply=reply,
    )


def parse_llm_media_control(data: dict) -> MediaControlIntent | None:
    if str(data.get("intent", "")).strip() != "media_control":
        return None
    action = str(data.get("action", "")).strip()
    if not action:
        return None

    media_type = data.get("media_type")
    category = data.get("category")
    track_id = data.get("track_id")
    return MediaControlIntent(
        action=action,
        media_type=str(media_type).strip() if media_type else None,
        category=str(category).strip() if category else None,
        track_id=str(track_id).strip() if track_id else None,
        source=MediaSource.USER_EXPLICIT.value,
        reply=str(data.get("reply", "")).strip(),
    )
