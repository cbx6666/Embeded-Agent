from __future__ import annotations

"""轻量 LLM 播报文案校验：去空白、拦截明显未完成句，不引入 NLP 依赖。"""

_INCOMPLETE_ENDINGS = (
    "如果",
    "因为",
    "但是",
    "要不要",
    "可以",
    "建议你",
    "的话",
    "是不是",
    "会不会",
    "要不要。",
    "可以。",
    "的话。",
    "如果。",
    "因为。",
    "但是。",
    "建议你。",
)

_MIN_REPLY_LEN = 3


def normalize_reply(text: object) -> str:
    return str(text or "").strip()


def validate_tts_reply(text: str) -> tuple[bool, str]:
    """返回 (是否可播报, 无效原因)。"""

    reply = normalize_reply(text)
    if not reply:
        return False, "empty"
    if len(reply) < _MIN_REPLY_LEN:
        return False, "too_short"

    if reply.endswith("，") or reply.endswith(","):
        return False, "trailing_comma"

    core = reply.rstrip("。！？…!?")
    for suffix in _INCOMPLETE_ENDINGS:
        if reply.endswith(suffix) or core.endswith(suffix):
            return False, f"incomplete_ending:{suffix}"

    if "，" in reply:
        last = reply.split("，")[-1].strip("。！？…!?")
        if 0 < len(last) <= 2:
            return False, "fragment_after_comma"

    return True, ""


def prepare_tts_reply(text: object) -> tuple[str | None, str | None]:
    """规范化并校验；无效时返回 (None, reason)。"""

    reply = normalize_reply(text)
    valid, reason = validate_tts_reply(reply)
    if not valid:
        return None, reason
    return reply, None
