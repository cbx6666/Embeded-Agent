from __future__ import annotations

"""结构化用户记忆模型。

``MemoryItem`` 是 LLM 从用户言谈中异步抽取出的一条长期有用的用户画像信息。它不是
短期 RuntimeHistory，也不是显式 UserProfile：UserProfile 是用户/系统明确写入的权威
资料，MemoryItem 是从自然对话里轻度归纳出来的“系统观察结论”，必须保留 evidence。

记忆类型闭集（``MEMORY_TYPES``）：

- ``preference``：显式偏好（喜欢简短回答、喜欢轻松语气）
- ``hobby``：兴趣爱好（音乐、相声、足球、篮球、游戏）
- ``habit``：行为习惯（经常熬夜、喜欢长时间专注、不喜欢番茄钟）
- ``emotion_pattern``：情绪/压力模式（考试前焦虑、卡住时烦躁）
- ``work_style``：学习/工作方式（先做难题、按计划推进、短目标拆解）
- ``interaction_style``：互动风格（自然一点、像朋友、别说教）
- ``care_strategy``：可用于主动关怀的建议素材（累了可建议听音乐）
- ``dislike``：明确禁止或反感（不要频繁提醒、不要太正式）
- ``fact``：其它长期有用事实
"""

import uuid
from dataclasses import dataclass, field, fields
from typing import Any

MEMORY_TYPES: frozenset[str] = frozenset(
    {
        "preference",
        "hobby",
        "habit",
        "emotion_pattern",
        "work_style",
        "interaction_style",
        "care_strategy",
        "dislike",
        "fact",
    }
)

# 记忆类型 -> 在 user_context.memories 里的分组键（复数、可读）。
GROUP_KEY_BY_TYPE: dict[str, str] = {
    "preference": "preferences",
    "hobby": "hobbies",
    "habit": "habits",
    "emotion_pattern": "emotion_patterns",
    "work_style": "work_styles",
    "interaction_style": "interaction_styles",
    "care_strategy": "care_strategies",
    "dislike": "dislikes",
    "fact": "facts",
}


@dataclass
class MemoryItem:
    """单条结构化用户记忆。"""

    user_id: str
    type: str
    content: str
    evidence: str = ""
    confidence: float = 0.0
    tags: list[str] = field(default_factory=list)
    source_event: str = "speech_recognized"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: int = 0
    updated_at: int = 0
    last_used_at: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type,
            "content": self.content,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "source_event": self.source_event,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in names}
        values["tags"] = _normalize_tags(values.get("tags"))
        values["user_id"] = str(values.get("user_id") or "default")
        values["type"] = str(values.get("type") or "fact")
        values["content"] = str(values.get("content") or "").strip()
        values["evidence"] = str(values.get("evidence") or "").strip()
        values["confidence"] = _clamp_confidence(values.get("confidence"))
        return cls(**values)


def make_memory_item(
    *,
    user_id: str,
    type: str,
    content: str,
    evidence: str = "",
    confidence: float = 0.0,
    tags: list[str] | None = None,
    source_event: str = "speech_recognized",
    timestamp: int,
) -> MemoryItem:
    """构造一条带时间戳与归一化字段的 MemoryItem。"""

    ts = int(timestamp)
    return MemoryItem(
        user_id=str(user_id or "default"),
        type=str(type),
        content=str(content).strip(),
        evidence=str(evidence or "").strip(),
        confidence=_clamp_confidence(confidence),
        tags=_normalize_tags(tags),
        source_event=str(source_event or "speech_recognized"),
        created_at=ts,
        updated_at=ts,
        last_used_at=None,
    )


def _clamp_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, conf))


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [str(value)]
    seen: list[str] = []
    for item in items:
        tag = str(item).strip().lower()
        if tag and tag not in seen:
            seen.append(tag)
    return seen
