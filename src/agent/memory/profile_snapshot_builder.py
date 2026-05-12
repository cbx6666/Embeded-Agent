"""
ProfileSnapshot 构建模块。

本模块位于 MemoryStore 之后、AgentContextBuilder 之前，负责把长期记忆、
显式 profile 偏好和短期消息压缩成决策可消费的 ProfileSnapshot。上游输入是
MemoryStore、UserProfileService 和 AgentState，下游输出是 ProfileSnapshot。

本模块不调用 LLM、不写入 MemoryStore、不生成 Intent，也不直接影响设备执行。
它的职责是让决策层只读快照，而不是直接读取长期记忆数据库。
"""

from __future__ import annotations

"""
ProfileSnapshot 构建模块。

本模块位于 MemoryStore 之后、AgentContextBuilder 之前，负责把长期记忆、
显式 profile 偏好和短期消息压缩成决策可消费的 ProfileSnapshot。上游输入是
MemoryStore、UserProfileService 和 AgentState，下游输出是 ProfileSnapshot。

本模块不调用 LLM、不写入 MemoryStore、不生成 Intent，也不直接影响设备执行。
它的职责是让决策层只读快照，而不是直接读取长期记忆数据库。
"""

from dataclasses import dataclass, field
from typing import Any

from src.agent.event import Event
from src.agent.memory.memory_store import MemoryStore, StoredMemory
from src.agent.state import AgentState
from src.services.user_profile_service import UserProfileService


@dataclass
class ProfileSnapshot:
    """决策层唯一可见的用户画像快照。

    它把长期记忆按用途拆成显式偏好、行为模式、交互风格、当前约束、近期上下文
    和不确定记忆，供 AgentContextBuilder 进行相关性检索。
    """
    explicit_preferences: list[dict[str, Any]] = field(default_factory=list)
    behavior_patterns: list[dict[str, Any]] = field(default_factory=list)
    interaction_style: list[dict[str, Any]] = field(default_factory=list)
    active_constraints: list[dict[str, Any]] = field(default_factory=list)
    recent_context: list[dict[str, Any]] = field(default_factory=list)
    uncertain_memories: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转成 prompt 和 trace 都能使用的稳定字典。"""

        return {
            "explicit_preferences": list(self.explicit_preferences),
            "behavior_patterns": list(self.behavior_patterns),
            "interaction_style": list(self.interaction_style),
            "active_constraints": list(self.active_constraints),
            "recent_context": list(self.recent_context),
            "uncertain_memories": list(self.uncertain_memories),
        }

    def retrieve_relevant(self, *, event_type: str, text: str = "", limit: int = 8) -> list[dict[str, Any]]:
        """检索与当前事件相关的少量记忆。

        当前实现采用轻量词项匹配，只用于压缩上下文，不承担语义理解；真正的
        记忆价值判断和整合由 LLM-managed memory 角色完成。
        """

        del event_type
        pool = (
            self.active_constraints
            + self.explicit_preferences
            + self.interaction_style
            + self.behavior_patterns
            + self.recent_context
            + self.uncertain_memories
        )
        if not text:
            return pool[:limit]
        text_terms = {part.lower() for part in text.replace(",", " ").split() if len(part) > 1}
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in pool:
            content = str(item.get("content", "")).lower()
            score = sum(1 for term in text_terms if term in content)
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]


class ProfileSnapshotBuilder:
    """画像快照构建器。

    输入 user_id、当前状态、可选 profile service，输出 ProfileSnapshot。它是
    DecisionPipeline 读取记忆的唯一入口。
    """

    def __init__(self, memory_store: MemoryStore | None = None) -> None:
        self.memory_store = memory_store or MemoryStore()

    def build(
        self,
        *,
        user_id: str,
        state: AgentState,
        event: Event | None = None,
        profile_service: UserProfileService | None = None,
    ) -> ProfileSnapshot:
        """生成紧凑画像快照。

        长期记忆按类型分桶；显式 profile 偏好以最高置信度注入；短期消息只取
        最近几条，避免一次对话污染长期画像。
        """

        del event
        memories = self.memory_store.list(user_id)
        snapshot = ProfileSnapshot()

        for memory in sorted(memories, key=lambda item: item.updated_at, reverse=True):
            rendered = _render_memory(memory)
            if memory.confidence < 0.55 or memory.memory_type == "uncertain":
                snapshot.uncertain_memories.append(rendered)
            elif memory.memory_type == "explicit_preference":
                snapshot.explicit_preferences.append(rendered)
            elif memory.memory_type == "behavior_pattern":
                snapshot.behavior_patterns.append(rendered)
            elif memory.memory_type == "interaction_style":
                snapshot.interaction_style.append(rendered)
            elif memory.memory_type == "active_constraint":
                snapshot.active_constraints.append(rendered)
            else:
                snapshot.recent_context.append(rendered)

        if profile_service is not None:
            profile = profile_service.get_user(user_id)
            preference = profile.preference
            explicit = []
            for key in (
                "favorite_content_types",
                "favorite_music_styles",
                "disliked_topics",
                "reminder_style",
                "speech_style",
                "tts_voice",
                "tts_speed",
                "tts_volume",
            ):
                value = getattr(preference, key)
                if value is None or value == []:
                    continue
                explicit.append({"memory_type": "explicit_preference", "content": f"{key}: {value}", "confidence": 1.0})
            snapshot.explicit_preferences = explicit + snapshot.explicit_preferences

        if state.memory.recent_messages:
            for message in state.memory.recent_messages[-4:]:
                snapshot.recent_context.append(
                    {
                        "memory_type": "recent_context",
                        "content": f"{message.get('role')}: {message.get('text')}",
                        "confidence": 1.0,
                        "updated_at": message.get("timestamp"),
                    }
                )

        return snapshot


def _render_memory(memory: StoredMemory) -> dict[str, Any]:
    """把 StoredMemory 渲染成 prompt 安全的摘要，不暴露 store 内部实现。"""

    return {
        "id": memory.id,
        "memory_type": memory.memory_type,
        "content": memory.content,
        "confidence": memory.confidence,
        "updated_at": memory.updated_at,
        "evidence_count": len(memory.evidence),
    }
