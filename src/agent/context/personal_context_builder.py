from __future__ import annotations

"""PersonalContext 构建器。

它是什么：
PersonalContextBuilder 是决策层个性化上下文的唯一构建入口，组合 RuntimeHistory、
LongTermMemory 和 UserProfile。

它不是什么：
它不是长期记忆管线，不调用 LLM，不写 store；也不是 DecisionPipeline，不生成 intent。

为什么存在：
系统必须有一个地方明确 Authoritative Source：显式偏好来自 UserProfile，行为偏好来自
LongTermMemory，最近对话来自 RuntimeHistory，决策上下文来自 PersonalContextBuilder。

边界：
DecisionPipeline 只接收 PersonalContext，不直接读取 LongTermMemoryStore 或 UserProfileStore。
"""

from typing import Any

from src.agent.context.personal_context import PersonalContext
from src.agent.event import Event
from src.agent.memory.long_term_memory import LongTermMemory
from src.agent.state import AgentState
from src.services.user_profile_service import UserProfileService
from src.storage.long_term_memory_store import LongTermMemoryStore


class PersonalContextBuilder:
    """组合三个权威来源，生成只读 PersonalContext。"""

    def __init__(
        self,
        *,
        long_term_memory_store: LongTermMemoryStore | None = None,
        user_profile_service: UserProfileService | None = None,
        max_recent_messages: int = 6,
        max_recent_events: int = 8,
        max_recent_actions: int = 8,
    ) -> None:
        self.long_term_memory_store = long_term_memory_store or LongTermMemoryStore()
        self.user_profile_service = user_profile_service
        self.max_recent_messages = max_recent_messages
        self.max_recent_events = max_recent_events
        self.max_recent_actions = max_recent_actions

    def build(
        self,
        *,
        user_id: str,
        state: AgentState,
        event: Event | None = None,
    ) -> PersonalContext:
        """生成当前决策可读取的不可变人格上下文。"""

        del event
        memories = self.long_term_memory_store.list(user_id)
        profile = (
            self.user_profile_service.profile_context(user_id)
            if self.user_profile_service is not None
            else {}
        )
        buckets: dict[str, list[dict[str, Any]]] = {
            "behavior_preference": [],
            "behavior_pattern": [],
            "interaction_style": [],
            "active_constraint": [],
            "uncertain": [],
        }
        for memory in sorted(memories, key=lambda item: item.updated_at, reverse=True):
            rendered = _render_memory(memory)
            if memory.confidence < 0.55 or memory.memory_type == "uncertain":
                buckets["uncertain"].append(rendered)
            elif memory.memory_type in buckets:
                buckets[memory.memory_type].append(rendered)
            else:
                buckets["uncertain"].append(rendered)

        history = state.runtime_history
        runtime_history = {
            "recent_events": list(history.recent_events[-self.max_recent_events :]),
            "recent_messages": list(history.recent_messages[-self.max_recent_messages :]),
            "recent_actions": list(history.recent_actions[-self.max_recent_actions :]),
            "attention_summary": list(history.attention_records[-self.max_recent_events :]),
            "environment_summary": list(history.environment_records[-self.max_recent_events :]),
            "emotion_summaries": list(history.emotion_summaries[-self.max_recent_events :]),
        }
        return PersonalContext(
            user_id=user_id,
            user_profile=profile,
            behavior_preferences=tuple(buckets["behavior_preference"]),
            behavior_patterns=tuple(buckets["behavior_pattern"]),
            interaction_style=tuple(buckets["interaction_style"]),
            active_constraints=tuple(buckets["active_constraint"]),
            uncertain_memories=tuple(buckets["uncertain"]),
            runtime_history=runtime_history,
            authoritative_sources={
                "explicit_user_preference": "UserProfile",
                "user_identity": "UserProfile",
                "behavior_preference": "LongTermMemory",
                "behavior_pattern": "LongTermMemory",
                "recent_conversation": "RuntimeHistory",
                "recent_action": "RuntimeHistory",
                "decision_context": "PersonalContextBuilder",
            },
        )


def _render_memory(memory: LongTermMemory) -> dict[str, Any]:
    """把 LongTermMemory 渲染成 prompt 安全摘要，不暴露仓库内部实现。"""

    return {
        "id": memory.id,
        "memory_type": memory.memory_type,
        "content": memory.content,
        "confidence": memory.confidence,
        "updated_at": memory.updated_at,
        "evidence_count": len(memory.evidence),
        "decay": memory.decay,
        "source": "LongTermMemory",
    }
