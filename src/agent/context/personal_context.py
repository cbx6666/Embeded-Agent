from __future__ import annotations

"""决策层人格上下文快照。

它是什么：
PersonalContext 是决策层唯一允许读取的人格上下文快照，由 RuntimeHistory、
LongTermMemory 和 UserProfile 组合生成。

它不是什么：
它不是 store，不负责持久化；不是 profile，不拥有权威用户资料；不是长期记忆，不做学习；
也不是 runtime history，不无限保留会话窗口。

为什么存在：
决策层需要一个稳定、只读、可序列化的个性化上下文。把多个来源压缩为 PersonalContext，
能避免 DecisionPipeline 直接读取不同 store，防止数据来源分散。

边界：
PersonalContext 由 PersonalContextBuilder 构建后不可变使用；业务层不允许绕过它读取
LongTermMemoryStore 或 UserProfileStore。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PersonalContext:
    """面向 decision/planning 的只读人格上下文快照。"""

    user_id: str
    user_profile: dict[str, Any] = field(default_factory=dict)
    behavior_preferences: tuple[dict[str, Any], ...] = ()
    behavior_patterns: tuple[dict[str, Any], ...] = ()
    interaction_style: tuple[dict[str, Any], ...] = ()
    active_constraints: tuple[dict[str, Any], ...] = ()
    uncertain_memories: tuple[dict[str, Any], ...] = ()
    runtime_history: dict[str, Any] = field(default_factory=dict)
    authoritative_sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 prompt 和 trace 可用的稳定字典。"""

        return {
            "user_id": self.user_id,
            "user_profile": dict(self.user_profile),
            "behavior_preferences": list(self.behavior_preferences),
            "behavior_patterns": list(self.behavior_patterns),
            "interaction_style": list(self.interaction_style),
            "active_constraints": list(self.active_constraints),
            "uncertain_memories": list(self.uncertain_memories),
            "runtime_history": dict(self.runtime_history),
            "authoritative_sources": dict(self.authoritative_sources),
        }

    def retrieve_relevant(self, *, event_type: str, text: str = "", limit: int = 8) -> list[dict[str, Any]]:
        """检索与当前事件相关的少量长期个性化材料。

        这里的检索只用于压缩 prompt，不承担长期记忆价值判断；价值判断在
        LongTermMemoryPipeline 中完成。
        """

        del event_type
        pool = (
            list(self.active_constraints)
            + list(self.behavior_preferences)
            + list(self.interaction_style)
            + list(self.behavior_patterns)
            + list(self.uncertain_memories)
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
