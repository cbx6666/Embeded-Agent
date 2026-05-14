from __future__ import annotations

"""决策层人格上下文快照。

它是什么：
PersonalContext 是决策层唯一允许读取的人格上下文快照，由 RuntimeHistory、
LongTermMemory 和 UserProfile 组合生成。

它不是什么：
它不是 store，不负责持久化；不是 profile，不拥有权威用户资料；不是长期记忆，
不做学习；也不是 runtime history，不无限保留会话窗口。

为什么存在：
决策层需要稳定、只读、可压缩的人格上下文。把多个来源压成 PersonalContext，可以
避免 DecisionPipeline 直接读取不同 store，防止数据来源分散。

边界：
PersonalContext 由 PersonalContextBuilder 构建后不可变使用；业务层不允许绕过它读取
LongTermMemoryStore 或 UserProfileStore。
"""

from dataclasses import dataclass, field
from typing import Any

from src.agent.config.retrieval_policy import RetrievalPolicyConfig


DEFAULT_RETRIEVAL_POLICY = RetrievalPolicyConfig()


@dataclass(frozen=True)
class PersonalContext:
    """面向 decision/planning 的只读人格上下文快照。"""

    user_id: str
    user_profile: dict[str, Any] = field(default_factory=dict)
    profile_items: tuple[dict[str, Any], ...] = ()
    behavior_preferences: tuple[dict[str, Any], ...] = ()
    behavior_patterns: tuple[dict[str, Any], ...] = ()
    interaction_style: tuple[dict[str, Any], ...] = ()
    active_constraints: tuple[dict[str, Any], ...] = ()
    uncertain_memories: tuple[dict[str, Any], ...] = ()
    runtime_history: dict[str, Any] = field(default_factory=dict)
    runtime_items: tuple[dict[str, Any], ...] = ()
    compression: dict[str, Any] = field(default_factory=dict)
    authoritative_sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 prompt 和 trace 可用的稳定字典。"""

        return {
            "user_id": self.user_id,
            "user_profile": dict(self.user_profile),
            "profile_items": list(self.profile_items),
            "behavior_preferences": list(self.behavior_preferences),
            "behavior_patterns": list(self.behavior_patterns),
            "interaction_style": list(self.interaction_style),
            "active_constraints": list(self.active_constraints),
            "uncertain_memories": list(self.uncertain_memories),
            "runtime_history": dict(self.runtime_history),
            "runtime_items": list(self.runtime_items),
            "compression": dict(self.compression),
            "authoritative_sources": dict(self.authoritative_sources),
        }

    def retrieve_relevant(self, *, event_type: str, text: str = "", limit: int = 8) -> list[dict[str, Any]]:
        """检索当前事件最需要的少量个性化材料。

        这里不是重新判断长期记忆价值，而是做 prompt 压缩：UserProfile 的显式偏好优先，
        LongTermMemory 按有效置信度和事件类型排序，RuntimeHistory 只提供短期窗口。
        """

        pool = (
            list(self.profile_items)
            + list(self.active_constraints)
            + list(self.behavior_preferences)
            + list(self.interaction_style)
            + list(self.behavior_patterns)
            + list(self.runtime_items)
            + list(self.uncertain_memories)
        )
        terms = _terms(text)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, item in enumerate(pool):
            scored.append(
                (
                    _relevance_score(
                        item,
                        event_type=event_type,
                        terms=terms,
                        policy=DEFAULT_RETRIEVAL_POLICY,
                    ),
                    index,
                    item,
                )
            )
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [item for score, _, item in scored[: max(0, limit)] if score > -100]


def _terms(text: str) -> set[str]:
    return {part.lower() for part in text.replace(",", " ").split() if len(part) > 1}


def _relevance_score(
    item: dict[str, Any],
    *,
    event_type: str,
    terms: set[str],
    policy: RetrievalPolicyConfig,
) -> float:
    """Execute the configured retrieval policy for one candidate item."""

    source = str(item.get("source", ""))
    item_type = str(item.get("memory_type") or item.get("item_type") or "")
    content = str(item.get("content", "")).lower()
    score = _source_weight(source, policy) + _event_type_weight(event_type, item_type, policy)
    score += float(item.get("priority_score", 0.0))
    score += float(item.get("effective_confidence", item.get("confidence", 0.0))) * policy.confidence_weight
    evidence_bonus = int(item.get("evidence_count", 0)) * policy.evidence_weight
    score += min(policy.max_evidence_bonus, evidence_bonus)
    if item.get("conflict_with") and source != _most_authoritative_source(policy):
        score -= policy.conflict_penalty
    if terms:
        score += sum(policy.content_term_weight for term in terms if term in content)
        tags = item.get("tags", [])
        if isinstance(tags, list):
            score += sum(policy.tag_term_weight for term in terms if term in {str(tag).lower() for tag in tags})
    return score


def _source_weight(source: str, policy: RetrievalPolicyConfig) -> float:
    return float(policy.source_weights.get(source, 0.0))


def _event_type_weight(event_type: str, item_type: str, policy: RetrievalPolicyConfig) -> float:
    return float(policy.event_type_weights.get(event_type, {}).get(item_type, 0.0))


def _most_authoritative_source(policy: RetrievalPolicyConfig) -> str:
    if not policy.source_weights:
        return ""
    return max(policy.source_weights.items(), key=lambda item: item[1])[0]
