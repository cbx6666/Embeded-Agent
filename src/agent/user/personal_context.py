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

import copy
from dataclasses import dataclass, field
from typing import Any

from src.agent.config.policy_config import RetrievalPolicyConfig


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

        pool = _retrieval_pool(self)
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

    def retrieve_relevant_with_scores(
        self,
        *,
        event_type: str,
        text: str = "",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Return retrieval results with deterministic score explanations.

        The default `retrieve_relevant` method intentionally keeps returning the
        original item dictionaries. This method is for debug/experiments and
        returns deep-copied items with `score_breakdown`, `retrieval_rank`, and
        `original_index` added.
        """

        pool = _retrieval_pool(self)
        terms = _terms(text)
        scored: list[tuple[float, int, dict[str, Any], dict[str, float]]] = []
        for index, item in enumerate(pool):
            breakdown = _score_breakdown(
                item,
                event_type=event_type,
                terms=terms,
                policy=DEFAULT_RETRIEVAL_POLICY,
            )
            scored.append((breakdown["final_score"], index, item, breakdown))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        explained: list[dict[str, Any]] = []
        for rank, (score, index, item, breakdown) in enumerate(scored[: max(0, limit)], start=1):
            if score <= -100:
                continue
            rendered = copy.deepcopy(item)
            rendered["score_breakdown"] = dict(breakdown)
            rendered["retrieval_rank"] = rank
            rendered["original_index"] = index
            explained.append(rendered)
        return explained

    def explain_retrieval(
        self,
        *,
        event_type: str,
        text: str = "",
        limit: int = 8,
    ) -> dict[str, Any]:
        """Return a compact retrieval explanation payload for CLI and experiments."""

        pool = _retrieval_pool(self)
        results = self.retrieve_relevant_with_scores(event_type=event_type, text=text, limit=limit)
        return {
            "query": {
                "event_type": event_type,
                "text": text,
                "terms": sorted(_terms(text)),
                "limit": limit,
            },
            "candidate_count": len(pool),
            "returned_count": len(results),
            "results": results,
        }


def _retrieval_pool(context: PersonalContext) -> list[dict[str, Any]]:
    return (
        list(context.profile_items)
        + list(context.active_constraints)
        + list(context.behavior_preferences)
        + list(context.interaction_style)
        + list(context.behavior_patterns)
        + list(context.runtime_items)
        + list(context.uncertain_memories)
    )


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

    return _score_breakdown(item, event_type=event_type, terms=terms, policy=policy)["final_score"]


def _score_breakdown(
    item: dict[str, Any],
    *,
    event_type: str,
    terms: set[str],
    policy: RetrievalPolicyConfig,
) -> dict[str, float]:
    """Explain the same hand-weighted score used by `retrieve_relevant`."""

    source = str(item.get("source", ""))
    item_type = str(item.get("memory_type") or item.get("item_type") or "")
    content = str(item.get("content", "")).lower()
    source_weight = _source_weight(source, policy)
    event_type_weight = _event_type_weight(event_type, item_type, policy)
    priority_score = float(item.get("priority_score", 0.0))
    confidence_bonus = (
        float(item.get("effective_confidence", item.get("confidence", 0.0))) * policy.confidence_weight
    )
    raw_evidence_bonus = int(item.get("evidence_count", 0)) * policy.evidence_weight
    evidence_bonus = min(policy.max_evidence_bonus, raw_evidence_bonus)
    conflict_penalty = 0.0
    if item.get("conflict_with") and source != _most_authoritative_source(policy):
        conflict_penalty = -float(policy.conflict_penalty)
    content_term_bonus = 0.0
    tag_term_bonus = 0.0
    if terms:
        content_term_bonus = float(sum(policy.content_term_weight for term in terms if term in content))
        tags = item.get("tags", [])
        if isinstance(tags, list):
            tag_terms = {str(tag).lower() for tag in tags}
            tag_term_bonus = float(sum(policy.tag_term_weight for term in terms if term in tag_terms))
    final_score = (
        source_weight
        + event_type_weight
        + priority_score
        + confidence_bonus
        + evidence_bonus
        + conflict_penalty
        + content_term_bonus
        + tag_term_bonus
    )
    return {
        "source_weight": source_weight,
        "event_type_weight": event_type_weight,
        "priority_score": priority_score,
        "confidence_bonus": confidence_bonus,
        "evidence_bonus": evidence_bonus,
        "conflict_penalty": conflict_penalty,
        "content_term_bonus": content_term_bonus,
        "tag_term_bonus": tag_term_bonus,
        "final_score": final_score,
    }


def _source_weight(source: str, policy: RetrievalPolicyConfig) -> float:
    return float(policy.source_weights.get(source, 0.0))


def _event_type_weight(event_type: str, item_type: str, policy: RetrievalPolicyConfig) -> float:
    return float(policy.event_type_weights.get(event_type, {}).get(item_type, 0.0))


def _most_authoritative_source(policy: RetrievalPolicyConfig) -> str:
    if not policy.source_weights:
        return ""
    return max(policy.source_weights.items(), key=lambda item: item[1])[0]
