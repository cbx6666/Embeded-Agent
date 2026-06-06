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
DecisionPipeline 只接收 PersonalContext，不直接读取 LongTermMemoryStore 或
UserProfileStore。
"""

from typing import Any

from src.agent.config.policy_config import ContextPolicyConfig, RetrievalPolicyConfig
from src.agent.user.personal_context import PersonalContext
from src.agent.event import Event
from src.agent.memory.long_term_memory import LongTermMemory
from src.agent.state import AgentState
from src.agent.state.runtime_history import compact_signal_trends
from src.services.user_profile_service import UserProfileService
from src.storage.long_term_memory_store import LongTermMemoryStore


class PersonalContextBuilder:
    """组合三个权威来源，生成只读 PersonalContext。"""

    def __init__(
        self,
        *,
        long_term_memory_store: LongTermMemoryStore | None = None,
        user_profile_service: UserProfileService | None = None,
        policy_config: ContextPolicyConfig | None = None,
        retrieval_policy: RetrievalPolicyConfig | None = None,
    ) -> None:
        self.long_term_memory_store = long_term_memory_store or LongTermMemoryStore()
        self.user_profile_service = user_profile_service
        self.policy_config = policy_config or ContextPolicyConfig()
        self.retrieval_policy = retrieval_policy or RetrievalPolicyConfig()

    def _get_profile(self, user_id: str) -> dict[str, Any]:
        """读取用户显式画像；未配置 UserProfileService 时返回空字典。"""
        if self.user_profile_service is not None:
            return self.user_profile_service.profile_context(user_id)
        return {}

    def _collect_memory_items(
        self,
        memories: list[LongTermMemory],
        profile_values: dict[str, str],
    ) -> dict[str, list[dict[str, Any]]]:
        """渲染长期记忆，做冲突检测和置信度过滤，分入 bucket。"""
        buckets: dict[str, list[dict[str, Any]]] = {
            "behavior_preference": [],
            "behavior_pattern": [],
            "interaction_style": [],
            "active_constraint": [],
            "uncertain": [],
        }
        threshold = self.policy_config.uncertain_confidence_threshold
        for memory in sorted(memories, key=_memory_sort_key, reverse=True):
            rendered = _render_memory(
                memory,
                confidence_weight=self.retrieval_policy.memory_priority_confidence_weight,
                evidence_weight=self.retrieval_policy.memory_priority_evidence_weight,
                max_evidence=self.retrieval_policy.memory_priority_max_evidence,
            )
            conflict = _profile_conflict(memory, profile_values)
            if conflict:
                rendered["conflict_with"] = conflict
                rendered["conflict_policy"] = "UserProfile wins; LongTermMemory is kept as uncertain evidence."
                buckets["uncertain"].append(rendered)
                continue

            effective_confidence = float(rendered["effective_confidence"])
            if effective_confidence < threshold or memory.memory_type == "uncertain":
                buckets["uncertain"].append(rendered)
            elif memory.memory_type in buckets:
                buckets[memory.memory_type].append(rendered)
            else:
                buckets["uncertain"].append(rendered)
        return buckets

    def _build_runtime_snapshot(self, state: AgentState) -> dict[str, Any]:
        """从 AgentState.runtime_history 提取裁剪后的短期窗口快照。"""
        history = state.runtime_history
        recent_events = _decision_recent_events(history.recent_events, self.policy_config)
        cfg = self.policy_config
        signal_summaries = compact_signal_trends(history.signal_trends)
        return {
            "recent_events": list(recent_events[-cfg.max_recent_events :]),
            "recent_messages": list(history.recent_messages[-cfg.max_recent_messages :]),
            "recent_actions": list(history.recent_actions[-cfg.max_recent_actions :]),
            "attention_summary": list(history.attention_records[-cfg.max_recent_events :]),
            "environment_summary": list(history.environment_records[-cfg.max_recent_events :]),
            "emotion_summaries": list(history.emotion_summaries[-cfg.max_recent_events :]),
            "signal_summaries": signal_summaries,
            "fatigue_summary": signal_summaries.get("fatigue", {}),
            "attention_trend_summary": signal_summaries.get("attention", {}),
            "posture_summary": signal_summaries.get("posture", {}),
            "activity_summary": signal_summaries.get("activity", {}),
            "environment_trend_summary": {
                name: signal_summaries.get(name, {})
                for name in ("light", "temperature", "humidity", "noise")
            },
        }

    def build(
        self,
        *,
        user_id: str,
        state: AgentState,
        event: Event | None = None,
    ) -> PersonalContext:
        """生成当前决策可读取的不可变人格上下文。"""

        now = int(event.timestamp) if event is not None else None
        memories = self.long_term_memory_store.list(user_id, now=now)
        profile = self._get_profile(user_id)
        profile_values = _profile_value_index(profile)
        profile_items = _profile_items(profile)

        buckets = self._collect_memory_items(memories, profile_values)

        compressed_buckets, compression = _compress_buckets(
            buckets,
            max_items_per_bucket=self.policy_config.max_memory_items_per_bucket,
        )

        runtime_history = self._build_runtime_snapshot(state)
        runtime_items = _runtime_items(runtime_history)

        return PersonalContext(
            user_id=user_id,
            user_profile=profile,
            profile_items=tuple(profile_items),
            behavior_preferences=tuple(compressed_buckets["behavior_preference"]),
            behavior_patterns=tuple(compressed_buckets["behavior_pattern"]),
            interaction_style=tuple(compressed_buckets["interaction_style"]),
            active_constraints=tuple(compressed_buckets["active_constraint"]),
            uncertain_memories=tuple(compressed_buckets["uncertain"]),
            runtime_history=runtime_history,
            runtime_items=tuple(runtime_items),
            compression=compression,
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


def _render_memory(
    memory: LongTermMemory,
    *,
    confidence_weight: float,
    evidence_weight: float,
    max_evidence: int,
) -> dict[str, Any]:
    """把 LongTermMemory 渲染成 prompt 安全摘要，不暴露仓库内部实现。"""

    effective_confidence = round(memory.confidence * memory.decay, 4)
    metadata = dict(memory.metadata)
    preference_key = _metadata_value(metadata, "preference_key", "profile_key")
    preference_value = _metadata_value(metadata, "preference_value", "profile_value")
    tags = [memory.memory_type]
    if preference_key:
        tags.append(preference_key)
    return {
        "id": memory.id,
        "memory_type": memory.memory_type,
        "content": memory.content,
        "confidence": memory.confidence,
        "effective_confidence": effective_confidence,
        "updated_at": memory.updated_at,
        "evidence_count": len(memory.evidence),
        "decay": memory.decay,
        "source": "LongTermMemory",
        "priority_score": round(
            effective_confidence * confidence_weight + min(max_evidence, len(memory.evidence)) * evidence_weight, 4
        ),
        "tags": tags,
        "preference_key": preference_key,
        "preference_value": preference_value,
    }


def _memory_sort_key(memory: LongTermMemory) -> tuple[float, int, int]:
    return (memory.confidence * memory.decay, len(memory.evidence), memory.updated_at)


def _profile_items(profile: dict[str, Any]) -> list[dict[str, Any]]:
    preference = profile.get("preference", {}) if isinstance(profile, dict) else {}
    if not isinstance(preference, dict):
        return []
    items: list[dict[str, Any]] = []
    for key, value in preference.items():
        if value is None or value == []:
            continue
        rendered = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
        items.append(
            {
                "item_type": "explicit_user_preference",
                "content": f"{key}: {rendered}",
                "source": "UserProfile",
                "profile_key": key,
                "profile_value": rendered,
                "priority_score": 40.0,
                "tags": [key, "explicit_user_preference"],
            }
        )
    return items


def _profile_value_index(profile: dict[str, Any]) -> dict[str, str]:
    preference = profile.get("preference", {}) if isinstance(profile, dict) else {}
    if not isinstance(preference, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in preference.items():
        normalized = _normalize_profile_value(value)
        if normalized:
            result[str(key).strip().lower()] = normalized
    return result


def _normalize_profile_value(value: object) -> str:
    if value is None or value == []:
        return ""
    if isinstance(value, list):
        return ",".join(str(item).strip().lower() for item in value if str(item).strip())
    return str(value).strip().lower()


def _profile_conflict(memory: LongTermMemory, profile_values: dict[str, str]) -> str | None:
    key = _metadata_value(memory.metadata, "profile_key", "preference_key")
    value = _metadata_value(memory.metadata, "profile_value", "preference_value")
    if not key or not value:
        return None
    profile_value = profile_values.get(key)
    if profile_value and profile_value != value:
        return f"UserProfile:{key}"
    return None


def _metadata_value(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return str(value).strip().lower()
    return ""


def _compress_buckets(
    buckets: dict[str, list[dict[str, Any]]],
    *,
    max_items_per_bucket: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    ordered_names = [
        "active_constraint",
        "behavior_preference",
        "interaction_style",
        "behavior_pattern",
        "uncertain",
    ]
    compressed = {name: [] for name in buckets}
    input_counts = {name: len(items) for name, items in buckets.items()}

    for name in ordered_names:
        items = sorted(buckets[name], key=lambda item: float(item.get("priority_score", 0.0)), reverse=True)
        compressed[name] = items[: max(0, max_items_per_bucket)]

    return compressed, {
        "strategy": "source_priority_then_effective_confidence",
        "input_counts": input_counts,
        "output_counts": {name: len(items) for name, items in compressed.items()},
        "max_memory_items_per_bucket": max_items_per_bucket,
    }


def _runtime_items(runtime_history: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in list(runtime_history.get("recent_messages", []))[-3:]:
        if isinstance(message, dict):
            role = str(message.get("role", "")).strip()
            text = str(message.get("text", "")).strip()
            if text:
                items.append(
                    {
                        "item_type": "recent_message",
                        "content": f"{role}: {text}" if role else text,
                        "source": "RuntimeHistory",
                        "timestamp": message.get("timestamp"),
                        "priority_score": 4.0,
                        "tags": ["recent_message", role],
                    }
                )
    for action in list(runtime_history.get("recent_actions", []))[-3:]:
        if isinstance(action, dict):
            action_type = str(action.get("type") or action.get("action_type") or "").strip()
            if action_type:
                items.append(
                    {
                        "item_type": "recent_action",
                        "content": action_type,
                        "source": "RuntimeHistory",
                        "timestamp": action.get("timestamp"),
                        "priority_score": 3.0,
                        "tags": ["recent_action", action_type],
                    }
                )
    for event in list(runtime_history.get("recent_events", []))[-3:]:
        if isinstance(event, dict):
            event_type = str(event.get("type", "")).strip()
            if event_type:
                items.append(
                    {
                        "item_type": "recent_event",
                        "content": event_type,
                        "source": "RuntimeHistory",
                        "timestamp": event.get("timestamp"),
                        "priority_score": 2.0,
                        "tags": ["recent_event", event_type],
                    }
                )
    return items


def _decision_recent_events(events: list[dict[str, Any]], policy: ContextPolicyConfig) -> list[dict[str, Any]]:
    return [event for event in events if not _is_noisy_runtime_event(event, policy)]


def _is_noisy_runtime_event(event: dict[str, Any], policy: ContextPolicyConfig) -> bool:
    event_type = str(event.get("type", "")).strip()
    if event_type in policy.noisy_runtime_event_types:
        return True
    payload = event.get("payload", {})
    trigger = str(payload.get("trigger", "")).strip() if isinstance(payload, dict) else ""
    return trigger in policy.noisy_runtime_trigger_types
