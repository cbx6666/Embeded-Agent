from __future__ import annotations

"""决策 AgentContext 构建器。

它是什么：
AgentContextBuilder 把 Event、AgentState 和 PersonalContext 压缩成 LLM 角色可消费的
AgentContext。

它不是什么：
它不是 PersonalContextBuilder，不读取 store；不是 DecisionPipeline，不生成 Intent；
也不是 LongTermMemoryPipeline，不写长期记忆。

为什么存在：
PersonalContext 是人格上下文快照，AgentContext 是本轮 LLM prompt 上下文。二者分开后，
决策 prompt 可以演化，而不破坏 personalization 的权威来源边界。

边界：
输入必须是已经构建好的 PersonalContext；禁止在这里读取 LongTermMemoryStore 或
UserProfileService。
"""

import json
from dataclasses import dataclass, field
from typing import Any

from src.agent.user.personal_context import PersonalContext
from src.agent.event import Event
from src.agent.state import AgentState


@dataclass
class AgentContext:
    """传给 LLM 角色的本轮紧凑上下文。"""

    event_type: str
    event_payload: dict[str, Any]
    timestamp: int
    user_text: str = ""
    state_summary: dict[str, Any] = field(default_factory=dict)
    previous_state_summary: dict[str, Any] = field(default_factory=dict)
    personal_context: PersonalContext | None = None
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    relevant_memories: list[dict[str, Any]] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        """转换为 prompt 和 trace 使用的稳定字典结构。"""

        personal = self.personal_context.to_dict() if self.personal_context is not None else {}
        return {
            "event": {
                "type": self.event_type,
                "timestamp": self.timestamp,
                "payload": self.event_payload,
                "user_text": self.user_text,
            },
            "state": self.state_summary,
            "previous_state": self.previous_state_summary,
            "personal_context": personal,
            "personalization_guidance": _personalization_guidance(self.personal_context, self.relevant_memories),
            "recent_messages": self.recent_messages,
            "relevant_memories": self.relevant_memories,
        }

    def to_prompt_json(self) -> str:
        """转换为格式化 JSON，作为各 LLM 角色 prompt 的上下文块。"""

        return json.dumps(self.to_prompt_dict(), ensure_ascii=False, indent=2)


class AgentContextBuilder:
    """把状态、事件和 PersonalContext 压缩为 LLM prompt 上下文。"""

    def __init__(
        self,
        *,
        max_recent_messages: int = 6,
        max_relevant_memories: int = 8,
    ) -> None:
        self.max_recent_messages = max_recent_messages
        self.max_relevant_memories = max_relevant_memories

    def build(
        self,
        *,
        previous_state: AgentState | None,
        current_state: AgentState,
        event: Event,
        personal_context: PersonalContext | None,
    ) -> AgentContext:
        """构建当前事件所需的最小决策上下文。"""

        user_text = ""
        if event.type in {"user_text_input", "speech_recognized"}:
            user_text = str(event.payload.get("text", "")).strip()

        relevant = []
        recent_messages = list(current_state.runtime_history.recent_messages[-self.max_recent_messages :])
        if personal_context is not None:
            relevant = personal_context.retrieve_relevant(
                event_type=str(event.type),
                text=user_text,
                limit=self.max_relevant_memories,
            )
            runtime_history = personal_context.runtime_history
            recent_messages = list(runtime_history.get("recent_messages", recent_messages))

        return AgentContext(
            event_type=str(event.type),
            event_payload=dict(event.payload),
            timestamp=int(event.timestamp),
            user_text=user_text,
            state_summary=_summarize_state(current_state),
            previous_state_summary=_summarize_state(previous_state) if previous_state else {},
            personal_context=personal_context,
            recent_messages=recent_messages,
            relevant_memories=relevant,
        )


def _summarize_state(state: AgentState | None) -> dict[str, Any]:
    """提取 LLM 和 guard 需要的结构化状态事实。"""

    if state is None:
        return {}
    return {
        "current_user_id": state.current_user_id,
        "interaction": {
            "mode": state.interaction.mode,
            "dialogue_state": state.interaction.dialogue_state,
            "in_conversation": state.interaction.in_conversation,
        },
        "focus": {
            "active": state.focus.active,
            "elapsed_sec": state.focus.elapsed_sec,
            "remaining_sec": state.focus.remaining_sec,
            "target_duration_sec": state.focus.target_duration_sec,
        },
        "user": {
            "presence": state.user.presence,
            "attention": state.user.attention,
            "emotion": state.user.emotion,
            "fatigue_level": state.user.fatigue_level,
        },
        "environment": {
            "light_level": state.environment.light_level,
            "noise_level": state.environment.noise_level,
            "temperature_level": state.environment.temperature_level,
            "humidity_level": state.environment.humidity_level,
        },
        "cooldowns": dict(state.cooldown.reminder_last_ts),
    }


def _personalization_guidance(
    personal_context: PersonalContext | None,
    relevant_memories: list[dict[str, Any]],
) -> dict[str, Any]:
    """把显式画像和相关长期记忆压成 ResponseWriter/Planner 容易消费的提示块。"""

    if personal_context is None:
        return {}
    conflicts = [
        {
            "content": item.get("content"),
            "conflict_with": item.get("conflict_with"),
            "policy": item.get("conflict_policy"),
        }
        for item in personal_context.uncertain_memories
        if item.get("conflict_with")
    ]
    style_hints = _style_hints(personal_context.profile_items, relevant_memories)
    return {
        "explicit_user_preferences": [
            {
                "content": item.get("content"),
                "source": item.get("source"),
                "profile_key": item.get("profile_key"),
                "profile_value": item.get("profile_value"),
            }
            for item in personal_context.profile_items
        ],
        "relevant_long_term_memory": [
            {
                "content": item.get("content"),
                "source": item.get("source"),
                "confidence": item.get("confidence"),
                "effective_confidence": item.get("effective_confidence"),
                "memory_type": item.get("memory_type"),
                "conflict_with": item.get("conflict_with"),
            }
            for item in relevant_memories
            if item.get("source") == "LongTermMemory"
        ],
        "profile_memory_conflicts": conflicts,
        "response_style_hints": style_hints,
    }


def _style_hints(
    profile_items: tuple[dict[str, Any], ...],
    relevant_memories: list[dict[str, Any]],
) -> list[str]:
    text = " ".join(
        str(item.get("content", ""))
        + " "
        + str(item.get("profile_key", item.get("preference_key", "")))
        + " "
        + str(item.get("profile_value", item.get("preference_value", "")))
        for item in list(profile_items) + relevant_memories
    ).lower()
    hints: list[str] = []
    if any(term in text for term in ["gentle", "温和", "柔和"]):
        hints.append("Use a gentle tone; avoid commands.")
    if any(term in text for term in ["low_frequency", "低频", "不要频繁", "不频繁", "少提醒"]):
        hints.append("Avoid promising that settings were changed unless an action/profile update really happened.")
    return hints
