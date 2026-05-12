"""
LLM Agent 上下文构建模块。

本模块负责把 Event、AgentState 和 ProfileSnapshot 压缩成 AgentContext，
供 SituationAnalyst、IntentPlanner、SafetyCritic 和 ResponseWriter 使用。
上游输入是 Reducer 后的状态、当前事件和画像快照，下游输出是可序列化 prompt
上下文。

本模块不做决策、不生成 Intent 或 Action、不调用 LLM，也不读取 MemoryStore。
它只做 context engineering：摘要、裁剪和相关记忆检索。
"""

from __future__ import annotations

"""
LLM Agent 上下文构建模块。

本模块负责把 Event、AgentState 和 ProfileSnapshot 压缩成 AgentContext，
供 SituationAnalyst、IntentPlanner、SafetyCritic 和 ResponseWriter 使用。
上游输入是 Reducer 后的状态、当前事件和画像快照，下游输出是可序列化 prompt
上下文。

本模块不做决策、不生成 Intent 或 Action、不调用 LLM，也不读取 MemoryStore。
它只做 context engineering：摘要、裁剪和相关记忆检索。
"""

import json
from dataclasses import dataclass, field
from typing import Any

from src.agent.event import Event
from src.agent.memory.profile_snapshot_builder import ProfileSnapshot
from src.agent.state import AgentState


@dataclass
class AgentContext:
    """传给 LLM 角色的紧凑上下文。

    它只包含当前事件、状态摘要、画像快照、最近消息和相关记忆，避免把完整
    历史塞进 prompt。
    """

    event_type: str
    event_payload: dict[str, Any]
    timestamp: int
    user_text: str = ""
    state_summary: dict[str, Any] = field(default_factory=dict)
    previous_state_summary: dict[str, Any] = field(default_factory=dict)
    profile_snapshot: ProfileSnapshot | None = None
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    relevant_memories: list[dict[str, Any]] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        """转成 prompt 使用的稳定字典结构，便于 trace 和测试复用。"""

        return {
            "event": {
                "type": self.event_type,
                "timestamp": self.timestamp,
                "payload": self.event_payload,
                "user_text": self.user_text,
            },
            "state": self.state_summary,
            "previous_state": self.previous_state_summary,
            "profile_snapshot": (
                self.profile_snapshot.to_dict() if self.profile_snapshot is not None else {}
            ),
            "recent_messages": self.recent_messages,
            "relevant_memories": self.relevant_memories,
        }

    def to_prompt_json(self) -> str:
        """转成格式化 JSON，作为各 LLM 角色 prompt 的上下文块。"""

        return json.dumps(self.to_prompt_dict(), ensure_ascii=False, indent=2)


class AgentContextBuilder:
    """上下文工程构建器。

    输入前后状态、当前事件和 ProfileSnapshot，输出 AgentContext。它通过固定
    上限裁剪最近消息和相关记忆，避免 prompt 无界膨胀。
    """

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
        profile_snapshot: ProfileSnapshot | None,
    ) -> AgentContext:
        """构建当前事件所需的最小上下文。

        用户文本事件会提取 `user_text`；若有 ProfileSnapshot，则只检索与文本
        相关的少量记忆。
        """

        user_text = ""
        if event.type in {"user_text_input", "speech_recognized"}:
            user_text = str(event.payload.get("text", "")).strip()

        relevant = []
        if profile_snapshot is not None:
            relevant = profile_snapshot.retrieve_relevant(
                event_type=str(event.type),
                text=user_text,
                limit=self.max_relevant_memories,
            )

        return AgentContext(
            event_type=str(event.type),
            event_payload=dict(event.payload),
            timestamp=int(event.timestamp),
            user_text=user_text,
            state_summary=_summarize_state(current_state),
            previous_state_summary=_summarize_state(previous_state) if previous_state else {},
            profile_snapshot=profile_snapshot,
            recent_messages=list(current_state.memory.recent_messages[-self.max_recent_messages :]),
            relevant_memories=relevant,
        )


def _summarize_state(state: AgentState | None) -> dict[str, Any]:
    """提取 LLM 需要的状态摘要。

    这里保留结构化字段而不是自然语言描述，是为了让 LLM 看到清晰边界，同时
    让 Guard 和 trace 可以复用同一份状态事实。
    """

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
