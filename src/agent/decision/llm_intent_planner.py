from __future__ import annotations

"""LLM 辅助意图选择层。"""

import json

from src.agent.decision.intent import AgentIntent
from src.agent.event import Event
from src.agent.state import AgentState
from src.services.llm_service import LLMService


def plan_intents_with_llm(
    *,
    event: Event,
    state: AgentState,
    candidate_intents: list[AgentIntent],
    llm_service: LLMService,
) -> list[AgentIntent]:
    """在候选意图范围内，让 LLM 辅助选择更合适的意图。"""
    if not candidate_intents:
        return []
    if event.type not in {"user_text_input", "speech_recognized"}:
        return candidate_intents

    allowed_intent_types = sorted({intent.type for intent in candidate_intents})
    if not allowed_intent_types:
        return candidate_intents

    prompt = _build_intent_prompt(event=event, state=state, candidate_intents=candidate_intents)

    try:
        raw_output = llm_service.choose_intents(prompt, allowed_intent_types)
        parsed = json.loads(raw_output)
        items = parsed.get("intents")
        if not isinstance(items, list) or not items:
            return candidate_intents
        intents = _parse_llm_intents(items, allowed_intent_types, candidate_intents)
        return intents or candidate_intents
    except Exception:
        return candidate_intents


def _build_intent_prompt(
    *,
    event: Event,
    state: AgentState,
    candidate_intents: list[AgentIntent],
) -> str:
    """构造给 LLM 的意图选择提示词。"""
    recent_messages = state.memory.recent_messages[-3:]
    return (
        f"用户输入：{event.payload.get('text', '')}\n"
        f"允许的意图类型：{[intent.type for intent in candidate_intents]}\n"
        f"当前模式：{state.interaction.mode}\n"
        f"是否专注中：{state.focus.active}\n"
        f"用户在场：{state.user.presence}\n"
        f"注意力：{state.user.attention}\n"
        f"疲劳：{state.user.fatigue_level}\n"
        f"最近消息：{recent_messages}\n"
        "请只从允许的意图类型中选择，并返回 JSON。"
    )


def _parse_llm_intents(
    items: list[object],
    allowed_intent_types: list[str],
    candidate_intents: list[AgentIntent],
) -> list[AgentIntent]:
    """将 LLM 返回的 JSON 条目解析为受限的 AgentIntent。"""
    candidate_by_type = {intent.type: intent for intent in candidate_intents}
    parsed_intents: list[AgentIntent] = []

    for item in items:
        if not isinstance(item, dict):
            return []

        intent_type = str(item.get("type", "")).strip()
        if intent_type not in allowed_intent_types:
            return []

        base_intent = candidate_by_type.get(intent_type)
        payload = dict(base_intent.payload) if base_intent is not None else {}
        raw_payload = item.get("payload")
        if isinstance(raw_payload, dict):
            payload.update(raw_payload)
        payload["llm_selected"] = True

        parsed_intents.append(
            AgentIntent(
                type=intent_type,  # type: ignore[arg-type]
                priority=int(item.get("priority", base_intent.priority if base_intent else 0)),
                reason=str(item.get("reason", base_intent.reason if base_intent else "llm_selected")),
                payload=payload,
                requires_llm=bool(item.get("requires_llm", base_intent.requires_llm if base_intent else False)),
            )
        )

    return parsed_intents
