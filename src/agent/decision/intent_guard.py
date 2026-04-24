from __future__ import annotations

"""Intent 安全校验层。"""

from typing import get_args

from src.agent.decision.intent import AgentIntent, IntentType
from src.agent.event import Event
from src.agent.state import AgentState

ALLOWED_INTENT_TYPES = set(get_args(IntentType))
PROACTIVE_INTENT_TYPES = {"suggest_rest", "remind_distraction", "adjust_environment_feedback"}
AUTONOMOUS_TRIGGERS = {"periodic_check", "user_idle_check", "focus_health_check", "environment_check"}
COOLDOWN_BY_INTENT = {
    "suggest_rest": "rest_reminder",
    "remind_distraction": "distraction_reminder",
    "adjust_environment_feedback": "environment_warning",
}
COOLDOWN_SEC = {
    "rest_reminder": 300,
    "distraction_reminder": 180,
    "environment_warning": 300,
    "fatigue_warning": 300,
    "idle_check": 600,
}


def guard_intents(
    intents: list[AgentIntent],
    *,
    state: AgentState,
    event: Event,
    fallback_intents: list[AgentIntent] | None = None,
) -> list[AgentIntent]:
    """统一过滤不安全、不合法或不符合边界条件的意图。"""
    safe_intents = _filter_intents(intents, state=state, event=event)
    if safe_intents:
        return safe_intents

    if fallback_intents is not None and fallback_intents is not intents:
        fallback_safe_intents = _filter_intents(fallback_intents, state=state, event=event)
        if fallback_safe_intents:
            return fallback_safe_intents

    return [AgentIntent(type="no_op", reason="intent_guard_filtered")]


def _filter_intents(
    intents: list[AgentIntent],
    *,
    state: AgentState,
    event: Event,
) -> list[AgentIntent]:
    """逐个检查意图，并保留符合约束的部分。"""
    safe_intents: list[AgentIntent] = []
    for intent in intents:
        if intent.type not in ALLOWED_INTENT_TYPES:
            continue
        if _is_blocked_by_presence(intent, state, event):
            continue
        if _is_blocked_by_cooldown(intent, state, event):
            continue
        if _is_blocked_by_focus_boundary(intent, state, event):
            continue
        if _is_blocked_by_autonomous_llm_boundary(intent, event):
            continue
        safe_intents.append(intent)
    return safe_intents


def _is_blocked_by_presence(intent: AgentIntent, state: AgentState, event: Event) -> bool:
    """判断用户离场时是否应该屏蔽主动提醒类意图。"""
    if state.user.presence != "away":
        return False
    if intent.type not in PROACTIVE_INTENT_TYPES:
        return False
    return event.type not in {"user_text_input", "speech_recognized"}


def _is_blocked_by_cooldown(intent: AgentIntent, state: AgentState, event: Event) -> bool:
    """判断提醒类意图是否仍处于冷却期内。"""
    reason = COOLDOWN_BY_INTENT.get(intent.type)
    if reason is None:
        return False
    last_ts = state.cooldown.reminder_last_ts.get(reason)
    if last_ts is None:
        return False
    cooldown_sec = COOLDOWN_SEC.get(reason, 300)
    return event.timestamp - int(last_ts) < cooldown_sec


def _is_blocked_by_focus_boundary(intent: AgentIntent, state: AgentState, event: Event) -> bool:
    """判断专注状态下是否应屏蔽系统主动闲聊类意图。"""
    if not state.focus.active:
        return False
    if event.type in {"user_text_input", "speech_recognized"}:
        return False
    return intent.type == "answer_user"


def _is_blocked_by_autonomous_llm_boundary(intent: AgentIntent, event: Event) -> bool:
    """判断自主检查类内部事件是否违规携带 requires_llm。"""
    if event.type != "system_triggered":
        return False
    trigger = str(event.payload.get("trigger", "")).strip()
    if trigger not in AUTONOMOUS_TRIGGERS:
        return False
    return intent.requires_llm
