from __future__ import annotations

"""长期记忆处理的确定性准入规则。

MemoryGate 位于 LLM Observer 之前，先用低成本规则过滤传感器噪声、内部回流、
空动作和低价值文本。Gate 只判断是否值得进入昂贵的 Memory Pipeline，不负责
最终决定某条候选记忆是否应写入长期存储。
"""

import re

from src.agent.action import Action
from src.agent.config.policy_config import MemoryGatePolicyConfig
from src.agent.event import Event
from src.agent.execution.action_result import ActionResult


_POLICY = MemoryGatePolicyConfig()
_STATUS_QUERY_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in _POLICY.status_query_patterns
)


def should_process_event_memory(event: Event) -> tuple[bool, str]:
    """判断事件是否允许进入长期记忆 Pipeline，并返回可追踪的原因码。"""

    event_type = str(event.type)
    if event_type in _POLICY.skipped_event_types:
        if event_type in {
            "light_level_updated",
            "temperature_humidity_updated",
            "noise_level_updated",
            "display_sensor_updated",
        }:
            return False, "skipped_sensor_event"
        if event_type in {"focus_start_requested", "focus_stop_requested", "timer_finished", "timer_ticked"}:
            return False, "skipped_focus_or_timer_event"
        return False, "skipped_runtime_control_event"

    if event_type == "system_triggered":
        source = str(event.payload.get("source", "")).strip()
        trigger = str(event.payload.get("trigger", "")).strip()
        if source == "agent_action_result" or event.payload.get("system_triggered") is True:
            return False, "skipped_internal_event"
        if trigger in _POLICY.internal_triggers:
            return False, "skipped_internal_event"
        return False, "skipped_system_triggered_event"

    if event_type in _POLICY.feedback_event_types:
        return True, "allowed_feedback_signal"

    if event_type not in _POLICY.user_semantic_event_types:
        return False, "skipped_unsupported_event"

    text = _normalize_text(event.payload.get("text"))
    if not text:
        return False, "skipped_empty_text"
    if _is_trivial_text(text):
        return False, "skipped_trivial_user_text"
    if any(marker in text for marker in _POLICY.long_term_markers):
        return True, "allowed_explicit_long_term_signal"
    return True, "allowed_user_semantic_event"


def should_process_action_memory(
    actions: list[Action],
    results: list[ActionResult],
    source_event: Event,
) -> tuple[bool, str]:
    """判断动作结果是否具有进入长期记忆 Pipeline 的最低价值。"""

    if not actions:
        return False, "skipped_empty_actions"
    if not results:
        return False, "skipped_empty_results"

    if source_event.type == "system_triggered":
        source = str(source_event.payload.get("source", "")).strip()
        trigger = str(source_event.payload.get("trigger", "")).strip()
        if (
            source == "agent_action_result"
            or source_event.payload.get("system_triggered") is True
            or trigger in _POLICY.internal_triggers
        ):
            return False, "skipped_internal_event"

    action_types = {str(action.type) for action in actions}
    if action_types and action_types <= _POLICY.no_long_term_action_types:
        if action_types <= {"speak", "display"}:
            return False, "skipped_speak_display_only"
        return False, "skipped_non_memory_actions"

    return True, "allowed_action_outcome"


def _normalize_text(value: object) -> str:
    """统一大小写、空白和末尾标点，保证规则匹配稳定。"""

    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n。！？!?，,；;：:、")


def _is_trivial_text(text: str) -> bool:
    """识别无需调用 Memory LLM 的低信息量文本。"""

    if text in _POLICY.trivial_texts:
        return True
    if len(text) <= 1:
        return True
    return any(pattern.fullmatch(text) for pattern in _STATUS_QUERY_PATTERNS)
