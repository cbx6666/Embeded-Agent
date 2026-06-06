from __future__ import annotations

"""长期记忆处理的确定性准入规则。

MemoryGate 位于 LLM Observer 之前，先用低成本规则过滤传感器噪声、内部回流、
空动作和低价值文本。Gate 只判断是否值得进入昂贵的 Memory Pipeline，不负责
最终决定某条候选记忆是否应写入长期存储。
"""

import re

from src.agent.action import Action
from src.agent.event import Event
from src.agent.execution.action_result import ActionResult


_SKIPPED_EVENT_TYPES = frozenset(
    {
        "voice_wake_detected",
        "voice_input_started",
        "voice_input_stopped",
        "tts_started",
        "tts_finished",
        "voice_volume_changed",
        "voice_timbre_changed",
        "voice_speed_changed",
        "light_level_updated",
        "temperature_humidity_updated",
        "noise_level_updated",
        "display_sensor_updated",
        "timer_ticked",
        "focus_start_requested",
        "focus_stop_requested",
        "timer_finished",
    }
)

# 这些 trigger 来自 Agent 自身执行回流，不代表新的用户长期事实。
_INTERNAL_TRIGGERS = frozenset(
    {
        "focus_timer_started",
        "focus_timer_stopped",
        "agent_response_completed",
        "action_result",
        "action_failed",
        "device_action_completed",
        "timer_internal_tick",
    }
)

_USER_SEMANTIC_EVENT_TYPES = frozenset({"user_text_input", "speech_recognized"})

# 明确表达长期偏好、习惯或默认设置的中英文信号。
_LONG_TERM_MARKERS = (
    "以后",
    "以后默认",
    "记住",
    "帮我记住",
    "我喜欢",
    "我不喜欢",
    "我更喜欢",
    "我讨厌",
    "我习惯",
    "我通常",
    "我经常",
    "我希望你以后",
    "从现在开始",
    "默认",
    "不要再",
    "每次",
    "remember",
    "from now on",
    "i prefer",
    "i like",
    "i dislike",
    "i usually",
    "always",
    "by default",
)

# 无法单独形成长期知识的寒暄、确认和即时查询。
_TRIVIAL_TEXTS = frozenset(
    {
        "你好",
        "您好",
        "嗨",
        "hello",
        "hi",
        "hey",
        "嗯",
        "嗯嗯",
        "好的",
        "好",
        "可以",
        "知道了",
        "谢谢",
        "感谢",
        "开始",
        "停止",
        "现在几点",
        "几点了",
        "what time is it",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yes",
        "no",
    }
)

_STATUS_QUERY_PATTERNS = (
    re.compile(r"^(现在)?几点(了)?$"),
    re.compile(r"^(现在)?什么时间(了)?$"),
    re.compile(r"^what time is it$"),
    re.compile(r"^(当前|现在)?状态(怎么样|如何|是什么)?$"),
)

_NO_LONG_TERM_ACTION_TYPES = frozenset(
    {
        "speak",
        "display",
        "render_pet_expression",
        "start_voice_capture",
        "stop_voice_capture",
        "set_tts_voice",
        "set_tts_volume",
        "set_tts_speed",
    }
)


def should_process_event_memory(event: Event) -> tuple[bool, str]:
    """判断事件是否允许进入长期记忆 Pipeline，并返回可追踪的原因码。"""

    event_type = str(event.type)
    if event_type in _SKIPPED_EVENT_TYPES:
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
        if trigger in _INTERNAL_TRIGGERS:
            return False, "skipped_internal_event"
        return False, "skipped_system_triggered_event"

    if event_type not in _USER_SEMANTIC_EVENT_TYPES:
        return False, "skipped_unsupported_event"

    text = _normalize_text(event.payload.get("text"))
    if not text:
        return False, "skipped_empty_text"
    if _is_trivial_text(text):
        return False, "skipped_trivial_user_text"
    if any(marker in text for marker in _LONG_TERM_MARKERS):
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
            or trigger in _INTERNAL_TRIGGERS
        ):
            return False, "skipped_internal_event"

    action_types = {str(action.type) for action in actions}
    if action_types and action_types <= _NO_LONG_TERM_ACTION_TYPES:
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

    if text in _TRIVIAL_TEXTS:
        return True
    if len(text) <= 1:
        return True
    return any(pattern.fullmatch(text) for pattern in _STATUS_QUERY_PATTERNS)
