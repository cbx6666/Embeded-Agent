from __future__ import annotations

"""Intent 到 Action 的确定性落地模块。

本模块位于 DeterministicGuard 之后、DeviceAdapter 之前，负责把已经通过边界过滤的
IntentPlan 转换为系统注册 Action。这里不调用 LLM、不修改 AgentState、不执行硬件。
策略数值和默认文案来自 policy_config；协议分支仍然直接保留，避免引入 registry。
"""

from src.agent.action import (
    Action,
    display,
    set_light_state,
    set_tts_volume,
    speak,
    start_timer,
    start_voice_capture,
    stop_timer,
    stop_voice_capture,
)
from src.agent.config.policy_config import ActionPolicyConfig, CopyPolicyConfig
from src.agent.decision.intent_model import AgentIntent, IntentPlan
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.llm_agent.schemas import ResponseDraft


class ActionRealizer:
    """确定性动作落地器。"""

    def __init__(
        self,
        *,
        action_policy: ActionPolicyConfig | None = None,
        copy_policy: CopyPolicyConfig | None = None,
    ) -> None:
        self.action_policy = action_policy or ActionPolicyConfig()
        self.copy_policy = copy_policy or CopyPolicyConfig()

    def realize(
        self,
        plan: IntentPlan,
        *,
        response: ResponseDraft,
        context: AgentContext,
    ) -> list[Action]:
        """按优先级把已通过 Validator/Guard 的 intent 转成 Action。

        Rule 和 LLM 在这里没有特权差异；Realizer 只看注册 intent 和配置，不重新
        解释自然语言，也不允许上游直接夹带设备命令。
        """

        actions: list[Action] = []
        for intent in sorted(plan.intents, key=lambda item: item.priority, reverse=True):
            actions.extend(self._realize_intent(intent, response=response, context=context))
        return _deduplicate_actions(actions)

    def _realize_intent(
        self,
        intent: AgentIntent,
        *,
        response: ResponseDraft,
        context: AgentContext,
    ) -> list[Action]:
        """落地单个 intent。

        主分支仍按协议 intent 类型显式展开；本轮只治理策略值和 fallback 文案。
        """

        if intent.type == "no_op":
            return []

        if intent.type == "answer_user":
            text = _response_text(intent, response) or self.copy_policy.fallback_answer_text
            actions: list[Action] = []
            if not bool(getattr(response, "already_spoken", False)):
                actions.append(speak(text, reason=intent.reason))
            actions.append(display(response.display_text or text, reason=intent.reason))
            return actions

        if intent.type == "start_focus":
            event_default = _clamp_int(
                context.event_payload.get("duration_sec", self.action_policy.default_focus_duration_sec),
                self.action_policy.min_duration_sec,
                self.action_policy.max_duration_sec,
            )
            duration = _duration_from_intent(intent, default_sec=event_default, policy=self.action_policy)
            return [
                start_timer(duration),
                display(_format_duration_copy(self.copy_policy.focus_started_template, duration), reason="focus_start"),
            ]

        if intent.type == "continue_focus":
            duration = _duration_from_intent(
                intent,
                default_sec=self.action_policy.default_continue_focus_sec,
                policy=self.action_policy,
            )
            return [
                start_timer(duration),
                display(
                    _format_duration_copy(self.copy_policy.continue_focus_template, duration),
                    reason="continue_focus",
                ),
            ]

        if intent.type == "stop_focus":
            return [stop_timer(), display(self.copy_policy.focus_stopped_text, reason="focus_stop")]

        if intent.type == "complete_focus":
            text = _response_text(intent, response) or self.copy_policy.focus_complete_text
            return [
                stop_timer(),
                speak(text, kind="notification", reason="focus_complete"),
                display(text, reason="focus_complete"),
            ]

        if intent.type == "suggest_rest":
            text = _response_text(intent, response) or self.copy_policy.rest_reminder_text
            return [
                speak(text, kind="notification", level="gentle", reason="rest_reminder"),
                display(text, kind="notification", level="gentle", reason="rest_reminder"),
            ]

        if intent.type == "remind_distraction":
            text = _response_text(intent, response) or self.copy_policy.distraction_reminder_text
            return [
                speak(text, kind="notification", level="gentle", reason="distraction_reminder"),
                display(text, kind="notification", level="gentle", reason="distraction_reminder"),
            ]

        if intent.type == "update_status_feedback":
            text = _response_text(intent, response) or _status_text(context, self.copy_policy)
            return [display(text, reason="status_update"), speak(text, reason="status_update")]

        if intent.type == "adjust_environment_feedback":
            text = _response_text(intent, response) or self.copy_policy.environment_warning_text
            return [
                display(text, kind="notification", reason="environment_warning"),
                set_light_state("attention", pattern="soft", reason="environment_warning"),
            ]

        if intent.type == "voice_interaction":
            mode = str(intent.payload.get("mode", "start"))
            if mode == "stop":
                return [stop_voice_capture(source="agent", reason=intent.reason)]
            return [start_voice_capture(source="agent", trigger=intent.reason or "intent")]

        if intent.type == "display_update":
            text = _response_text(intent, response) or str(intent.payload.get("text", "")).strip()
            return [display(text or self.copy_policy.display_updated_text, reason=intent.reason)]

        if intent.type == "reduce_reminder_frequency":
            text = _response_text(intent, response) or self.copy_policy.reduce_reminder_frequency_text
            return [display(text, reason="reduce_reminder_frequency")]

        if intent.type == "set_tts_volume":
            volume = intent.payload.get("volume", intent.payload.get("level", 50))
            return [
                set_tts_volume(
                    _clamp_int(volume, self.action_policy.min_tts_volume, self.action_policy.max_tts_volume)
                )
            ]

        return []


def _response_text(intent: AgentIntent, response: ResponseDraft) -> str:
    """选择用户可见文本，优先使用 intent payload，再使用 ResponseWriter 草稿。"""

    text = str(intent.payload.get("text", "")).strip()
    return text or response.speak_text or response.display_text


def _duration_from_intent(intent: AgentIntent, *, default_sec: int, policy: ActionPolicyConfig) -> int:
    """从 intent payload 提取专注时长，并按策略边界裁剪。"""

    if "duration_sec" in intent.payload:
        return _clamp_int(intent.payload.get("duration_sec"), policy.min_duration_sec, policy.max_duration_sec)
    if "duration_minutes" in intent.payload:
        min_minutes = max(1, (policy.min_duration_sec + 59) // 60)
        max_minutes = max(min_minutes, policy.max_duration_sec // 60)
        return _clamp_int(intent.payload.get("duration_minutes"), min_minutes, max_minutes) * 60
    return _clamp_int(default_sec, policy.min_duration_sec, policy.max_duration_sec)


def _format_duration_copy(template: str, duration_sec: int) -> str:
    minutes = duration_sec // 60
    return template.format(duration_sec=duration_sec, duration_minutes=minutes, minutes=minutes)


def _clamp_int(value: object, low: int, high: int) -> int:
    """把外部或 LLM 给出的整数裁剪到策略边界。"""

    try:
        number = int(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def _status_text(context: AgentContext, copy_policy: CopyPolicyConfig) -> str:
    """从结构化状态生成兜底状态文本；不做语义推断。"""

    state = context.state_summary
    focus = state.get("focus", {}) if isinstance(state, dict) else {}
    user = state.get("user", {}) if isinstance(state, dict) else {}
    if focus.get("active"):
        return copy_policy.status_focus_active_template.format(remaining_sec=focus.get("remaining_sec"))
    return copy_policy.status_user_state_template.format(
        presence=user.get("presence"),
        attention=user.get("attention"),
        fatigue_level=user.get("fatigue_level"),
    )


def _deduplicate_actions(actions: list[Action]) -> list[Action]:
    """Collapse repeated visible actions without changing the action schema."""

    deduplicated: list[Action] = []
    exact_seen: set[tuple[str, str, str]] = set()
    display_text_seen: set[str] = set()

    for action in actions:
        text = str(action.payload.get("text", "")).strip()
        reason = str(action.payload.get("reason", "")).strip()
        exact_key = (action.type, text, reason)
        if exact_key in exact_seen:
            continue
        if action.type == "display" and text:
            if text in display_text_seen:
                continue
            display_text_seen.add(text)
        exact_seen.add(exact_key)
        deduplicated.append(action)
    return deduplicated
