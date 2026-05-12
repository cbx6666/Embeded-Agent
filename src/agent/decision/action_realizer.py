"""
Intent 到 Action 的确定性落地模块。

本模块位于 DeterministicGuard 之后、DeviceAdapter 之前，负责把已通过边界
过滤的 IntentPlan 转换为系统注册 Action。上游输入是 IntentPlan、ResponseDraft
和 AgentContext，下游输出是 Action 列表。

本模块不调用 LLM、不理解用户自然语言、不修改 AgentState，也不直接执行硬件。
它只做可审计的参数裁剪和动作构造。
"""

from __future__ import annotations

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
from src.agent.decision.intent_model import AgentIntent, IntentPlan
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.llm_agent.schemas import ResponseDraft


class ActionRealizer:
    """确定性动作落地器。

    输入已验证和过滤的 intent，输出标准 Action。它是 Event -> Intent -> Action
    主线中的最后一个纯决策步骤；真实设备执行由 runtime/device_adapter.py 完成。
    """

    def realize(
        self,
        plan: IntentPlan,
        *,
        response: ResponseDraft,
        context: AgentContext,
    ) -> list[Action]:
        """按优先级把 intent 转成 Action。

        排序只影响多个已批准 intent 的执行顺序，不重新解释语义。未知 intent
        不生成动作，避免新 intent 在未实现前误触设备。
        """

        actions: list[Action] = []
        for intent in sorted(plan.intents, key=lambda item: item.priority, reverse=True):
            actions.extend(self._realize_intent(intent, response=response, context=context))
        return actions

    def _realize_intent(
        self,
        intent: AgentIntent,
        *,
        response: ResponseDraft,
        context: AgentContext,
    ) -> list[Action]:
        """落地单个 intent。

        每个分支只处理注册 intent 到注册 action 的映射。涉及用户表达的文本
        来自 ResponseWriter 或 intent payload；涉及数值的字段在这里做裁剪。
        """

        if intent.type == "no_op":
            return []

        if intent.type == "answer_user":
            text = _response_text(intent, response) or "I am here."
            return [
                speak(text, reason=intent.reason),
                display(response.display_text or text, reason=intent.reason),
            ]

        if intent.type == "start_focus":
            event_default = _clamp_int(context.event_payload.get("duration_sec", 1500), 1, 24 * 3600)
            duration = _duration_from_intent(intent, default_sec=event_default)
            return [
                start_timer(duration),
                display(f"Focus timer started for {duration // 60} minutes.", reason="focus_start"),
            ]

        if intent.type == "continue_focus":
            duration = _duration_from_intent(intent, default_sec=1200)
            return [
                start_timer(duration),
                display(f"Continuing focus for {duration // 60} minutes.", reason="continue_focus"),
            ]

        if intent.type == "stop_focus":
            return [stop_timer(), display("Focus timer stopped.", reason="focus_stop")]

        if intent.type == "complete_focus":
            text = _response_text(intent, response) or "Focus time is complete."
            return [
                stop_timer(),
                speak(text, kind="notification", reason="focus_complete"),
                display(text, reason="focus_complete"),
            ]

        if intent.type == "suggest_rest":
            text = _response_text(intent, response) or "You may want to take a short rest."
            return [
                speak(text, kind="notification", level="gentle", reason="rest_reminder"),
                display(text, kind="notification", level="gentle", reason="rest_reminder"),
            ]

        if intent.type == "remind_distraction":
            text = _response_text(intent, response) or "Let's gently return to the current task."
            return [
                speak(text, kind="notification", level="gentle", reason="distraction_reminder"),
                display(text, kind="notification", level="gentle", reason="distraction_reminder"),
            ]

        if intent.type == "update_status_feedback":
            text = _response_text(intent, response) or _status_text(context)
            return [display(text, reason="status_update"), speak(text, reason="status_update")]

        if intent.type == "adjust_environment_feedback":
            text = _response_text(intent, response) or "The environment may need adjustment."
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
            return [display(text or "Updated.", reason=intent.reason)]

        if intent.type == "reduce_reminder_frequency":
            text = _response_text(intent, response) or "I will keep reminders lighter."
            return [display(text, reason="reduce_reminder_frequency")]

        if intent.type == "set_tts_volume":
            volume = intent.payload.get("volume", intent.payload.get("level", 50))
            return [set_tts_volume(_clamp_int(volume, 0, 100))]

        return []


def _response_text(intent: AgentIntent, response: ResponseDraft) -> str:
    """选择用户可见文本，优先使用 intent 明确文本，再使用 ResponseWriter 草稿。"""

    text = str(intent.payload.get("text", "")).strip()
    return text or response.speak_text or response.display_text


def _duration_from_intent(intent: AgentIntent, *, default_sec: int) -> int:
    """从 intent payload 提取专注时长，并裁剪到设备可接受范围。"""

    if "duration_sec" in intent.payload:
        return max(1, _clamp_int(intent.payload.get("duration_sec"), 1, 24 * 3600))
    if "duration_minutes" in intent.payload:
        return max(1, _clamp_int(intent.payload.get("duration_minutes"), 1, 24 * 60) * 60)
    return default_sec


def _clamp_int(value: object, low: int, high: int) -> int:
    """把 LLM 或外部输入给出的数值裁剪到安全闭区间。"""

    try:
        number = int(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def _status_text(context: AgentContext) -> str:
    """从结构化状态生成兜底状态文本；不做语义推断。"""

    state = context.state_summary
    focus = state.get("focus", {}) if isinstance(state, dict) else {}
    user = state.get("user", {}) if isinstance(state, dict) else {}
    if focus.get("active"):
        return f"Focus is active. Remaining: {focus.get('remaining_sec')} seconds."
    return (
        f"Status: presence={user.get('presence')}, "
        f"attention={user.get('attention')}, fatigue={user.get('fatigue_level')}."
    )
