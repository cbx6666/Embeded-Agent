from __future__ import annotations

"""意图落地层。"""

from src.agent.action import (
    Action,
    display,
    set_light_state,
    set_tts_speed,
    set_tts_voice,
    set_tts_volume,
    speak,
    start_timer,
    start_voice_capture,
    stop_timer,
    stop_voice_capture,
)
from src.agent.decision.intent import AgentIntent
from src.agent.decision.personalization import (
    build_distraction_reminder_text,
    build_llm_prompt,
    build_rest_reminder_text,
    tts_payload_for_user,
)
from src.agent.event import Event
from src.agent.memory.policy.personalization_policy import PersonalizedPolicy
from src.agent.state import AgentState
from src.services.llm_service import LLMService
from src.services.user_profile_service import UserProfileService


def realize_actions(
    intents: list[AgentIntent],
    current_state: AgentState,
    event: Event,
    llm_service: LLMService,
    profile_service: UserProfileService | None = None,
    personalized_policy: PersonalizedPolicy | None = None,
) -> list[Action]:
    """把 intent 序列转换为具体 action。"""
    actions: list[Action] = []
    for intent in sorted(intents, key=lambda item: item.priority, reverse=True):
        actions.extend(_realize_intent(intent, current_state, event, llm_service, profile_service, personalized_policy))
    return actions


def _realize_intent(
    intent: AgentIntent,
    current_state: AgentState,
    event: Event,
    llm_service: LLMService,
    profile_service: UserProfileService | None,
    personalized_policy: PersonalizedPolicy | None,
) -> list[Action]:
    """根据意图类型分发到对应的动作生成函数。"""
    if intent.type == "answer_user":
        return _realize_answer_user(intent, current_state, event, llm_service, profile_service, personalized_policy)
    if intent.type == "start_focus":
        return _realize_start_focus(current_state, event, profile_service, personalized_policy)
    if intent.type == "stop_focus":
        return _realize_stop_focus(current_state, event, profile_service, personalized_policy)
    if intent.type == "complete_focus":
        return _realize_complete_focus(current_state, event, profile_service, personalized_policy)
    if intent.type == "suggest_rest":
        return _realize_suggest_rest(intent.reason, current_state, event, profile_service, personalized_policy)
    if intent.type == "remind_distraction":
        return _realize_distraction_reminder(current_state, event, profile_service, personalized_policy)
    if intent.type == "adjust_environment_feedback":
        return _realize_environment_feedback(event.type, intent.payload.get("level"))
    if intent.type == "voice_interaction":
        return _realize_voice_interaction(event)
    if intent.type == "display_update":
        return _realize_display_update(event)
    if intent.type == "update_status_feedback":
        return _realize_status_feedback(event)
    return []


def _realize_answer_user(
    intent: AgentIntent,
    current_state: AgentState,
    event: Event,
    llm_service: LLMService,
    profile_service: UserProfileService | None,
    personalized_policy: PersonalizedPolicy | None,
) -> list[Action]:
    """生成面向用户的回复动作，必要时调用 LLM 生成文本。"""
    text = str(intent.payload.get("text", "")).strip()

    if intent.payload.get("response_mode") == "fixed_text":
        fixed_text = text or "收到。"
        if intent.payload.get("display_only"):
            return [
                display(
                    fixed_text,
                    kind="result",
                    level="info",
                    reason=intent.reason or "status_feedback",
                )
            ]
        return _build_response_actions(
            fixed_text,
            current_state,
            event,
            kind="result",
            level="info",
            reason=intent.reason or "status_feedback",
            profile_service=profile_service,
            personalized_policy=personalized_policy,
        )

    if intent.payload.get("response_mode") == "status_summary":
        summary = _build_status_summary(current_state)
        return _build_response_actions(
            summary,
            current_state,
            event,
            kind="result",
            level="info",
            reason="status_query",
            profile_service=profile_service,
            personalized_policy=personalized_policy,
        )

    if not text:
        return _build_response_actions(
            "请输入内容。",
            current_state,
            event,
            kind="result",
            level="info",
            reason="empty_input",
            profile_service=profile_service,
            personalized_policy=personalized_policy,
        )

    if intent.requires_llm:
        llm_input = build_llm_prompt(text, current_state, personalized_policy, profile_service)
        reply = llm_service.generate_reply(llm_input, current_state)
    else:
        reply = "收到。我会继续保持简洁反馈。"

    return _build_response_actions(
        reply,
        current_state,
        event,
        kind="dialog",
        level="info",
        reason="assistant_reply",
        profile_service=profile_service,
        personalized_policy=personalized_policy,
    )


def _realize_start_focus(
    current_state: AgentState,
    event: Event,
    profile_service: UserProfileService | None,
    personalized_policy: PersonalizedPolicy | None,
) -> list[Action]:
    """生成开始专注时需要执行的一组动作。"""
    duration_sec = current_state.focus.target_duration_sec or 0
    minutes = duration_sec // 60 if duration_sec else 0
    actions = [start_timer(duration_sec)]
    actions.extend(
        _build_response_actions(
            f"已开始 {minutes} 分钟专注。",
            current_state,
            event,
            kind="notification",
            level="info",
            reason="focus_start",
            status="focus",
            profile_service=profile_service,
            personalized_policy=personalized_policy,
        )
    )
    actions.append(set_light_state("thinking", kind="status", level="info", reason="focus_start"))
    return actions


def _realize_stop_focus(
    current_state: AgentState,
    event: Event,
    profile_service: UserProfileService | None,
    personalized_policy: PersonalizedPolicy | None,
) -> list[Action]:
    """生成结束专注时需要执行的一组动作。"""
    if not current_state.memory.focus_sessions:
        return _build_response_actions(
            "当前没有正在进行的专注。",
            current_state,
            event,
            kind="result",
            level="info",
            reason="focus_not_active",
            profile_service=profile_service,
            personalized_policy=personalized_policy,
        )

    actual_duration = int(current_state.memory.focus_sessions[-1]["actual_duration_sec"])
    actions = [stop_timer()]
    actions.extend(
        _build_response_actions(
            f"已结束本次专注，本次持续 {actual_duration} 秒。",
            current_state,
            event,
            kind="notification",
            level="info",
            reason="focus_stop",
            status="idle",
            profile_service=profile_service,
            personalized_policy=personalized_policy,
        )
    )
    actions.append(set_light_state("idle", kind="status", level="info", reason="focus_stop"))
    return actions


def _realize_complete_focus(
    current_state: AgentState,
    event: Event,
    profile_service: UserProfileService | None,
    personalized_policy: PersonalizedPolicy | None,
) -> list[Action]:
    """生成专注自动完成后的提醒动作。"""
    actions = [stop_timer()]
    actions.extend(
        _build_response_actions(
            "专注时间到了，休息一下吧。",
            current_state,
            event,
            kind="notification",
            level="remind",
            reason="focus_complete",
            status="idle",
            proactive=True,
            profile_service=profile_service,
            personalized_policy=personalized_policy,
        )
    )
    actions.append(set_light_state("alert", kind="notification", level="remind", reason="focus_complete"))
    return actions


def _realize_suggest_rest(
    reason: str,
    current_state: AgentState,
    event: Event,
    profile_service: UserProfileService | None,
    personalized_policy: PersonalizedPolicy | None,
) -> list[Action]:
    """生成休息建议对应的动作集合。"""
    reminder_text = build_rest_reminder_text(current_state, personalized_policy, profile_service)
    actions = _build_response_actions(
        reminder_text,
        current_state,
        event,
        kind="notification",
        level="remind",
        reason=reason or "rest_reminder",
        status="alert",
        proactive=True,
        profile_service=profile_service,
        personalized_policy=personalized_policy,
    )
    actions.append(set_light_state("alert", kind="notification", level="remind", reason=reason or "rest_reminder"))
    return actions


def _realize_distraction_reminder(
    current_state: AgentState,
    event: Event,
    profile_service: UserProfileService | None,
    personalized_policy: PersonalizedPolicy | None,
) -> list[Action]:
    """生成分心提醒对应的动作集合。"""
    text = build_distraction_reminder_text(current_state, personalized_policy, profile_service)
    actions = _build_response_actions(
        text,
        current_state,
        event,
        kind="notification",
        level="remind",
        reason="distraction_reminder",
        status="focus",
        proactive=True,
        profile_service=profile_service,
        personalized_policy=personalized_policy,
    )
    actions.append(set_light_state("thinking", kind="notification", level="remind", reason="distraction_reminder"))
    return actions


def _realize_environment_feedback(event_type: str, level: object) -> list[Action]:
    """生成环境异常反馈动作。"""
    text = f"环境状态已更新：{event_type}({level})。"
    return [
        display(
            text,
            kind="notification",
            level="info",
            reason="environment_warning",
        )
    ]


def _realize_voice_interaction(event: Event) -> list[Action]:
    """生成语音交互相关动作。"""
    if event.type == "voice_wake_detected":
        return [start_voice_capture(source=str(event.payload.get("source", "voice")), trigger="wake_word")]
    if event.type == "voice_input_started":
        return [display("语音输入开始。", kind="status", reason="voice_input_started")]
    if event.type == "voice_input_stopped":
        return [stop_voice_capture(source=str(event.payload.get("source", "voice")), reason="voice_input_stopped")]
    return []


def _realize_display_update(event: Event) -> list[Action]:
    """生成显示状态和 TTS 参数更新动作。"""
    if event.type == "tts_started":
        return [display("语音播报中。", kind="status", reason="tts_started")]
    if event.type == "tts_finished":
        return [display("语音播报结束。", kind="status", reason="tts_finished")]
    if event.type == "voice_volume_changed":
        return [set_tts_volume(int(event.payload.get("volume", 50)))]
    if event.type == "voice_timbre_changed":
        return [set_tts_voice(str(event.payload.get("voice_id", "default")))]
    if event.type == "voice_speed_changed":
        return [set_tts_speed(float(event.payload.get("speed", 1.0)))]
    return []


def _realize_status_feedback(event: Event) -> list[Action]:
    """将状态更新事件转换为低优先级显示反馈。"""
    if event.type == "user_presence_updated":
        return [display(f"状态已更新：presence = {event.payload.get('presence')}", kind="status")]
    if event.type == "user_attention_updated":
        return [
            display(
                f"状态已更新：attention = {event.payload.get('attention')}，behavior = {event.payload.get('behavior')}",
                kind="status",
            )
        ]
    if event.type == "user_emotion_updated":
        return [display(f"状态已更新：emotion = {event.payload.get('emotion')}", kind="status")]
    if event.type == "user_fatigue_updated":
        return [display(f"状态已更新：fatigue = {event.payload.get('fatigue_level')}", kind="status")]
    return []


def _build_response_actions(
    text: str,
    current_state: AgentState,
    event: Event,
    *,
    kind: str,
    level: str,
    reason: str,
    status: str | None = None,
    proactive: bool = False,
    profile_service: UserProfileService | None = None,
    personalized_policy: PersonalizedPolicy | None = None,
) -> list[Action]:
    """统一生成 speak 和 display 组合动作。"""
    actions: list[Action] = []
    should_speak = _can_emit_speak(current_state, event, proactive=proactive)

    if should_speak:
        actions.append(
            speak(
                text,
                kind=kind,
                level=level,
                reason=reason,
                **tts_payload_for_user(current_state, personalized_policy, profile_service),
            )
        )

    if not should_speak or kind in {"notification", "dialog", "result"}:
        actions.append(display(text, status=status, kind=kind, level=level, reason=reason))

    return actions


def _can_emit_speak(current_state: AgentState, event: Event, *, proactive: bool) -> bool:
    """统一判断当前场景下是否允许 speak。"""
    if current_state.interaction.mode == "silent":
        return False
    if current_state.interaction.dialogue_state in {"speaking", "listening"}:
        return False
    if proactive and current_state.user.presence == "away":
        return False
    if event.type == "system_triggered" and current_state.user.presence == "away":
        return False
    return True


def _build_status_summary(state: AgentState) -> str:
    """基于当前状态构造规则式摘要文本。"""
    if state.focus.active and state.focus.remaining_sec is not None:
        focus_part = f"正在专注，剩余 {state.focus.remaining_sec} 秒。"
    else:
        focus_part = "当前没有进行中的专注。"

    emotion_summary_part = _latest_emotion_summary_text(state)
    return (
        f"当前模式：{state.interaction.mode}；"
        f"在场状态：{state.user.presence}；"
        f"注意力：{state.user.attention}；"
        f"行为：{state.user.behavior}；"
        f"情绪：{state.user.emotion}；"
        f"疲劳：{state.user.fatigue_level}；"
        f"{focus_part}"
        f"{emotion_summary_part}"
    )


def _latest_emotion_summary_text(state: AgentState) -> str:
    """提取最近一条情绪摘要并格式化为文本。"""
    if not state.memory.emotion_summaries:
        return ""
    latest = state.memory.emotion_summaries[-1]
    dominant = latest.get("dominant_emotion", "unknown")
    avg_confidence = latest.get("avg_confidence")
    if avg_confidence is None:
        return f"最近情绪摘要：主导情绪 {dominant}。"
    return f"最近情绪摘要：主导情绪 {dominant}，平均置信度 {float(avg_confidence):.2f}。"


