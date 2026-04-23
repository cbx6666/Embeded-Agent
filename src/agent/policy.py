from __future__ import annotations

"""策略决策模块。

本模块根据事件和当前状态生成动作，不直接修改状态。
"""

from src.agent.action import (
    Action,
    display,
    set_light_state,
    speak,
    start_timer,
    stop_timer,
)
from src.agent.event import Event
from src.agent.state import AgentState
from src.services.llm_service import LLMService

TIRED_REMINDER_MIN_FOCUS_SEC = 300
TIRED_REMINDER_COOLDOWN_SEC = 300


def decide_actions(
    previous_state: AgentState,
    current_state: AgentState,
    event: Event,
    llm_service: LLMService,
) -> list[Action]:
    """根据事件类型路由到对应策略分支。"""
    if event.type in {"user_text_input", "speech_recognized"}:
        return _decide_user_input(current_state, event, llm_service)
    if event.type == "focus_start_requested":
        return _decide_focus_start_requested(previous_state, current_state)
    if event.type == "focus_stop_requested":
        return _decide_focus_stop_requested(previous_state, current_state)
    if event.type == "timer_ticked":
        return _decide_timer_ticked(current_state, event)
    if event.type == "timer_finished":
        return _decide_timer_finished(previous_state)
    if event.type in {
        "user_presence_updated",
        "user_attention_updated",
        "user_emotion_updated",
        "user_fatigue_updated",
        "light_level_updated",
        "temperature_humidity_updated",
        "noise_level_updated",
    }:
        return _decide_state_feedback(event)
    return []


def _decide_user_input(
    current_state: AgentState,
    event: Event,
    llm_service: LLMService,
) -> list[Action]:
    text = str(event.payload.get("text", "")).strip()
    if not text:
        return [speak("请输入内容。", kind="result", level="info", reason="empty_input")]

    if _is_status_query(text):
        summary = _build_status_summary(current_state)
        return [
            speak(summary, kind="result", level="info", reason="status_query"),
            display(summary, kind="result", level="info", reason="status_query"),
        ]

    reply = llm_service.generate_reply(text, current_state)
    return [
        speak(reply, kind="dialog", level="info", reason="assistant_reply"),
        display(reply, kind="result", level="info", reason="assistant_reply"),
    ]


def _decide_focus_start_requested(
    previous_state: AgentState,
    current_state: AgentState,
) -> list[Action]:
    if previous_state.focus.active:
        return [speak("当前已经在专注中。", kind="result", level="info", reason="focus_already_active")]

    duration_sec = current_state.focus.target_duration_sec or 0
    minutes = duration_sec // 60 if duration_sec else 0
    return [
        start_timer(duration_sec),
        speak(
            f"已开始 {minutes} 分钟专注。",
            kind="notification",
            level="info",
            reason="focus_start",
        ),
        display(
            f"专注倒计时已启动：{minutes} 分钟。",
            status="focus",
            kind="notification",
            level="info",
            reason="focus_start",
        ),
        set_light_state(
            "thinking",
            kind="status",
            level="info",
            reason="focus_start",
        ),
    ]


def _decide_focus_stop_requested(
    previous_state: AgentState,
    current_state: AgentState,
) -> list[Action]:
    if not previous_state.focus.active:
        return [speak("当前没有正在进行的专注。", kind="result", level="info", reason="focus_not_active")]

    actual_duration = 0
    if current_state.memory.focus_sessions:
        actual_duration = int(current_state.memory.focus_sessions[-1]["actual_duration_sec"])
    return [
        stop_timer(),
        speak(
            f"已结束本次专注，本次持续 {actual_duration} 秒。",
            kind="notification",
            level="info",
            reason="focus_stop",
        ),
        display(
            "专注已结束。",
            status="idle",
            kind="notification",
            level="info",
            reason="focus_stop",
        ),
        set_light_state("idle", kind="status", level="info", reason="focus_stop"),
    ]


def _decide_timer_ticked(current_state: AgentState, now_event: Event) -> list[Action]:
    if not _should_trigger_rest_reminder(current_state, now_event.timestamp):
        return []

    reminder_text = "你已经专注了一段时间，而且看起来有点疲惫，建议休息一下。"
    actions: list[Action] = [
        display(
            reminder_text,
            status="alert",
            kind="notification",
            level="remind",
            reason="rest_reminder",
        ),
        set_light_state(
            "alert",
            kind="notification",
            level="remind",
            reason="rest_reminder",
        ),
    ]
    if current_state.user.presence != "away":
        actions.insert(
            0,
            speak(
                reminder_text,
                kind="notification",
                level="remind",
                reason="rest_reminder",
            ),
        )
    return actions


def _decide_timer_finished(previous_state: AgentState) -> list[Action]:
    if not previous_state.focus.active:
        return []
    return [
        stop_timer(),
        speak(
            "专注时间到了，休息一下吧。",
            kind="notification",
            level="remind",
            reason="focus_complete",
        ),
        display(
            "本轮专注已完成。",
            status="idle",
            kind="notification",
            level="remind",
            reason="focus_complete",
        ),
        set_light_state(
            "alert",
            kind="notification",
            level="remind",
            reason="focus_complete",
        ),
    ]


def _decide_state_feedback(event: Event) -> list[Action]:
    if event.type == "user_presence_updated":
        return [display(f"状态已更新：presence = {event.payload.get('presence')}", kind="status")]
    if event.type == "user_attention_updated":
        attention = event.payload.get("attention")
        behavior = event.payload.get("behavior")
        return [display(f"状态已更新：attention = {attention}，behavior = {behavior}", kind="status")]
    if event.type == "user_emotion_updated":
        return [display(f"状态已更新：emotion = {event.payload.get('emotion')}", kind="status")]
    if event.type == "user_fatigue_updated":
        return [display(f"状态已更新：fatigue = {event.payload.get('fatigue_level')}", kind="status")]
    return [display("环境状态已更新。", kind="status")]


def _should_trigger_rest_reminder(state: AgentState, now_ts: int) -> bool:
    if not state.focus.active or state.focus.start_ts is None:
        return False
    if state.user.attention != "focused":
        return False
    if state.user.fatigue_level not in {"moderate", "high"} and state.user.emotion != "tired":
        return False
    if state.focus.elapsed_sec < TIRED_REMINDER_MIN_FOCUS_SEC:
        return False

    last_ts = state.cooldown.reminder_last_ts.get("rest_reminder")
    if last_ts is not None and now_ts - last_ts < TIRED_REMINDER_COOLDOWN_SEC:
        return False
    return True


def _is_status_query(text: str) -> bool:
    keywords = ("现在状态如何", "当前状态", "state", "status")
    lowered = text.strip().lower()
    return any(keyword in lowered for keyword in keywords)


def _build_status_summary(state: AgentState) -> str:
    if state.focus.active and state.focus.remaining_sec is not None:
        focus_part = f"正在专注，剩余 {state.focus.remaining_sec} 秒。"
    else:
        focus_part = "当前没有进行中的专注。"

    return (
        f"当前模式：{state.interaction.mode}；"
        f"在场状态：{state.user.presence}；"
        f"注意力：{state.user.attention}；"
        f"行为：{state.user.behavior}；"
        f"情绪：{state.user.emotion}；"
        f"疲劳：{state.user.fatigue_level}；"
        f"{focus_part}"
    )
