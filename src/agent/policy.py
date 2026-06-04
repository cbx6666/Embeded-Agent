from __future__ import annotations

"""策略决策模块。

本模块根据事件和当前状态生成动作，不直接修改状态。
"""

from src.agent.action import Action, display, speak, start_timer, stop_timer
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
    if event.type == "user_text_input":
        return _decide_user_text_input(current_state, event, llm_service)
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
        "environment_updated",
    }:
        return _decide_state_feedback(event)
    return []



def _decide_user_text_input(
    current_state: AgentState,
    event: Event,
    llm_service: LLMService,
) -> list[Action]:
    """处理普通文本输入对应的动作决策。"""
    text = str(event.payload.get("text", "")).strip()
    if not text:
        return [speak("请输入内容。")]

    if _is_status_query(text):
        return [speak(_build_status_summary(current_state))]

    return [speak(llm_service.generate_reply(text, current_state))]



def _decide_focus_start_requested(
    previous_state: AgentState,
    current_state: AgentState,
) -> list[Action]:
    """处理开始专注请求。"""
    if previous_state.focus.active:
        return [speak("当前已经在专注中。")]

    duration_sec = current_state.focus.target_duration_sec or 0
    minutes = duration_sec // 60 if duration_sec else 0
    return [
        start_timer(duration_sec),
        speak(f"已开始 {minutes} 分钟专注。", kind="focus_start"),
        display(f"专注倒计时已启动：{minutes} 分钟。"),
    ]



def _decide_focus_stop_requested(
    previous_state: AgentState,
    current_state: AgentState,
) -> list[Action]:
    """处理结束专注请求。"""
    if not previous_state.focus.active:
        return [speak("当前没有正在进行的专注。")]

    actual_duration = 0
    if current_state.memory.focus_sessions:
        actual_duration = int(current_state.memory.focus_sessions[-1]["actual_duration_sec"])
    return [
        stop_timer(),
        speak(f"已结束本次专注，本次持续 {actual_duration} 秒。", kind="focus_stop"),
        display("专注已结束。"),
    ]



def _decide_timer_ticked(current_state: AgentState, event: Event) -> list[Action]:
    """处理计时过程中的提醒决策。"""
    if not _should_trigger_rest_reminder(current_state, event.timestamp):
        return []

    reminder_text = "你已经专注了一段时间，而且看起来有点疲惫，建议休息一下。"
    if current_state.user.presence == "away":
        return [display(reminder_text, kind="rest_reminder")]
    return [speak(reminder_text, kind="rest_reminder")]



def _decide_timer_finished(previous_state: AgentState) -> list[Action]:
    """处理倒计时结束后的完成提醒。"""
    if not previous_state.focus.active:
        return []
    return [
        stop_timer(),
        speak("专注时间到了，休息一下吧。", kind="focus_complete"),
        display("本轮专注已完成。"),
    ]



def _decide_state_feedback(event: Event) -> list[Action]:
    """为状态更新生成一条轻量确认提示。"""
    if event.type == "user_presence_updated":
        return [display(f"状态已更新：presence = {event.payload.get('presence')}")]
    if event.type == "user_attention_updated":
        return [display(f"状态已更新：attention = {event.payload.get('attention')}")]
    if event.type == "user_emotion_updated":
        return [display(f"状态已更新：emotion = {event.payload.get('emotion')}")]
    return [display("环境状态已更新。")]



def _should_trigger_rest_reminder(state: AgentState, now_ts: int) -> bool:
    """判断当前是否应该触发疲劳休息提醒。"""
    if not state.focus.active or state.focus.start_ts is None:
        return False
    if state.user.attention != "focused":
        return False
    if state.user.emotion != "tired":
        return False
    if state.focus.elapsed_sec < TIRED_REMINDER_MIN_FOCUS_SEC:
        return False

    last_ts = state.cooldown.reminder_last_ts.get("rest_reminder")
    if last_ts is not None and now_ts - last_ts < TIRED_REMINDER_COOLDOWN_SEC:
        return False
    return True



def _is_status_query(text: str) -> bool:
    """判断文本是否是在询问当前状态。"""
    keywords = ("现在状态如何", "当前状态", "state", "status")
    lowered = text.strip().lower()
    return any(keyword in lowered for keyword in keywords)



def _build_status_summary(state: AgentState) -> str:
    """构造一段面向用户的状态摘要。"""
    if state.focus.active and state.focus.remaining_sec is not None:
        focus_part = f"正在专注，剩余 {state.focus.remaining_sec} 秒。"
    else:
        focus_part = "当前没有进行中的专注。"
    return (
        f"当前模式：{state.interaction.mode}；"
        f"在场状态：{state.user.presence}；"
        f"注意力：{state.user.attention}；"
        f"情绪：{state.user.emotion}；"
        f"{focus_part}"
    )
