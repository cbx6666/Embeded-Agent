from __future__ import annotations

"""状态归约模块。

本模块只负责“事件如何更新状态”，不负责决定要不要输出动作。
事件来源可能是 CLI、mock 输入、未来的摄像头、麦克风、传感器或内部定时器。
"""

from src.agent.event import Event
from src.agent.state import AgentState



def reduce_state(state: AgentState, event: Event) -> AgentState:
    """根据事件直接在当前状态对象上完成归约更新。"""
    if event.type == "user_text_input":
        _handle_user_text_input(state, event)
    elif event.type == "focus_start_requested":
        _handle_focus_start_requested(state, event)
    elif event.type == "focus_stop_requested":
        _handle_focus_stop_requested(state, event)
    elif event.type == "user_presence_updated":
        state.user.presence = event.payload.get("presence", state.user.presence)
    elif event.type == "user_attention_updated":
        state.user.attention = event.payload.get("attention", state.user.attention)
        state.user.attention_confidence = event.payload.get("confidence")
    elif event.type == "user_emotion_updated":
        state.user.emotion = event.payload.get("emotion", state.user.emotion)
        state.user.emotion_confidence = event.payload.get("confidence")
    elif event.type == "environment_updated":
        _handle_environment_updated(state, event)
    elif event.type == "timer_ticked":
        _handle_timer_ticked(state, event)
    elif event.type == "timer_finished":
        _handle_timer_finished(state, event)
    return state



def _handle_user_text_input(state: AgentState, event: Event) -> None:
    """更新用户输入后带来的交互状态。"""
    text = str(event.payload.get("text", "")).strip()
    state.interaction.in_conversation = bool(text)
    state.interaction.dialogue_state = "listening" if text else "idle"
    state.interaction.last_user_time = event.timestamp



def _handle_focus_start_requested(state: AgentState, event: Event) -> None:
    """进入专注模式，并初始化计时字段。"""
    if state.focus.active:
        return

    duration_sec = int(event.payload.get("duration_sec", 0))
    state.interaction.mode = "focus"
    state.focus.active = True
    state.focus.start_ts = event.timestamp
    state.focus.target_duration_sec = duration_sec
    state.focus.elapsed_sec = 0
    state.focus.remaining_sec = duration_sec
    state.focus.triggered_by = str(event.payload.get("source", "user"))



def _handle_focus_stop_requested(state: AgentState, event: Event) -> None:
    """手动结束专注。"""
    if not state.focus.active:
        return
    _complete_focus_session(state, event.timestamp, reason="manual_stop")



def _handle_environment_updated(state: AgentState, event: Event) -> None:
    """更新环境状态。"""
    if "light" in event.payload:
        state.environment.light = event.payload.get("light")
    if "noise" in event.payload:
        state.environment.noise = event.payload.get("noise")
    if "temperature" in event.payload:
        state.environment.temperature = event.payload.get("temperature")
    if "humidity" in event.payload:
        state.environment.humidity = event.payload.get("humidity")



def _handle_timer_ticked(state: AgentState, event: Event) -> None:
    """处理专注倒计时过程中的普通 tick。"""
    if not state.focus.active or state.focus.start_ts is None:
        return

    remaining_sec = int(event.payload.get("remaining_sec", 0))
    state.focus.remaining_sec = remaining_sec
    state.focus.elapsed_sec = max(0, event.timestamp - state.focus.start_ts)



def _handle_timer_finished(state: AgentState, event: Event) -> None:
    """处理专注倒计时结束事件。"""
    if not state.focus.active:
        return
    _complete_focus_session(state, event.timestamp, reason="timer_complete")



def _complete_focus_session(state: AgentState, end_ts: int, reason: str) -> None:
    """结束专注并写入专注会话历史。"""
    start_ts = state.focus.start_ts
    target_duration_sec = state.focus.target_duration_sec or 0
    actual_duration_sec = 0 if start_ts is None else max(0, end_ts - start_ts)

    state.memory.focus_sessions.append(
        {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "planned_duration_sec": target_duration_sec,
            "actual_duration_sec": actual_duration_sec,
            "triggered_by": state.focus.triggered_by,
            "reason": reason,
        }
    )

    state.interaction.mode = "normal"
    state.interaction.dialogue_state = "idle"
    state.focus.active = False
    state.focus.start_ts = None
    state.focus.target_duration_sec = None
    state.focus.elapsed_sec = 0
    state.focus.remaining_sec = None
    state.focus.triggered_by = None
    state.focus.last_focus_end_ts = end_ts
