from __future__ import annotations

"""状态归约模块。

本模块只负责“事件如何更新状态”，不负责决定要不要输出动作。
"""

from src.agent.event import Event
from src.agent.state import AgentState


def reduce_state(state: AgentState, event: Event) -> AgentState:
    """根据事件直接在当前状态对象上完成归约更新。"""
    if event.type in {"user_text_input", "speech_recognized"}:
        _handle_user_input_event(state, event)
    elif event.type == "focus_start_requested":
        _handle_focus_start_requested(state, event)
    elif event.type == "focus_stop_requested":
        _handle_focus_stop_requested(state, event)
    elif event.type == "user_presence_updated":
        state.user.presence = str(event.payload.get("presence", state.user.presence))
        state.user.presence_confidence = _optional_float(event.payload.get("confidence"))
    elif event.type == "user_attention_updated":
        state.user.attention = str(event.payload.get("attention", state.user.attention))
        state.user.behavior = str(event.payload.get("behavior", state.user.behavior))
        confidence = _optional_float(event.payload.get("confidence"))
        state.user.attention_confidence = confidence
        state.user.behavior_confidence = confidence
    elif event.type == "user_emotion_updated":
        state.user.emotion = str(event.payload.get("emotion", state.user.emotion))
        state.user.emotion_confidence = _optional_float(event.payload.get("confidence"))
    elif event.type == "user_fatigue_updated":
        state.user.fatigue_level = str(event.payload.get("fatigue_level", state.user.fatigue_level))
        state.user.fatigue_confidence = _optional_float(event.payload.get("confidence"))
    elif event.type == "light_level_updated":
        state.environment.light_lux = _optional_int(event.payload.get("light_lux"))
        state.environment.light_level = _optional_str(event.payload.get("level"))
    elif event.type == "temperature_humidity_updated":
        state.environment.temperature_c = _optional_float(event.payload.get("temperature_c"))
        state.environment.humidity_pct = _optional_float(event.payload.get("humidity_pct"))
        state.environment.temperature_level = _optional_str(event.payload.get("temperature_level"))
        state.environment.humidity_level = _optional_str(event.payload.get("humidity_level"))
    elif event.type == "noise_level_updated":
        state.environment.noise_db = _optional_int(event.payload.get("noise_db"))
        state.environment.noise_level = _optional_str(event.payload.get("level"))
    elif event.type == "user_behavior_signal_updated":
        pass
    elif event.type == "user_behavior_summary_updated":
        pass
    elif event.type == "voice_input_started":
        state.interaction.dialogue_state = "listening"
    elif event.type == "voice_input_stopped":
        state.interaction.dialogue_state = "thinking"
    elif event.type == "tts_started":
        state.interaction.dialogue_state = "speaking"
    elif event.type == "tts_finished":
        state.interaction.dialogue_state = "idle"
    elif event.type == "timer_ticked":
        _handle_timer_ticked(state, event)
    elif event.type == "timer_finished":
        _handle_timer_finished(state, event)
    return state


def _handle_user_input_event(state: AgentState, event: Event) -> None:
    text = str(event.payload.get("text", "")).strip()
    state.interaction.in_conversation = bool(text)
    state.interaction.dialogue_state = "thinking" if text else "idle"
    state.interaction.last_user_time = event.timestamp


def _handle_focus_start_requested(state: AgentState, event: Event) -> None:
    if state.focus.active:
        return

    duration_sec = int(event.payload.get("duration_sec", 0))
    state.interaction.mode = "focus"
    state.interaction.dialogue_state = "idle"
    state.focus.active = True
    state.focus.start_ts = event.timestamp
    state.focus.target_duration_sec = duration_sec
    state.focus.elapsed_sec = 0
    state.focus.remaining_sec = duration_sec
    state.focus.triggered_by = str(event.payload.get("source", "user"))


def _handle_focus_stop_requested(state: AgentState, event: Event) -> None:
    if not state.focus.active:
        return
    _complete_focus_session(state, event.timestamp, reason="manual_stop")

def _handle_timer_ticked(state: AgentState, event: Event) -> None:
    if not state.focus.active or state.focus.start_ts is None:
        return

    state.focus.remaining_sec = int(event.payload.get("remaining_sec", 0))
    state.focus.elapsed_sec = max(0, event.timestamp - state.focus.start_ts)


def _handle_timer_finished(state: AgentState, event: Event) -> None:
    if not state.focus.active:
        return
    _complete_focus_session(state, event.timestamp, reason="timer_complete")


def _complete_focus_session(state: AgentState, end_ts: int, reason: str) -> None:
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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
