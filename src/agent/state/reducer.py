from __future__ import annotations

"""状态归约模块。

把标准 Event 确定性地归约到 AgentState：
- 只更新运行时状态；
- 不生成 Intent / Action；
- 不调用 LLM；
- 不写偏好记忆；
- 不读取策略配置。

事件分发通过 ``REDUCERS`` 映射完成，未注册事件被安全忽略并返回原状态。
"""

from collections.abc import Callable

from src.agent.event.event_model import Event
from src.agent.state.agent_state import AgentState

StateReducer = Callable[[AgentState, Event], None]


def reduce_state(state: AgentState, event: Event) -> AgentState:
    """根据事件类型分发到对应 handler，并返回同一个 state 对象。"""

    handler = REDUCERS.get(str(event.type))
    if handler is None:
        return state
    handler(state, event)
    return state


def _handle_speech_recognized(state: AgentState, event: Event) -> None:
    """语音识别后更新对话状态与最后输入时间。"""
    text = str(event.payload.get("text", "")).strip()
    state.interaction.in_conversation = bool(text)
    state.interaction.dialogue_state = "thinking" if text else "idle"
    state.interaction.last_user_time = event.timestamp


def _handle_focus_start_requested(state: AgentState, event: Event) -> None:
    if state.focus.active:
        return
    duration_sec = _optional_int(event.payload.get("duration_sec"), 0) or 0
    _start_focus_session(
        state,
        start_ts=event.timestamp,
        duration_sec=duration_sec,
        triggered_by=str(event.payload.get("source", "user")),
    )


def _handle_focus_stop_requested(state: AgentState, event: Event) -> None:
    if not state.focus.active:
        return
    _complete_focus_session(state, event.timestamp, reason="manual_stop")


def _handle_system_triggered(state: AgentState, event: Event) -> None:
    """消费由 start_timer/stop_timer 动作回流的计时器事件，让 FocusState 跟随真实计时。"""
    trigger = str(event.payload.get("trigger", "")).strip()
    if trigger == "focus_timer_started":
        duration_sec = _optional_int(event.payload.get("duration_sec"), state.focus.target_duration_sec) or 0
        _start_focus_session(
            state,
            start_ts=event.timestamp,
            duration_sec=duration_sec,
            triggered_by=str(event.payload.get("source_event_type") or event.payload.get("source") or "timer"),
        )
        return
    if trigger == "focus_timer_stopped" and state.focus.active:
        _complete_focus_session(state, event.timestamp, reason="timer_stopped")


def _start_focus_session(state: AgentState, *, start_ts: int, duration_sec: int, triggered_by: str) -> None:
    state.interaction.mode = "focus"
    state.interaction.dialogue_state = "idle"
    state.focus.active = True
    state.focus.start_ts = start_ts
    state.focus.target_duration_sec = duration_sec
    state.focus.elapsed_sec = 0
    state.focus.remaining_sec = duration_sec
    state.focus.triggered_by = triggered_by


def _handle_user_presence_updated(state: AgentState, event: Event) -> None:
    state.user.presence = str(event.payload.get("presence", state.user.presence))
    state.user.presence_confidence = _optional_float(event.payload.get("confidence"))


def _handle_user_attention_updated(state: AgentState, event: Event) -> None:
    state.user.attention = str(event.payload.get("attention", state.user.attention))
    state.user.behavior = str(event.payload.get("behavior", state.user.behavior))
    confidence = _optional_float(event.payload.get("confidence"))
    state.user.attention_confidence = confidence
    state.user.behavior_confidence = confidence


def _handle_user_emotion_updated(state: AgentState, event: Event) -> None:
    state.user.emotion = str(event.payload.get("emotion", state.user.emotion))
    state.user.emotion_confidence = _optional_float(event.payload.get("confidence"))


def _handle_user_fatigue_updated(state: AgentState, event: Event) -> None:
    state.user.fatigue_level = str(event.payload.get("fatigue_level", state.user.fatigue_level))
    state.user.fatigue_confidence = _optional_float(event.payload.get("confidence"))


def _handle_user_posture_updated(state: AgentState, event: Event) -> None:
    state.user.posture = str(event.payload.get("posture", state.user.posture))
    state.user.posture_confidence = _optional_float(event.payload.get("confidence"))


def _handle_user_activity_updated(state: AgentState, event: Event) -> None:
    state.user.current_activity = str(event.payload.get("activity", state.user.current_activity))


def _handle_light_level_updated(state: AgentState, event: Event) -> None:
    state.environment.light_lux = _optional_int(event.payload.get("light_lux"))
    state.environment.light_level = _optional_str(event.payload.get("level"))


def _handle_temperature_humidity_updated(state: AgentState, event: Event) -> None:
    state.environment.temperature_c = _optional_float(event.payload.get("temperature_c"))
    state.environment.humidity_pct = _optional_float(event.payload.get("humidity_pct"))
    state.environment.temperature_level = _optional_str(event.payload.get("temperature_level"))
    state.environment.humidity_level = _optional_str(event.payload.get("humidity_level"))


def _handle_noise_level_updated(state: AgentState, event: Event) -> None:
    state.environment.noise_db = _optional_int(event.payload.get("noise_db"))
    state.environment.noise_level = _optional_str(event.payload.get("level"))


def _handle_voice_input_started(state: AgentState, event: Event) -> None:
    del event
    state.interaction.dialogue_state = "listening"


def _handle_voice_input_stopped(state: AgentState, event: Event) -> None:
    del event
    state.interaction.dialogue_state = "thinking"


def _handle_tts_started(state: AgentState, event: Event) -> None:
    del event
    state.interaction.dialogue_state = "speaking"


def _handle_tts_finished(state: AgentState, event: Event) -> None:
    del event
    state.interaction.dialogue_state = "idle"


def _handle_timer_ticked(state: AgentState, event: Event) -> None:
    if not state.focus.active or state.focus.start_ts is None:
        return
    remaining_sec = max(0, _optional_int(event.payload.get("remaining_sec"), 0) or 0)
    elapsed_sec = max(0, event.timestamp - state.focus.start_ts)
    target_sec = state.focus.target_duration_sec or 0
    state.focus.remaining_sec = remaining_sec
    state.focus.elapsed_sec = min(elapsed_sec, target_sec) if target_sec > 0 else elapsed_sec


def _handle_timer_finished(state: AgentState, event: Event) -> None:
    if not state.focus.active:
        return
    _complete_focus_session(state, event.timestamp, reason="timer_complete")


def _complete_focus_session(state: AgentState, end_ts: int, reason: str) -> None:
    start_ts = state.focus.start_ts
    target_duration_sec = state.focus.target_duration_sec or 0
    actual_duration_sec = 0 if start_ts is None else max(0, end_ts - start_ts)

    state.runtime_history.focus_sessions.append(
        {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "planned_duration_sec": target_duration_sec,
            "actual_duration_sec": actual_duration_sec,
            "triggered_by": state.focus.triggered_by,
            "reason": reason,
        }
    )
    state.runtime_history.focus_session_count += 1
    state.runtime_history.focus_total_duration_sec += actual_duration_sec

    state.interaction.mode = "normal"
    state.interaction.dialogue_state = "idle"
    state.focus.active = False
    state.focus.start_ts = None
    state.focus.target_duration_sec = None
    state.focus.elapsed_sec = 0
    state.focus.remaining_sec = None
    state.focus.triggered_by = None
    state.focus.last_focus_end_ts = end_ts


def _optional_int(value: object, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_str(value: object, default: str | None = None) -> str | None:
    if value is None:
        return default
    return str(value)


REDUCERS: dict[str, StateReducer] = {
    "speech_recognized": _handle_speech_recognized,
    "focus_start_requested": _handle_focus_start_requested,
    "focus_stop_requested": _handle_focus_stop_requested,
    "user_presence_updated": _handle_user_presence_updated,
    "user_attention_updated": _handle_user_attention_updated,
    "user_emotion_updated": _handle_user_emotion_updated,
    "user_fatigue_updated": _handle_user_fatigue_updated,
    "user_posture_updated": _handle_user_posture_updated,
    "user_activity_updated": _handle_user_activity_updated,
    "light_level_updated": _handle_light_level_updated,
    "temperature_humidity_updated": _handle_temperature_humidity_updated,
    "noise_level_updated": _handle_noise_level_updated,
    "voice_input_started": _handle_voice_input_started,
    "voice_input_stopped": _handle_voice_input_stopped,
    "tts_started": _handle_tts_started,
    "tts_finished": _handle_tts_finished,
    "timer_ticked": _handle_timer_ticked,
    "timer_finished": _handle_timer_finished,
    "system_triggered": _handle_system_triggered,
}
