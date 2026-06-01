"""
状态归约模块。

本模块只负责把标准 Event 确定性归约到 AgentState。它位于 AgentCore 主链路
最前段：上游输入是 Event，下游输出是更新后的运行时状态。

Reducer 不调用 LLM、不生成 Intent、不生成 Action、不写长期 LongTermMemoryStore，也
不读取策略配置。未注册事件会被安全忽略并返回原状态。
"""

from __future__ import annotations

"""状态归约模块。

reducer 只负责把标准 Event 确定性地归约到 AgentState：
- 只更新运行时状态。
- 不生成 Intent。
- 不生成 Action。
- 不调用 LLM。
- 不写长期 profile。
- 不读取 YAML 策略。

事件分发通过 REDUCERS 映射完成，避免把所有 event.type 分支堆在
reduce_state 一个函数里。未注册事件会被安全忽略并返回原状态对象。
"""

from collections.abc import Callable

from src.agent.event import Event
from src.agent.state import AgentState

StateReducer = Callable[[AgentState, Event], None]


def reduce_state(state: AgentState, event: Event) -> AgentState:
    """根据事件类型分发到对应 handler，并返回同一个 state 对象。

    这里不做智能判断，只做状态事实更新；语义理解交给 LLM Agent。
    """

    """根据事件类型分发到对应 handler，并返回同一个 state 对象。"""
    handler = REDUCERS.get(str(event.type))
    if handler is None:
        return state
    handler(state, event)
    return state


def _handle_user_input_event(state: AgentState, event: Event) -> None:
    """更新用户输入后的对话状态、会话活跃状态和最后输入时间。"""
    text = str(event.payload.get("text", "")).strip()
    state.interaction.in_conversation = bool(text)
    state.interaction.dialogue_state = "thinking" if text else "idle"
    state.interaction.last_user_time = event.timestamp


def _handle_focus_start_requested(state: AgentState, event: Event) -> None:
    """启动一轮新的专注状态；已专注时保持现状，避免重复启动。"""
    if state.focus.active:
        return

    duration_sec = _optional_int(event.payload.get("duration_sec"), 0) or 0
    _start_focus_session(
        state,
        start_ts=event.timestamp,
        duration_sec=duration_sec,
        triggered_by=str(event.payload.get("source", "user")),
    )


def _handle_system_triggered(state: AgentState, event: Event) -> None:
    """Consume observable timer result events so FocusState follows the real timer."""

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


def _start_focus_session(
    state: AgentState,
    *,
    start_ts: int,
    duration_sec: int,
    triggered_by: str,
) -> None:
    """Set FocusState to an active session from an observable event."""

    state.interaction.mode = "focus"
    state.interaction.dialogue_state = "idle"
    state.focus.active = True
    state.focus.start_ts = start_ts
    state.focus.target_duration_sec = duration_sec
    state.focus.elapsed_sec = 0
    state.focus.remaining_sec = duration_sec
    state.focus.triggered_by = triggered_by


def _handle_focus_stop_requested(state: AgentState, event: Event) -> None:
    """结束当前专注状态并写入短期 focus session 记录。"""
    if not state.focus.active:
        return
    _complete_focus_session(state, event.timestamp, reason="manual_stop")


def _handle_user_switched(state: AgentState, event: Event) -> None:
    """切换当前用户 ID；长期 profile 创建由 service 层负责。"""
    state.current_user_id = str(event.payload.get("user_id", "default")).strip() or "default"


def _handle_user_presence_updated(state: AgentState, event: Event) -> None:
    """更新用户在场状态和识别置信度。"""
    state.user.presence = str(event.payload.get("presence", state.user.presence))
    state.user.presence_confidence = _optional_float(event.payload.get("confidence"))


def _handle_user_attention_updated(state: AgentState, event: Event) -> None:
    """更新注意力、行为状态以及对应置信度。"""
    state.user.attention = str(event.payload.get("attention", state.user.attention))
    state.user.behavior = str(event.payload.get("behavior", state.user.behavior))
    confidence = _optional_float(event.payload.get("confidence"))
    state.user.attention_confidence = confidence
    state.user.behavior_confidence = confidence


def _handle_user_emotion_updated(state: AgentState, event: Event) -> None:
    """更新用户情绪状态和情绪识别置信度。"""
    state.user.emotion = str(event.payload.get("emotion", state.user.emotion))
    state.user.emotion_confidence = _optional_float(event.payload.get("confidence"))


def _handle_user_fatigue_updated(state: AgentState, event: Event) -> None:
    """更新用户疲劳等级和疲劳识别置信度。"""
    state.user.fatigue_level = str(event.payload.get("fatigue_level", state.user.fatigue_level))
    state.user.fatigue_confidence = _optional_float(event.payload.get("confidence"))


def _handle_user_posture_updated(state: AgentState, event: Event) -> None:
    """更新用户姿势状态和姿势识别置信度。"""
    state.user.posture = str(event.payload.get("posture", state.user.posture))
    state.user.posture_confidence = _optional_float(event.payload.get("confidence"))


def _handle_user_activity_updated(state: AgentState, event: Event) -> None:
    """更新用户当前活动状态。"""
    state.user.current_activity = str(event.payload.get("activity", state.user.current_activity))


def _handle_light_level_updated(state: AgentState, event: Event) -> None:
    """更新光照数值和标准化光照等级。"""
    state.environment.light_lux = _optional_int(event.payload.get("light_lux"))
    state.environment.light_level = _optional_str(event.payload.get("level"))


def _handle_temperature_humidity_updated(state: AgentState, event: Event) -> None:
    """更新温湿度数值和标准化等级。"""
    state.environment.temperature_c = _optional_float(event.payload.get("temperature_c"))
    state.environment.humidity_pct = _optional_float(event.payload.get("humidity_pct"))
    state.environment.temperature_level = _optional_str(event.payload.get("temperature_level"))
    state.environment.humidity_level = _optional_str(event.payload.get("humidity_level"))


def _handle_noise_level_updated(state: AgentState, event: Event) -> None:
    """更新噪声分贝和标准化噪声等级。"""
    state.environment.noise_db = _optional_int(event.payload.get("noise_db"))
    state.environment.noise_level = _optional_str(event.payload.get("level"))


def _handle_voice_input_started(state: AgentState, event: Event) -> None:
    """进入语音监听状态。"""
    del event
    state.interaction.dialogue_state = "listening"


def _handle_voice_input_stopped(state: AgentState, event: Event) -> None:
    """语音输入结束后进入思考状态。"""
    del event
    state.interaction.dialogue_state = "thinking"


def _handle_tts_started(state: AgentState, event: Event) -> None:
    """TTS 开始后进入播报状态。"""
    del event
    state.interaction.dialogue_state = "speaking"


def _handle_tts_finished(state: AgentState, event: Event) -> None:
    """TTS 结束后回到空闲状态。"""
    del event
    state.interaction.dialogue_state = "idle"


def _handle_timer_ticked(state: AgentState, event: Event) -> None:
    """根据 timer tick 更新专注剩余时间和已用时长。"""
    if not state.focus.active or state.focus.start_ts is None:
        return

    remaining_sec = max(0, _optional_int(event.payload.get("remaining_sec"), 0) or 0)
    elapsed_sec = max(0, event.timestamp - state.focus.start_ts)
    target_sec = state.focus.target_duration_sec or 0
    state.focus.remaining_sec = remaining_sec
    state.focus.elapsed_sec = min(elapsed_sec, target_sec) if target_sec > 0 else elapsed_sec
    if remaining_sec <= 0 or (target_sec > 0 and elapsed_sec >= target_sec):
        _complete_focus_session(state, event.timestamp, reason="timer_complete")


def _handle_timer_finished(state: AgentState, event: Event) -> None:
    """在专注计时结束时关闭本轮专注并归档会话。"""
    if not state.focus.active:
        return
    _complete_focus_session(state, event.timestamp, reason="timer_complete")


def _complete_focus_session(state: AgentState, end_ts: int, reason: str) -> None:
    """归档当前专注会话并重置专注相关状态。"""
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
    """将可选输入安全转换为整数；非法值回退到 default。"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: object, default: float | None = None) -> float | None:
    """将可选输入安全转换为浮点数；非法值回退到 default。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_str(value: object, default: str | None = None) -> str | None:
    """将可选输入安全转换为字符串。"""
    if value is None:
        return default
    return str(value)


REDUCERS: dict[str, StateReducer] = {
    "user_text_input": _handle_user_input_event,
    "speech_recognized": _handle_user_input_event,
    "focus_start_requested": _handle_focus_start_requested,
    "focus_stop_requested": _handle_focus_stop_requested,
    "user_switched": _handle_user_switched,
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
