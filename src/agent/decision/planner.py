from __future__ import annotations

"""意图规划层。"""

from src.agent.decision.intent import AgentIntent
from src.agent.decision.intent_guard import guard_intents
from src.agent.decision.llm_intent_planner import plan_intents_with_llm
from src.agent.event import Event
from src.agent.state import AgentState
from src.services.llm_service import LLMService

TIRED_REMINDER_MIN_FOCUS_SEC = 300
IDLE_CHECK_MIN_INTERVAL_SEC = 600


def build_candidate_intents(
    previous_state: AgentState,
    current_state: AgentState,
    event: Event,
) -> list[AgentIntent]:
    """先由规则层生成候选意图，不直接决定具体动作。"""
    if event.type in {"user_text_input", "speech_recognized"}:
        return _plan_user_text_like(current_state, event)
    if event.type == "focus_start_requested":
        if previous_state.focus.active:
            return [
                AgentIntent(
                    type="answer_user",
                    priority=90,
                    reason="focus_already_active",
                    payload={"response_mode": "fixed_text", "text": "当前已经在专注中了。"},
                )
            ]
        return [AgentIntent(type="start_focus", priority=90, reason="focus_start_requested")]
    if event.type == "focus_stop_requested":
        if not previous_state.focus.active:
            return [
                AgentIntent(
                    type="answer_user",
                    priority=90,
                    reason="focus_not_active",
                    payload={"response_mode": "fixed_text", "text": "当前没有正在进行的专注。"},
                )
            ]
        return [AgentIntent(type="stop_focus", priority=90, reason="focus_stop_requested")]
    if event.type == "timer_finished":
        if previous_state.focus.active:
            return [AgentIntent(type="complete_focus", priority=95, reason="timer_finished")]
        return [AgentIntent(type="no_op", reason="timer_finished_ignored")]
    if event.type == "timer_ticked":
        return _plan_timer_tick(current_state, event)
    if event.type == "user_attention_updated":
        return _plan_attention_feedback(current_state, event)
    if event.type in {"user_emotion_updated", "user_fatigue_updated"}:
        return _plan_fatigue_feedback(current_state, event)
    if event.type == "system_triggered":
        return _plan_system_trigger(current_state, event)
    if event.type in {"user_presence_updated", "display_sensor_updated"}:
        return [AgentIntent(type="update_status_feedback", priority=5, reason=event.type)]
    if event.type in {"light_level_updated", "temperature_humidity_updated", "noise_level_updated"}:
        return _plan_environment_feedback(current_state, event)
    if event.type in {"voice_wake_detected", "voice_input_started", "voice_input_stopped"}:
        return [AgentIntent(type="voice_interaction", priority=10, reason=event.type, payload=event.payload)]
    if event.type in {"tts_started", "tts_finished", "voice_volume_changed", "voice_timbre_changed", "voice_speed_changed"}:
        return [AgentIntent(type="display_update", priority=5, reason=event.type, payload=event.payload)]
    return [AgentIntent(type="no_op", reason="unknown_event")]


def plan_intents(
    previous_state: AgentState,
    current_state: AgentState,
    event: Event,
    llm_service: LLMService | None = None,
) -> list[AgentIntent]:
    """生成最终意图列表，并在需要时接入 LLM 辅助判断。"""
    candidates = build_candidate_intents(previous_state, current_state, event)

    if llm_service is not None and _should_use_llm_intent_planner(event, candidates):
        llm_intents = plan_intents_with_llm(
            event=event,
            state=current_state,
            candidate_intents=candidates,
            llm_service=llm_service,
        )
        return guard_intents(
            llm_intents,
            state=current_state,
            event=event,
            fallback_intents=candidates,
        )

    return guard_intents(
        candidates,
        state=current_state,
        event=event,
        fallback_intents=candidates,
    )


def _should_use_llm_intent_planner(event: Event, candidate_intents: list[AgentIntent]) -> bool:
    """只在模糊的用户文本输入场景下启用 LLM 意图辅助。"""
    if not candidate_intents:
        return False
    if event.type not in {"user_text_input", "speech_recognized"}:
        return False

    text = str(event.payload.get("text", "")).strip()
    if not text:
        return False

    semantic = _classify_user_text(text)
    if semantic in {"status_query", "start_focus", "stop_focus"}:
        return False

    return any(intent.requires_llm for intent in candidate_intents)


def _plan_user_text_like(current_state: AgentState, event: Event) -> list[AgentIntent]:
    """先用轻量规则识别文本语义，再决定候选意图。"""
    text = str(event.payload.get("text", "")).strip()
    if not text:
        return [AgentIntent(type="no_op", reason="empty_input")]

    semantic = _classify_user_text(text)
    if semantic == "status_query":
        return [
            AgentIntent(
                type="answer_user",
                priority=100,
                reason="status_query",
                payload={"text": text, "response_mode": "status_summary"},
                requires_llm=False,
            )
        ]
    if semantic == "start_focus":
        return [AgentIntent(type="start_focus", priority=100, reason="text_start_focus", payload={"text": text})]
    if semantic == "stop_focus":
        return [AgentIntent(type="stop_focus", priority=100, reason="text_stop_focus", payload={"text": text})]

    return [
        AgentIntent(
            type="answer_user",
            priority=70,
            reason="user_dialogue",
            payload={
                "text": text,
                "response_mode": "dialogue",
                "focused_mode": current_state.focus.active,
            },
            requires_llm=semantic in {"chat", "open_question"},
        )
    ]


def _plan_timer_tick(current_state: AgentState, event: Event) -> list[AgentIntent]:
    """在专注进行中检查是否满足休息提醒条件。"""
    if not current_state.focus.active:
        return []
    if _should_trigger_rest_reminder(current_state, event.timestamp):
        return [
            AgentIntent(
                type="suggest_rest",
                priority=85,
                reason="rest_reminder",
                payload={"source_event": "timer_ticked"},
            )
        ]
    return []


def _plan_attention_feedback(current_state: AgentState, event: Event) -> list[AgentIntent]:
    """处理注意力更新事件，必要时升级为分心提醒。"""
    attention = str(event.payload.get("attention", current_state.user.attention))
    if attention != "distracted":
        return [AgentIntent(type="update_status_feedback", priority=5, reason="attention_updated")]
    if not current_state.focus.active:
        return [AgentIntent(type="update_status_feedback", priority=5, reason="distracted_but_not_focusing")]
    if _is_in_cooldown(current_state, "distraction_reminder", event.timestamp):
        return [AgentIntent(type="no_op", reason="distraction_in_cooldown")]
    return [
        AgentIntent(
            type="remind_distraction",
            priority=80,
            reason="distraction_reminder",
            payload={"attention": attention, "behavior": event.payload.get("behavior")},
        )
    ]


def _plan_fatigue_feedback(current_state: AgentState, event: Event) -> list[AgentIntent]:
    """处理疲劳和情绪变化，必要时转成休息建议。"""
    fatigue = current_state.user.fatigue_level
    emotion = current_state.user.emotion
    if fatigue not in {"moderate", "high"} and emotion not in {"tired", "stressed"}:
        return [AgentIntent(type="update_status_feedback", priority=5, reason="fatigue_or_emotion_updated")]
    if not current_state.focus.active:
        return [AgentIntent(type="update_status_feedback", priority=5, reason="fatigue_not_in_focus")]
    if current_state.focus.elapsed_sec < TIRED_REMINDER_MIN_FOCUS_SEC:
        return [AgentIntent(type="no_op", reason="fatigue_too_early")]
    if _is_in_cooldown(current_state, "fatigue_warning", event.timestamp):
        return [AgentIntent(type="no_op", reason="fatigue_in_cooldown")]
    return [
        AgentIntent(
            type="suggest_rest",
            priority=82,
            reason="fatigue_warning",
            payload={"fatigue_level": fatigue, "emotion": emotion},
        )
    ]


def _plan_environment_feedback(current_state: AgentState, event: Event) -> list[AgentIntent]:
    """处理环境传感器事件，必要时给出低打扰反馈。"""
    if _is_in_cooldown(current_state, "environment_warning", event.timestamp):
        return [AgentIntent(type="no_op", reason="environment_in_cooldown")]
    level = str(event.payload.get("level") or event.payload.get("temperature_level") or "unknown")
    if level in {"low", "high", "noisy", "dark", "hot", "cold"}:
        return [
            AgentIntent(
                type="adjust_environment_feedback",
                priority=20,
                reason="environment_warning",
                payload={"event_type": event.type, "level": level},
            )
        ]
    return [AgentIntent(type="no_op", reason="environment_normal")]


def _plan_system_trigger(current_state: AgentState, event: Event) -> list[AgentIntent]:
    """处理内部回流事件和自主检查事件。"""
    trigger = str(event.payload.get("trigger", "")).strip()

    if trigger in {"agent_response_completed", "focus_timer_started", "focus_timer_stopped"}:
        return [AgentIntent(type="no_op", reason=trigger or "system_triggered")]
    if trigger == "action_failed":
        action_type = str(event.payload.get("action_type", "unknown_action"))
        reason = str(event.payload.get("reason", "")).strip()
        text = f"动作执行失败：{action_type}。"
        if reason:
            text = f"{text}原因：{reason}"
        return [
            AgentIntent(
                type="answer_user",
                priority=70,
                reason="action_failed",
                payload={"response_mode": "fixed_text", "text": text},
                requires_llm=False,
            )
        ]
    if trigger == "periodic_check":
        return _plan_periodic_check(current_state, event)
    if trigger == "user_idle_check":
        return _plan_user_idle_check(current_state, event)
    if trigger == "focus_health_check":
        return _plan_focus_health_check(current_state, event)
    if trigger == "environment_check":
        return _plan_environment_check(current_state, event)
    return [AgentIntent(type="no_op", reason="unknown_system_trigger")]


def _plan_periodic_check(current_state: AgentState, event: Event) -> list[AgentIntent]:
    """周期检查先判断专注健康，再判断环境是否需要提醒。"""
    if current_state.user.presence == "away":
        return [AgentIntent(type="no_op", reason="periodic_check_user_away")]
    if current_state.focus.active:
        health_intents = _plan_focus_health_check(current_state, event)
        if not _only_no_op(health_intents):
            return health_intents
        environment_intents = _plan_environment_check(current_state, event)
        if not _only_no_op(environment_intents):
            return environment_intents
    return [AgentIntent(type="no_op", reason="periodic_check_idle")]


def _plan_user_idle_check(current_state: AgentState, event: Event) -> list[AgentIntent]:
    """长时间无输入时，仅生成低打扰的显示型提示。"""
    if current_state.user.presence == "away":
        return [AgentIntent(type="no_op", reason="idle_check_user_away")]
    if current_state.focus.active:
        return [AgentIntent(type="no_op", reason="idle_check_focus_active")]
    if current_state.interaction.in_conversation:
        return [AgentIntent(type="no_op", reason="idle_check_in_conversation")]

    last_user_time = current_state.interaction.last_user_time
    if last_user_time is None:
        return [AgentIntent(type="no_op", reason="idle_check_no_context")]
    if event.timestamp - int(last_user_time) < IDLE_CHECK_MIN_INTERVAL_SEC:
        return [AgentIntent(type="no_op", reason="idle_check_too_early")]
    if _is_in_cooldown(current_state, "idle_check", event.timestamp):
        return [AgentIntent(type="no_op", reason="idle_check_in_cooldown")]

    return [
        AgentIntent(
            type="answer_user",
            priority=15,
            reason="idle_check",
            payload={
                "response_mode": "fixed_text",
                "text": "如果需要，我可以帮助你继续当前任务或开始一轮专注。",
                "display_only": True,
            },
            requires_llm=False,
        )
    ]


def _plan_focus_health_check(current_state: AgentState, event: Event) -> list[AgentIntent]:
    """专注健康检查优先看疲劳，再看是否分心。"""
    if current_state.user.presence == "away":
        return [AgentIntent(type="no_op", reason="focus_health_user_away")]
    if not current_state.focus.active:
        return [AgentIntent(type="no_op", reason="focus_health_not_active")]

    if current_state.user.fatigue_level in {"moderate", "high"} or current_state.user.emotion in {"tired", "stressed"}:
        if current_state.focus.elapsed_sec < TIRED_REMINDER_MIN_FOCUS_SEC:
            return [AgentIntent(type="no_op", reason="focus_health_too_early")]
        if _is_in_cooldown(current_state, "rest_reminder", event.timestamp):
            return [AgentIntent(type="no_op", reason="focus_health_rest_in_cooldown")]
        return [
            AgentIntent(
                type="suggest_rest",
                priority=84,
                reason="rest_reminder",
                payload={"source_event": "focus_health_check"},
            )
        ]

    if current_state.user.attention == "distracted":
        if _is_in_cooldown(current_state, "distraction_reminder", event.timestamp):
            return [AgentIntent(type="no_op", reason="focus_health_distraction_in_cooldown")]
        return [
            AgentIntent(
                type="remind_distraction",
                priority=78,
                reason="distraction_reminder",
                payload={"source_event": "focus_health_check", "behavior": current_state.user.behavior},
            )
        ]

    return [AgentIntent(type="no_op", reason="focus_health_normal")]


def _plan_environment_check(current_state: AgentState, event: Event) -> list[AgentIntent]:
    """直接从当前状态读取环境结论，决定是否需要环境提醒。"""
    if _is_in_cooldown(current_state, "environment_warning", event.timestamp):
        return [AgentIntent(type="no_op", reason="environment_check_in_cooldown")]

    level = _current_environment_level(current_state)
    if level is None:
        return [AgentIntent(type="no_op", reason="environment_check_normal")]

    return [
        AgentIntent(
            type="adjust_environment_feedback",
            priority=18,
            reason="environment_warning",
            payload={"event_type": "environment_check", "level": level},
        )
    ]


def _current_environment_level(state: AgentState) -> str | None:
    """将多个环境字段收敛为统一的异常级别。"""
    if state.environment.light_level in {"low", "dark"}:
        return str(state.environment.light_level)
    if state.environment.noise_level in {"high", "noisy"}:
        return str(state.environment.noise_level)
    if state.environment.temperature_level in {"high", "hot", "low", "cold"}:
        return str(state.environment.temperature_level)
    if state.environment.humidity_level in {"high", "low"}:
        return str(state.environment.humidity_level)
    return None


def _should_trigger_rest_reminder(state: AgentState, now_ts: int) -> bool:
    """统一判断当前是否满足休息提醒条件。"""
    if not state.focus.active or state.focus.start_ts is None:
        return False
    if state.user.presence == "away":
        return False
    if state.user.attention != "focused":
        return False
    if state.user.fatigue_level not in {"moderate", "high"} and state.user.emotion != "tired":
        return False
    if state.focus.elapsed_sec < TIRED_REMINDER_MIN_FOCUS_SEC:
        return False
    return not _is_in_cooldown(state, "rest_reminder", now_ts)


def _is_in_cooldown(state: AgentState, reason: str, now_ts: int) -> bool:
    """判断某类提醒是否仍处于冷却期。"""
    last_ts = state.cooldown.reminder_last_ts.get(reason)
    if last_ts is None:
        return False
    cooldown_sec = _cooldown_seconds(reason)
    return now_ts - int(last_ts) < cooldown_sec


def _cooldown_seconds(reason: str) -> int:
    """返回不同提醒原因对应的冷却时长。"""
    if reason == "distraction_reminder":
        return 180
    if reason == "idle_check":
        return 600
    return 300


def _only_no_op(intents: list[AgentIntent]) -> bool:
    """判断一组意图是否只有 no_op。"""
    return not intents or all(intent.type == "no_op" for intent in intents)


def _classify_user_text(text: str) -> str:
    """用轻量规则做语义分类，避免所有输入都依赖 LLM。"""
    lowered = text.strip().lower()
    if any(keyword in lowered for keyword in ("现在状态如何", "当前状态", "state", "status", "情绪", "疲劳")):
        return "status_query"
    if any(keyword in lowered for keyword in ("开始专注", "开始番茄", "继续学", "继续学习", "start focus")):
        return "start_focus"
    if any(keyword in lowered for keyword in ("结束专注", "停止专注", "stop focus")):
        return "stop_focus"
    if any(keyword in lowered for keyword in ("为什么", "怎么", "如何", "帮我", "?")):
        return "open_question"
    return "chat"
