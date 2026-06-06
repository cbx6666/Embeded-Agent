from __future__ import annotations

"""Agent 策略配置。

它是什么：
集中保存事件路由、低频调度、自治检查门控、趋势聚合、冷却规则、动作边界、
默认文案、检索排序权重和 RuntimeHistory 窗口大小。

它不是什么：
它不是协议定义，不替代 EventType / IntentType / ActionType；也不是 Prompt 管理器，
本轮不接管 LLM prompt 或 mock LLM 关键词。

为什么存在：
这些值会影响智能行为策略，但不是稳定协议。集中到这里后，后续调整策略时不需要在
Router、Scheduler、Guard、ActionRealizer、PersonalContextBuilder 和 RuntimeHistoryService
之间来回找。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EventRoutingPolicyConfig:
    """EventPriorityRouter 使用的事件分类数据。

    priority 表示处理时效，handling 表示处理机制。两者分开配置，避免把
    “立即处理”误写成“必须调用 LLM”。
    """

    open_semantic_events: frozenset[str] = field(
        default_factory=lambda: frozenset({"user_text_input", "speech_recognized"})
    )
    structured_decision_events: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"focus_start_requested", "focus_stop_requested", "timer_finished"}
        )
    )
    low_frequency_triggers: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"focus_health_check", "environment_check", "periodic_check", "user_idle_check"}
        )
    )
    user_state_events: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "user_presence_updated",
                "user_attention_updated",
                "user_emotion_updated",
                "user_fatigue_updated",
                "user_posture_updated",
                "user_activity_updated",
            }
        )
    )
    telemetry_events: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "timer_ticked",
                "display_sensor_updated",
                "voice_wake_detected",
                "voice_input_started",
                "voice_input_stopped",
                "tts_started",
                "tts_finished",
                "light_level_updated",
                "temperature_humidity_updated",
                "noise_level_updated",
            }
        )
    )
    p4_handling_by_event: dict[str, str] = field(
        default_factory=lambda: {
            "user_switched": "profile_handler",
            "user_profile_updated": "profile_handler",
            "user_preference_update_requested": "settings_handler",
            "break_suggestion_accepted": "feedback_signal",
            "break_suggestion_rejected": "feedback_signal",
            "voice_volume_changed": "settings_handler",
            "voice_timbre_changed": "settings_handler",
            "voice_speed_changed": "settings_handler",
        }
    )


@dataclass(frozen=True)
class SignalAggregationPolicyConfig:
    """RuntimeHistory rolling summary 的事件字段映射。"""

    fields_by_event: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = field(
        default_factory=lambda: {
            "user_presence_updated": (("presence", ("presence",)),),
            "user_attention_updated": (
                ("attention", ("attention",)),
                ("behavior", ("behavior",)),
            ),
            "user_emotion_updated": (("emotion", ("emotion",)),),
            "user_fatigue_updated": (("fatigue", ("fatigue_level",)),),
            "user_posture_updated": (("posture", ("posture",)),),
            "user_activity_updated": (("activity", ("activity",)),),
            "light_level_updated": (("light", ("level", "light_lux")),),
            "temperature_humidity_updated": (
                ("temperature", ("temperature_level", "temperature_c")),
                ("humidity", ("humidity_level", "humidity_pct")),
            ),
            "noise_level_updated": (("noise", ("level", "noise_db")),),
        }
    )


@dataclass(frozen=True)
class AutonomousCheckPolicyConfig:
    """P1 自主检查进入 Rule/LLM 前的确定性门控配置。"""

    trusted_source: str = "agent_autonomy"
    focus_min_elapsed_sec: int = 10 * 60
    idle_min_duration_sec: int = 15 * 60
    sustained_min_samples: int = 3
    sustained_min_ratio: float = 0.6
    sustained_min_consecutive: int = 2
    minimum_average_confidence: float = 0.5
    periodic_check_enabled: bool = False
    present_values: frozenset[str] = field(
        default_factory=lambda: frozenset({"present"})
    )
    working_modes: frozenset[str] = field(
        default_factory=lambda: frozenset({"focus", "working"})
    )
    working_activities: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"working", "studying", "reading", "typing", "coding"}
        )
    )
    abnormal_values_by_signal: dict[str, frozenset[str]] = field(
        default_factory=lambda: {
            "fatigue": frozenset({"moderate", "high", "severe", "tired"}),
            "attention": frozenset({"distracted", "inattentive"}),
            "posture": frozenset({"slouching", "poor", "bad", "unhealthy"}),
            "light": frozenset({"low", "dark", "dim", "too_bright", "bright"}),
            "temperature": frozenset({"low", "high", "cold", "hot"}),
            "humidity": frozenset({"low", "high", "dry", "humid"}),
            "noise": frozenset({"high", "noisy", "loud"}),
        }
    )
    reminder_reasons_by_signal: dict[str, str] = field(
        default_factory=lambda: {
            "fatigue": "rest_reminder",
            "posture": "rest_reminder",
            "attention": "distraction_reminder",
            "light": "environment_warning",
            "temperature": "environment_warning",
            "humidity": "environment_warning",
            "noise": "environment_warning",
            "idle": "distraction_reminder",
        }
    )
    check_cooldown_sec: dict[str, int] = field(
        default_factory=lambda: {
            "focus_health_check": 5 * 60,
            "environment_check": 5 * 60,
            "user_idle_check": 10 * 60,
            "periodic_check": 30 * 60,
        }
    )


@dataclass(frozen=True)
class AutonomousScheduleConfig:
    """系统时间驱动的 P1 检查调度配置。"""

    event_source: str = "agent_autonomy"
    poll_interval_sec: float = 1.0
    emit_immediately_on_start: bool = False
    intervals_sec: dict[str, int] = field(
        default_factory=lambda: {
            "focus_health_check": 5 * 60,
            "environment_check": 5 * 60,
            "user_idle_check": 5 * 60,
        }
    )
    disabled_triggers: frozenset[str] = field(
        default_factory=lambda: frozenset({"periodic_check"})
    )


@dataclass(frozen=True)
class MemoryGatePolicyConfig:
    """长期记忆前置 Gate 使用的事件和文本信号数据。"""

    skipped_event_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "voice_wake_detected",
                "voice_input_started",
                "voice_input_stopped",
                "tts_started",
                "tts_finished",
                "voice_volume_changed",
                "voice_timbre_changed",
                "voice_speed_changed",
                "light_level_updated",
                "temperature_humidity_updated",
                "noise_level_updated",
                "display_sensor_updated",
                "timer_ticked",
                "focus_start_requested",
                "focus_stop_requested",
                "timer_finished",
            }
        )
    )
    internal_triggers: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "focus_timer_started",
                "focus_timer_stopped",
                "agent_response_completed",
                "action_result",
                "action_failed",
                "device_action_completed",
                "timer_internal_tick",
            }
        )
    )
    user_semantic_event_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"user_text_input", "speech_recognized"})
    )
    feedback_event_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"break_suggestion_accepted", "break_suggestion_rejected"}
        )
    )
    long_term_markers: tuple[str, ...] = (
        "以后",
        "以后默认",
        "记住",
        "帮我记住",
        "我喜欢",
        "我不喜欢",
        "我更喜欢",
        "我讨厌",
        "我习惯",
        "我通常",
        "我经常",
        "我希望你以后",
        "从现在开始",
        "默认",
        "不要再",
        "每次",
        "remember",
        "from now on",
        "i prefer",
        "i like",
        "i dislike",
        "i usually",
        "always",
        "by default",
    )
    trivial_texts: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "你好",
                "您好",
                "嗨",
                "hello",
                "hi",
                "hey",
                "嗯",
                "嗯嗯",
                "好的",
                "好",
                "可以",
                "知道了",
                "谢谢",
                "感谢",
                "开始",
                "停止",
                "现在几点",
                "几点了",
                "what time is it",
                "thanks",
                "thank you",
                "ok",
                "okay",
                "yes",
                "no",
            }
        )
    )
    status_query_patterns: tuple[str, ...] = (
        r"^(现在)?几点(了)?$",
        r"^(现在)?什么时间(了)?$",
        r"^what time is it$",
        r"^(当前|现在)?状态(怎么样|如何|是什么)?$",
    )
    no_long_term_action_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "speak",
                "display",
                "render_pet_expression",
                "start_voice_capture",
                "stop_voice_capture",
                "set_tts_voice",
                "set_tts_volume",
                "set_tts_speed",
            }
        )
    )


@dataclass(frozen=True)
class DedicatedEventPolicyConfig:
    """P4 profile/settings 事件到显式 UserProfile 字段的映射。"""

    voice_preference_fields: dict[str, tuple[str, tuple[str, ...]]] = field(
        default_factory=lambda: {
            "voice_volume_changed": ("tts_volume", ("volume", "value")),
            "voice_timbre_changed": ("tts_voice", ("voice", "voice_id", "timbre", "value")),
            "voice_speed_changed": ("tts_speed", ("speed", "value")),
        }
    )


@dataclass(frozen=True)
class GuardPolicyConfig:
    """DeterministicGuard 的硬边界策略值。"""

    reminder_cooldown_sec: int = 300
    interruptive_intents: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "suggest_rest",
                "remind_distraction",
                "adjust_environment_feedback",
                "voice_interaction",
            }
        )
    )
    cooldown_reasons: dict[str, str] = field(
        default_factory=lambda: {
            "suggest_rest": "rest_reminder",
            "remind_distraction": "distraction_reminder",
            "adjust_environment_feedback": "environment_warning",
        }
    )
    user_initiated_event_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"user_text_input", "speech_recognized"})
    )
    block_interruptive_when_presence: str = "away"
    allow_on_invalid_cooldown_timestamp: bool = True


@dataclass(frozen=True)
class DecisionPolicyConfig:
    """DecisionPipeline 对 system_triggered 的轻量入口策略。"""

    allowed_autonomous_triggers: frozenset[str] = field(
        default_factory=lambda: EventRoutingPolicyConfig().low_frequency_triggers
    )
    internal_system_triggers: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "agent_response_completed",
                "focus_timer_started",
                "focus_timer_stopped",
                "action_result",
                "device_action_completed",
                "timer_internal_tick",
            }
        )
    )
    action_result_source: str = "agent_action_result"
    ignored_system_trigger_reason: str = "internal system trigger ignored by decision policy"
    llm_skipped_event_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "voice_wake_detected",
                "voice_input_started",
                "voice_input_stopped",
                "tts_started",
                "tts_finished",
                "light_level_updated",
                "temperature_humidity_updated",
                "noise_level_updated",
                # 高频感知 Event 只更新 state，不走四角色 LLM（否则每条约 4 次 API，阻塞视觉/行为线程）
                "user_fatigue_updated",
                "user_emotion_updated",
                "user_presence_updated",
                "user_attention_updated",
                "user_posture_updated",
                "user_activity_updated",
            }
        )
    )
    # fast：统一规划 1 次，必要时追加 SafetyCritic；full：四角色串行。
    # 库默认 full 以兼容旧调用；main 默认 fast。
    llm_mode: str = "full"

    def is_allowed_trigger(self, trigger: str, source: str) -> bool:
        """Return True if this system_triggered event should be processed."""
        if source == self.action_result_source:
            return False
        if trigger in self.internal_system_triggers:
            return False
        if trigger not in self.allowed_autonomous_triggers:
            return False
        return True


@dataclass(frozen=True)
class LLMRolePolicyConfig:
    """fast/adaptive 模式下的条件角色调用策略。"""

    safety_review_risk_levels: frozenset[str] = field(
        default_factory=lambda: frozenset({"medium", "high"})
    )
    safety_review_min_intents: int = 2
    safety_review_intent_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"voice_interaction"})
    )
    deterministic_fast_fallback_text: str = "我在，请再说一遍。"


@dataclass(frozen=True)
class RuleIntentPolicyConfig:
    """RuleIntentBuilder 的结构化事件映射数据。"""

    intent_by_event: dict[str, str] = field(
        default_factory=lambda: {
            "focus_start_requested": "start_focus",
            "focus_stop_requested": "stop_focus",
            "timer_finished": "complete_focus",
        }
    )
    action_priority: int = 80
    no_op_priority: int = 10


@dataclass(frozen=True)
class ContextPolicyConfig:
    """PersonalContextBuilder 的上下文裁剪和长期记忆分桶策略。"""

    max_recent_messages: int = 6
    max_recent_events: int = 8
    max_recent_actions: int = 8
    uncertain_confidence_threshold: float = 0.55
    max_memory_items_per_bucket: int = 6
    max_relevant_memories: int = 8
    noisy_runtime_event_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "timer_ticked",
                "agent_response_completed",
                "focus_timer_started",
                "focus_timer_stopped",
                "light_level_updated",
                "temperature_humidity_updated",
                "noise_level_updated",
            }
        )
    )
    noisy_runtime_trigger_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"agent_response_completed", "focus_timer_started", "focus_timer_stopped", "timer_ticked"}
        )
    )


@dataclass(frozen=True)
class ActionPolicyConfig:
    """ActionRealizer 的数值边界和默认时长策略。"""

    default_focus_duration_sec: int = 1500
    default_continue_focus_sec: int = 1200
    min_duration_sec: int = 1
    max_duration_sec: int = 24 * 3600
    min_tts_volume: int = 0
    max_tts_volume: int = 100


@dataclass(frozen=True)
class CopyPolicyConfig:
    """ActionRealizer 的默认文案。

    ResponseWriter 或 intent payload 给出文本时仍优先使用上游文本；这些值只作为确定性
    fallback。
    """

    fallback_answer_text: str = "我在。"
    focus_started_template: str = "已开始专注 {minutes} 分钟。"
    continue_focus_template: str = "继续专注 {minutes} 分钟。"
    focus_stopped_text: str = "已结束专注。"
    focus_complete_text: str = "这轮专注完成了。"
    rest_reminder_text: str = "你已经专注一会儿了，要不要稍微休息一下？"
    distraction_reminder_text: str = "我们慢慢把注意力拉回当前任务。"
    environment_warning_text: str = "环境可能有点不舒服，我们可以稍微调整一下。"
    display_updated_text: str = "已更新。"
    reduce_reminder_frequency_text: str = "好的，我会尽量少打扰你。"
    status_focus_active_template: str = "专注正在进行，剩余 {remaining_sec} 秒。"
    status_user_state_template: str = (
        "当前状态：presence={presence}，attention={attention}，fatigue={fatigue_level}。"
    )


@dataclass(frozen=True)
class RuntimeHistoryPolicyConfig:
    """RuntimeHistoryService 的短期窗口大小。"""

    max_recent_events: int = 20
    max_recent_messages: int = 20
    max_recent_actions: int = 20
    max_reminder_records: int = 50
    max_attention_records: int = 120
    max_environment_records: int = 120
    max_focus_sessions: int = 10
    max_emotion_samples: int = 120
    max_emotion_summaries: int = 60
    max_signal_recent_values: int = 50
    emotion_summary_window_sec: int = 60


def _default_source_weights() -> dict[str, float]:
    return {
        "UserProfile": 100.0,
        "LongTermMemory": 50.0,
        "RuntimeHistory": 25.0,
    }


def _default_event_type_weights() -> dict[str, dict[str, float]]:
    dialogue_weights = {
        "explicit_user_preference": 14.0,
        "interaction_style": 12.0,
        "behavior_preference": 10.0,
        "recent_message": 5.0,
    }
    focus_weights = {
        "active_constraint": 14.0,
        "behavior_pattern": 10.0,
        "behavior_preference": 8.0,
        "recent_action": 4.0,
    }
    user_state_weights = {
        "behavior_pattern": 12.0,
        "active_constraint": 10.0,
        "recent_event": 5.0,
    }
    system_weights = {
        "active_constraint": 12.0,
        "behavior_pattern": 8.0,
        "recent_action": 5.0,
    }
    return {
        "user_text_input": dict(dialogue_weights),
        "speech_recognized": dict(dialogue_weights),
        "focus_start_requested": dict(focus_weights),
        "focus_stop_requested": dict(focus_weights),
        "timer_ticked": dict(focus_weights),
        "timer_finished": dict(focus_weights),
        "user_presence_updated": dict(user_state_weights),
        "user_attention_updated": dict(user_state_weights),
        "user_emotion_updated": dict(user_state_weights),
        "user_fatigue_updated": dict(user_state_weights),
        "system_triggered": dict(system_weights),
    }


@dataclass(frozen=True)
class RetrievalPolicyConfig:
    """PersonalContext.retrieve_relevant 的检索排序策略。"""

    source_weights: dict[str, float] = field(default_factory=_default_source_weights)
    event_type_weights: dict[str, dict[str, float]] = field(default_factory=_default_event_type_weights)
    confidence_weight: float = 10.0
    evidence_weight: float = 0.5
    conflict_penalty: float = 30.0
    content_term_weight: float = 3.0
    tag_term_weight: float = 2.0
    max_evidence_bonus: float = 4.0
    memory_priority_confidence_weight: float = 20.0
    memory_priority_evidence_weight: float = 1.5
    memory_priority_max_evidence: int = 5
