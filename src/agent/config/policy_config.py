from __future__ import annotations

"""Agent 策略配置。

它是什么：
集中保存冷却规则、上下文裁剪、动作边界、默认文案和 RuntimeHistory 窗口大小。

它不是什么：
它不是协议定义，不替代 EventType / IntentType / ActionType；也不是 Prompt 管理器，
本轮不接管 LLM prompt 或 mock LLM 关键词。

为什么存在：
这些值会影响智能行为策略，但不是稳定协议。集中到这里后，后续调整策略时不需要在
Guard、ActionRealizer、PersonalContextBuilder 和 RuntimeHistoryService 之间来回找。
"""

from dataclasses import dataclass, field


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
        default_factory=lambda: frozenset(
            {
                "focus_health_check",
                "periodic_check",
                "environment_check",
                "user_idle_check",
            }
        )
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


@dataclass(frozen=True)
class ContextPolicyConfig:
    """PersonalContextBuilder 的上下文裁剪和长期记忆分桶策略。"""

    max_recent_messages: int = 6
    max_recent_events: int = 8
    max_recent_actions: int = 8
    uncertain_confidence_threshold: float = 0.55
    max_memory_items_per_bucket: int = 6


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
    emotion_summary_window_sec: int = 60
