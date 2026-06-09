from __future__ import annotations

"""Agent 策略配置。

- ``LLMRoutingPolicy``：事件分流与 LLM prompt role。
- ``SchedulePolicy`` / ``ScheduledTaskPolicy``：多任务周期调度（周期与优先级）。
- ``WellnessCareCheckPolicy`` / ``BehaviorDistractionCheckPolicy`` /
  ``EnvironmentCareCheckPolicy``：各自检的候选/触发阈值。
- ``SensorReportPolicy``：sensor_status_report 播报边界。
- ``GuardPolicy``：提醒类 Intent 的防刷屏与打断保护。
- ``ActionPolicy``：动作数值边界与默认文案。
- ``MemoryPolicy``：异步记忆抽取与检索。

协议（EventType / ActionType）不在这里定义。
"""

from dataclasses import dataclass, field


# 任务优先级（数字越小优先级越高），用于调度防打断与同一时刻单 LLM 任务约束。
# 调度优先级：分心 > 疲劳/情绪关怀 > 环境关怀 > 环境详细强制播报。
SPEECH_PRIORITY = 0
BEHAVIOR_DISTRACTION_PRIORITY = 1
WELLNESS_CARE_PRIORITY = 2
ENVIRONMENT_CARE_PRIORITY = 3
SENSOR_REPORT_PRIORITY = 4
STATE_ONLY_PRIORITY = 5


@dataclass(frozen=True)
class LLMRoutingPolicy:
    """事件分流策略：决定一条事件如何被处理。"""

    speech_event: str = "speech_recognized"
    behavior_distraction_trigger: str = "behavior_distraction_check"
    wellness_care_trigger: str = "wellness_care_check"
    environment_care_trigger: str = "environment_care_check"
    sensor_trigger: str = "sensor_status_report"
    trusted_source: str = "agent_autonomy"
    rule_events: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"focus_start_requested", "focus_stop_requested", "timer_finished"}
        )
    )
    speech_prompt: str = "speech_recognized"
    behavior_distraction_prompt: str = "behavior_distraction_check"
    wellness_prompt: str = "wellness_care_check"
    environment_care_prompt: str = "environment_care_check"
    # 专注结束关怀（timer_finished -> complete_focus）走个性化轮换关怀的 prompt。
    focus_complete_prompt: str = "focus_complete_care"
    # 面向用户的回复类决策温度：偏高让 reply 文案更自然多变、不重复。
    reply_temperature: float = 0.85


@dataclass(frozen=True)
class ScheduledTaskPolicy:
    """单个周期任务定义：名称 / trigger / 周期 / 优先级。"""

    name: str
    trigger: str
    interval_sec: int
    priority: int
    enabled: bool = True


@dataclass(frozen=True)
class SchedulePolicy:
    """系统时间驱动的多任务周期调度。

    每个任务有独立的 ``interval_sec`` 与优先级；调度器按 ``remaining_sec`` 倒计时，
    高优先级任务运行时低优先级任务的倒计时冻结（不被重置），完成后从完整周期重启。
    """

    event_source: str = "agent_autonomy"
    poll_interval_sec: float = 1.0
    emit_immediately_on_start: bool = False
    tasks: tuple[ScheduledTaskPolicy, ...] = (
        ScheduledTaskPolicy(
            name="behavior_distraction_check",
            trigger="behavior_distraction_check",
            interval_sec=20,
            priority=BEHAVIOR_DISTRACTION_PRIORITY,
        ),
        ScheduledTaskPolicy(
            name="wellness_care_check",
            trigger="wellness_care_check",
            interval_sec=30,
            priority=WELLNESS_CARE_PRIORITY,
        ),
        ScheduledTaskPolicy(
            name="environment_care_check",
            trigger="environment_care_check",
            interval_sec=60,
            priority=ENVIRONMENT_CARE_PRIORITY,
        ),
        ScheduledTaskPolicy(
            name="sensor_status_report",
            trigger="sensor_status_report",
            interval_sec=300,
            priority=SENSOR_REPORT_PRIORITY,
        ),
    )

    def interval_for(self, trigger: str) -> int | None:
        """按 trigger 查任务周期（秒）；未配置时返回 None。"""

        for task in self.tasks:
            if task.trigger == trigger:
                return max(1, int(task.interval_sec))
        return None


@dataclass(frozen=True)
class BehaviorDistractionCheckPolicy:
    """behavior_distraction_check 的严格分心判定阈值。

    判定基于「最近窗口的占比 + 最近是否仍在玩」，并**硬性要求 YOLO 真的检出手机**，
    避免「行为分类误判为 phone_use 但画面里并没有手机」造成的误报：

    - 窗口内 phone_use 记录数 ≥ ``min_phone_use_events``；
    - 窗口内 **YOLO 直接检出手机**的记录数 ≥ ``min_yolo_phone_events``（默认 1，必须有手机）；
    - phone_use 占窗口内全部行为记录的比例 ≥ ``min_phone_use_ratio``；
    - 最近一次 phone_use 记录距检查时刻 ≤ ``recent_active_sec``（确认「还在玩」）；
    - ``require_yolo_phone_on_latest=True`` 时，最近一条 phone_use 记录也必须带 YOLO 手机框
      （确认「此刻仍真的拿着手机」）。

    是否真的提醒仍由 prompt + Guard 决定。检查周期见 ``SchedulePolicy.tasks``（20s）。
    """

    window_sec: int = 30
    min_confidence: float = 0.5
    min_phone_use_events: int = 1
    # 硬性要求：窗口内必须至少有一帧 YOLO 真正检出手机，否则不判定为玩手机。
    min_yolo_phone_events: int = 1
    min_phone_use_ratio: float = 0.15
    recent_active_sec: int = 15
    # 硬性要求：最近一条 phone_use 记录必须带 YOLO 手机框，确认此刻仍真的拿着手机。
    require_yolo_phone_on_latest: bool = True
    require_presence: str = "present"


@dataclass(frozen=True)
class WellnessCareCheckPolicy:
    """wellness_care_check（疲劳 / 负面情绪 / 姿态）的窗口化触发阈值。

    只负责疲劳、负面情绪、姿态不佳；不含环境光照 / 温度 / 湿度 / 噪声。
    疲劳、情绪、姿态三者是 OR 关系：任一成立即触发关怀。Python 用这些阈值在
    ``build_wellness_care_summary`` 里算出 ``should_care`` 与 selected intent；
    LLM 只负责文案，不能把强触发改成 no_op，也不能改成环境提醒。
    """

    # 统计最近窗口（秒），用于 ratio / streak 计算。
    recent_window_sec: int = 60

    # 疲劳触发阈值。
    fatigue_sustained_high_sec: int = 20
    fatigue_sustained_moderate_or_high_sec: int = 45
    fatigue_high_ratio_threshold: float = 0.35
    fatigue_peak_confidence_threshold: float = 0.85
    fatigue_high_levels: frozenset[str] = field(default_factory=lambda: frozenset({"high", "severe", "exhausted"}))
    fatigue_moderate_levels: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"moderate", "fatigued", "weary", "sleepy", "drowsy", "high", "severe", "exhausted"}
        )
    )

    # 负面情绪触发阈值。
    emotion_negative_streak_sec: int = 20
    emotion_negative_ratio_threshold: float = 0.4
    emotion_negative_labels: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "sad",
                "angry",
                "anxious",
                "fear",
                "fearful",
                "disgust",
                "frustrated",
                "stressed",
                "worried",
                "nervous",
                "irritated",
                "upset",
                "unhappy",
                "depressed",
                "panic",
                "down",
                "low",
            }
        )
    )
    emotion_strong_labels: frozenset[str] = field(
        default_factory=lambda: frozenset({"sad", "angry", "anxious", "fear", "fearful", "disgust"})
    )

    # 姿态触发阈值：仍要求「当前确实是坏姿态」，略收紧以减少 leaning 误报刷屏。
    posture_sustained_bad_sec: int = 45
    posture_bad_ratio_threshold: float = 0.55
    # 当前姿态也必须在 bad 集合内，才允许触发（避免窗口里偶发坏姿态误报）。
    posture_require_current_bad: bool = True
    posture_bad_levels: frozenset[str] = field(
        default_factory=lambda: frozenset({"slouching", "poor", "bad", "unhealthy", "lying", "leaning"})
    )

    require_presence: str = "present"


@dataclass(frozen=True)
class EnvironmentCareCheckPolicy:
    """environment_care_check（每 60s 环境关怀）阈值。

    只看环境（光照 / 温度 / 湿度 / 噪声）；是否播由 LLM 判断，可 no_op。
    只能产出 environment_warning，不能生成 rest_reminder / emotion_reminder / posture_reminder。
    """

    require_presence: str = "present"


@dataclass(frozen=True)
class SensorReportPolicy:
    """sensor_status_report 的确定性播报边界（周期与优先级见 ``SchedulePolicy.tasks``）。"""

    block_when_away: bool = True


@dataclass(frozen=True)
class GuardPolicy:
    """提醒类动作的确定性安全边界。"""

    # 默认冷却（未在 cooldown_by_reason 中列出的 reason 用它兜底）。
    reminder_cooldown_sec: int = 60
    # 按 reason 区分的冷却时间（秒）。
    cooldown_by_reason: dict[str, int] = field(
        default_factory=lambda: {
            "distraction_reminder": 60,
            "rest_reminder": 60,
            "joke_reminder": 120,
            "emotion_reminder": 60,
            "posture_reminder": 120,
            "environment_warning": 90,
            "status_report": 300,
        }
    )
    interruptive_intents: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "suggest_rest",
                "offer_emotion_care",
                "remind_distraction",
                "adjust_environment_feedback",
                "suggest_media",
            }
        )
    )
    cooldown_reasons: dict[str, str] = field(
        default_factory=lambda: {
            "suggest_rest": "rest_reminder",
            "offer_emotion_care": "emotion_reminder",
            "remind_distraction": "distraction_reminder",
            "adjust_environment_feedback": "environment_warning",
            "suggest_media": "media_suggestion",
        }
    )
    block_interruptive_when_presence: str = "away"
    block_interruptive_when_speaking: bool = False
    # 这些意图在 TTS 播放中仍允许入队（由语音适配器串行播放，不静默丢弃提醒）。
    # 分心提醒 + 疲劳/情绪关怀（wellness）都不应因 speaking 而被丢弃。
    speaking_exempt_intents: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"remind_distraction", "suggest_rest", "offer_emotion_care"}
        )
    )
    # suggest_media 由 wellness 计数策略控制询问频率，不走时间冷却。
    cooldown_exempt_intents: frozenset[str] = field(
        default_factory=lambda: frozenset({"suggest_media"})
    )


@dataclass(frozen=True)
class MediaPolicy:
    """本地音乐/相声陪伴：曲库根路径。"""

    music_root: str = "data/music"


@dataclass(frozen=True)
class ActionPolicy:
    """动作落地时的数值边界。

    ``*_text`` 字段仅用于 Realizer 在 LLM 已给出 intent 且携带 reply 时的落地；
    不得作为 LLM 失败时的 fallback 话术。
    """

    default_focus_duration_sec: int = 1500
    min_duration_sec: int = 60
    max_duration_sec: int = 7200
    min_tts_volume: int = 0
    max_tts_volume: int = 100

    focus_started_template: str = "好的，开始专注 {minutes} 分钟。"
    focus_stopped_text: str = "已经结束这次专注啦。"
    focus_complete_text: str = "这轮专注完成了，先歇一会儿吧。"
    rest_reminder_text: str = "我感觉你有点累了，要不要稍微休息一下？"
    emotion_care_text: str = "我感觉你现在情绪有点低，可以先慢慢呼吸一下。"
    posture_reminder_text: str = "坐姿好像有点塌了，挺直一下背会舒服点。"
    distraction_reminder_text: str = "好像有点分心了，我们一起把注意力收回来吧。"
    environment_warning_text: str = "环境好像不太舒适，我们可以稍微调整一下。"
    media_suggestion_text: str = "要不要听点音乐放松一下？"
    joke_care_text: str = "给你讲个笑话换换脑子吧，笑一笑会轻松点。"
    media_no_track_text: str = "本地还没有可播放的音频，你可以先把音乐放到 data/music 目录。"
    media_play_ack_text: str = "好，这就给你放。"


@dataclass(frozen=True)
class MemoryPolicy:
    """异步 LLM 记忆抽取的写入门槛、容量与检索配置。

    记忆抽取由 ``MemoryExtractor`` 的单次 LLM 调用完成（后台线程）；这里只配置：
    - 抽取前的廉价预过滤（跳过空串 / 纯寒暄，省一次 LLM 调用）；
    - 每用户记忆容量上限与默认检索数量；
    - 各 context_type 下的记忆类型权重，用于轻量相关性打分。
    """

    store_path: str = "data/memory/user_memory.json"
    async_write: bool = True
    max_memories_per_user: int = 100
    retrieve_top_k: int = 8
    min_extract_chars: int = 2
    # 一次抽取的最低置信度门槛，低于此值的记忆不落库。
    min_keep_confidence: float = 0.45

    # 各 context_type 下记忆类型的相关性权重；未列出的类型用 default。
    # 只为真实主链路 context_type 配置：speech / wellness_care /
    # behavior_distraction / environment_care。``sensor_status_report`` 不检索 Memory，故不配置。
    type_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "speech": {
                "preference": 3.0,
                "interaction_style": 3.0,
                "work_style": 2.5,
                "hobby": 2.0,
                "care_strategy": 2.0,
                "dislike": 2.0,
                "fact": 2.0,
                "habit": 1.0,
                "emotion_pattern": 1.0,
            },
            "wellness_care": {
                "care_strategy": 3.0,
                "emotion_pattern": 2.5,
                "hobby": 2.5,
                "preference": 2.5,
                "habit": 2.0,
                "work_style": 2.0,
                "dislike": 2.0,
                "interaction_style": 1.5,
                "fact": 0.5,
            },
            "behavior_distraction": {
                "work_style": 3.0,
                "habit": 2.5,
                "dislike": 2.5,
                "interaction_style": 2.5,
                "care_strategy": 2.0,
                "preference": 1.0,
                "hobby": 0.5,
                "fact": 0.5,
            },
            "environment_care": {
                "preference": 3.0,
                "dislike": 3.0,
                "habit": 1.5,
                "work_style": 1.5,
                "interaction_style": 1.5,
                "care_strategy": 1.0,
                "hobby": 0.5,
                "emotion_pattern": 0.5,
                "fact": 0.5,
            },
        }
    )
    default_type_weight: float = 1.0

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
                "好的",
                "好",
                "谢谢",
                "谢谢你",
                "开始",
                "停止",
                "ok",
                "okay",
                "yes",
                "no",
                "几点了",
                "现在几点",
            }
        )
    )
