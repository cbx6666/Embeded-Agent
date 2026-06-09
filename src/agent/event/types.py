from typing import Literal

# 面向真实嵌入式硬件 Agent 保留的标准事件类型。
# 只保留真实有生产者、有消费者、有落地意义的事件。
EventType = Literal[
    # 真实语音事件
    "speech_recognized",
    "voice_wake_detected",
    "voice_input_started",
    "voice_input_stopped",
    # TTS 状态事件
    "tts_started",
    "tts_finished",
    # 用户状态事件（高频，仅更新 State / RuntimeHistory）
    "user_presence_updated",
    "user_attention_updated",
    "user_emotion_updated",
    "user_fatigue_updated",
    "user_posture_updated",
    "user_activity_updated",
    # 环境事件（高频，仅更新 State / RuntimeHistory）
    "light_level_updated",
    "temperature_humidity_updated",
    "noise_level_updated",
    # 计时器事件
    "timer_ticked",
    "timer_finished",
    # 系统定时事件（wellness/behavior/environment 自检、sensor_status_report，由调度器产生）
    "system_triggered",
    # 结构化控制事件（走规则，不走 LLM）
    "focus_start_requested",
    "focus_stop_requested",
]
