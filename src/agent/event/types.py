from typing import Literal

# 面向真实世界输入抽象出来的标准事件类型。
EventType = Literal[
    "user_text_input",
    "focus_start_requested",
    "focus_stop_requested",
    "user_presence_updated",
    "user_attention_updated",
    "user_emotion_updated",

    # posture / fatigue / sensor
    "user_posture_updated",
    "user_posture_summary",
    "user_fatigue_updated",
    "display_sensor_updated",

    # voice pipeline
    "voice_input_captured",
    "voice_wake_detected",
    "voice_input_started",
    "voice_input_stopped",
    "speech_recognized",
    "tts_started",
    "tts_finished",
    "voice_volume_changed",
    "voice_timbre_changed",
    "voice_speed_changed",

    # environment
    "environment_updated",
    "light_level_updated",
    "temperature_humidity_updated",
    "noise_level_updated",

    # timer / system
    "timer_ticked",
    "timer_finished",
    "system_triggered",
]
