from typing import Literal

# 面向真实世界输入抽象出来的标准事件类型。
EventType = Literal[
    # text / command
    "user_text_input",
    "focus_start_requested",
    "focus_stop_requested",

    # behavior
    "user_presence_updated",
    "user_attention_updated",
    "user_emotion_updated",
    "user_fatigue_updated",
    "user_posture_updated",
    "user_activity_updated",

    # display / voice
    "display_sensor_updated",
    "voice_wake_detected",
    "voice_input_started",
    "voice_input_stopped",
    "speech_recognized",
    "user_switched",
    "user_profile_updated",
    "user_preference_update_requested",
    "break_suggestion_accepted",
    "break_suggestion_rejected",
    "tts_started",
    "tts_finished",
    "voice_volume_changed",
    "voice_timbre_changed",
    "voice_speed_changed",

    # environment
    "light_level_updated",
    "temperature_humidity_updated",
    "noise_level_updated",

    # timer / system
    "timer_ticked",
    "timer_finished",
    "system_triggered",
]
