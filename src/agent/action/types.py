from typing import Literal

# Agent 当前支持的最小动作集合。
ActionType = Literal[
    "speak",
    "display",
    "start_timer",
    "stop_timer",
    "none",
    "start_voice_capture",
    "stop_voice_capture",
    "set_tts_voice",
    "set_tts_volume",
    "set_tts_speed",
    "environment_alert",
    "render_pet_expression",
    "play_voice",
]
