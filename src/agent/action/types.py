from typing import Literal

# Agent 当前支持的最小动作集合。
ActionType = Literal[
    # output
    "speak",
    "display",
    "render_pet_expression",
    "set_light_state",

    # timer
    "start_timer",
    "stop_timer",

    # voice
    "start_voice_capture",
    "stop_voice_capture",
    "set_tts_voice",
    "set_tts_volume",
    "set_tts_speed",
    "none",
]
