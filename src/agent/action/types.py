from typing import Literal

# Agent 当前支持的最小动作集合。
ActionType = Literal[
    "speak",
    "display",
    "render_pet_expression",
    "play_voice",
    "start_timer",
    "stop_timer",
    "none",
]
