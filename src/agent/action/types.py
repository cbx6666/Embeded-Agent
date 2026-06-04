from typing import Literal, cast, get_args

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
]

# 运行时可校验的 ActionType 闭集。
# Literal 只在静态类型检查时生效，因此工厂函数还需要用这个集合拦截非法动作。
ACTION_TYPES = cast(tuple[ActionType, ...], get_args(ActionType))
ACTION_TYPE_SET = frozenset(ACTION_TYPES)
