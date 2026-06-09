from typing import Literal, cast, get_args

# Agent 当前真实可执行的最小动作集合。
# 只保留 ActionRealizer 会生成、DeviceAdapter / 适配器能真实执行的动作。
ActionType = Literal[
    "speak",
    "display",
    "start_timer",
    "stop_timer",
    "set_tts_volume",
    "play_media",
    "stop_media",
    "pause_media",
    "resume_media",
    "next_media",
]

# 运行时可校验的 ActionType 闭集；拦截未注册动作进入设备层。
ACTION_TYPES = cast(tuple[ActionType, ...], get_args(ActionType))
ACTION_TYPE_SET = frozenset(ACTION_TYPES)
