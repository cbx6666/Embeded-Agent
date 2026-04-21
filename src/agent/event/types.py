from typing import Literal

# 面向真实世界输入抽象出来的标准事件类型。
EventType = Literal[
    "user_text_input",
    "focus_start_requested",
    "focus_stop_requested",
    "user_presence_updated",
    "user_attention_updated",
    "user_emotion_updated",
    "user_fatigue_updated",
    "environment_updated",
    "timer_ticked",
    "timer_finished",
    "system_triggered",
]
