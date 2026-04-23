from typing import Literal

# Agent 的高层运行模式。
Mode = Literal["normal", "focus", "silent"]

# 用户相关状态。
UserPresence = Literal["present", "away", "unknown"]
UserAttention = Literal["focused", "distracted", "idle"]
UserEmotion = Literal["neutral", "tired", "stressed", "happy"]
UserFatigueLevel = Literal["none", "mild", "moderate", "high"]
UserBehavior = Literal["working", "phone_use", "staring", "desk_rest", "away", "unknown"]

# 输出与交互状态。
DialogueState = Literal["idle", "listening", "thinking", "speaking"]
LightState = Literal["idle", "wake", "thinking", "alert", "error"]
