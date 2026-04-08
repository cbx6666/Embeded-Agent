from typing import Literal

# Agent 的高层运行模式。
Mode = Literal["normal", "focus", "silent"]

# 用户相关状态。
UserPresence = Literal["present", "away", "unknown"]
UserAttention = Literal["focused", "distracted", "idle"]
UserEmotion = Literal["neutral", "tired", "stressed", "happy"]
CurrentActivity = Literal["studying", "working", "resting", "unknown"]

# 当前对话阶段。
DialogueState = Literal["idle", "listening", "thinking", "speaking"]
