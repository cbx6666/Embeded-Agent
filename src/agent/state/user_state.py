from dataclasses import dataclass

from src.agent.state.types import CurrentActivity, UserAttention, UserEmotion, UserPresence


@dataclass
class UserState:
    """用户状态：反映用户是否在场、是否专注、当前情绪与活动。"""

    presence: UserPresence = "unknown"
    attention: UserAttention = "idle"
    emotion: UserEmotion = "neutral"
    current_activity: CurrentActivity = "unknown"
    emotion_confidence: float | None = None
    attention_confidence: float | None = None
