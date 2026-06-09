from __future__ import annotations

from dataclasses import dataclass

from src.agent.state.types import (
    CurrentActivity,
    UserAttention,
    UserBehavior,
    UserEmotion,
    UserFatigueLevel,
    UserPosture,
    UserPresence,
)


@dataclass
class UserState:
    """用户状态：反映用户是否在场、是否专注、当前行为、情绪与疲劳。"""

    presence: UserPresence = "unknown"
    attention: UserAttention = "idle"
    behavior: UserBehavior = "unknown"
    emotion: UserEmotion = "neutral"
    fatigue_level: UserFatigueLevel = "none"
    posture: UserPosture = "unknown"
    current_activity: CurrentActivity = "unknown"
    presence_confidence: float | None = None
    attention_confidence: float | None = None
    behavior_confidence: float | None = None
    emotion_confidence: float | None = None
    fatigue_confidence: float | None = None
    posture_confidence: float | None = None
