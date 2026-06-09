"""桌宠显示上下文：整合 Agent 状态、传感器、统计与语音字幕。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PetDisplayContext:
    """一帧桌宠所需的全部显示数据。"""

    agent_state: str = "idle"
    expression: str = "neutral"
    status_label: str = ""
    speak_text: str = ""
    user_speech_text: str = ""
    agent_speech_text: str = ""
    speech_mode: str = ""  # listening | recognizing | user | agent | ""
    focus_remaining: int = 0
    focus_duration: int = 0

    temperature_c: float | None = None
    humidity_pct: float | None = None
    light_lux: int | None = None
    noise_db: int | None = None
    temperature_level: str = ""
    humidity_level: str = ""
    light_level: str = ""
    noise_level: str = ""

    emotion: str = "neutral"
    fatigue: str = "none"
    emotion_confidence: float | None = None
    fatigue_confidence: float | None = None

    emotion_pie: dict[str, int] = field(default_factory=dict)
    fatigue_pie: dict[str, int] = field(default_factory=dict)
    emotion_timeline: list[tuple[int, float]] = field(default_factory=list)
    fatigue_timeline: list[tuple[int, float]] = field(default_factory=list)

    @classmethod
    def from_legacy(
        cls,
        *,
        agent_state: str,
        speak_text: str = "",
        focus_remaining: int = 0,
        focus_duration: int = 0,
        status_label: str = "",
        **extra: Any,
    ) -> PetDisplayContext:
        ctx = cls(
            agent_state=agent_state,
            speak_text=speak_text,
            focus_remaining=focus_remaining,
            focus_duration=focus_duration,
            status_label=status_label,
        )
        for key, value in extra.items():
            if hasattr(ctx, key):
                setattr(ctx, key, value)
        return ctx
