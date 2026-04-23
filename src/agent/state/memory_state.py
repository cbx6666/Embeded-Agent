from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryState:
    """记忆状态：只保留最近工作集，不承担长期归档职责。"""

    recent_events: list[dict[str, Any]] = field(default_factory=list)
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    reminder_records: list[dict[str, Any]] = field(default_factory=list)
    attention_records: list[dict[str, Any]] = field(default_factory=list)
    environment_records: list[dict[str, Any]] = field(default_factory=list)
    focus_sessions: list[dict[str, Any]] = field(default_factory=list)
    focus_session_count: int = 0
    focus_total_duration_sec: int = 0
    distraction_event_count: int = 0
    state_change_counts: dict[str, int] = field(default_factory=dict)
    emotion_samples: list[dict[str, Any]] = field(default_factory=list)
    emotion_summaries: list[dict[str, Any]] = field(default_factory=list)
