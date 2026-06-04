from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryState:
    """记忆状态：只保留最近工作集，不承担长期归档职责。"""

    recent_events: list[dict[str, Any]] = field(default_factory=list)
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    focus_sessions: list[dict[str, Any]] = field(default_factory=list)
