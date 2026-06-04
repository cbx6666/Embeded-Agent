from __future__ import annotations

"""统一事件模型。

事件示例：
- Event(type="user_text_input", payload={"text": "你好", "source": "cli"})
- Event(type="focus_start_requested", payload={"duration_sec": 1500, "source": "cli"})
- Event(type="user_presence_updated", payload={"presence": "away", "source": "camera"})
- Event(type="user_fatigue_updated", payload={"fatigue_level": "moderate", "perclos": 0.32, "source": "mediapipe_pipeline"})
- Event(type="timer_finished", payload={"timer": "focus"})
"""

from dataclasses import dataclass, field
from typing import Any

from src.agent.event.types import EventType


@dataclass
class Event:
    """统一事件模型：描述“外部世界或系统内部发生了什么”。"""

    type: EventType
    timestamp: int
    payload: dict[str, Any] = field(default_factory=dict)
