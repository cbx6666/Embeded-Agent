from __future__ import annotations

"""最近记忆维护服务模块。"""

from src.agent.event import Event
from src.agent.state import AgentState


class MemoryService:
    """维护最近事件、最近消息和最近专注记录。"""

    def __init__(
        self,
        max_recent_events: int = 20,
        max_recent_messages: int = 20,
        max_focus_sessions: int = 10,
    ) -> None:
        self.max_recent_events = max_recent_events
        self.max_recent_messages = max_recent_messages
        self.max_focus_sessions = max_focus_sessions

    def record_event(self, state: AgentState, event: Event) -> None:
        state.memory.recent_events.append(
            {
                "type": event.type,
                "timestamp": event.timestamp,
                "payload": event.payload,
            }
        )

    def record_message(
        self,
        state: AgentState,
        role: str,
        text: str,
        timestamp: int,
    ) -> None:
        state.memory.recent_messages.append(
            {
                "role": role,
                "text": text,
                "timestamp": timestamp,
            }
        )

    def trim(self, state: AgentState) -> None:
        state.memory.recent_events = state.memory.recent_events[-self.max_recent_events :]
        state.memory.recent_messages = state.memory.recent_messages[-self.max_recent_messages :]
        state.memory.focus_sessions = state.memory.focus_sessions[-self.max_focus_sessions :]
