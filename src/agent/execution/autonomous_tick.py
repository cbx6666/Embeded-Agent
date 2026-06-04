from __future__ import annotations

"""自主检查事件入口。"""

from src.agent.event import Event
from src.agent.state import AgentState


def build_autonomous_check_event(
    state: AgentState,
    now_ts: int,
    reason: str = "periodic_check",
) -> Event:
    """构造一个标准的自主检查内部事件。"""
    return Event(
        type="system_triggered",
        timestamp=int(now_ts),
        payload={
            "trigger": reason,
            "source": "agent_autonomy",
            "mode": state.interaction.mode,
            "focus_active": state.focus.active,
        },
    )
