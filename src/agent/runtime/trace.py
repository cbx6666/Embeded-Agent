from __future__ import annotations

"""决策 trace 模型。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentDecisionTrace:
    """记录一次闭环步骤中的核心决策信息。"""

    event_type: str
    timestamp: int
    state_summary: dict[str, Any]
    intents: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    loop_step: int = 0
