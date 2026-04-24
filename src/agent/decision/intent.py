from __future__ import annotations

"""Agent 意图模型。"""

from dataclasses import dataclass, field
from typing import Any, Literal

IntentType = Literal[
    "answer_user",
    "start_focus",
    "stop_focus",
    "complete_focus",
    "suggest_rest",
    "remind_distraction",
    "update_status_feedback",
    "adjust_environment_feedback",
    "voice_interaction",
    "display_update",
    "no_op",
]


@dataclass
class AgentIntent:
    """Planner 输出的中间语义层，不直接绑定具体动作。"""

    type: IntentType
    priority: int = 0
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    requires_llm: bool = False
