from __future__ import annotations

"""动作执行结果模型。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionResult:
    """记录单个动作的执行结果。"""

    action_type: str
    success: bool
    timestamp: int
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
