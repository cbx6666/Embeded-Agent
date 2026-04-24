from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionResult:
    action_type: str
    success: bool
    timestamp: int
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
