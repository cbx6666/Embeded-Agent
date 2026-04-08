from __future__ import annotations

"""统一动作模型。

动作示例：
- Action(type="speak", payload={"text": "你好，我在。"})
- Action(type="display", payload={"text": "专注倒计时开始"})
- Action(type="start_timer", payload={"duration_sec": 1500})
- Action(type="stop_timer", payload={})
"""

from dataclasses import dataclass, field
from typing import Any

from src.agent.action.types import ActionType


@dataclass
class Action:
    """统一动作模型：描述“系统接下来要执行什么”。"""

    type: ActionType
    payload: dict[str, Any] = field(default_factory=dict)
