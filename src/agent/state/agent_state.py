from __future__ import annotations

"""系统总状态模型。

状态示例：
- user.presence = "present"
- focus.active = True
- focus.remaining_sec = 1200
- memory.focus_sessions = [{"actual_duration_sec": 1500, "reason": "timer_complete"}]
"""

from dataclasses import asdict, dataclass, field

from src.agent.state.cooldown_state import CooldownState
from src.agent.state.environment_state import EnvironmentState
from src.agent.state.focus_state import FocusState
from src.agent.state.interaction_state import InteractionState
from src.agent.state.memory_state import MemoryState
from src.agent.state.user_state import UserState


@dataclass
class AgentState:
    """系统总状态。

    通过组合多个子状态块，避免所有字段平铺在一个大对象里。
    """

    user: UserState = field(default_factory=UserState)
    interaction: InteractionState = field(default_factory=InteractionState)
    focus: FocusState = field(default_factory=FocusState)
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    cooldown: CooldownState = field(default_factory=CooldownState)
    memory: MemoryState = field(default_factory=MemoryState)

    def to_dict(self) -> dict:
        """将嵌套 dataclass 转成可持久化字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "AgentState":
        """从持久化字典恢复完整状态；没有数据时返回默认状态。"""
        if not data:
            return cls()

        return cls(
            user=UserState(**data.get("user", {})),
            interaction=InteractionState(**data.get("interaction", {})),
            focus=FocusState(**data.get("focus", {})),
            environment=EnvironmentState(**data.get("environment", {})),
            cooldown=CooldownState(**data.get("cooldown", {})),
            memory=MemoryState(**data.get("memory", {})),
        )
