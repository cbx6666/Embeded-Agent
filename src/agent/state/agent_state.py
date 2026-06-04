from __future__ import annotations

"""系统总状态模型。

状态示例：
- user.presence = "present"
- focus.active = True
- focus.remaining_sec = 1200
- runtime_history.focus_sessions = [{"actual_duration_sec": 1500, "reason": "timer_complete"}]
"""

from dataclasses import asdict, dataclass, field, fields

from src.agent.state.runtime_history import RuntimeHistory
from src.agent.state.cooldown_state import CooldownState
from src.agent.state.environment_state import EnvironmentState
from src.agent.state.focus_state import FocusState
from src.agent.state.interaction_state import InteractionState
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
    runtime_history: RuntimeHistory = field(default_factory=RuntimeHistory)
    current_user_id: str = "default"

    def to_dict(self) -> dict:
        """将嵌套 dataclass 转成可持久化字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "AgentState":
        """从持久化字典恢复完整状态；没有数据时返回默认状态。"""
        if not data:
            return cls()

        def _only_fields(d: dict, dc: type) -> dict:
            """仅保留目标 dataclass 定义过的字段。"""
            names = {f.name for f in fields(dc)}
            return {k: v for k, v in d.items() if k in names}

        return cls(
            user=UserState(**_only_fields(data.get("user", {}), UserState)),
            interaction=InteractionState(**_only_fields(data.get("interaction", {}), InteractionState)),
            focus=FocusState(**_only_fields(data.get("focus", {}), FocusState)),
            environment=EnvironmentState(**_only_fields(data.get("environment", {}), EnvironmentState)),
            cooldown=CooldownState(**_only_fields(data.get("cooldown", {}), CooldownState)),
            runtime_history=RuntimeHistory(
                **_only_fields(
                    data.get("runtime_history", data.get("memory", {})),
                    RuntimeHistory,
                )
            ),
            current_user_id=str(data.get("current_user_id") or "default"),
        )
