from dataclasses import dataclass, field


@dataclass
class CooldownState:
    """冷却状态：记录不同提醒类型最近一次触发时间。"""

    reminder_last_ts: dict[str, int] = field(default_factory=dict)
