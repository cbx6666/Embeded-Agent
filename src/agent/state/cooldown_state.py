from dataclasses import dataclass, field


@dataclass
class CooldownState:
    """冷却状态：分别记录提醒执行时间和自主检查准入时间。"""

    reminder_last_ts: dict[str, int] = field(default_factory=dict)
    autonomous_check_last_ts: dict[str, int] = field(default_factory=dict)
