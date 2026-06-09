from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CooldownState:
    """冷却状态：分别记录提醒执行时间和自主检查准入时间。"""

    reminder_last_ts: dict[str, int] = field(default_factory=dict)
    autonomous_check_last_ts: dict[str, int] = field(default_factory=dict)
    # 媒体询问：首次可问放歌；之后每累计 2 次纯 wellness 关怀播报后才可再问。
    media_suggestion_ever_asked: bool = False
    wellness_cares_since_media_ask: int = 0
