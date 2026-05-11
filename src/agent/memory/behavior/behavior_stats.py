from __future__ import annotations

"""长期行为统计数据结构。

BehaviorStats 表达“长期统计事实”，不是画像结论，也不直接做决策。
例如：用户几点开始专注、疲劳事件出现了多少次、音乐休息建议被接受多少次。
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class BehaviorStats:
    """单个用户的长期行为聚合结果。

    focus_start_by_hour：按小时统计专注开始次数，key 为 0-23 的字符串。
    focus_session_count：已完成专注会话数量。
    total_focus_duration_sec：所有已完成专注会话总时长。
    fatigue_event_count：疲劳/压力相关事件数量。
    distraction_event_count：分心事件数量。
    break_suggestion_count：Agent 给出休息建议的次数。
    accepted_break_suggestions：用户明确接受休息建议次数。
    rejected_break_suggestions：用户明确拒绝休息建议次数。
    music_break_suggestion_count：Agent 给出音乐类休息建议次数。
    accepted_music_breaks：用户明确接受音乐休息建议次数。
    rejected_music_breaks：用户明确拒绝音乐休息建议次数。
    fatigue_after_focus_duration_total_sec：发生疲劳时累计的专注已持续时长。
    fatigue_after_focus_count：有专注时长上下文的疲劳样本数。
    last_updated_at：最近一次统计更新时间。

    这些字段只描述事实，不表达“用户是什么样的人”；抽象结论由 InsightExtractor 生成。
    """

    focus_start_by_hour: dict[str, int] = field(default_factory=dict)
    focus_session_count: int = 0
    total_focus_duration_sec: int = 0
    fatigue_event_count: int = 0
    distraction_event_count: int = 0
    break_suggestion_count: int = 0
    accepted_break_suggestions: int = 0
    rejected_break_suggestions: int = 0
    music_break_suggestion_count: int = 0
    accepted_music_breaks: int = 0
    rejected_music_breaks: int = 0
    fatigue_after_focus_duration_total_sec: int = 0
    fatigue_after_focus_count: int = 0
    last_updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的普通字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BehaviorStats":
        """从 JSON 原始字典恢复 BehaviorStats，并忽略未知字段。"""
        if not isinstance(data, dict):
            return cls()
        names = {field.name for field in fields(cls)}
        values = {key: value for key, value in data.items() if key in names}
        values["focus_start_by_hour"] = _normalize_hour_counts(values.get("focus_start_by_hour"))
        return cls(**values)

    @property
    def average_focus_duration_sec(self) -> float:
        """返回平均专注时长；这是统计事实，不是决策规则。"""
        if self.focus_session_count <= 0:
            return 0.0
        return self.total_focus_duration_sec / self.focus_session_count

    @property
    def average_focus_before_fatigue_sec(self) -> float:
        """返回疲劳出现前平均已专注时长。"""
        if self.fatigue_after_focus_count <= 0:
            return 0.0
        return self.fatigue_after_focus_duration_total_sec / self.fatigue_after_focus_count

    @property
    def break_accept_rate(self) -> float:
        """返回普通休息建议接受率。"""
        total = self.accepted_break_suggestions + self.rejected_break_suggestions
        if total <= 0:
            return 0.0
        return self.accepted_break_suggestions / total

    @property
    def music_break_accept_rate(self) -> float:
        """返回音乐休息建议接受率。"""
        total = self.accepted_music_breaks + self.rejected_music_breaks
        if total <= 0:
            return 0.0
        return self.accepted_music_breaks / total

    @property
    def night_focus_ratio(self) -> float:
        """返回夜间开始专注的比例，夜间定义为 20:00-23:59 和 00:00-05:59。"""
        total = sum(int(value) for value in self.focus_start_by_hour.values())
        if total <= 0:
            return 0.0
        night_count = sum(
            int(count)
            for hour, count in self.focus_start_by_hour.items()
            if _is_night_hour(int(hour))
        )
        return night_count / total


def _normalize_hour_counts(value: object) -> dict[str, int]:
    """规范化小时统计，避免 JSON 读回后 key/value 类型不稳定。"""
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, int] = {}
    for raw_hour, raw_count in value.items():
        hour = int(raw_hour)
        if hour < 0 or hour > 23:
            continue
        normalized[str(hour)] = int(raw_count)
    return normalized


def _is_night_hour(hour: int) -> bool:
    """判断小时是否属于夜间学习统计窗口。"""
    return hour >= 20 or hour <= 5
