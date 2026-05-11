from __future__ import annotations

"""长期行为统计更新器。

BehaviorUpdater 只负责把标准化 BehaviorSignal 累计进 BehaviorStats。
它不生成 insight、不操作 Planner、不写 Action，也不关心 profile 如何持久化。
"""

from src.agent.memory.behavior.behavior_stats import BehaviorStats
from src.agent.memory.extractor.behavior_extractor import BehaviorSignal


class BehaviorUpdater:
    """根据行为信号更新长期统计事实。"""

    def update(self, stats: BehaviorStats, signal: BehaviorSignal) -> bool:
        """消费单个行为信号；统计发生变化时返回 True。"""
        if signal.type == "focus_started":
            hour = str(int(signal.payload.get("hour", 0)))
            stats.focus_start_by_hour[hour] = stats.focus_start_by_hour.get(hour, 0) + 1
            stats.last_updated_at = signal.timestamp
            return True

        if signal.type == "focus_completed":
            duration_sec = max(0, int(signal.payload.get("duration_sec", 0)))
            stats.focus_session_count += 1
            stats.total_focus_duration_sec += duration_sec
            stats.last_updated_at = signal.timestamp
            return True

        if signal.type == "fatigue_detected":
            stats.fatigue_event_count += 1
            focus_elapsed_sec = int(signal.payload.get("focus_elapsed_sec") or 0)
            if focus_elapsed_sec > 0:
                stats.fatigue_after_focus_count += 1
                stats.fatigue_after_focus_duration_total_sec += focus_elapsed_sec
            stats.last_updated_at = signal.timestamp
            return True

        if signal.type == "distraction_detected":
            stats.distraction_event_count += 1
            stats.last_updated_at = signal.timestamp
            return True

        if signal.type == "break_suggestion_shown":
            stats.break_suggestion_count += 1
            if signal.payload.get("content_type") == "音乐":
                stats.music_break_suggestion_count += 1
            stats.last_updated_at = signal.timestamp
            return True

        if signal.type == "break_suggestion_accepted":
            stats.accepted_break_suggestions += 1
            if signal.payload.get("content_type") == "音乐":
                stats.accepted_music_breaks += 1
            stats.last_updated_at = signal.timestamp
            return True

        if signal.type == "break_suggestion_rejected":
            stats.rejected_break_suggestions += 1
            if signal.payload.get("content_type") == "音乐":
                stats.rejected_music_breaks += 1
            stats.last_updated_at = signal.timestamp
            return True

        return False
