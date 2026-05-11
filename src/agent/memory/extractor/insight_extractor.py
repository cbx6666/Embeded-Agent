from __future__ import annotations

"""从长期行为统计中抽取画像候选。

InsightExtractor 只做“统计事实 -> 抽象候选”的转换，不直接处理 Event，
也不直接写 UserProfile。是否能进入长期画像由 MemoryPolicy 决定。
"""

from src.agent.memory.behavior.behavior_stats import BehaviorStats
from src.agent.memory.memory_candidate import MemoryCandidate

NIGHT_FOCUS_MIN_EVIDENCE = 3
MUSIC_BREAK_MIN_EVIDENCE = 3
FATIGUE_MIN_EVIDENCE = 3
NIGHT_FOCUS_RATIO_THRESHOLD = 0.7
MUSIC_ACCEPT_RATE_THRESHOLD = 0.8
SHORT_FATIGUE_THRESHOLD_SEC = 25 * 60


class InsightExtractor:
    """根据行为统计生成可解释的画像候选。"""

    def extract(self, stats: BehaviorStats) -> list[MemoryCandidate]:
        """从当前 BehaviorStats 生成候选画像列表。"""
        candidates: list[MemoryCandidate] = []
        candidates.extend(self._extract_focus_time_pattern(stats))
        candidates.extend(self._extract_break_recovery_preference(stats))
        candidates.extend(self._extract_fatigue_sensitivity(stats))
        return candidates

    def _extract_focus_time_pattern(self, stats: BehaviorStats) -> list[MemoryCandidate]:
        """根据专注开始时段抽取作息倾向画像。"""
        focus_count = sum(stats.focus_start_by_hour.values())
        if focus_count < NIGHT_FOCUS_MIN_EVIDENCE:
            return []

        ratio = stats.night_focus_ratio
        if ratio < NIGHT_FOCUS_RATIO_THRESHOLD:
            return []

        confidence = min(0.95, max(0.0, ratio))
        return [
            MemoryCandidate(
                insight_type="study_time_pattern",
                content="用户倾向于夜间学习",
                confidence=confidence,
                evidence_count=focus_count,
                source="behavior_stats.focus_start_by_hour",
                contradiction_group="study_time_pattern",
                explanation=f"夜间专注开始占比 {ratio:.2f}，样本数 {focus_count}",
            )
        ]

    def _extract_break_recovery_preference(self, stats: BehaviorStats) -> list[MemoryCandidate]:
        """根据休息建议接受情况抽取恢复方式偏好。"""
        music_evidence = stats.accepted_music_breaks + stats.rejected_music_breaks
        if music_evidence < MUSIC_BREAK_MIN_EVIDENCE:
            return []

        accept_rate = stats.music_break_accept_rate
        if accept_rate < MUSIC_ACCEPT_RATE_THRESHOLD:
            return []

        confidence = min(0.95, accept_rate)
        return [
            MemoryCandidate(
                insight_type="break_recovery_preference",
                content="用户疲劳时更容易接受音乐休息",
                confidence=confidence,
                evidence_count=music_evidence,
                source="behavior_stats.music_break_accept_rate",
                contradiction_group="break_recovery_preference",
                explanation=f"音乐休息接受率 {accept_rate:.2f}，样本数 {music_evidence}",
            )
        ]

    def _extract_fatigue_sensitivity(self, stats: BehaviorStats) -> list[MemoryCandidate]:
        """根据疲劳出现前专注时长抽取疲劳敏感性画像。"""
        if stats.fatigue_after_focus_count < FATIGUE_MIN_EVIDENCE:
            return []

        average_sec = stats.average_focus_before_fatigue_sec
        if average_sec <= 0 or average_sec >= SHORT_FATIGUE_THRESHOLD_SEC:
            return []

        # 越早出现疲劳，置信度越高；同时受证据数约束，避免少量样本过度放大。
        early_ratio = 1.0 - (average_sec / SHORT_FATIGUE_THRESHOLD_SEC)
        evidence_boost = min(0.2, stats.fatigue_after_focus_count * 0.03)
        confidence = min(0.9, 0.6 + early_ratio * 0.2 + evidence_boost)
        return [
            MemoryCandidate(
                insight_type="fatigue_sensitivity",
                content="用户较短专注后也容易疲劳",
                confidence=confidence,
                evidence_count=stats.fatigue_after_focus_count,
                source="behavior_stats.average_focus_before_fatigue_sec",
                contradiction_group="fatigue_sensitivity",
                explanation=f"疲劳前平均专注 {average_sec:.0f} 秒，样本数 {stats.fatigue_after_focus_count}",
            )
        ]
