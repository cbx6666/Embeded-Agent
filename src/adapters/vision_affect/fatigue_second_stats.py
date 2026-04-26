from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.adapters.vision_affect.pipeline import FatigueLevel


@dataclass
class FatigueSecondSummary:
    timestamp: int
    fatigue_level: FatigueLevel
    avg_confidence: float | None


class FatigueSecondStats:
    """按秒聚合 fatigue 帧结果，产出该秒多数票状态。"""

    def __init__(self) -> None:
        self._sec: int | None = None
        self._votes: Counter[str] = Counter()
        self._conf_sum: float = 0.0
        self._conf_n: int = 0

    def push(self, timestamp_sec: int, fatigue_level: FatigueLevel, confidence: float | None) -> FatigueSecondSummary | None:
        ready = None
        if self._sec is None:
            self._sec = int(timestamp_sec)
        elif int(timestamp_sec) != self._sec:
            ready = self._close_current_second()
            self._sec = int(timestamp_sec)

        self._votes[fatigue_level] += 1
        if confidence is not None:
            self._conf_sum += float(max(0.0, min(1.0, confidence)))
            self._conf_n += 1
        return ready

    def flush(self) -> FatigueSecondSummary | None:
        return self._close_current_second()

    def _close_current_second(self) -> FatigueSecondSummary | None:
        if self._sec is None or not self._votes:
            return None
        fatigue_state = self._votes.most_common(1)[0][0]
        if fatigue_state not in {"none", "mild", "moderate", "severe"}:
            self._reset_bucket()
            return None
        avg_conf = (self._conf_sum / self._conf_n) if self._conf_n > 0 else None
        result = FatigueSecondSummary(
            timestamp=self._sec,
            fatigue_level=fatigue_state,
            avg_confidence=avg_conf,
        )
        self._reset_bucket()
        return result

    def _reset_bucket(self) -> None:
        self._votes.clear()
        self._conf_sum = 0.0
        self._conf_n = 0
