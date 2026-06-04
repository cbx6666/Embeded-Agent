from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.adapters.vision_affect.pipeline import FatigueLevel

VALID_FATIGUE_LEVELS = frozenset({"none", "mild", "moderate", "high"})


@dataclass
class FatigueSecondSummary:
    timestamp: int
    fatigue_level: FatigueLevel
    avg_confidence: float | None


class FatigueSecondStats:
    """按秒聚合 fatigue 帧结果；秒号跳变时补齐中间缺失秒的上报。"""

    def __init__(self) -> None:
        self._sec: int | None = None
        self._votes: Counter[str] = Counter()
        self._conf_sum: float = 0.0
        self._conf_n: int = 0

    def push(
        self,
        timestamp_sec: int,
        fatigue_level: FatigueLevel,
        confidence: float | None,
    ) -> list[FatigueSecondSummary]:
        ready: list[FatigueSecondSummary] = []
        ts = int(timestamp_sec)

        if self._sec is None:
            self._sec = ts
        elif ts > self._sec:
            closed = self._close_current_second()
            if closed is not None:
                ready.append(closed)
                for gap_ts in range(self._sec + 1, ts):
                    ready.append(
                        FatigueSecondSummary(
                            timestamp=gap_ts,
                            fatigue_level=closed.fatigue_level,
                            avg_confidence=closed.avg_confidence,
                        )
                    )
            self._reset_bucket()
            self._sec = ts
        elif ts < self._sec:
            self._reset_bucket()
            self._sec = ts

        if fatigue_level in VALID_FATIGUE_LEVELS:
            self._votes[fatigue_level] += 1
            if confidence is not None:
                self._conf_sum += float(max(0.0, min(1.0, confidence)))
                self._conf_n += 1
        return ready

    def flush(self) -> FatigueSecondSummary | None:
        closed = self._close_current_second()
        self._reset_bucket()
        self._sec = None
        return closed

    def _close_current_second(self) -> FatigueSecondSummary | None:
        if self._sec is None or not self._votes:
            return None
        fatigue_state = self._votes.most_common(1)[0][0]
        if fatigue_state not in VALID_FATIGUE_LEVELS:
            return None
        avg_conf = (self._conf_sum / self._conf_n) if self._conf_n > 0 else None
        return FatigueSecondSummary(
            timestamp=self._sec,
            fatigue_level=fatigue_state,
            avg_confidence=avg_conf,
        )

    def _reset_bucket(self) -> None:
        self._votes.clear()
        self._conf_sum = 0.0
        self._conf_n = 0
