from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class EmotionSecondSummary:
    timestamp: int
    emotion_key: str
    avg_confidence: float | None


class EmotionSecondStats:
    """按秒聚合 emotion 帧结果，产出该秒多数票状态。"""

    def __init__(self) -> None:
        self._sec: int | None = None
        self._votes: Counter[str] = Counter()
        self._conf_sum: float = 0.0
        self._conf_n: int = 0

    def push(self, timestamp_sec: int, emotion_key: str, confidence: float | None) -> EmotionSecondSummary | None:
        ready = None
        if self._sec is None:
            self._sec = int(timestamp_sec)
        elif int(timestamp_sec) != self._sec:
            ready = self._close_current_second()
            self._sec = int(timestamp_sec)

        self._votes[emotion_key] += 1
        if confidence is not None:
            self._conf_sum += float(max(0.0, min(1.0, confidence)))
            self._conf_n += 1
        return ready

    def flush(self) -> EmotionSecondSummary | None:
        return self._close_current_second()

    def _close_current_second(self) -> EmotionSecondSummary | None:
        if self._sec is None or not self._votes:
            return None
        result = EmotionSecondSummary(
            timestamp=self._sec,
            emotion_key=self._votes.most_common(1)[0][0],
            avg_confidence=(self._conf_sum / self._conf_n) if self._conf_n > 0 else None,
        )
        self._reset_bucket()
        return result

    def _reset_bucket(self) -> None:
        self._votes.clear()
        self._conf_sum = 0.0
        self._conf_n = 0
