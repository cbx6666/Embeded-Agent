from __future__ import annotations

import unittest

from src.adapters.vision_affect.emotion_second_stats import EmotionSecondStats
from src.adapters.vision_affect.fatigue_second_stats import FatigueSecondStats


class FatigueSecondStatsTest(unittest.TestCase):
    def test_high_level_emits(self) -> None:
        stats = FatigueSecondStats()
        for _ in range(5):
            stats.push(0, "high", 0.9)
        ready = stats.push(1, "high", 0.85)
        self.assertTrue(any(s.fatigue_level == "high" for s in ready))

    def test_fills_gap_seconds(self) -> None:
        stats = FatigueSecondStats()
        stats.push(0, "mild", 0.2)
        ready = stats.push(3, "moderate", 0.5)
        # 当前秒 3 仅开始投票，需下次 push 或 flush 才上报
        self.assertEqual([s.timestamp for s in ready], [0, 1, 2])


class EmotionSecondStatsTest(unittest.TestCase):
    def test_fills_gap_seconds(self) -> None:
        stats = EmotionSecondStats()
        stats.push(10, "emo:happy", 0.9)
        ready = stats.push(13, "emo:stressed", 0.8)
        self.assertEqual([s.timestamp for s in ready], [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
