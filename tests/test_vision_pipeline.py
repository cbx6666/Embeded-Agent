from __future__ import annotations

import unittest

from src.adapters.vision_affect.pipeline import (
    combined_fatigue_score,
    map_fatigue_with_hysteresis,
    monotonic_ts,
    PercLosWindow,
)


class VisionPipelineTest(unittest.TestCase):
    def test_perclos_window_ratio(self) -> None:
        w = PercLosWindow(window_sec=1.0)
        t0 = monotonic_ts()
        for i in range(10):
            w.push(t0 + i * 0.05, eye_closed=(i % 2 == 0))
        self.assertAlmostEqual(w.ratio(), 0.5, places=1)

    def test_hysteresis_does_not_oscillate(self) -> None:
        self.assertEqual(map_fatigue_with_hysteresis(0.25, "none"), "mild")
        self.assertEqual(map_fatigue_with_hysteresis(0.25, "mild"), "mild")
        self.assertEqual(map_fatigue_with_hysteresis(0.15, "mild"), "mild")
        self.assertEqual(map_fatigue_with_hysteresis(0.10, "mild"), "none")

    def test_combined_fatigue_is_weighted(self) -> None:
        c = combined_fatigue_score(0.2, 0.8, eye_weight=0.5, mouth_weight=0.5)
        self.assertAlmostEqual(c, 0.5, places=2)
        c2 = combined_fatigue_score(0.0, 1.0, eye_weight=0.5, mouth_weight=0.5)
        self.assertAlmostEqual(c2, 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
