from __future__ import annotations

import unittest

from src.adapters.screen.pet_display_context import PetDisplayContext
from src.adapters.screen.sensor_format import format_sensor_lines, format_user_lines


class SensorFormatTest(unittest.TestCase):
    def test_numeric_sensor_lines(self) -> None:
        ctx = PetDisplayContext(
            temperature_c=26.5,
            humidity_pct=62.0,
            light_lux=300,
            noise_db=45,
        )
        lines = format_sensor_lines(ctx)
        self.assertIn("TEMP 26.5C", lines)
        self.assertIn("HUM  62%", lines)
        self.assertIn("LUX  300", lines)
        self.assertIn("NOISE 45dB", lines)

    def test_level_fallback_when_no_numeric(self) -> None:
        ctx = PetDisplayContext(temperature_level="high", humidity_level="dry")
        lines = format_sensor_lines(ctx)
        self.assertIn("TEMP high", lines)
        self.assertIn("HUM  dry", lines)

    def test_placeholder_when_empty(self) -> None:
        self.assertEqual(format_sensor_lines(PetDisplayContext()), ["SENSOR --"])

    def test_user_lines(self) -> None:
        ctx = PetDisplayContext(
            emotion="happy", fatigue="mild",
            emotion_confidence=0.85, fatigue_confidence=0.7,
        )
        lines = format_user_lines(ctx)
        self.assertIn("EMO  happy", lines)
        self.assertIn("FAT  mild", lines)
        self.assertIn("EMO% 85", lines)
        self.assertIn("FAT% 70", lines)


if __name__ == "__main__":
    unittest.main()
