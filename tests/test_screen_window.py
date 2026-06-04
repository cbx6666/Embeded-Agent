from __future__ import annotations

import unittest

from src.adapters.screen.screen_window import parse_screen_size_arg


class ScreenWindowConfigTestCase(unittest.TestCase):
    def test_parse_screen_size_arg(self) -> None:
        self.assertEqual(parse_screen_size_arg("1920x1080"), (1920, 1080))
        self.assertIsNone(parse_screen_size_arg("bad"))
        self.assertIsNone(parse_screen_size_arg("100x100"))


if __name__ == "__main__":
    unittest.main()
