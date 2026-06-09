from __future__ import annotations

import os
import unittest


class CjkFontTestCase(unittest.TestCase):
    def test_resolve_finds_system_font_or_none(self) -> None:
        from src.adapters.screen.cjk_font import resolve_cjk_font_path

        path = resolve_cjk_font_path()
        if path is not None:
            self.assertTrue(os.path.isfile(path))

    def test_get_font_renders_chinese(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        from src.adapters.screen.cjk_font import get_font, resolve_cjk_font_path

        if resolve_cjk_font_path() is None:
            self.skipTest("no CJK font on system")

        pygame.init()
        pygame.font.init()
        surf = get_font(24).render("你好", True, (255, 255, 255))
        self.assertGreater(surf.get_width(), 20)


if __name__ == "__main__":
    unittest.main()
