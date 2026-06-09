from __future__ import annotations

import os
import unittest


class PixelFontTest(unittest.TestCase):
    def test_ascii_uses_mono_not_boxes(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        from src.adapters.screen.pixel_font import render_pixel_text

        pygame.init()
        pygame.font.init()
        surf = render_pixel_text("// sensor TEMP 26.8C", 14, (200, 255, 200))
        self.assertGreater(surf.get_width(), 80)
        # 方框字体会让有效像素极少；正常英文应有足够宽度
        self.assertGreater(surf.get_height(), 8)

    def test_mixed_en_zh(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        from src.adapters.screen.cjk_font import resolve_cjk_font_path
        from src.adapters.screen.pixel_font import render_pixel_text

        if resolve_cjk_font_path() is None:
            self.skipTest("no CJK font")

        pygame.init()
        pygame.font.init()
        surf = render_pixel_text("> pet: 你好世界", 14, (255, 255, 255))
        self.assertGreater(surf.get_width(), 60)


if __name__ == "__main__":
    unittest.main()
