"""Pygame 中文字体：默认 Font(None) 仅含 ASCII，需加载系统 CJK TTF。"""

from __future__ import annotations

import os
from functools import lru_cache

_CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)


def resolve_cjk_font_path() -> str | None:
    """Return first usable CJK font path, or None."""
    custom = os.environ.get("EMBED_DISPLAY_FONT", "").strip()
    if custom and os.path.isfile(custom):
        return custom
    for path in _CJK_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


@lru_cache(maxsize=32)
def get_font(size: int):
    """Cached pygame Font that supports Chinese; falls back to default."""
    import pygame

    size = max(12, int(size))
    path = resolve_cjk_font_path()
    if path:
        try:
            return pygame.font.Font(path, size)
        except (OSError, pygame.error):
            pass
    return pygame.font.Font(None, size)
