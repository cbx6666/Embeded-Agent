"""像素风 / 代码风字体：英文等宽 + 中文分段混排。"""

from __future__ import annotations

import os
import re
from functools import lru_cache

_MONO_CANDIDATES = (
    "/usr/share/fonts/truetype/hack/Hack-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "C:/Windows/Fonts/consola.ttf",
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff，。！？、；：…（）【】《》「」『』￥％]")


def resolve_mono_font_path() -> str | None:
    custom = os.environ.get("EMBED_MONO_FONT", "").strip()
    if custom and os.path.isfile(custom):
        return custom
    for path in _MONO_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


@lru_cache(maxsize=32)
def get_mono_font(size: int):
    import pygame

    size = max(10, int(size))
    path = resolve_mono_font_path()
    if path:
        try:
            return pygame.font.Font(path, size)
        except (OSError, pygame.error):
            pass
    return pygame.font.SysFont("monospace", size)


@lru_cache(maxsize=32)
def get_cjk_font(size: int):
    from src.adapters.screen.cjk_font import get_font

    return get_font(max(12, int(size)))


def _is_cjk_char(ch: str) -> bool:
    return bool(ch) and _CJK_RE.match(ch) is not None


def _segment_text(text: str) -> list[tuple[str, bool]]:
    """按 ASCII / 中文切分，避免用 CJK 字体渲染英文时出现方框。"""
    if not text:
        return []
    parts: list[tuple[str, bool]] = []
    i = 0
    while i < len(text):
        cjk = _is_cjk_char(text[i])
        j = i + 1
        while j < len(text) and _is_cjk_char(text[j]) == cjk:
            j += 1
        parts.append((text[i:j], cjk))
        i = j
    return parts


def render_pixel_text(
    text: str,
    size: int,
    color: tuple[int, int, int],
) -> "object":
    """渲染文字：英文走等宽字体，中文走 CJK 字体，混排时横向拼接。"""
    import pygame

    size = max(12, int(size))
    segments = _segment_text(str(text))
    if not segments:
        return pygame.Surface((1, 1), pygame.SRCALPHA)

    surfaces = []
    total_w = 0
    max_h = 0
    for chunk, is_cjk in segments:
        if not chunk:
            continue
        font = get_cjk_font(size) if is_cjk else get_mono_font(size)
        surf = font.render(chunk, True, color)
        surfaces.append(surf)
        total_w += surf.get_width()
        max_h = max(max_h, surf.get_height())

    if len(surfaces) == 1:
        return surfaces[0]

    out = pygame.Surface((max(1, total_w), max(1, max_h)), pygame.SRCALPHA)
    x = 0
    for surf in surfaces:
        out.blit(surf, (x, 0))
        x += surf.get_width()
    return out


def blit_pixel_text(
    surface,
    text: str,
    pos: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
    *,
    anchor: str = "topleft",
) -> tuple[int, int, int, int]:
    rendered = render_pixel_text(text, size, color)
    rect = rendered.get_rect()
    anchors = {
        "topleft": lambda r, p: setattr(r, "topleft", p) or r,
        "topright": lambda r, p: setattr(r, "topright", p) or r,
        "center": lambda r, p: setattr(r, "center", p) or r,
        "midleft": lambda r, p: setattr(r, "midleft", p) or r,
        "midright": lambda r, p: setattr(r, "midright", p) or r,
        "bottomleft": lambda r, p: setattr(r, "bottomleft", p) or r,
        "bottomright": lambda r, p: setattr(r, "bottomright", p) or r,
    }
    anchors.get(anchor, anchors["topleft"])(rect, pos)
    surface.blit(rendered, rect)
    return rect
