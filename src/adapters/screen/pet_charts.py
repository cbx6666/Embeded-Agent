"""透明角落迷你图表（像素折线，无背景框）。"""

from __future__ import annotations

from typing import Sequence

from src.adapters.screen.pixel_font import blit_pixel_text

_LABEL = (100, 180, 140)
_LINE_EMOTION = (90, 200, 255)
_LINE_FATIGUE = (255, 160, 90)
_DOT = (200, 220, 200)


def draw_corner_sparkline(
    surface,
    rect: tuple[int, int, int, int],
    *,
    label: str,
    timeline: Sequence[tuple[int, float]],
    y_max: float,
    line_color: tuple[int, int, int],
    font_size: int,
) -> None:
    x, y, w, h = rect
    blit_pixel_text(surface, f"// {label}", (x, y), font_size, _LABEL, anchor="topleft")

    plot_y = y + font_size + 4
    plot_h = max(16, h - font_size - 8)
    plot = (x, plot_y, w, plot_h)

    if not timeline:
        blit_pixel_text(surface, "..", (x, plot_y + 2), font_size, (80, 90, 100), anchor="topleft")
        return

    import pygame

    values = [v for _, v in timeline]
    ymax = max(y_max, max(values) if values else 1.0)
    px0, py0, pw, ph = plot
    pts: list[tuple[int, int]] = []
    count = len(timeline)
    for i, (_, val) in enumerate(timeline):
        px = px0 + int(i * max(1, pw - 1) / max(1, count - 1))
        py = py0 + ph - 1 - int((val / ymax) * max(1, ph - 2))
        pts.append((px, py))

    if len(pts) >= 2:
        pygame.draw.lines(surface, line_color, False, pts, 1)
    for px, py in pts[-8:]:
        pygame.draw.rect(surface, _DOT, (px, py, 2, 2))


def draw_corner_pie_hint(
    surface,
    pos: tuple[int, int],
    *,
    label: str,
    data: dict[str, int],
    font_size: int,
    anchor: str = "topleft",
) -> None:
    """用代码风文本行代替饼图（透明、简约）。"""
    x, y = pos
    blit_pixel_text(surface, f"// {label}", (x, y), font_size, _LABEL, anchor=anchor)
    total = sum(data.values()) or 1
    line_y = y + font_size + 2
    for key, val in sorted(data.items())[:4]:
        pct = int(100 * val / total)
        blit_pixel_text(
            surface, f"{key}:{pct}%", (x, line_y), font_size - 2, (160, 175, 195), anchor=anchor,
        )
        line_y += font_size
