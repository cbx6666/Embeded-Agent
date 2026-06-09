from __future__ import annotations

"""
Pygame-based desktop pet renderer.

Renders a cute desktop pet character with expression animations, speech bubbles,
and background light effects. The window is transparent, always-on-top, and
frameless — appearing as a desktop companion.

Expression set: neutral, happy, sad, thinking, surprised, sleepy, excited, annoyed
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import pygame

from src.adapters.screen.cjk_font import get_font

# ── layout constants ───────────────────────────────────────────

PET_SIZE = 180  # base diameter
BUBBLE_MAX_WIDTH = 260
BUBBLE_FONT_SIZE = 16
BUBBLE_PADDING = 14
BUBBLE_DURATION_MS = 6000  # default text bubble lifetime
LIGHT_FADE_MS = 800

# ── colour palette ────────────────────────────────────────────

BODY_COLOR = (255, 220, 180)  # warm cream
BODY_OUTLINE = (180, 140, 100)
EYE_COLOR = (60, 40, 30)
MOUTH_COLOR = (60, 40, 30)
BLUSH_COLOR = (255, 160, 160)
BUBBLE_BG = (255, 255, 255, 230)
BUBBLE_OUTLINE = (200, 200, 210)
LIGHT_COLORS: dict[str, tuple[int, int, int]] = {
    "on": (255, 240, 200),
    "warm": (255, 200, 140),
    "cool": (180, 210, 255),
    "alert": (255, 140, 140),
}

EXPRESSION_CONFIGS: dict[str, dict[str, Any]] = {
    "neutral": {
        "eye_h": 0.28,
        "eye_w": 0.18,
        "eye_curve": 0.0,
        "pupil_r": 0.07,
        "mouth_type": "line",
        "mouth_val": 0.08,
        "brow_angle": 0.0,
        "blush": 0.0,
    },
    "happy": {
        "eye_h": 0.10,
        "eye_w": 0.20,
        "eye_curve": 0.06,
        "pupil_r": 0.06,
        "mouth_type": "smile",
        "mouth_val": 0.25,
        "brow_angle": 0.0,
        "blush": 0.7,
    },
    "sad": {
        "eye_h": 0.22,
        "eye_w": 0.16,
        "eye_curve": -0.04,
        "pupil_r": 0.08,
        "mouth_type": "frown",
        "mouth_val": 0.12,
        "brow_angle": -0.15,
        "blush": 0.2,
    },
    "thinking": {
        "eye_h": 0.20,
        "eye_w": 0.16,
        "eye_curve": 0.0,
        "pupil_r": 0.06,
        "mouth_type": "dot",
        "mouth_val": 0.06,
        "brow_angle": 0.10,
        "blush": 0.0,
    },
    "surprised": {
        "eye_h": 0.32,
        "eye_w": 0.20,
        "eye_curve": 0.0,
        "pupil_r": 0.05,
        "mouth_type": "open",
        "mouth_val": 0.20,
        "brow_angle": 0.0,
        "blush": 0.0,
    },
    "sleepy": {
        "eye_h": 0.06,
        "eye_w": 0.20,
        "eye_curve": 0.03,
        "pupil_r": 0.06,
        "mouth_type": "yawn",
        "mouth_val": 0.14,
        "brow_angle": 0.0,
        "blush": 0.3,
    },
    "excited": {
        "eye_h": 0.26,
        "eye_w": 0.22,
        "eye_curve": 0.0,
        "pupil_r": 0.05,
        "mouth_type": "smile",
        "mouth_val": 0.28,
        "brow_angle": 0.10,
        "blush": 0.9,
    },
    "annoyed": {
        "eye_h": 0.18,
        "eye_w": 0.17,
        "eye_curve": 0.0,
        "pupil_r": 0.07,
        "mouth_type": "flat",
        "mouth_val": 0.06,
        "brow_angle": -0.20,
        "blush": 0.0,
    },
}


@dataclass
class Bubble:
    text: str = ""
    appeared_at: float = 0.0
    duration_ms: int = BUBBLE_DURATION_MS

    @property
    def alive(self) -> bool:
        if not self.text:
            return False
        return (time.time() * 1000 - self.appeared_at) < self.duration_ms


@dataclass
class LightState:
    state: str = "off"
    color: str = ""
    pattern: str = ""
    brightness: int = 128
    started_at: float = 0.0
    duration_ms: int = 0
    _prev_color: tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))

    @property
    def current_color(self) -> tuple[int, int, int]:
        rgb = LIGHT_COLORS.get(self.state, (0, 0, 0))
        if self.color:
            try:
                rgb = _parse_color(self.color)
            except ValueError:
                rgb = LIGHT_COLORS.get(self.state, (0, 0, 0))
        elapsed = time.time() * 1000 - self.started_at
        if self.duration_ms > 0 and elapsed > self.duration_ms:
            return self._prev_color
        alpha = min(1.0, elapsed / LIGHT_FADE_MS) if self._prev_color != rgb else 1.0
        r = int(self._prev_color[0] + (rgb[0] - self._prev_color[0]) * alpha)
        g = int(self._prev_color[1] + (rgb[1] - self._prev_color[1]) * alpha)
        b = int(self._prev_color[2] + (rgb[2] - self._prev_color[2]) * alpha)
        return (r, g, b)

    @property
    def active(self) -> bool:
        if self.state == "off":
            return False
        if self.duration_ms <= 0:
            return True
        return (time.time() * 1000 - self.started_at) < self.duration_ms


def _parse_color(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PetRenderer:
    """Desktop pet window using Pygame."""

    def __init__(
        self,
        width: int = 320,
        height: int = 360,
        x: int = 100,
        y: int = 100,
        scale: float = 1.0,
        fps: int = 30,
    ) -> None:
        pygame.init()
        self.width = int(width * scale)
        self.height = int(height * scale)
        self.scale = scale
        self.fps = fps
        self._running = False
        self._clock = pygame.time.Clock()

        self._expression: str = "neutral"
        self._prev_expression: str = "neutral"
        self._expression_start: float = time.time()
        self._expression_duration: float = 0.5  # morph seconds

        self._bubble: Bubble = Bubble()
        self._light: LightState = LightState()
        self._bob_phase: float = 0.0

        # command queue (thread-safe via deque + GIL)
        self._cmd_queue: deque[DisplayCommand] = deque()

        self._screen: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None

        # window position state
        self._pos = (x, y)
        self._dragging = False
        self._drag_offset = (0, 0)

        self._init_window()

    # ── public API ─────────────────────────────────────────────

    def push_command(self, cmd: DisplayCommand) -> None:
        self._cmd_queue.append(cmd)

    def set_expression(
        self, expression: str, duration_ms: Optional[int] = None
    ) -> None:
        if expression not in EXPRESSION_CONFIGS:
            expression = "neutral"
        self._prev_expression = self._expression
        self._expression = expression
        self._expression_start = time.time()
        self._expression_duration = (
            max(0.1, duration_ms / 1000.0) if duration_ms else 0.5
        )

    def show_bubble(self, text: str, duration_ms: Optional[int] = None) -> None:
        if not text.strip():
            return
        self._bubble = Bubble(
            text=text.strip(),
            appeared_at=time.time() * 1000,
            duration_ms=duration_ms or BUBBLE_DURATION_MS,
        )

    def set_light(
        self,
        state: str,
        color: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        prev_color = self._light.current_color if self._light.active else (0, 0, 0)
        self._light = LightState(
            state=state,
            color=color or "",
            started_at=time.time() * 1000,
            duration_ms=duration_ms or 0,
            _prev_color=prev_color,
        )

    def run(self) -> None:
        self._running = True
        while self._running:
            self._handle_events()
            self._process_queue()
            self._update()
            self._draw()
            self._clock.tick(self.fps)
        pygame.quit()

    def stop(self) -> None:
        self._running = False

    # ── internal ───────────────────────────────────────────────

    def _init_window(self) -> None:
        flags = pygame.NOFRAME | pygame.SRCALPHA
        self._screen = pygame.display.set_mode(
            (self.width, self.height), flags
        )
        pygame.display.set_caption("Desktop Pet")

        if hasattr(pygame, "_sdl2"):
            try:
                from pygame._sdl2 import Window

                sdl_window = Window.from_display_module()
                sdl_window.set_always_on_top(True)
            except Exception:
                pass

        try:
            self._font = get_font(BUBBLE_FONT_SIZE)
            self._small_font = get_font(BUBBLE_FONT_SIZE - 2)
        except Exception:
            self._font = None

        self._screen.fill((0, 0, 0, 0))

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                cx, cy = self._pet_center()
                if math.hypot(mx - cx, my - cy) < PET_SIZE * self.scale * 0.55:
                    self._dragging = True
                    wx, wy = pygame.display.get_window_position()
                    self._drag_offset = (mx - wx, my - wy)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._dragging = False
            elif event.type == pygame.MOUSEMOTION and self._dragging:
                mx, my = event.pos
                pygame.display.set_window_position(
                    (mx - self._drag_offset[0], my - self._drag_offset[1])
                )

    def _process_queue(self) -> None:
        while self._cmd_queue:
            cmd = self._cmd_queue.popleft()
            self._dispatch_command(cmd)

    def _dispatch_command(self, cmd: DisplayCommand) -> None:
        if cmd.type == "expression":
            self.set_expression(
                cmd.payload.get("expression", "neutral"),
                duration_ms=cmd.payload.get("duration_ms"),
            )
        elif cmd.type == "display":
            self.show_bubble(
                cmd.payload.get("text", ""),
                duration_ms=cmd.payload.get("duration_ms"),
            )
        elif cmd.type == "light":
            self.set_light(
                state=cmd.payload.get("state", "off"),
                color=cmd.payload.get("color"),
                duration_ms=cmd.payload.get("duration_ms"),
            )

    def _update(self) -> None:
        self._bob_phase += 0.04

    def _morph_progress(self) -> float:
        elapsed = time.time() - self._expression_start
        return _clamp(elapsed / max(0.05, self._expression_duration), 0.0, 1.0)

    def _get_config(self, key: str) -> float:
        prev = EXPRESSION_CONFIGS.get(self._prev_expression, EXPRESSION_CONFIGS["neutral"])
        curr = EXPRESSION_CONFIGS.get(self._expression, EXPRESSION_CONFIGS["neutral"])
        t = self._morph_progress()
        return _lerp(float(prev.get(key, 0)), float(curr.get(key, 0)), t)

    def _get_mouth_config(self) -> tuple[str, float]:
        """Return (mouth_type, mouth_val) with morphing for simple values."""
        # mouth_type snaps to current after 50% morph
        t = self._morph_progress()
        prev_cfg = EXPRESSION_CONFIGS.get(self._prev_expression, EXPRESSION_CONFIGS["neutral"])
        curr_cfg = EXPRESSION_CONFIGS.get(self._expression, EXPRESSION_CONFIGS["neutral"])
        mtype = curr_cfg["mouth_type"] if t > 0.5 else prev_cfg["mouth_type"]
        mval = _lerp(float(prev_cfg["mouth_val"]), float(curr_cfg["mouth_val"]), t)
        return mtype, mval

    def _pet_center(self) -> tuple[float, float]:
        return (self.width / 2, self.height * 0.45)

    # ── drawing ────────────────────────────────────────────────

    def _draw(self) -> None:
        if self._screen is None:
            return

        # transparent background
        self._screen.fill((0, 0, 0, 0))

        # light glow
        if self._light.active:
            lc = self._light.current_color
            glow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            cx, cy = self._pet_center()
            glow_r = int(PET_SIZE * self.scale * 0.9)
            for r in range(glow_r, glow_r - 60, -4):
                alpha = max(0, int(30 * (r / glow_r)))
                pygame.draw.circle(glow_surf, (*lc, alpha), (int(cx), int(cy)), r)
            self._screen.blit(glow_surf, (0, 0))

        self._draw_body()
        self._draw_expression()
        self._draw_bubble()

        pygame.display.flip()

    def _draw_body(self) -> None:
        cx, cy = self._pet_center()
        s = self.scale
        bob = math.sin(self._bob_phase) * 6 * s
        r = PET_SIZE * s / 2

        # shadow
        shadow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.ellipse(
            shadow_surf,
            (0, 0, 0, 25),
            (
                cx - r * 0.85,
                cy + r * 0.65,
                r * 1.7,
                r * 0.18,
            ),
        )
        self._screen.blit(shadow_surf, (0, int(bob)))

        body_cy = int(cy + bob)

        # ears / antenna
        ear_off = r * 0.55
        pygame.draw.circle(
            self._screen, BODY_COLOR,
            (int(cx - ear_off), int(body_cy - r * 0.75)), int(r * 0.22),
        )
        pygame.draw.circle(
            self._screen, BODY_COLOR,
            (int(cx + ear_off), int(body_cy - r * 0.75)), int(r * 0.22),
        )
        pygame.draw.circle(
            self._screen, BODY_OUTLINE,
            (int(cx - ear_off), int(body_cy - r * 0.75)), int(r * 0.22), 2,
        )
        pygame.draw.circle(
            self._screen, BODY_OUTLINE,
            (int(cx + ear_off), int(body_cy - r * 0.75)), int(r * 0.22), 2,
        )

        # body
        pygame.draw.circle(
            self._screen, BODY_COLOR,
            (int(cx), int(body_cy)), int(r),
        )
        pygame.draw.circle(
            self._screen, BODY_OUTLINE,
            (int(cx), int(body_cy)), int(r), 3,
        )

        # tiny feet
        foot_y = int(body_cy + r * 0.78)
        for fx in [cx - r * 0.35, cx + r * 0.35]:
            pygame.draw.ellipse(
                self._screen, BODY_COLOR,
                (fx - r * 0.18, foot_y - r * 0.08, r * 0.36, r * 0.18),
            )
            pygame.draw.ellipse(
                self._screen, BODY_OUTLINE,
                (fx - r * 0.18, foot_y - r * 0.08, r * 0.36, r * 0.18), 2,
            )

    def _draw_expression(self) -> None:
        cx, cy = self._pet_center()
        s = self.scale
        bob = math.sin(self._bob_phase) * 6 * s
        cy = int(cy + bob)
        r = PET_SIZE * s / 2

        eye_y = cy - r * 0.08
        eye_spacing = r * 0.32
        eye_w = r * self._get_config("eye_w")
        eye_h = r * self._get_config("eye_h")
        pupil_r = r * self._get_config("pupil_r")
        brow_angle = self._get_config("brow_angle")

        blush_alpha = self._get_config("blush")

        # blush
        if blush_alpha > 0.01:
            blush_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            blush_r = int(r * 0.18)
            for side in [-1, 1]:
                bx = int(cx + side * r * 0.45)
                by = int(eye_y + r * 0.14)
                pygame.draw.circle(
                    blush_surf,
                    (*BLUSH_COLOR, int(blush_alpha * 100)),
                    (bx, by), blush_r,
                )
            self._screen.blit(blush_surf, (0, 0))

        # eyebrows
        brow_y = eye_y - eye_h - r * 0.06
        brow_len = eye_w * 1.2
        for side in [-1, 1]:
            bx = cx + side * eye_spacing
            start = (int(bx - brow_len), int(brow_y - brow_angle * r * 0.3))
            end = (int(bx + brow_len), int(brow_y + brow_angle * r * 0.3))
            pygame.draw.line(self._screen, EYE_COLOR, start, end, max(2, int(3 * s)))

        # eyes
        eye_curve = self._get_config("eye_curve")
        for side in [-1, 1]:
            ex = int(cx + side * eye_spacing)
            points = _eye_polygon(
                ex, int(eye_y), int(eye_w), int(eye_h), eye_curve,
            )
            if len(points) >= 3:
                pygame.draw.polygon(self._screen, (255, 255, 255), points)
                pygame.draw.polygon(self._screen, EYE_COLOR, points, 2)

            # pupil
            px = ex + int(side * r * 0.03)
            pygame.draw.circle(
                self._screen, EYE_COLOR, (px, int(eye_y)), int(pupil_r),
            )
            # pupil highlight
            hl = max(1, int(pupil_r * 0.4))
            pygame.draw.circle(
                self._screen, (255, 255, 255),
                (px - hl, int(eye_y) - hl), hl,
            )

        # mouth
        mtype, mval = self._get_mouth_config()
        mouth_y = int(cy + r * 0.22)
        if mtype == "line":
            mw = int(r * mval)
            pygame.draw.line(
                self._screen, MOUTH_COLOR,
                (int(cx - mw), mouth_y), (int(cx + mw), mouth_y),
                max(2, int(2.5 * s)),
            )
        elif mtype == "smile":
            mw = int(r * 0.30)
            mh = int(r * mval)
            _draw_arc_smile(self._screen, cx, mouth_y, mw, mh, s, MOUTH_COLOR)
        elif mtype == "frown":
            mw = int(r * 0.28)
            mh = int(r * mval)
            _draw_arc_frown(self._screen, cx, mouth_y, mw, mh, s, MOUTH_COLOR)
        elif mtype == "open":
            r_m = int(r * mval * 0.6)
            pygame.draw.ellipse(
                self._screen, (80, 40, 40),
                (int(cx - r_m * 1.2), mouth_y - r_m // 2, int(r_m * 2.4), r_m),
            )
            pygame.draw.ellipse(
                self._screen, MOUTH_COLOR,
                (int(cx - r_m * 1.2), mouth_y - r_m // 2, int(r_m * 2.4), r_m), 2,
            )
        elif mtype == "dot":
            dot_r = int(r * mval)
            pygame.draw.circle(
                self._screen, MOUTH_COLOR,
                (int(cx + r * 0.12), mouth_y), dot_r,
            )
        elif mtype == "yawn":
            y_r = int(r * mval)
            pygame.draw.ellipse(
                self._screen, (80, 40, 40),
                (int(cx - y_r), mouth_y - y_r // 2, int(y_r * 2), y_r),
            )
            pygame.draw.ellipse(
                self._screen, MOUTH_COLOR,
                (int(cx - y_r), mouth_y - y_r // 2, int(y_r * 2), y_r), 2,
            )
        elif mtype == "flat":
            mw = int(r * 0.25)
            pygame.draw.line(
                self._screen, MOUTH_COLOR,
                (int(cx - mw), mouth_y + int(r * 0.02)),
                (int(cx + mw), mouth_y + int(r * 0.02)),
                max(2, int(2.5 * s)),
            )

    def _draw_bubble(self) -> None:
        if not self._bubble.alive:
            return

        cx, cy = self._pet_center()
        s = self.scale
        bob = math.sin(self._bob_phase) * 6 * s
        pet_top = int(cy + bob) - int(PET_SIZE * s / 2) - int(20 * s)

        text = self._bubble.text
        font = self._font or get_font(BUBBLE_FONT_SIZE)
        lines = _wrap_text(text, font, BUBBLE_MAX_WIDTH - BUBBLE_PADDING * 2)

        line_h = font.get_height() + 3
        bubble_w = min(
            BUBBLE_MAX_WIDTH,
            max(font.size(ln)[0] for ln in lines) + BUBBLE_PADDING * 2 + 10,
        )
        bubble_h = len(lines) * line_h + BUBBLE_PADDING * 2

        bubble_x = int(cx - bubble_w / 2)
        bubble_x = max(5, min(bubble_x, self.width - bubble_w - 5))
        bubble_y = max(5, pet_top - bubble_h)

        # bubble background
        bubble_surf = pygame.Surface((bubble_w, bubble_h), pygame.SRCALPHA)
        pygame.draw.rect(
            bubble_surf,
            BUBBLE_BG,
            (0, 0, bubble_w, bubble_h),
            border_radius=12,
        )
        pygame.draw.rect(
            bubble_surf,
            BUBBLE_OUTLINE,
            (0, 0, bubble_w, bubble_h),
            border_radius=12,
            width=1,
        )

        # tail
        tail_h = 10
        tail_x = int(cx - bubble_x)
        tail_x = _clamp(tail_x, 15, bubble_w - 15)
        pygame.draw.polygon(
            bubble_surf,
            BUBBLE_BG,
            [
                (tail_x - 6, bubble_h - 1),
                (tail_x + 6, bubble_h - 1),
                (tail_x, bubble_h + tail_h),
            ],
        )
        pygame.draw.polygon(
            bubble_surf,
            BUBBLE_OUTLINE,
            [
                (tail_x - 6, bubble_h - 1),
                (tail_x + 6, bubble_h - 1),
                (tail_x, bubble_h + tail_h),
            ],
            width=1,
        )

        for i, ln in enumerate(lines):
            txt_surf = font.render(ln, True, (40, 40, 50))
            bubble_surf.blit(
                txt_surf,
                (BUBBLE_PADDING, BUBBLE_PADDING + i * line_h),
            )

        # fade out near end
        remaining = (
            self._bubble.duration_ms
            - (time.time() * 1000 - self._bubble.appeared_at)
        )
        fade = _clamp(remaining / 400.0, 0.0, 1.0)
        bubble_surf.set_alpha(int(255 * fade))

        self._screen.blit(bubble_surf, (bubble_x, bubble_y))


# ── drawing helpers ────────────────────────────────────────────


def _eye_polygon(
    cx: int, cy: int, w: int, h: int, curve: float
) -> list[tuple[int, int]]:
    """Build eye polygon with optional upward/downward curve."""
    left = cx - w
    right = cx + w
    top = cy - h
    bottom = cy + h
    mid_y = cy + int(curve * h * 2)
    return [
        (left, top),
        (right, top),
        (right, bottom),
        (cx, mid_y),
        (left, bottom),
    ]


def _draw_arc_smile(
    surf: pygame.Surface,
    cx: float,
    cy: float,
    w: int,
    h: int,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a smiling arc using an ellipse clip."""
    # draw top half of ellipse
    rect = pygame.Rect(int(cx - w), int(cy - h), w * 2, h * 2)
    # use arc if available, otherwise approximate
    try:
        pygame.draw.arc(
            surf,
            color,
            rect,
            math.pi * 0.15,
            math.pi * 0.85,
            max(2, int(2.5 * scale)),
        )
    except Exception:
        pygame.draw.ellipse(surf, color, rect, max(2, int(2.5 * scale)))


def _draw_arc_frown(
    surf: pygame.Surface,
    cx: float,
    cy: float,
    w: int,
    h: int,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a frowning arc."""
    rect = pygame.Rect(int(cx - w), int(cy), w * 2, h * 2)
    try:
        pygame.draw.arc(
            surf, color, rect, math.pi * 1.15, math.pi * 1.85, max(2, int(2.5 * scale)),
        )
    except Exception:
        pygame.draw.ellipse(surf, color, rect, max(2, int(2.5 * scale)))


def _wrap_text(text: str, font: pygame.font.Font, max_w: int) -> list[str]:
    """Simple word-wrap for single-font rendering."""
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = cur + (" " if cur else "") + w
        if font.size(trial)[0] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


# ── re-export for type hints in other PC modules ──────────────

from src.adapters.usb_display.serial_protocol import DisplayCommand  # noqa: E402, F811
