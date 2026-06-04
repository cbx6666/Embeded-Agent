"""Pygame-based screen window for pet display."""

from __future__ import annotations

import os
import threading
from enum import Enum

import pygame

DEFAULT_WINDOW_SIZE = (400, 320)
BG_COLOR = (30, 30, 40)
FPS = 30


class AgentState(Enum):
    """Agent display states."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOCUS_MODE = "focus_mode"


def _parse_size_env() -> tuple[int, int] | None:
    raw = (os.environ.get("EMBED_SCREEN_SIZE") or "").strip().lower()
    if not raw:
        return None
    if "x" in raw:
        w, _, h = raw.partition("x")
        try:
            return int(w), int(h)
        except ValueError:
            return None
    return None


def _env_fullscreen() -> bool:
    return os.environ.get("EMBED_SCREEN_FULLSCREEN", "").strip().lower() in {"1", "true", "yes", "on"}


def parse_screen_size_arg(raw: str | None) -> tuple[int, int] | None:
    """Parse ``1920x1080`` style CLI size."""
    if not raw:
        return None
    text = raw.strip().lower()
    if "x" not in text:
        return None
    w, _, h = text.partition("x")
    try:
        width, height = int(w), int(h)
    except ValueError:
        return None
    if width < 160 or height < 120:
        return None
    return width, height


def create_screen_window(
    *,
    fullscreen: bool = False,
    size: tuple[int, int] | None = None,
    size_arg: str | None = None,
) -> ScreenWindow:
    """Factory used by ``main`` and test scripts."""
    resolved = size or parse_screen_size_arg(size_arg)
    return ScreenWindow(fullscreen=fullscreen, size=resolved)


class ScreenWindow:
    """Pygame window for displaying pet and agent status.

    Supports fixed window, custom size, or fullscreen on the current display
    (HDMI / VNC ``DISPLAY=:1`` etc.). Layout scales with resolution.
    """

    def __init__(
        self,
        *,
        fullscreen: bool | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        pygame.init()
        self._fullscreen = _env_fullscreen() if fullscreen is None else bool(fullscreen)
        self._requested_size = size or _parse_size_env() or DEFAULT_WINDOW_SIZE

        if self._fullscreen:
            display_info = pygame.display.Info()
            w = max(display_info.current_w, 320)
            h = max(display_info.current_h, 240)
            self._screen = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
        else:
            self._screen = pygame.display.set_mode(self._requested_size)

        self._width, self._height = self._screen.get_size()
        pygame.display.set_caption("Embeded-Agent Pet")
        self._clock = pygame.time.Clock()
        self._running = False
        self._lock = threading.Lock()

        self._agent_state = AgentState.IDLE
        self._speak_text = ""
        self._focus_remaining = 0
        self._focus_duration = 0
        self._mouth_open = False

    @property
    def size(self) -> tuple[int, int]:
        return self._width, self._height

    @property
    def fullscreen(self) -> bool:
        return self._fullscreen

    def start(self) -> None:
        """Start the pygame event loop in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="ScreenWindow", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the pygame event loop."""
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2)
        try:
            pygame.quit()
        except Exception:
            pass

    def update(
        self,
        agent_state: str,
        speak_text: str = "",
        focus_remaining: int = 0,
        focus_duration: int = 0,
    ) -> None:
        """Update display state (thread-safe)."""
        with self._lock:
            try:
                self._agent_state = AgentState(agent_state)
            except ValueError:
                self._agent_state = AgentState.IDLE
            self._speak_text = speak_text
            self._focus_remaining = focus_remaining
            self._focus_duration = focus_duration

    def _run_loop(self) -> None:
        """Main pygame event loop."""
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False

            with self._lock:
                self._draw()

            self._clock.tick(FPS)
            pygame.display.flip()

    def _dim(self, fraction: float) -> int:
        """Scale a layout unit against the shorter screen edge."""
        return max(1, int(min(self._width, self._height) * fraction))

    def _draw(self) -> None:
        """Draw current frame."""
        self._screen.fill(BG_COLOR)
        state = self._agent_state

        self._draw_face(state)

        if self._speak_text:
            self._draw_text(
                self._speak_text,
                (self._width // 2, int(self._height * 0.72)),
                size=self._dim(0.045),
            )

        if state == AgentState.FOCUS_MODE and self._focus_duration > 0:
            self._draw_focus_timer()

        self._draw_text(
            state.value.upper(),
            (self._width // 2, int(self._height * 0.08)),
            size=self._dim(0.05),
        )

    def _draw_face(self, state: AgentState) -> None:
        """Draw simple face: two eyes + one mouth."""
        cx = self._width // 2
        cy = int(self._height * 0.38)
        eye_spacing = self._dim(0.125)
        eye_y = cy - self._dim(0.03)
        eye_radius = self._dim(0.03)

        pygame.draw.circle(self._screen, (255, 255, 255), (cx - eye_spacing, eye_y), eye_radius)
        pygame.draw.circle(self._screen, (255, 255, 255), (cx + eye_spacing, eye_y), eye_radius)

        pupil_color = (50, 50, 50)
        if state == AgentState.LISTENING:
            pupil_r = max(4, eye_radius * 7 // 12)
            pygame.draw.circle(self._screen, pupil_color, (cx - eye_spacing, eye_y), pupil_r)
            pygame.draw.circle(self._screen, pupil_color, (cx + eye_spacing, eye_y), pupil_r)
        elif state == AgentState.THINKING:
            pupil_r = max(3, eye_radius // 3)
            pygame.draw.circle(self._screen, pupil_color, (cx - eye_spacing, eye_y), pupil_r)
            pygame.draw.circle(self._screen, pupil_color, (cx + eye_spacing, eye_y), pupil_r)
        elif state == AgentState.SPEAKING:
            pupil_r = max(4, eye_radius // 2)
            pygame.draw.circle(self._screen, pupil_color, (cx - eye_spacing, eye_y), pupil_r)
            pygame.draw.circle(self._screen, pupil_color, (cx + eye_spacing, eye_y), pupil_r)
            self._draw_mouth(cx, cy + self._dim(0.12), open=True)
            return
        elif state == AgentState.FOCUS_MODE:
            pupil_r = max(3, eye_radius * 5 // 12)
            pygame.draw.circle(self._screen, pupil_color, (cx - eye_spacing, eye_y), pupil_r)
            pygame.draw.circle(self._screen, pupil_color, (cx + eye_spacing, eye_y), pupil_r)
        else:
            pupil_r = max(4, eye_radius // 2)
            pygame.draw.circle(self._screen, pupil_color, (cx - eye_spacing, eye_y), pupil_r)
            pygame.draw.circle(self._screen, pupil_color, (cx + eye_spacing, eye_y), pupil_r)

        self._draw_mouth(cx, cy + self._dim(0.12), open=False)

    def _draw_mouth(self, cx: int, cy: int, open: bool) -> None:
        """Draw mouth as a line or open oval."""
        mouth_w = self._dim(0.075)
        mouth_h = max(8, self._dim(0.035))
        line_w = max(2, self._dim(0.008))
        if open:
            pygame.draw.ellipse(
                self._screen,
                (255, 100, 100),
                (cx - mouth_w, cy - mouth_h // 2, mouth_w * 2, mouth_h),
            )
        else:
            pygame.draw.line(
                self._screen,
                (255, 255, 255),
                (cx - mouth_w, cy),
                (cx + mouth_w, cy),
                line_w,
            )

    def _draw_text(self, text: str, pos: tuple[int, int], size: int = 24) -> None:
        """Draw text at position."""
        font = pygame.font.Font(None, max(12, size))
        surface = font.render(text, True, (255, 255, 255))
        rect = surface.get_rect(center=pos)
        self._screen.blit(surface, rect)

    def _draw_focus_timer(self) -> None:
        """Draw focus countdown."""
        if self._focus_duration <= 0:
            return
        minutes = self._focus_remaining // 60
        seconds = self._focus_remaining % 60
        time_str = f"{minutes:02d}:{seconds:02d}"
        self._draw_text(time_str, (self._width // 2, int(self._height * 0.82)), size=self._dim(0.12))

        progress = self._focus_remaining / self._focus_duration if self._focus_duration > 0 else 0
        bar_width = int(self._width * 0.5)
        bar_height = max(8, self._dim(0.025))
        bar_x = (self._width - bar_width) // 2
        bar_y = int(self._height * 0.9)
        pygame.draw.rect(self._screen, (60, 60, 80), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(
            self._screen,
            (100, 200, 100),
            (bar_x, bar_y, int(bar_width * progress), bar_height),
        )
