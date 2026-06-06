"""Pygame-based screen window for pet display."""

from __future__ import annotations

import os
import threading
import time

import pygame

from src.adapters.screen.pet_renderer import AgentState, draw_pet_frame

DEFAULT_WINDOW_SIZE = (400, 320)
BG_COLOR = (30, 30, 40)
FPS = 30


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
        if not os.environ.get("DISPLAY"):
            # VNC 默认 :1；SSH 无 DISPLAY 时 pygame 无法开窗口
            if os.path.exists("/tmp/.X11-unix/X1"):
                os.environ.setdefault("DISPLAY", ":1")
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
        self._fullscreen = _env_fullscreen() if fullscreen is None else bool(fullscreen)
        self._requested_size = size or _parse_size_env() or DEFAULT_WINDOW_SIZE

        try:
            if self._fullscreen:
                display_info = pygame.display.Info()
                w = max(display_info.current_w, 320)
                h = max(display_info.current_h, 240)
                self._screen = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            else:
                self._screen = pygame.display.set_mode(self._requested_size)
        except pygame.error:
            # 全屏/Info 失败时回退固定窗口，避免整栈退出
            self._fullscreen = False
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
        egl_errors = 0
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False

            with self._lock:
                self._draw()

            self._clock.tick(FPS)
            try:
                pygame.display.flip()
            except pygame.error as exc:
                egl_errors += 1
                if egl_errors == 1:
                    print(f"[ScreenWindow] 显示刷新失败（{exc}），桌宠画面已停用。", flush=True)
                if egl_errors >= 3:
                    self._running = False
                time.sleep(0.05)

    def _dim(self, fraction: float) -> int:
        """Scale a layout unit against the shorter screen edge."""
        return max(1, int(min(self._width, self._height) * fraction))

    def _draw(self) -> None:
        """Draw current frame."""
        draw_pet_frame(
            self._screen,
            width=self._width,
            height=self._height,
            agent_state=self._agent_state,
            speak_text=self._speak_text,
            focus_remaining=self._focus_remaining,
            focus_duration=self._focus_duration,
        )
