"""无 DISPLAY 的桌宠输出：渲染 PNG 并推送到 PetPreviewServer。"""

from __future__ import annotations

from src.adapters.screen.pet_preview_server import PetPreviewServer
from src.adapters.screen.pet_renderer import render_pet_png_bytes


class HeadlessPetDisplay:
    """实现 DisplayHardware 协议，供 ScreenDisplayAdapter 驱动。"""

    def __init__(
        self,
        preview_server: PetPreviewServer,
        *,
        size: tuple[int, int] = (480, 360),
    ) -> None:
        self._server = preview_server
        self._width, self._height = size
        self._agent_state = "idle"
        self._speak_text = ""
        self._focus_remaining = 0
        self._focus_duration = 0

    @property
    def size(self) -> tuple[int, int]:
        return self._width, self._height

    @property
    def fullscreen(self) -> bool:
        return False

    def start(self) -> None:
        self._push_frame()

    def stop(self) -> None:
        self._server.stop()

    def update(
        self,
        agent_state: str,
        speak_text: str = "",
        focus_remaining: int = 0,
        focus_duration: int = 0,
    ) -> None:
        self._agent_state = agent_state
        self._speak_text = speak_text
        self._focus_remaining = focus_remaining
        self._focus_duration = focus_duration
        self._push_frame()

    def _push_frame(self) -> None:
        png = render_pet_png_bytes(
            agent_state=self._agent_state,
            speak_text=self._speak_text,
            focus_remaining=self._focus_remaining,
            focus_duration=self._focus_duration,
            size=(self._width, self._height),
        )
        self._server.set_frame(png, state_label=self._agent_state)
