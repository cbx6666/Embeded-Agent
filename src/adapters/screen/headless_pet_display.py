"""无 DISPLAY 的桌宠输出：渲染 PNG 并推送到 PetPreviewServer。"""

from __future__ import annotations

from src.adapters.screen.pet_display_context import PetDisplayContext
from src.adapters.screen.pet_preview_server import PetPreviewServer
from src.adapters.screen.pet_renderer import render_pet_png_bytes


class HeadlessPetDisplay:
    """实现 DisplayHardware 协议，供 ScreenDisplayAdapter 驱动。"""

    def __init__(
        self,
        preview_server: PetPreviewServer,
        *,
        size: tuple[int, int] = (960, 540),
    ) -> None:
        self._server = preview_server
        self._width, self._height = size
        self._context = PetDisplayContext()

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

    def update(self, context: PetDisplayContext) -> None:
        self._context = context
        self._push_frame()

    def _push_frame(self) -> None:
        ctx = self._context
        png = render_pet_png_bytes(
            agent_state=ctx.agent_state,
            speak_text=ctx.speak_text,
            focus_remaining=ctx.focus_remaining,
            focus_duration=ctx.focus_duration,
            size=(self._width, self._height),
            status_label=ctx.status_label,
            context=ctx,
        )
        self._server.set_frame(png, state_label=ctx.agent_state)
