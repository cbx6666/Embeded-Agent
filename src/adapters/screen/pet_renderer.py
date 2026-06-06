"""桌宠帧绘制（pygame Surface，可用于窗口或 headless PNG 导出）。"""

from __future__ import annotations

import io
import os
from enum import Enum

BG_COLOR = (30, 30, 40)


class AgentState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOCUS_MODE = "focus_mode"


def agent_state_from_str(name: str) -> AgentState:
    try:
        return AgentState(name)
    except ValueError:
        return AgentState.IDLE


def _dim(width: int, height: int, fraction: float) -> int:
    return max(1, int(min(width, height) * fraction))


def draw_pet_frame(
    surface,
    *,
    width: int,
    height: int,
    agent_state: AgentState,
    speak_text: str = "",
    focus_remaining: int = 0,
    focus_duration: int = 0,
) -> None:
    import pygame

    surface.fill(BG_COLOR)
    _draw_face(surface, width, height, agent_state)

    if speak_text:
        _draw_text(surface, speak_text, (width // 2, int(height * 0.72)), _dim(width, height, 0.045))

    if agent_state == AgentState.FOCUS_MODE and focus_duration > 0:
        _draw_focus_timer(surface, width, height, focus_remaining, focus_duration)

    _draw_text(
        surface,
        agent_state.value.upper(),
        (width // 2, int(height * 0.08)),
        _dim(width, height, 0.05),
    )


def _draw_face(surface, width: int, height: int, state: AgentState) -> None:
    import pygame

    cx = width // 2
    cy = int(height * 0.38)
    eye_spacing = _dim(width, height, 0.125)
    eye_y = cy - _dim(width, height, 0.03)
    eye_radius = _dim(width, height, 0.03)

    pygame.draw.circle(surface, (255, 255, 255), (cx - eye_spacing, eye_y), eye_radius)
    pygame.draw.circle(surface, (255, 255, 255), (cx + eye_spacing, eye_y), eye_radius)

    pupil_color = (50, 50, 50)
    if state == AgentState.LISTENING:
        pupil_r = max(4, eye_radius * 7 // 12)
    elif state == AgentState.THINKING:
        pupil_r = max(3, eye_radius // 3)
    elif state == AgentState.SPEAKING:
        pupil_r = max(4, eye_radius // 2)
        pygame.draw.circle(surface, pupil_color, (cx - eye_spacing, eye_y), pupil_r)
        pygame.draw.circle(surface, pupil_color, (cx + eye_spacing, eye_y), pupil_r)
        _draw_mouth(surface, width, height, cx, cy + _dim(width, height, 0.12), open_mouth=True)
        return
    elif state == AgentState.FOCUS_MODE:
        pupil_r = max(3, eye_radius * 5 // 12)
    else:
        pupil_r = max(4, eye_radius // 2)

    pygame.draw.circle(surface, pupil_color, (cx - eye_spacing, eye_y), pupil_r)
    pygame.draw.circle(surface, pupil_color, (cx + eye_spacing, eye_y), pupil_r)
    _draw_mouth(surface, width, height, cx, cy + _dim(width, height, 0.12), open_mouth=False)


def _draw_mouth(surface, width: int, height: int, cx: int, cy: int, *, open_mouth: bool) -> None:
    import pygame

    mouth_w = _dim(width, height, 0.075)
    mouth_h = max(8, _dim(width, height, 0.035))
    line_w = max(2, _dim(width, height, 0.008))
    if open_mouth:
        pygame.draw.ellipse(
            surface,
            (255, 100, 100),
            (cx - mouth_w, cy - mouth_h // 2, mouth_w * 2, mouth_h),
        )
    else:
        pygame.draw.line(
            surface,
            (255, 255, 255),
            (cx - mouth_w, cy),
            (cx + mouth_w, cy),
            line_w,
        )


def _draw_text(surface, text: str, pos: tuple[int, int], size: int) -> None:
    import pygame

    font = pygame.font.Font(None, max(12, size))
    rendered = font.render(text, True, (255, 255, 255))
    rect = rendered.get_rect(center=pos)
    surface.blit(rendered, rect)


def _draw_focus_timer(
    surface,
    width: int,
    height: int,
    focus_remaining: int,
    focus_duration: int,
) -> None:
    import pygame

    minutes = focus_remaining // 60
    seconds = focus_remaining % 60
    time_str = f"{minutes:02d}:{seconds:02d}"
    _draw_text(surface, time_str, (width // 2, int(height * 0.82)), _dim(width, height, 0.12))

    progress = focus_remaining / focus_duration if focus_duration > 0 else 0
    bar_width = int(width * 0.5)
    bar_height = max(8, _dim(width, height, 0.025))
    bar_x = (width - bar_width) // 2
    bar_y = int(height * 0.9)
    pygame.draw.rect(surface, (60, 60, 80), (bar_x, bar_y, bar_width, bar_height))
    pygame.draw.rect(
        surface,
        (100, 200, 100),
        (bar_x, bar_y, int(bar_width * progress), bar_height),
    )


def render_pet_png_bytes(
    *,
    agent_state: str,
    speak_text: str = "",
    focus_remaining: int = 0,
    focus_duration: int = 0,
    size: tuple[int, int] = (480, 360),
) -> bytes:
    """Headless 渲染 PNG（SDL dummy，无需 DISPLAY / VNC）。"""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()

    width, height = size
    surface = pygame.Surface((width, height))
    draw_pet_frame(
        surface,
        width=width,
        height=height,
        agent_state=agent_state_from_str(agent_state),
        speak_text=speak_text,
        focus_remaining=focus_remaining,
        focus_duration=focus_duration,
    )
    buf = io.BytesIO()
    pygame.image.save(surface, buf, "PNG")
    return buf.getvalue()
