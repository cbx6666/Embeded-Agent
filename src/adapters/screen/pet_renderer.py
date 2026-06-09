"""桌宠帧绘制：像素风 / 代码风布局，中心表情+对话，角落传感器与用户数据。"""

from __future__ import annotations

import io
import math
import os
import time
from enum import Enum

from src.adapters.screen.expression_styles import resolve_expression
from src.adapters.screen.pet_charts import draw_corner_pie_hint, draw_corner_sparkline
from src.adapters.screen.pet_display_context import PetDisplayContext
from src.adapters.screen.pixel_font import blit_pixel_text
from src.adapters.screen.sensor_format import format_sensor_lines, format_user_lines

BG_COLOR = (8, 12, 18)
GRID_COLOR = (20, 28, 38)
EYE_COLOR = (220, 230, 245)
MOUTH_COLOR = (200, 210, 225)

_COL_SENSOR = (100, 220, 160)
_COL_USER = (140, 190, 255)
_COL_SPEECH = (230, 235, 245)
_COL_DIM = (90, 100, 120)
_COL_STATE = (180, 200, 220)
_COL_FOCUS = (100, 230, 150)


class AgentState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOCUS_MODE = "focus_mode"


_STATE_LABELS: dict[AgentState, str] = {
    AgentState.IDLE: "IDLE",
    AgentState.LISTENING: "LISTEN",
    AgentState.THINKING: "THINK",
    AgentState.SPEAKING: "SPEAK",
    AgentState.FOCUS_MODE: "FOCUS",
}

_REASON_LABELS: dict[str, str] = {
    "rest_reminder": "REST",
    "distraction_reminder": "FOCUS_ALERT",
    "environment_warning": "ENV_WARN",
    "emotion_reminder": "EMO_CARE",
    "focus_complete": "FOCUS_DONE",
    "reduce_reminder_frequency": "REMIND_CFG",
}


def state_label_for(agent_state: AgentState, *, status_label: str = "") -> str:
    if status_label.strip():
        return status_label.strip()
    return _STATE_LABELS.get(agent_state, "IDLE")


def reason_status_label(reason: str) -> str:
    return _REASON_LABELS.get(reason.strip(), "NOTICE")


def agent_state_from_str(name: str) -> AgentState:
    try:
        return AgentState(name)
    except ValueError:
        return AgentState.IDLE


def _dim(width: int, height: int, fraction: float) -> int:
    return max(1, int(min(width, height) * fraction))


def _font_size(width: int, height: int) -> int:
    return max(12, _dim(width, height, 0.028))


def _listening_dots() -> str:
    n = int(time.time() * 2) % 4
    return "> listen" + "." * n


def _speech_lines(ctx: PetDisplayContext) -> list[tuple[str, tuple[int, int, int]]]:
    mode = ctx.speech_mode
    if mode == "listening":
        return [(_listening_dots(), (100, 210, 255))]
    if mode == "recognizing":
        dots = int(time.time() * 3) % 4
        return [(f"> asr{'.' * dots}", (150, 200, 255))]
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if mode == "user" and ctx.user_speech_text:
        lines.append((f"> user: {ctx.user_speech_text}", (180, 220, 255)))
    if mode == "agent" and ctx.agent_speech_text:
        lines.append((f"> pet:  {ctx.agent_speech_text}", (255, 220, 150)))
    elif ctx.speak_text and not lines:
        lines.append((f"> pet:  {ctx.speak_text}", _COL_SPEECH))
    return lines


def draw_pet_frame(
    surface,
    *,
    width: int,
    height: int,
    agent_state: AgentState | None = None,
    speak_text: str = "",
    focus_remaining: int = 0,
    focus_duration: int = 0,
    status_label: str = "",
    context: PetDisplayContext | None = None,
) -> None:
    if context is None:
        ctx = PetDisplayContext.from_legacy(
            agent_state=agent_state.value if agent_state else "idle",
            speak_text=speak_text,
            focus_remaining=focus_remaining,
            focus_duration=focus_duration,
            status_label=status_label,
        )
    else:
        ctx = context
        if agent_state is None:
            agent_state = agent_state_from_str(ctx.agent_state)

    ctx.expression = ctx.expression or resolve_expression(
        agent_state=ctx.agent_state,
        user_emotion=ctx.emotion,
    )

    surface.fill(BG_COLOR)
    fs = _font_size(width, height)
    margin = max(10, fs)

    _draw_code_grid(surface, width, height)
    _draw_corner_sensors(surface, margin, margin, fs, ctx)
    _draw_corner_user(surface, width - margin, margin, fs, ctx)
    _draw_corner_stats(surface, width, height, margin, fs, ctx)
    _draw_center_stage(surface, width, height, ctx, agent_state, fs)

    if agent_state == AgentState.FOCUS_MODE and ctx.focus_duration > 0:
        _draw_focus_timer(surface, width, height, ctx.focus_remaining, ctx.focus_duration, fs)


def _draw_code_grid(surface, width: int, height: int) -> None:
    import pygame

    step = 24
    for x in range(0, width, step):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, height))
    for y in range(0, height, step):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (width, y))


def _draw_corner_sensors(surface, x: int, y: int, fs: int, ctx: PetDisplayContext) -> None:
    blit_pixel_text(surface, "// sensor", (x, y), fs, _COL_DIM, anchor="topleft")
    for i, line in enumerate(format_sensor_lines(ctx)):
        blit_pixel_text(surface, line, (x, y + fs + 4 + i * (fs + 2)), fs, _COL_SENSOR, anchor="topleft")


def _draw_corner_user(surface, x: int, y: int, fs: int, ctx: PetDisplayContext) -> None:
    blit_pixel_text(surface, "// user", (x, y), fs, _COL_DIM, anchor="topright")
    for i, line in enumerate(format_user_lines(ctx)):
        blit_pixel_text(surface, line, (x, y + fs + 4 + i * (fs + 2)), fs, _COL_USER, anchor="topright")


def _draw_corner_stats(surface, width: int, height: int, margin: int, fs: int, ctx: PetDisplayContext) -> None:
    spark_h = max(36, fs * 3)
    bottom = height - margin

    draw_corner_sparkline(
        surface, (margin, bottom - spark_h, width // 3, spark_h),
        label="emo_trend", timeline=ctx.emotion_timeline, y_max=3.0,
        line_color=(90, 200, 255), font_size=fs - 2,
    )
    draw_corner_sparkline(
        surface, (width - margin - width // 3, bottom - spark_h, width // 3, spark_h),
        label="fat_trend", timeline=ctx.fatigue_timeline, y_max=3.0,
        line_color=(255, 160, 90), font_size=fs - 2,
    )

    mid_y = bottom - spark_h - fs * 3
    draw_corner_pie_hint(
        surface, (margin, mid_y), label="emo_dist",
        data=ctx.emotion_pie, font_size=fs - 2,
    )
    draw_corner_pie_hint(
        surface, (width - margin, mid_y), label="fat_dist",
        data=ctx.fatigue_pie, font_size=fs - 2, anchor="topright",
    )


def _draw_center_stage(
    surface, width: int, height: int,
    ctx: PetDisplayContext, agent_state: AgentState, fs: int,
) -> None:
    cx, cy = width // 2, int(height * 0.36)
    unit = max(10, min(width, height) // 12)
    _draw_expression_only(
        surface, cx, cy, unit,
        expression=ctx.expression,
        agent_state=ctx.agent_state,
    )

    state_txt = state_label_for(agent_state, status_label=ctx.status_label)
    tag = f"[{state_txt}] {ctx.expression}"
    blit_pixel_text(surface, tag, (cx, cy + unit * 4), fs, _COL_STATE, anchor="center")

    speech = _speech_lines(ctx)
    speech_y = cy + unit * 5 + fs
    for i, (line, color) in enumerate(speech[:3]):
        blit_pixel_text(
            surface, line, (cx, speech_y + i * (fs + 4)), fs, color, anchor="center",
        )


def _draw_expression_only(
    surface,
    cx: int,
    cy: int,
    unit: int,
    *,
    expression: str,
    agent_state: str,
) -> None:
    """仅绘制五官：两眼 + 嘴，无脸框、无装饰。"""
    import pygame

    lw = max(2, unit // 6)
    eye_y = cy - unit
    eye_dx = unit * 2
    eye_s = max(6, unit // 2)
    mouth_y = cy + unit
    mouth_w = unit * 2

    # 眼
    if expression == "sleepy" or agent_state == "focus_mode":
        for ex in (cx - eye_dx, cx + eye_dx):
            pygame.draw.line(surface, EYE_COLOR, (ex - eye_s, eye_y), (ex + eye_s, eye_y), lw)
    elif expression == "happy" or agent_state in {"listening", "speaking"}:
        for ex in (cx - eye_dx, cx + eye_dx):
            pygame.draw.rect(surface, EYE_COLOR, (ex - eye_s // 2, eye_y - eye_s // 2, eye_s, eye_s))
    elif expression == "angry":
        for ex in (cx - eye_dx, cx + eye_dx):
            pygame.draw.rect(surface, EYE_COLOR, (ex - eye_s // 2, eye_y - eye_s // 2, eye_s, eye_s))
    else:
        # idle / neutral：两眼方块
        for ex in (cx - eye_dx, cx + eye_dx):
            pygame.draw.rect(surface, EYE_COLOR, (ex - eye_s // 2, eye_y - eye_s // 2, eye_s, eye_s))

    # 嘴
    if agent_state == "speaking":
        bob = int(math.sin(time.time() * 10) * 2)
        h = max(4, unit // 3)
        pygame.draw.rect(
            surface, MOUTH_COLOR,
            (cx - mouth_w // 4, mouth_y - h // 2 + bob, mouth_w // 2, h),
        )
    elif agent_state == "thinking":
        pygame.draw.rect(surface, MOUTH_COLOR, (cx - lw, mouth_y - lw, lw * 2, lw * 2))
    elif expression == "happy":
        pygame.draw.line(surface, MOUTH_COLOR, (cx - mouth_w // 2, mouth_y), (cx, mouth_y + unit // 3), lw)
        pygame.draw.line(surface, MOUTH_COLOR, (cx, mouth_y + unit // 3), (cx + mouth_w // 2, mouth_y), lw)
    elif expression == "angry":
        pygame.draw.line(surface, MOUTH_COLOR, (cx - mouth_w // 2, mouth_y + unit // 4), (cx + mouth_w // 2, mouth_y), lw)
    elif expression == "sleepy":
        pygame.draw.line(surface, MOUTH_COLOR, (cx - mouth_w // 3, mouth_y), (cx + mouth_w // 3, mouth_y), lw)
    else:
        # idle：一横杠嘴
        pygame.draw.line(surface, MOUTH_COLOR, (cx - mouth_w // 2, mouth_y), (cx + mouth_w // 2, mouth_y), lw)


def _draw_focus_timer(
    surface, width: int, height: int,
    focus_remaining: int, focus_duration: int, fs: int,
) -> None:
    import pygame

    minutes = focus_remaining // 60
    seconds = focus_remaining % 60
    blit_pixel_text(
        surface, f"TIMER {minutes:02d}:{seconds:02d}",
        (width // 2, height - fs * 2), fs, _COL_FOCUS, anchor="center",
    )
    progress = focus_remaining / focus_duration if focus_duration > 0 else 0
    bar_w = width // 3
    bar_x = (width - bar_w) // 2
    bar_y = height - fs - 6
    pygame.draw.rect(surface, (40, 50, 60), (bar_x, bar_y, bar_w, 4))
    fill = int(bar_w * progress)
    if fill > 0:
        pygame.draw.rect(surface, _COL_FOCUS, (bar_x, bar_y, fill, 4))


def render_pet_png_bytes(
    *,
    agent_state: str,
    speak_text: str = "",
    focus_remaining: int = 0,
    focus_duration: int = 0,
    size: tuple[int, int] = (960, 540),
    status_label: str = "",
    context: PetDisplayContext | None = None,
) -> bytes:
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
        status_label=status_label,
        context=context,
    )
    buf = io.BytesIO()
    pygame.image.save(surface, buf, "PNG")
    return buf.getvalue()
