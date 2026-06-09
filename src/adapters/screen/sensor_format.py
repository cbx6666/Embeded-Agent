"""传感器读数格式化（代码风标签，数值缺失时回退 level）。"""

from __future__ import annotations

from src.adapters.screen.pet_display_context import PetDisplayContext


def format_sensor_lines(ctx: PetDisplayContext) -> list[str]:
    lines: list[str] = []

    if ctx.temperature_c is not None:
        lines.append(f"TEMP {ctx.temperature_c:.1f}C")
    elif ctx.temperature_level:
        lines.append(f"TEMP {ctx.temperature_level}")

    if ctx.humidity_pct is not None:
        lines.append(f"HUM  {ctx.humidity_pct:.0f}%")
    elif ctx.humidity_level:
        lines.append(f"HUM  {ctx.humidity_level}")

    if ctx.light_lux is not None:
        lines.append(f"LUX  {ctx.light_lux}")
    elif ctx.light_level:
        lines.append(f"LUX  {ctx.light_level}")

    if ctx.noise_db is not None:
        lines.append(f"NOISE {ctx.noise_db}dB")
    elif ctx.noise_level:
        lines.append(f"NOISE {ctx.noise_level}")

    if not lines:
        lines.append("SENSOR --")
    return lines


def format_user_lines(ctx: PetDisplayContext) -> list[str]:
    emo = ctx.emotion or "neutral"
    fat = ctx.fatigue or "none"
    lines = [f"EMO  {emo}", f"FAT  {fat}"]
    if ctx.emotion_confidence is not None:
        lines.append(f"EMO% {int(ctx.emotion_confidence * 100)}")
    if ctx.fatigue_confidence is not None:
        lines.append(f"FAT% {int(ctx.fatigue_confidence * 100)}")
    return lines
