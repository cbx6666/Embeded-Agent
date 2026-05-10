from __future__ import annotations

from typing import Any

from src.agent.action.action_model import Action


def _build_action(action_type: str, **payload: Any) -> Action:
    """构造标准 Action，并自动剔除值为 None 的字段。"""
    normalized_payload = {key: value for key, value in payload.items() if value is not None}
    return Action(type=action_type, payload=normalized_payload)


def speak(
    text: str,
    *,
    kind: str | None = None,
    level: str | None = None,
    reason: str | None = None,
    interrupt: bool | None = None,
    voice: str | None = None,
    volume: int | None = None,
    speed: float | None = None,
    emotion: str | None = None,
) -> Action:
    """构造标准语音输出动作。"""
    return _build_action(
        "speak",
        text=text,
        kind=kind,
        level=level,
        reason=reason,
        interrupt=interrupt,
        voice=voice,
        volume=volume,
        speed=speed,
        emotion=emotion,
    )


def display(
    text: str,
    *,
    title: str | None = None,
    status: str | None = None,
    kind: str | None = None,
    level: str | None = None,
    reason: str | None = None,
) -> Action:
    """构造标准显示输出动作。"""
    return _build_action(
        "display",
        text=text,
        title=title,
        status=status,
        kind=kind,
        level=level,
        reason=reason,
    )


def render_pet_expression(
    expression: str,
    *,
    style: str | None = None,
    intensity: float | None = None,
    duration_ms: int | None = None,
) -> Action:
    """构造桌宠表情渲染动作。"""
    return _build_action(
        "render_pet_expression",
        expression=expression,
        style=style,
        intensity=intensity,
        duration_ms=duration_ms,
    )


def set_light_state(
    state: str,
    *,
    color: str | None = None,
    pattern: str | None = None,
    brightness: int | None = None,
    duration_ms: int | None = None,
    kind: str | None = None,
    level: str | None = None,
    reason: str | None = None,
) -> Action:
    """构造灯光控制动作。"""
    return _build_action(
        "set_light_state",
        state=state,
        color=color,
        pattern=pattern,
        brightness=brightness,
        duration_ms=duration_ms,
        kind=kind,
        level=level,
        reason=reason,
    )


def start_timer(duration_sec: int) -> Action:
    """构造启动计时器动作。"""
    return _build_action("start_timer", duration_sec=int(duration_sec))


def stop_timer() -> Action:
    """构造停止计时器动作。"""
    return _build_action("stop_timer")


def start_voice_capture(*, source: str, trigger: str | None = None) -> Action:
    """构造开始采集语音输入的动作。"""
    return _build_action("start_voice_capture", source=source, trigger=trigger)


def stop_voice_capture(*, source: str, reason: str | None = None) -> Action:
    """构造停止采集语音输入的动作。"""
    return _build_action("stop_voice_capture", source=source, reason=reason)


def set_tts_voice(voice_id: str) -> Action:
    """构造设置 TTS 音色的动作。"""
    return _build_action("set_tts_voice", voice_id=voice_id)


def set_tts_volume(volume: int) -> Action:
    """构造设置 TTS 音量的动作。"""
    return _build_action("set_tts_volume", volume=int(volume))


def set_tts_speed(speed: float) -> Action:
    """构造设置 TTS 语速的动作。"""
    return _build_action("set_tts_speed", speed=float(speed))


def none_action() -> Action:
    """构造空动作，用于显式表示本轮不执行任何操作。"""
    return _build_action("none")
