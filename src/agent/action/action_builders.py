from __future__ import annotations

"""
标准 Action 构造模块。

本文件位于 `ActionRealizer -> DeviceAdapter` 之间的确定性动作边界，负责
把已经通过校验的动作参数封装为统一 `Action`。它的上游是
`decision/action_realizer.py`，下游是 `runtime/device_adapter.py` 和具体
硬件/输出适配器。

本模块不理解用户语义、不调用 LLM、不修改 AgentState，也不直接执行硬件。
这里保留运行时白名单校验，是为了防止未注册动作绕过动作模型进入设备层。
"""

from typing import Any

from src.agent.action.action_model import Action
from src.agent.action.types import ACTION_TYPE_SET, ActionType


def _build_action(action_type: ActionType, **payload: Any) -> Action:
    """构造标准 Action，并拒绝未注册动作类型。

    ActionRealizer 负责决定“要做什么”，本函数只负责把参数落到统一动作
    模型里。过滤 `None` 可以让上层构造函数保持简洁，同时避免设备层收到
    含义不明确的空字段。
    """

    if action_type not in ACTION_TYPE_SET:
        raise ValueError(f"unknown action_type: {action_type}")
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
    """构造语音输出动作；只描述输出内容，不负责 TTS 执行。"""

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
    """构造屏幕显示动作；只描述显示文本，不绑定具体屏幕实现。"""

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
    """构造桌宠表情渲染动作；具体动画由显示适配器实现。"""

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
    """构造灯光状态动作；安全边界仍由 DeviceAdapter 和硬件侧执行。"""

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
    """构造计时器启动动作；计时线程由 TimerService 负责。"""

    return _build_action("start_timer", duration_sec=int(duration_sec))


def stop_timer() -> Action:
    """构造计时器停止动作；不直接改变 FocusState。"""

    return _build_action("stop_timer")


def start_voice_capture(*, source: str, trigger: str | None = None) -> Action:
    """构造开始语音采集动作；唤醒和录音设备由语音适配器处理。"""

    return _build_action("start_voice_capture", source=source, trigger=trigger)


def stop_voice_capture(*, source: str, reason: str | None = None) -> Action:
    """构造停止语音采集动作；只表达意图，不读取音频。"""

    return _build_action("stop_voice_capture", source=source, reason=reason)


def set_tts_voice(voice_id: str) -> Action:
    """构造 TTS 音色设置动作；不直接访问 TTS 引擎。"""

    return _build_action("set_tts_voice", voice_id=voice_id)


def set_tts_volume(volume: int) -> Action:
    """构造 TTS 音量设置动作；数值范围由 ActionRealizer 先做裁剪。"""

    return _build_action("set_tts_volume", volume=int(volume))


def set_tts_speed(speed: float) -> Action:
    """构造 TTS 语速设置动作；设备层决定是否支持该参数。"""

    return _build_action("set_tts_speed", speed=float(speed))
