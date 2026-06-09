from __future__ import annotations

"""标准 Action 构造模块。

位于 ``ActionRealizer -> DeviceAdapter`` 之间的确定性动作边界，把已经通过决策
的动作参数封装为统一 ``Action``。只支持五个真实可执行动作：speak、display、
start_timer、stop_timer、set_tts_volume。

本模块不理解语义、不调用 LLM、不修改 AgentState、不执行硬件；运行时白名单校验
防止未注册动作绕过动作模型进入设备层。
"""

from typing import Any

from src.agent.action.action_model import Action
from src.agent.action.types import ACTION_TYPE_SET, ActionType


def _build_action(action_type: ActionType, **payload: Any) -> Action:
    """构造标准 Action，并拒绝未注册动作类型。"""

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
) -> Action:
    """构造语音输出动作；只描述输出内容，不负责 TTS 执行。"""

    return _build_action("speak", text=text, kind=kind, level=level, reason=reason)


def display(
    text: str,
    *,
    kind: str | None = None,
    level: str | None = None,
    reason: str | None = None,
) -> Action:
    """构造屏幕显示动作；只描述显示文本，不绑定具体屏幕实现。"""

    return _build_action("display", text=text, kind=kind, level=level, reason=reason)


def start_timer(duration_sec: int) -> Action:
    """构造计时器启动动作；计时线程由 TimerService 负责。"""

    return _build_action("start_timer", duration_sec=int(duration_sec))


def stop_timer() -> Action:
    """构造计时器停止动作；不直接改变 FocusState。"""

    return _build_action("stop_timer")


def set_tts_volume(volume: int) -> Action:
    """构造 TTS 音量设置动作；数值范围由 ActionRealizer 先做裁剪。"""

    return _build_action("set_tts_volume", volume=int(volume))


def play_media(
    *,
    track_id: str,
    path: str,
    title: str = "",
    media_type: str = "",
    category: str = "",
    source: str = "user_explicit",
    defer_after_speak: bool = False,
) -> Action:
    """构造媒体播放动作；defer_after_speak 时须等前置 speak 播完再执行。"""

    return _build_action(
        "play_media",
        track_id=track_id,
        path=path,
        title=title,
        media_type=media_type,
        category=category,
        source=source,
        defer_after_speak=bool(defer_after_speak),
    )


def stop_media(reason: str = "user") -> Action:
    return _build_action("stop_media", reason=reason)


def pause_media() -> Action:
    return _build_action("pause_media")


def resume_media() -> Action:
    return _build_action("resume_media")


def next_media() -> Action:
    return _build_action("next_media")
