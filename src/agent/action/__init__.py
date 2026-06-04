"""Action 包公开入口。

本包位于 IntentPlan 之后、设备执行之前，公开标准 Action 模型和受控构造
函数。上游是 `decision/action_realizer.py`，下游是 `runtime/device_adapter.py`。

本包不负责语义理解、策略判断或硬件执行；这些职责分别由 LLM Agent、
DeterministicGuard 和 DeviceAdapter 承担。
"""

from src.agent.action.action_builders import (
    display,
    render_pet_expression,
    set_light_state,
    set_tts_speed,
    set_tts_voice,
    set_tts_volume,
    speak,
    start_timer,
    start_voice_capture,
    stop_timer,
    stop_voice_capture,
)
from src.agent.action.action_model import Action
from src.agent.action.types import ActionType

__all__ = [
    "Action",
    "ActionType",
    "display",
    "render_pet_expression",
    "set_light_state",
    "set_tts_speed",
    "set_tts_voice",
    "set_tts_volume",
    "speak",
    "start_timer",
    "start_voice_capture",
    "stop_timer",
    "stop_voice_capture",
]
