"""事件模型包。

定义 Agent 认可的标准事件，不绑定具体输入设备。真实世界的摄像头、麦克风、
传感器等，先在 adapters 中被翻译成这里的事件，再进入核心状态机。
"""

from src.agent.event.event_model import Event
from src.agent.event.event_builders import (
    make_activity_event,
    make_behavior_attention_event,
    make_behavior_presence_event,
    make_fatigue_event,
    make_light_level_event,
    make_noise_level_event,
    make_posture_event,
    make_speech_recognized_event,
    make_temperature_humidity_event,
    user_emotion_updated_from_rafdb,
    user_emotion_updated_standard,
)
from src.agent.event.types import EventType
from src.agent.event.router import EventRouter, RouteDecision

__all__ = [
    "Event",
    "EventType",
    "EventRouter",
    "RouteDecision",
    "make_activity_event",
    "make_behavior_attention_event",
    "make_behavior_presence_event",
    "make_fatigue_event",
    "make_light_level_event",
    "make_noise_level_event",
    "make_posture_event",
    "make_speech_recognized_event",
    "make_temperature_humidity_event",
    "user_emotion_updated_from_rafdb",
    "user_emotion_updated_standard",
]
