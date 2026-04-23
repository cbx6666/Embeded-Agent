"""事件模型包。

这里定义的是 Agent 认可的标准事件，不绑定具体输入设备。
真实世界的摄像头、麦克风、按键、CLI、传感器等，都应先在 adapters
中被翻译成这里的事件，再进入核心状态机。
"""

from src.agent.event.event_model import Event
from src.agent.event.factories import (
    make_behavior_attention_event,
    make_behavior_presence_event,
    make_behavior_signal_event,
    make_behavior_summary_event,
    make_display_sensor_event,
    make_fatigue_event,
    make_light_level_event,
    make_noise_level_event,
    make_speech_recognized_event,
    make_temperature_humidity_event,
    user_emotion_updated_from_rafdb,
)
from src.agent.event.types import EventType

__all__ = [
    "Event",
    "EventType",
    "make_behavior_attention_event",
    "make_behavior_presence_event",
    "make_behavior_signal_event",
    "make_behavior_summary_event",
    "make_display_sensor_event",
    "make_fatigue_event",
    "make_light_level_event",
    "make_noise_level_event",
    "make_speech_recognized_event",
    "make_temperature_humidity_event",
    "user_emotion_updated_from_rafdb",
]
