from __future__ import annotations

"""mock 输入适配器模块。"""

import time

from src.agent.event import Event

FIELD_MAP = {
    "presence": ("user_presence_updated", "presence", {"present", "away", "unknown"}),
    "attention": ("user_attention_updated", "attention", {"focused", "distracted", "idle"}),
    "emotion": ("user_emotion_updated", "emotion", {"neutral", "tired", "stressed", "happy"}),
    "fatigue": (
        "user_fatigue_updated",
        "fatigue_level",
        {"none", "mild", "moderate", "high"},
    ),
}



def parse_mock_command(command: str) -> Event | None:
    """将 /mock 命令解析为标准事件。"""
    stripped = command.strip()
    if not stripped.startswith("/mock"):
        return None

    parts = stripped.split()
    if len(parts) != 3:
        raise ValueError("mock 命令格式应为: /mock <field> <value>")

    field_key, value = parts[1], parts[2]
    if field_key not in FIELD_MAP:
        raise ValueError(f"不支持的 mock 字段: {field_key}")

    event_type, payload_key, valid_values = FIELD_MAP[field_key]
    if value not in valid_values:
        raise ValueError(f"字段 {field_key} 不支持值: {value}")

    return Event(
        type=event_type,
        timestamp=int(time.time()),
        payload={payload_key: value, "source": "mock"},
    )
