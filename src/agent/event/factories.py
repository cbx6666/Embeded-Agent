from __future__ import annotations

"""事件构造辅助函数。"""

import time
from typing import Any

from src.agent.event.event_model import Event


def _resolve_timestamp(timestamp: int | None) -> int:
    """解析事件时间戳；为空时使用当前时间。"""
    return int(time.time()) if timestamp is None else int(timestamp)


def _build_event(event_type: str, timestamp: int | None = None, **payload: Any) -> Event:
    """构造标准事件，并自动剔除值为 None 的字段。"""
    normalized_payload = {key: value for key, value in payload.items() if value is not None}
    return Event(type=event_type, timestamp=_resolve_timestamp(timestamp), payload=normalized_payload)


def make_behavior_presence_event(
    presence: str,
    *,
    source: str = "camera_v1",
    confidence: float | None = None,
    timestamp: int | None = None,
) -> Event:
    """构造用户在场状态更新事件。"""
    return _build_event(
        "user_presence_updated",
        timestamp=timestamp,
        presence=presence,
        source=source,
        confidence=_normalize_confidence(confidence),
    )


def make_behavior_attention_event(
    attention: str,
    *,
    behavior: str,
    source: str = "camera_v1",
    confidence: float | None = None,
    timestamp: int | None = None,
) -> Event:
    """构造用户注意力与行为更新事件。"""
    return _build_event(
        "user_attention_updated",
        timestamp=timestamp,
        attention=attention,
        behavior=behavior,
        source=source,
        confidence=_normalize_confidence(confidence),
    )


def make_fatigue_event(
    fatigue_level: str,
    *,
    source: str,
    confidence: float | None = None,
    perclos: float | None = None,
    yawn_in_window: bool | None = None,
    window_sec: int | None = None,
    timestamp: int | None = None,
) -> Event:
    """构造疲劳状态更新事件。"""
    return _build_event(
        "user_fatigue_updated",
        timestamp=timestamp,
        fatigue_level=fatigue_level,
        source=source,
        confidence=_normalize_confidence(confidence),
        perclos=perclos,
        yawn_in_window=yawn_in_window,
        window_sec=window_sec,
    )


RAF_DB_LABELS: dict[int, str] = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happiness",
    5: "sadness",
    6: "anger",
    7: "neutral",
}

RAF_TO_AGENT_EMOTION: dict[str, str] = {
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "stressed",
    "anger": "stressed",
    "disgust": "stressed",
    "fear": "stressed",
    "surprise": "neutral",
}


def user_emotion_updated_from_rafdb(
    *,
    timestamp: int,
    label_id: int | None = None,
    label_name: str | None = None,
    confidence: float | None = None,
    person_id: str | None = None,
    source: str = "camera",
) -> Event:
    """将 RAF-DB 输出转换为系统标准情绪事件。"""
    raf_emotion = _resolve_raf_emotion(label_id=label_id, label_name=label_name)
    agent_emotion = RAF_TO_AGENT_EMOTION.get(raf_emotion, "neutral")
    return _build_event(
        "user_emotion_updated",
        timestamp=timestamp,
        emotion=agent_emotion,
        confidence=_normalize_confidence(confidence),
        source=source,
        model="raf-db",
        raf_emotion=raf_emotion,
        raf_label_id=label_id,
        person_id=person_id,
    )


def make_display_sensor_event(
    *,
    expression: str,
    source: str,
    brightness: int | None = None,
    fps: int | None = None,
    sensor_values: dict[str, object] | None = None,
    screen_id: str | None = None,
    timestamp: int | None = None,
) -> Event:
    """构造显示设备状态快照事件。"""
    return _build_event(
        "display_sensor_updated",
        timestamp=timestamp,
        expression=expression,
        source=source,
        brightness=int(brightness) if brightness is not None else None,
        fps=int(fps) if fps is not None else None,
        sensor_values=sensor_values,
        screen_id=screen_id,
    )


def make_speech_recognized_event(
    *,
    text: str,
    source: str,
    confidence: float | None = None,
    language: str | None = None,
    is_final: bool = True,
    audio_id: str | None = None,
    session_id: str | None = None,
    timestamp: int | None = None,
) -> Event:
    """构造语音识别结果事件。"""
    return _build_event(
        "speech_recognized",
        timestamp=timestamp,
        text=text,
        source=source,
        confidence=_normalize_confidence(confidence),
        language=language,
        is_final=bool(is_final),
        audio_id=audio_id,
        session_id=session_id,
    )


def make_light_level_event(
    *,
    light_lux: int,
    source: str,
    level: str | None = None,
    is_low_light: bool | None = None,
    timestamp: int | None = None,
) -> Event:
    """构造环境光照更新事件。"""
    return _build_event(
        "light_level_updated",
        timestamp=timestamp,
        light_lux=int(light_lux),
        source=source,
        level=level,
        is_low_light=is_low_light,
    )


def make_temperature_humidity_event(
    *,
    temperature_c: float,
    humidity_pct: float,
    source: str,
    temperature_level: str | None = None,
    humidity_level: str | None = None,
    timestamp: int | None = None,
) -> Event:
    """构造温湿度更新事件。"""
    return _build_event(
        "temperature_humidity_updated",
        timestamp=timestamp,
        temperature_c=float(temperature_c),
        humidity_pct=float(humidity_pct),
        source=source,
        temperature_level=temperature_level,
        humidity_level=humidity_level,
    )


def make_noise_level_event(
    *,
    noise_db: int,
    source: str,
    level: str | None = None,
    is_noisy: bool | None = None,
    timestamp: int | None = None,
) -> Event:
    """构造噪声等级更新事件。"""
    return _build_event(
        "noise_level_updated",
        timestamp=timestamp,
        noise_db=int(noise_db),
        source=source,
        level=level,
        is_noisy=is_noisy,
    )


def _resolve_raf_emotion(*, label_id: int | None, label_name: str | None) -> str:
    """解析 RAF-DB 标签，得到统一情绪名称。"""
    if label_name:
        return label_name.strip().lower()
    if label_id is None:
        raise ValueError("label_id 和 label_name 不能同时为空。")
    return RAF_DB_LABELS.get(label_id, "neutral")


def _normalize_confidence(confidence: float | None) -> float | None:
    """将置信度裁剪到 0 到 1 之间。"""
    if confidence is None:
        return None
    return max(0.0, min(1.0, float(confidence)))
