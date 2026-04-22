from __future__ import annotations

"""事件构造辅助函数。

该模块用于把各类输入侧原始结果，转换为核心可消费的标准 Event。
"""

import time
from typing import Any

from src.agent.event.event_model import Event


# =========================
# posture 相关（姿态识别）
# =========================

def make_posture_event(
    posture: str,
    confidence: float | None = None,
    frame_id: int | str | None = None,
    bbox: dict[str, Any] | None = None,
    duration_sec: float | None = None,
    person_id: str | int | None = None,
    keypoints_summary: dict[str, Any] | None = None,
    severity: str | None = None,
    source: str = "camera_v1",
    timestamp: int | None = None,
) -> Event:
    ts = int(time.time()) if timestamp is None else int(timestamp)
    payload: dict[str, Any] = {"posture": posture, "source": source}

    if confidence is not None:
        payload["confidence"] = float(confidence)
    if frame_id is not None:
        payload["frame_id"] = frame_id
    if bbox is not None:
        payload["bbox"] = bbox
    if duration_sec is not None:
        payload["duration_sec"] = float(duration_sec)
    if person_id is not None:
        payload["person_id"] = person_id
    if keypoints_summary is not None:
        payload["keypoints_summary"] = keypoints_summary
    if severity is not None:
        payload["severity"] = severity

    return Event(type="user_posture_updated", timestamp=ts, payload=payload)


def make_posture_summary_event(
    posture: str,
    accumulated_sec: float,
    confidence: float | None = None,
    source: str = "camera_v1",
    timestamp: int | None = None,
) -> Event:
    ts = int(time.time()) if timestamp is None else int(timestamp)
    payload = {
        "posture": posture,
        "accumulated_sec": float(accumulated_sec),
        "source": source,
    }
    if confidence is not None:
        payload["confidence"] = float(confidence)

    return Event(type="user_posture_summary", timestamp=ts, payload=payload)


# =========================
# emotion（RAF-DB）
# =========================

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
    raf_emotion = _resolve_raf_emotion(label_id=label_id, label_name=label_name)
    agent_emotion = RAF_TO_AGENT_EMOTION.get(raf_emotion, "neutral")

    payload: dict[str, object] = {
        "emotion": agent_emotion,
        "confidence": _normalize_confidence(confidence),
        "source": source,
        "model": "raf-db",
        "raf_emotion": raf_emotion,
    }

    if label_id is not None:
        payload["raf_label_id"] = label_id
    if person_id:
        payload["person_id"] = person_id

    return Event(type="user_emotion_updated", timestamp=timestamp, payload=payload)


# =========================
# display / voice
# =========================

def display_sensor_updated(
    *,
    timestamp: int,
    expression: str,
    source: str,
    brightness: int | None = None,
    fps: int | None = None,
    sensor_values: dict[str, object] | None = None,
    screen_id: str | None = None,
) -> Event:
    payload: dict[str, object] = {
        "expression": expression,
        "source": source,
    }

    if brightness is not None:
        payload["brightness"] = int(brightness)
    if fps is not None:
        payload["fps"] = int(fps)
    if sensor_values:
        payload["sensor_values"] = sensor_values
    if screen_id:
        payload["screen_id"] = screen_id

    return Event(type="display_sensor_updated", timestamp=timestamp, payload=payload)


def voice_input_captured(
    *,
    timestamp: int,
    text: str,
    source: str,
    confidence: float | None = None,
    language: str | None = None,
    is_final: bool = True,
    audio_id: str | None = None,
) -> Event:
    payload: dict[str, object] = {
        "text": text,
        "source": source,
        "is_final": bool(is_final),
    }

    normalized_confidence = _normalize_confidence(confidence)
    if normalized_confidence is not None:
        payload["confidence"] = normalized_confidence
    if language:
        payload["language"] = language
    if audio_id:
        payload["audio_id"] = audio_id

    return Event(type="voice_input_captured", timestamp=timestamp, payload=payload)


# =========================
# utils
# =========================

def _resolve_raf_emotion(*, label_id: int | None, label_name: str | None) -> str:
    if label_name:
        return label_name.strip().lower()
    if label_id is None:
        raise ValueError("label_id 和 label_name 不能同时为空。")
    return RAF_DB_LABELS.get(label_id, "neutral")


def _normalize_confidence(confidence: float | None) -> float | None:
    if confidence is None:
        return None
    return max(0.0, min(1.0, float(confidence)))
  