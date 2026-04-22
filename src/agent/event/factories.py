from __future__ import annotations

"""事件工厂：构造常用 Event 的便捷函数。"""

import time
from typing import Any

from src.agent.event.event_model import Event


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
    """构造一个标准的 user_posture_updated 事件。"""
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
    """构造姿势累计/告警汇总事件（例如连续不良姿势达到阈值）。"""
    ts = int(time.time()) if timestamp is None else int(timestamp)
    payload = {
        "posture": posture,
        "accumulated_sec": float(accumulated_sec),
        "source": source,
    }
    if confidence is not None:
        payload["confidence"] = float(confidence)
    return Event(type="user_posture_summary", timestamp=ts, payload=payload)
