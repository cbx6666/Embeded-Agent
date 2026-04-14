from __future__ import annotations

"""事件构造辅助函数。

该模块用于把各类输入侧原始结果，转换为核心可消费的标准 Event。
"""

from src.agent.event.event_model import Event

# RAF-DB Basic 七分类标签（常见 1-based 编号）。
RAF_DB_LABELS: dict[int, str] = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happiness",
    5: "sadness",
    6: "anger",
    7: "neutral",
}

# 将 RAF-DB 细粒度表情映射到当前 Agent 的高层情绪空间。
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
    """把 RAF-DB 预测结果转换为 user_emotion_updated 事件。

    至少提供 `label_id` 或 `label_name` 之一。
    """
    raf_emotion = _resolve_raf_emotion(label_id=label_id, label_name=label_name)
    agent_emotion = RAF_TO_AGENT_EMOTION.get(raf_emotion, "neutral")
    normalized_confidence = _normalize_confidence(confidence)

    payload: dict[str, object] = {
        "emotion": agent_emotion,
        "confidence": normalized_confidence,
        "source": source,
        "model": "raf-db",
        "raf_emotion": raf_emotion,
    }
    if label_id is not None:
        payload["raf_label_id"] = label_id
    if person_id:
        payload["person_id"] = person_id

    return Event(type="user_emotion_updated", timestamp=timestamp, payload=payload)


def _resolve_raf_emotion(*, label_id: int | None, label_name: str | None) -> str:
    """解析 RAF-DB 预测标签到规范表情字符串。"""
    if label_name:
        return label_name.strip().lower()
    if label_id is None:
        raise ValueError("label_id 和 label_name 不能同时为空。")
    return RAF_DB_LABELS.get(label_id, "neutral")


def _normalize_confidence(confidence: float | None) -> float | None:
    """把置信度规范到 [0, 1] 区间。"""
    if confidence is None:
        return None
    return max(0.0, min(1.0, float(confidence)))
