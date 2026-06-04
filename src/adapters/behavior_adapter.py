from __future__ import annotations

"""行为识别适配器。"""

import time
from typing import Any

from src.agent.event.event_builders import (
    make_activity_event,
    make_behavior_attention_event,
    make_behavior_presence_event,
    make_posture_event,
)


class BehaviorAdapter:
    """简单行为适配器：负责阈值、去抖并上报在场与注意力事件。"""

    def __init__(
        self,
        core: Any,
        min_confidence: float = 0.6,
        debounce_seconds: float = 2.0,
        summary_threshold_seconds: float = 120.0,
    ) -> None:
        self.core = core
        self.min_confidence = float(min_confidence)
        self.debounce_seconds = float(debounce_seconds)
        self.summary_threshold_seconds = float(summary_threshold_seconds)
        self._last_presence: str | None = None
        self._last_attention_key: tuple[str, str] | None = None
        self._last_presence_ts: float | None = None
        self._last_attention_ts: float | None = None
        self._last_posture: str | None = None
        self._last_activity: str | None = None
        self._last_posture_ts: float | None = None
        self._last_activity_ts: float | None = None

    def publish_presence(
        self,
        presence: str,
        confidence: float | None = None,
        source: str = "camera_v1",
        timestamp: int | None = None,
    ) -> bool:
        normalized_confidence = 1.0 if confidence is None else float(confidence)
        if normalized_confidence < self.min_confidence:
            return False
        now = time.time()
        if self._last_presence == presence and self._last_presence_ts is not None:
            if (now - self._last_presence_ts) < self.debounce_seconds:
                return False

        event = make_behavior_presence_event(
            presence=presence,
            confidence=normalized_confidence,
            source=source,
            timestamp=timestamp,
        )
        try:
            self.core.handle_event(event)
        except Exception:
            return False

        self._last_presence = presence
        self._last_presence_ts = now
        return True

    def publish_attention(
        self,
        attention: str,
        behavior: str,
        confidence: float | None = None,
        source: str = "camera_v1",
        timestamp: int | None = None,
    ) -> bool:
        normalized_confidence = 1.0 if confidence is None else float(confidence)
        if normalized_confidence < self.min_confidence:
            return False
        now = time.time()
        attention_key = (attention, behavior)
        if self._last_attention_key == attention_key and self._last_attention_ts is not None:
            if (now - self._last_attention_ts) < self.debounce_seconds:
                return False

        event = make_behavior_attention_event(
            attention=attention,
            behavior=behavior,
            confidence=normalized_confidence,
            source=source,
            timestamp=timestamp,
        )
        try:
            self.core.handle_event(event)
        except Exception:
            return False

        self._last_attention_key = attention_key
        self._last_attention_ts = now
        return True

    def publish_posture(
        self,
        posture: str,
        confidence: float | None = None,
        source: str = "yolo26_pose_om_v1",
        timestamp: int | None = None,
    ) -> bool:
        normalized_confidence = 1.0 if confidence is None else float(confidence)
        if normalized_confidence < self.min_confidence:
            return False
        now = time.time()
        if self._last_posture == posture and self._last_posture_ts is not None:
            if (now - self._last_posture_ts) < self.debounce_seconds:
                return False
        event = make_posture_event(
            posture=posture,
            confidence=normalized_confidence,
            source=source,
            timestamp=timestamp,
        )
        try:
            self.core.handle_event(event)
        except Exception:
            return False
        self._last_posture = posture
        self._last_posture_ts = now
        return True

    def publish_activity(
        self,
        activity: str,
        confidence: float | None = None,
        source: str = "yolo26_pose_om_v1",
        timestamp: int | None = None,
    ) -> bool:
        normalized_confidence = 1.0 if confidence is None else float(confidence)
        if normalized_confidence < self.min_confidence:
            return False
        now = time.time()
        if self._last_activity == activity and self._last_activity_ts is not None:
            if (now - self._last_activity_ts) < self.debounce_seconds:
                return False
        event = make_activity_event(
            activity=activity,
            confidence=normalized_confidence,
            source=source,
            timestamp=timestamp,
        )
        try:
            self.core.handle_event(event)
        except Exception:
            return False
        self._last_activity = activity
        self._last_activity_ts = now
        return True

    def publish_behavior(
        self,
        behavior: str,
        *,
        attention: str = "idle",
        confidence: float | None = None,
        source: str = "camera_v1",
        timestamp: int | None = None,
        **_: Any,
    ) -> bool:
        return self.publish_attention(
            attention=attention,
            behavior=behavior,
            confidence=confidence,
            source=source,
            timestamp=timestamp,
        )
