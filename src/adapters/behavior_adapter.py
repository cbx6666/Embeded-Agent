from __future__ import annotations

"""行为识别适配器（兼容旧 posture 命名）。"""

import time
from typing import Any

from src.agent.event.factories import make_behavior_signal_event, make_behavior_summary_event


class BehaviorAdapter:
    """简单行为适配器：负责阈值、去抖并上报标准行为事件。"""

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
        self._last_signal: str | None = None
        self._last_ts: float | None = None
        self._signal_accumulated_sec: dict[str, float] = {}
        self._last_summary_ts: dict[str, float] = {}

    def publish_behavior_signal(
        self,
        behavior_signal: str,
        confidence: float | None = None,
        frame_id: int | str | None = None,
        bbox: dict[str, Any] | None = None,
        source: str = "camera_v1",
        timestamp: int | None = None,
    ) -> bool:
        now = time.time()
        normalized_confidence = 1.0 if confidence is None else float(confidence)
        if normalized_confidence < self.min_confidence:
            return False

        if self._last_signal == behavior_signal and self._last_ts is not None:
            if (now - self._last_ts) < self.debounce_seconds:
                return False

        event = make_behavior_signal_event(
            behavior_signal=behavior_signal,
            confidence=normalized_confidence,
            frame_id=frame_id,
            bbox=bbox,
            source=source,
            timestamp=timestamp,
        )
        try:
            self.core.handle_event(event)
        except Exception:
            return False

        last_ts = self._last_ts or now
        delta = max(0.0, now - last_ts)
        summary_candidates = {"phone_use", "staring", "desk_rest", "lying", "slouch"}
        if behavior_signal in summary_candidates:
            self._signal_accumulated_sec[behavior_signal] = (
                self._signal_accumulated_sec.get(behavior_signal, 0.0) + delta
            )

        accumulated_sec = self._signal_accumulated_sec.get(behavior_signal, 0.0)
        last_summary_ts = self._last_summary_ts.get(behavior_signal, 0.0)
        if accumulated_sec >= self.summary_threshold_seconds:
            if (now - last_summary_ts) >= self.summary_threshold_seconds:
                try:
                    summary_event = make_behavior_summary_event(
                        behavior_signal=behavior_signal,
                        accumulated_sec=accumulated_sec,
                        confidence=normalized_confidence,
                        source=source,
                    )
                    self.core.handle_event(summary_event)
                    self._signal_accumulated_sec[behavior_signal] = 0.0
                    self._last_summary_ts[behavior_signal] = now
                except Exception:
                    pass

        self._last_signal = behavior_signal
        self._last_ts = now
        return True

    def publish_behavior(self, behavior_signal: str, **kwargs: Any) -> bool:
        return self.publish_behavior_signal(behavior_signal, **kwargs)
