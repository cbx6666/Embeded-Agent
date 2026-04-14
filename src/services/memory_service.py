from __future__ import annotations

"""最近记忆维护服务模块。"""

from collections import Counter

from src.agent.event import Event
from src.agent.state import AgentState


class MemoryService:
    """维护最近事件、最近消息和最近专注记录。"""

    def __init__(
        self,
        max_recent_events: int = 20,
        max_recent_messages: int = 20,
        max_focus_sessions: int = 10,
        max_emotion_samples: int = 120,
        max_emotion_summaries: int = 60,
        emotion_summary_window_sec: int = 60,
    ) -> None:
        self.max_recent_events = max_recent_events
        self.max_recent_messages = max_recent_messages
        self.max_focus_sessions = max_focus_sessions
        self.max_emotion_samples = max_emotion_samples
        self.max_emotion_summaries = max_emotion_summaries
        self.emotion_summary_window_sec = emotion_summary_window_sec

    def record_event(self, state: AgentState, event: Event) -> None:
        state.memory.recent_events.append(
            {
                "type": event.type,
                "timestamp": event.timestamp,
                "payload": event.payload,
            }
        )
        if event.type == "user_emotion_updated":
            self._record_emotion_sample(state, event)
            self._maybe_rollup_emotion_summary(state, event.timestamp)

    def record_message(
        self,
        state: AgentState,
        role: str,
        text: str,
        timestamp: int,
    ) -> None:
        state.memory.recent_messages.append(
            {
                "role": role,
                "text": text,
                "timestamp": timestamp,
            }
        )

    def trim(self, state: AgentState) -> None:
        state.memory.recent_events = state.memory.recent_events[-self.max_recent_events :]
        state.memory.recent_messages = state.memory.recent_messages[-self.max_recent_messages :]
        state.memory.focus_sessions = state.memory.focus_sessions[-self.max_focus_sessions :]
        state.memory.emotion_samples = state.memory.emotion_samples[-self.max_emotion_samples :]
        state.memory.emotion_summaries = state.memory.emotion_summaries[-self.max_emotion_summaries :]

    def _record_emotion_sample(self, state: AgentState, event: Event) -> None:
        """记录单条情绪样本，用于短时间窗口统计。"""
        sample = {
            "timestamp": event.timestamp,
            "emotion": str(event.payload.get("emotion", "neutral")),
            "confidence": event.payload.get("confidence"),
            "person_id": event.payload.get("person_id"),
            "source": event.payload.get("source"),
            "model": event.payload.get("model"),
            "raw_emotion": event.payload.get("raf_emotion"),
        }
        state.memory.emotion_samples.append(sample)

    def _maybe_rollup_emotion_summary(self, state: AgentState, now_ts: int) -> None:
        """按固定时间窗口生成情绪摘要，避免逐帧长期存储。"""
        latest_summary = (
            state.memory.emotion_summaries[-1] if state.memory.emotion_summaries else None
        )
        last_end_ts = int(latest_summary["end_ts"]) if latest_summary else None
        if last_end_ts is not None and now_ts - last_end_ts < self.emotion_summary_window_sec:
            return

        window_start_ts = now_ts - self.emotion_summary_window_sec
        window_samples = [
            item for item in state.memory.emotion_samples if int(item["timestamp"]) >= window_start_ts
        ]
        if not window_samples:
            return

        emotions = [str(item.get("emotion", "neutral")) for item in window_samples]
        counts = Counter(emotions)
        dominant_emotion = counts.most_common(1)[0][0]
        total = len(window_samples)

        confidences: list[float] = []
        for item in window_samples:
            value = item.get("confidence")
            if value is None:
                continue
            confidences.append(float(value))

        distribution = {emotion: round(count / total, 3) for emotion, count in counts.items()}
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None

        summary = {
            "start_ts": window_start_ts,
            "end_ts": now_ts,
            "window_sec": self.emotion_summary_window_sec,
            "sample_count": total,
            "dominant_emotion": dominant_emotion,
            "distribution": distribution,
            "avg_confidence": avg_confidence,
        }
        state.memory.emotion_summaries.append(summary)
