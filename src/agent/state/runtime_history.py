from __future__ import annotations

"""AgentState 的运行时历史子结构与维护服务。

``RuntimeHistory`` 是 AgentState 的字段，保存当前会话窗口里的短期工作历史：最近
事件、消息、动作、提醒记录、注意力/情绪/环境采样以及少量滚动统计。

``RuntimeHistoryService`` 是短期历史的唯一写入入口，负责记录与裁剪。它不是长期
记忆，不调用 LLM，不写偏好存储。窗口上限与信号聚合规则作为模块常量内聚在这里，
不进入 policy_config。
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - 仅类型提示
    from src.agent.action.action_model import Action
    from src.agent.event.event_model import Event
    from src.agent.state.agent_state import AgentState


# 短期历史窗口上限。
MAX_RECENT_EVENTS = 30
MAX_RECENT_MESSAGES = 20
MAX_RECENT_ACTIONS = 20
MAX_REMINDER_RECORDS = 20
MAX_ATTENTION_RECORDS = 30
MAX_ENVIRONMENT_RECORDS = 30
MAX_FOCUS_SESSIONS = 20
MAX_EMOTION_SAMPLES = 60
MAX_EMOTION_SUMMARIES = 20
MAX_SIGNAL_RECENT_VALUES = 50
EMOTION_SUMMARY_WINDOW_SEC = 30

# 事件到滚动信号趋势字段的映射。
_SIGNAL_FIELDS_BY_EVENT: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "user_presence_updated": (("presence", ("presence",)),),
    "user_attention_updated": (
        ("attention", ("attention",)),
        ("behavior", ("behavior",)),
    ),
    "user_emotion_updated": (("emotion", ("emotion",)),),
    "user_fatigue_updated": (("fatigue", ("fatigue_level",)),),
    "user_posture_updated": (("posture", ("posture",)),),
    "user_activity_updated": (("activity", ("activity",)),),
    "light_level_updated": (("light", ("level", "light_lux")),),
    "temperature_humidity_updated": (
        ("temperature", ("temperature_level", "temperature_c")),
        ("humidity", ("humidity_level", "humidity_pct")),
    ),
    "noise_level_updated": (("noise", ("level", "noise_db")),),
}

_USER_STATE_EVENTS = frozenset(
    {
        "user_presence_updated",
        "user_attention_updated",
        "user_emotion_updated",
        "user_fatigue_updated",
        "user_posture_updated",
        "user_activity_updated",
    }
)
_ENVIRONMENT_EVENTS = frozenset(
    {"light_level_updated", "temperature_humidity_updated", "noise_level_updated"}
)


@dataclass
class RuntimeHistory:
    """当前运行期的短期历史窗口。"""

    recent_events: list[dict[str, Any]] = field(default_factory=list)
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    recent_actions: list[dict[str, Any]] = field(default_factory=list)
    reminder_records: list[dict[str, Any]] = field(default_factory=list)
    attention_records: list[dict[str, Any]] = field(default_factory=list)
    environment_records: list[dict[str, Any]] = field(default_factory=list)
    focus_sessions: list[dict[str, Any]] = field(default_factory=list)
    focus_session_count: int = 0
    focus_total_duration_sec: int = 0
    distraction_event_count: int = 0
    state_change_counts: dict[str, int] = field(default_factory=dict)
    emotion_samples: list[dict[str, Any]] = field(default_factory=list)
    emotion_summaries: list[dict[str, Any]] = field(default_factory=list)
    signal_trends: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_decision_dict(self) -> dict[str, Any]:
        return {
            "recent_events": list(self.recent_events),
            "recent_messages": list(self.recent_messages),
            "recent_actions": list(self.recent_actions),
            "reminder_records": list(self.reminder_records),
            "attention_records": list(self.attention_records),
            "environment_records": list(self.environment_records),
            "focus_sessions": list(self.focus_sessions),
            "focus_session_count": self.focus_session_count,
            "focus_total_duration_sec": self.focus_total_duration_sec,
            "distraction_event_count": self.distraction_event_count,
            "state_change_counts": dict(self.state_change_counts),
            "emotion_samples": list(self.emotion_samples),
            "emotion_summaries": list(self.emotion_summaries),
            "signal_trends": {name: dict(s) for name, s in self.signal_trends.items()},
        }


class RuntimeHistoryService:
    """维护当前运行期的短期历史窗口。"""

    def record_event(self, state: "AgentState", event: "Event") -> None:
        history = state.runtime_history
        history.recent_events.append(
            {"type": event.type, "timestamp": event.timestamp, "payload": event.payload}
        )
        self._record_signal_trends(state, event)
        if event.type == "user_emotion_updated":
            self._record_emotion_sample(state, event)
            self._maybe_rollup_emotion_summary(state, event.timestamp)
        if event.type in _USER_STATE_EVENTS:
            history.state_change_counts[event.type] = (
                history.state_change_counts.get(event.type, 0) + 1
            )
        if event.type == "user_attention_updated":
            self._record_attention_event(state, event)
        if event.type in _ENVIRONMENT_EVENTS:
            history.environment_records.append(
                {"type": event.type, "timestamp": event.timestamp, "payload": event.payload}
            )

    def record_message(self, state: "AgentState", role: str, text: str, timestamp: int) -> None:
        state.runtime_history.recent_messages.append(
            {"role": role, "text": text, "timestamp": timestamp}
        )

    def record_action(self, state: "AgentState", action: "Action", timestamp: int) -> None:
        history = state.runtime_history
        history.recent_actions.append(
            {"type": action.type, "timestamp": timestamp, "payload": dict(action.payload)}
        )
        if action.payload.get("kind") != "notification":
            return
        history.reminder_records.append(
            {
                "action_type": action.type,
                "timestamp": timestamp,
                "reason": action.payload.get("reason"),
                "level": action.payload.get("level"),
                "text": action.payload.get("text"),
            }
        )

    def trim(self, state: "AgentState") -> None:
        history = state.runtime_history
        history.recent_events = history.recent_events[-MAX_RECENT_EVENTS:]
        history.recent_messages = history.recent_messages[-MAX_RECENT_MESSAGES:]
        history.recent_actions = history.recent_actions[-MAX_RECENT_ACTIONS:]
        history.reminder_records = history.reminder_records[-MAX_REMINDER_RECORDS:]
        history.attention_records = history.attention_records[-MAX_ATTENTION_RECORDS:]
        history.environment_records = history.environment_records[-MAX_ENVIRONMENT_RECORDS:]
        history.focus_sessions = history.focus_sessions[-MAX_FOCUS_SESSIONS:]
        history.emotion_samples = history.emotion_samples[-MAX_EMOTION_SAMPLES:]
        history.emotion_summaries = history.emotion_summaries[-MAX_EMOTION_SUMMARIES:]
        for trend in history.signal_trends.values():
            trend["recent_values"] = list(trend.get("recent_values", []))[-MAX_SIGNAL_RECENT_VALUES:]
            self._rebuild_window_summaries(trend)

    def _record_signal_trends(self, state: "AgentState", event: "Event") -> None:
        signal_fields = _SIGNAL_FIELDS_BY_EVENT.get(str(event.type), ())
        confidence = _optional_float(event.payload.get("confidence"))
        for signal_name, payload_keys in signal_fields:
            value = _first_payload_value(event.payload, payload_keys)
            if value is None:
                continue
            trend = state.runtime_history.signal_trends.setdefault(
                signal_name,
                {
                    "current": None,
                    "previous": None,
                    "updated_at": None,
                    "last_changed_at": None,
                    "consecutive_same_count": 0,
                    "value_counts": {},
                    "confidence_summary": {"count": 0, "average": None, "minimum": None, "maximum": None},
                    "recent_values": [],
                },
            )
            previous = trend.get("current")
            trend["previous"] = previous
            trend["current"] = value
            trend["updated_at"] = event.timestamp
            if previous != value:
                trend["last_changed_at"] = event.timestamp
                trend["consecutive_same_count"] = 1
            else:
                trend["consecutive_same_count"] = int(trend.get("consecutive_same_count", 0)) + 1
            recent_values = list(trend.get("recent_values", []))
            recent_values.append(
                {"timestamp": event.timestamp, "value": value, "confidence": confidence}
            )
            trend["recent_values"] = recent_values[-MAX_SIGNAL_RECENT_VALUES:]
            if previous != value:
                self._rebuild_window_summaries(trend)

    def _rebuild_window_summaries(self, trend: dict[str, Any]) -> None:
        recent_values = [item for item in trend.get("recent_values", []) if isinstance(item, dict)]
        counts = Counter(str(item.get("value")) for item in recent_values)
        trend["value_counts"] = dict(counts)
        confidences = [
            value
            for item in recent_values
            if (value := _optional_float(item.get("confidence"))) is not None
        ]
        trend["confidence_summary"] = {
            "count": len(confidences),
            "average": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "minimum": min(confidences) if confidences else None,
            "maximum": max(confidences) if confidences else None,
        }

    def _record_attention_event(self, state: "AgentState", event: "Event") -> None:
        history = state.runtime_history
        attention = str(event.payload.get("attention", "idle"))
        record = {
            "timestamp": event.timestamp,
            "attention": attention,
            "behavior": str(event.payload.get("behavior", "unknown")),
            "confidence": event.payload.get("confidence"),
            "source": event.payload.get("source"),
        }
        if "yolo_phone_detected" in event.payload:
            record["yolo_phone_detected"] = bool(event.payload.get("yolo_phone_detected"))
        history.attention_records.append(record)
        if attention == "distracted":
            history.distraction_event_count += 1

    def _record_emotion_sample(self, state: "AgentState", event: "Event") -> None:
        state.runtime_history.emotion_samples.append(
            {
                "timestamp": event.timestamp,
                "emotion": str(event.payload.get("emotion", "neutral")),
                "confidence": event.payload.get("confidence"),
                "source": event.payload.get("source"),
            }
        )

    def _maybe_rollup_emotion_summary(self, state: "AgentState", now_ts: int) -> None:
        history = state.runtime_history
        latest = history.emotion_summaries[-1] if history.emotion_summaries else None
        last_end_ts = int(latest["end_ts"]) if latest else None
        if last_end_ts is not None and now_ts - last_end_ts < EMOTION_SUMMARY_WINDOW_SEC:
            return
        window_start_ts = now_ts - EMOTION_SUMMARY_WINDOW_SEC
        window_samples = [
            item for item in history.emotion_samples if int(item["timestamp"]) >= window_start_ts
        ]
        if not window_samples:
            return
        counts = Counter(str(item.get("emotion", "neutral")) for item in window_samples)
        total = len(window_samples)
        confidences = [
            float(item["confidence"]) for item in window_samples if item.get("confidence") is not None
        ]
        history.emotion_summaries.append(
            {
                "start_ts": window_start_ts,
                "end_ts": now_ts,
                "window_sec": EMOTION_SUMMARY_WINDOW_SEC,
                "sample_count": total,
                "dominant_emotion": counts.most_common(1)[0][0],
                "distribution": {e: round(c / total, 3) for e, c in counts.items()},
                "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            }
        )


def _first_payload_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if payload.get(key) is not None:
            return payload[key]
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
