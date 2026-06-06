from __future__ import annotations

"""RuntimeHistory 维护服务。

它是什么：
RuntimeHistoryService 是短期历史的唯一写入入口，负责记录最近事件、消息、动作、
提醒、环境和情绪滚动摘要，并在一轮事件结束后裁剪窗口。

它不是什么：
它不是长期记忆管线，不做 LLM 提取，不写 LongTermMemoryStore，不维护 UserProfile。

为什么存在：
短期历史需要高频更新和容量控制，把这些逻辑集中在服务里，能让 Core 只表达事件流，
避免把“最近发生了什么”的细节散落到决策或长期记忆模块中。

边界：
上游只能是 Event/Action 执行结果；下游是 AgentState.runtime_history 和
PersonalContextBuilder。它不读取 LongTermMemory，也不依赖 UserProfile。
"""

from collections import Counter
from typing import Any

from src.agent.action import Action
from src.agent.config.policy_config import (
    RuntimeHistoryPolicyConfig,
    SignalAggregationPolicyConfig,
)
from src.agent.event import Event
from src.agent.state import AgentState


class RuntimeHistoryService:
    """维护当前运行期的短期历史窗口。"""

    def __init__(
        self,
        *,
        policy_config: RuntimeHistoryPolicyConfig | None = None,
        signal_policy_config: SignalAggregationPolicyConfig | None = None,
    ) -> None:
        self.policy_config = policy_config or RuntimeHistoryPolicyConfig()
        self.signal_policy_config = signal_policy_config or SignalAggregationPolicyConfig()

    def record_event(self, state: AgentState, event: Event) -> None:
        """记录一条标准事件，并维护与运行期相关的滚动统计。"""

        history = state.runtime_history
        history.recent_events.append(
            {
                "type": event.type,
                "timestamp": event.timestamp,
                "payload": event.payload,
            }
        )
        self._record_signal_trends(state, event)
        if event.type == "user_emotion_updated":
            self._record_emotion_sample(state, event)
            self._maybe_rollup_emotion_summary(state, event.timestamp)
        if event.type in {
            "user_presence_updated",
            "user_attention_updated",
            "user_emotion_updated",
            "user_fatigue_updated",
            "user_posture_updated",
            "user_activity_updated",
        }:
            self._record_state_change(state, event)
        if event.type == "user_attention_updated":
            self._record_attention_event(state, event)
        if event.type in {"light_level_updated", "temperature_humidity_updated", "noise_level_updated"}:
            self._record_environment_event(state, event)

    def record_message(self, state: AgentState, role: str, text: str, timestamp: int) -> None:
        """记录用户、Agent 或显示侧消息。"""

        state.runtime_history.recent_messages.append(
            {
                "role": role,
                "text": text,
                "timestamp": timestamp,
            }
        )

    def record_action(self, state: AgentState, action: Action, timestamp: int) -> None:
        """记录最近执行动作；提醒类动作额外进入 reminder_records。"""

        history = state.runtime_history
        history.recent_actions.append(
            {
                "type": action.type,
                "timestamp": timestamp,
                "payload": dict(action.payload),
            }
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
                "status": action.payload.get("status"),
                "state": action.payload.get("state"),
            }
        )

    def trim(self, state: AgentState) -> None:
        """按配置裁剪所有短期历史窗口，避免 runtime state 无界膨胀。"""

        history = state.runtime_history
        history.recent_events = history.recent_events[-self.policy_config.max_recent_events :]
        history.recent_messages = history.recent_messages[-self.policy_config.max_recent_messages :]
        history.recent_actions = history.recent_actions[-self.policy_config.max_recent_actions :]
        history.reminder_records = history.reminder_records[-self.policy_config.max_reminder_records :]
        history.attention_records = history.attention_records[-self.policy_config.max_attention_records :]
        history.environment_records = history.environment_records[-self.policy_config.max_environment_records :]
        history.focus_sessions = history.focus_sessions[-self.policy_config.max_focus_sessions :]
        history.emotion_samples = history.emotion_samples[-self.policy_config.max_emotion_samples :]
        history.emotion_summaries = history.emotion_summaries[-self.policy_config.max_emotion_summaries :]
        for trend in history.signal_trends.values():
            recent_values = list(trend.get("recent_values", []))
            trend["recent_values"] = recent_values[-self.policy_config.max_signal_recent_values :]
            self._rebuild_window_summaries(trend)

    def _record_signal_trends(self, state: AgentState, event: Event) -> None:
        signal_fields = self.signal_policy_config.fields_by_event.get(str(event.type), ())
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
                    "confidence_summary": {
                        "count": 0,
                        "average": None,
                        "minimum": None,
                        "maximum": None,
                    },
                    "recent_values": [],
                },
            )
            previous = trend.get("current")
            changed = previous != value
            trend["previous"] = previous
            trend["current"] = value
            trend["updated_at"] = event.timestamp
            if changed:
                trend["last_changed_at"] = event.timestamp
                trend["consecutive_same_count"] = 1
            else:
                trend["consecutive_same_count"] = int(trend.get("consecutive_same_count", 0)) + 1
            recent_values = list(trend.get("recent_values", []))
            recent_values.append(
                {
                    "timestamp": event.timestamp,
                    "value": value,
                    "confidence": confidence,
                }
            )
            trend["recent_values"] = recent_values[-self.policy_config.max_signal_recent_values :]
            self._rebuild_window_summaries(trend)

    def _rebuild_window_summaries(self, trend: dict[str, Any]) -> None:
        recent_values = [
            item
            for item in trend.get("recent_values", [])
            if isinstance(item, dict)
        ]
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

    def _record_state_change(self, state: AgentState, event: Event) -> None:
        history = state.runtime_history
        history.state_change_counts[event.type] = history.state_change_counts.get(event.type, 0) + 1

    def _record_attention_event(self, state: AgentState, event: Event) -> None:
        history = state.runtime_history
        attention = str(event.payload.get("attention", "idle"))
        behavior = str(event.payload.get("behavior", "unknown"))
        history.attention_records.append(
            {
                "timestamp": event.timestamp,
                "attention": attention,
                "behavior": behavior,
                "confidence": event.payload.get("confidence"),
                "source": event.payload.get("source"),
            }
        )
        if attention == "distracted":
            history.distraction_event_count += 1

    def _record_environment_event(self, state: AgentState, event: Event) -> None:
        state.runtime_history.environment_records.append(
            {
                "type": event.type,
                "timestamp": event.timestamp,
                "payload": event.payload,
            }
        )

    def _record_emotion_sample(self, state: AgentState, event: Event) -> None:
        state.runtime_history.emotion_samples.append(
            {
                "timestamp": event.timestamp,
                "emotion": str(event.payload.get("emotion", "neutral")),
                "confidence": event.payload.get("confidence"),
                "person_id": event.payload.get("person_id"),
                "source": event.payload.get("source"),
                "model": event.payload.get("model"),
                "raw_emotion": event.payload.get("raf_emotion"),
            }
        )

    def _maybe_rollup_emotion_summary(self, state: AgentState, now_ts: int) -> None:
        history = state.runtime_history
        latest_summary = history.emotion_summaries[-1] if history.emotion_summaries else None
        last_end_ts = int(latest_summary["end_ts"]) if latest_summary else None
        if last_end_ts is not None and now_ts - last_end_ts < self.policy_config.emotion_summary_window_sec:
            return

        window_start_ts = now_ts - self.policy_config.emotion_summary_window_sec
        window_samples = [
            item for item in history.emotion_samples if int(item["timestamp"]) >= window_start_ts
        ]
        if not window_samples:
            return

        emotions = [str(item.get("emotion", "neutral")) for item in window_samples]
        counts = Counter(emotions)
        dominant_emotion = counts.most_common(1)[0][0]
        total = len(window_samples)
        confidences = [
            float(item["confidence"])
            for item in window_samples
            if item.get("confidence") is not None
        ]
        history.emotion_summaries.append(
            {
                "start_ts": window_start_ts,
                "end_ts": now_ts,
                "window_sec": self.policy_config.emotion_summary_window_sec,
                "sample_count": total,
                "dominant_emotion": dominant_emotion,
                "distribution": {emotion: round(count / total, 3) for emotion, count in counts.items()},
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
