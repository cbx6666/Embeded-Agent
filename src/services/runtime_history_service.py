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

from src.agent.action import Action
from src.agent.config.policy_config import RuntimeHistoryPolicyConfig
from src.agent.event import Event
from src.agent.state import AgentState


class RuntimeHistoryService:
    """维护当前运行期的短期历史窗口。"""

    def __init__(
        self,
        *,
        policy_config: RuntimeHistoryPolicyConfig | None = None,
    ) -> None:
        self.policy_config = policy_config or RuntimeHistoryPolicyConfig()

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
        if event.type == "user_emotion_updated":
            self._record_emotion_sample(state, event)
            self._maybe_rollup_emotion_summary(state, event.timestamp)
        if event.type in {"user_presence_updated", "user_attention_updated", "user_emotion_updated", "user_fatigue_updated"}:
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
