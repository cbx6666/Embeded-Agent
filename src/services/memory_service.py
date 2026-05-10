from __future__ import annotations

"""最近记忆维护服务模块。

MemoryService 只维护 Agent 的短期工作集和轻量统计，
不承担长期用户画像或偏好管理；长期资料放在 UserProfileService。
"""

from collections import Counter

from src.agent.event import Event
from src.agent.state import AgentState


class MemoryService:
    """维护最近事件、最近消息、提醒记录和轻量统计。"""

    def __init__(
        self,
        max_recent_events: int = 20,
        max_recent_messages: int = 20,
        max_reminder_records: int = 50,
        max_attention_records: int = 120,
        max_environment_records: int = 120,
        max_focus_sessions: int = 10,
        max_emotion_samples: int = 120,
        max_emotion_summaries: int = 60,
        emotion_summary_window_sec: int = 60,
    ) -> None:
        # 各类上限控制内存和 JSON 状态体积，适合嵌入式 MVP 长时间运行。
        self.max_recent_events = max_recent_events
        self.max_recent_messages = max_recent_messages
        self.max_reminder_records = max_reminder_records
        self.max_attention_records = max_attention_records
        self.max_environment_records = max_environment_records
        self.max_focus_sessions = max_focus_sessions
        self.max_emotion_samples = max_emotion_samples
        self.max_emotion_summaries = max_emotion_summaries
        self.emotion_summary_window_sec = emotion_summary_window_sec

    def record_event(self, state: AgentState, event: Event) -> None:
        """记录一条标准事件，并按事件类型维护附加统计。"""
        # recent_events 是调试和 /history 展示用的统一事件窗口。
        state.memory.recent_events.append(
            {
                "type": event.type,
                "timestamp": event.timestamp,
                "payload": event.payload,
            }
        )
        # 情绪事件额外保留样本，并按时间窗口滚动生成摘要。
        if event.type == "user_emotion_updated":
            self._record_emotion_sample(state, event)
            self._maybe_rollup_emotion_summary(state, event.timestamp)
        # 用户状态类事件用于粗略统计状态变化次数。
        if event.type in {"user_presence_updated", "user_attention_updated", "user_emotion_updated", "user_fatigue_updated"}:
            self._record_state_change(state, event)
        # 注意力事件单独记录，方便后续统计分心次数和行为分布。
        if event.type == "user_attention_updated":
            self._record_attention_event(state, event)
        # 环境事件单独记录，避免从 recent_events 里再筛一遍。
        if event.type in {"light_level_updated", "temperature_humidity_updated", "noise_level_updated"}:
            self._record_environment_event(state, event)

    def record_message(
        self,
        state: AgentState,
        role: str,
        text: str,
        timestamp: int,
    ) -> None:
        """记录用户、Agent 或显示侧消息。"""
        state.memory.recent_messages.append(
            {
                "role": role,
                "text": text,
                "timestamp": timestamp,
            }
        )

    def record_action(self, state: AgentState, action_type: str, payload: dict[str, object], timestamp: int) -> None:
        """记录提醒类动作，非提醒动作不进入 reminder_records。"""
        if payload.get("kind") != "notification":
            return
        state.memory.reminder_records.append(
            {
                "action_type": action_type,
                "timestamp": timestamp,
                "reason": payload.get("reason"),
                "level": payload.get("level"),
                "text": payload.get("text"),
                "status": payload.get("status"),
                "state": payload.get("state"),
            }
        )

    def trim(self, state: AgentState) -> None:
        """按配置裁剪所有短期记忆窗口。"""
        # 统一在一轮事件处理结束后裁剪，避免每次 append 都重复切片。
        state.memory.recent_events = state.memory.recent_events[-self.max_recent_events :]
        state.memory.recent_messages = state.memory.recent_messages[-self.max_recent_messages :]
        state.memory.reminder_records = state.memory.reminder_records[-self.max_reminder_records :]
        state.memory.attention_records = state.memory.attention_records[-self.max_attention_records :]
        state.memory.environment_records = state.memory.environment_records[-self.max_environment_records :]
        state.memory.focus_sessions = state.memory.focus_sessions[-self.max_focus_sessions :]
        state.memory.emotion_samples = state.memory.emotion_samples[-self.max_emotion_samples :]
        state.memory.emotion_summaries = state.memory.emotion_summaries[-self.max_emotion_summaries :]

    def _record_state_change(self, state: AgentState, event: Event) -> None:
        """累加某类状态事件出现次数。"""
        state.memory.state_change_counts[event.type] = state.memory.state_change_counts.get(event.type, 0) + 1

    def _record_attention_event(self, state: AgentState, event: Event) -> None:
        """记录注意力/行为状态，并统计分心次数。"""
        attention = str(event.payload.get("attention", "idle"))
        behavior = str(event.payload.get("behavior", "unknown"))
        state.memory.attention_records.append(
            {
                "timestamp": event.timestamp,
                "attention": attention,
                "behavior": behavior,
                "confidence": event.payload.get("confidence"),
                "source": event.payload.get("source"),
            }
        )
        if attention == "distracted":
            state.memory.distraction_event_count += 1

    def _record_environment_event(self, state: AgentState, event: Event) -> None:
        """记录环境传感器事件。"""
        state.memory.environment_records.append(
            {
                "type": event.type,
                "timestamp": event.timestamp,
                "payload": event.payload,
            }
        )

    def _record_emotion_sample(self, state: AgentState, event: Event) -> None:
        """记录单条情绪样本，用于短时间窗口统计。"""
        # person_id/model/raw_emotion 保留给后续多用户识别和模型排障。
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
        # 如果距离上一次摘要还没到窗口长度，就继续累计样本。
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

        # 当前摘要只做简单多数票和置信度均值，保持第一版可解释。
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
