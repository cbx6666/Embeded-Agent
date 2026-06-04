from __future__ import annotations

"""AgentState 的运行时历史子结构。

它是什么：
RuntimeHistory 是 AgentState 的一个字段，保存当前进程和当前会话窗口里的短期工作
历史，包括最近事件、最近消息、最近动作、提醒记录、注意力/情绪/环境采样以及少量
滚动统计。

它不是什么：
它不是长期记忆，不保存稳定偏好，不保存用户身份资料，也不作为决策层的长期真相。

为什么存在：
Agent 需要知道"刚刚发生了什么"，但这类数据变化频繁、生命周期短。如果把它叫作
memory，很容易与 LongTermMemory 混淆，进而污染长期个性化数据。

为什么在这里：
RuntimeHistory 是 AgentState 的直接子结构（state.runtime_history），放在 state/
目录中与 AgentState 定义在一起，避免误导性的独立 history/ 目录。

边界：
RuntimeHistory 只能由事件归约和 RuntimeHistoryService 更新；LongTermMemoryPipeline
可以读取它的摘要作为观察材料，但 RuntimeHistory 不能依赖长期记忆。
"""

from dataclasses import dataclass, field
from typing import Any


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

    def to_decision_dict(self) -> dict[str, Any]:
        """生成 PersonalContext 可读取的只读短期历史摘要。"""

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
        }
