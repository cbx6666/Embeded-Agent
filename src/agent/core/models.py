from __future__ import annotations

"""Agent 核心数据模型。

集中定义决策与执行环节共享的小型模型：

- ``Intent``：决策层选择的语义意图（不是设备动作）。
- ``ActionResult``：单个动作的执行结果。
- ``DecisionResult``：一轮决策的结果（意图 + 动作 + 来源）。

并从这里转出 ``Event`` / ``Action``，方便核心层统一引用。
"""

from dataclasses import dataclass, field
from typing import Any

from src.agent.action.action_model import Action
from src.agent.event.event_model import Event

# 决策层允许产生的意图类型闭集。
REGISTERED_INTENT_TYPES: frozenset[str] = frozenset(
    {
        "no_op",
        "answer_user",
        "start_focus",
        "stop_focus",
        "complete_focus",
        "suggest_rest",
        "offer_emotion_care",
        "remind_distraction",
        "adjust_environment_feedback",
        "set_tts_volume",
        "report_sensor_status",
        "media_control",
        "suggest_media",
        "play_media",
        "stop_media",
        "pause_media",
        "resume_media",
        "next_media",
    }
)


@dataclass
class Intent:
    """决策层语义意图：描述"系统想做什么"，由 ActionRealizer 落地为 Action。"""

    type: str
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "reason": self.reason, "payload": dict(self.payload)}


@dataclass
class ActionResult:
    """单个动作的执行结果。"""

    action_type: str
    success: bool
    timestamp: int
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "success": self.success,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "payload": dict(self.payload),
        }


@dataclass
class DecisionResult:
    """一轮决策的结果。"""

    intents: list[Intent] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    used_llm: bool = False
    source: str = ""
    reason: str = ""
    reply_text: str = ""
    # 结构化日志字段：各自主检查 handler 填入，供 AgentCore 输出诊断 JSON。
    log_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intents": [intent.to_dict() for intent in self.intents],
            "actions": [{"type": a.type, "payload": dict(a.payload)} for a in self.actions],
            "used_llm": self.used_llm,
            "source": self.source,
            "reason": self.reason,
            "reply_text": self.reply_text,
            "log_fields": dict(self.log_fields),
        }


__all__ = [
    "Event",
    "Action",
    "Intent",
    "ActionResult",
    "DecisionResult",
    "REGISTERED_INTENT_TYPES",
]
