"""
Intent 数据模型模块。

本模块定义 LLM-centered 决策边界允许的注册 intent 类型、AgentIntent 和
IntentPlan。上游输入是 IntentPlanner/SafetyCritic 的 JSON 输出，下游是
IntentPlanValidator、DeterministicGuard 和 ActionRealizer。

本模块不生成 Action、不执行设备、不修改状态；它只提供结构化语义层，确保
LLM 只能在注册 intent 词表内表达计划。
"""

from __future__ import annotations

"""Intent model for the LLM-centered decision boundary."""

from dataclasses import dataclass, field
from typing import Any

REGISTERED_INTENT_TYPES: frozenset[str] = frozenset(
    {
        "answer_user",
        "start_focus",
        "stop_focus",
        "complete_focus",
        "suggest_rest",
        "remind_distraction",
        "update_status_feedback",
        "adjust_environment_feedback",
        "voice_interaction",
        "display_update",
        "continue_focus",
        "reduce_reminder_frequency",
        "set_tts_volume",
        "no_op",
    }
)


@dataclass
class AgentIntent:
    """LLM 生成、代码校验的语义意图。

    它描述“系统想做什么”，但不等同于设备动作。只有通过 validator 和 guard
    后，ActionRealizer 才能把它落地为 Action。
    """

    type: str
    priority: int = 0
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    requires_llm: bool = False

    @classmethod
    def from_dict(cls, data: object) -> "AgentIntent":
        if not isinstance(data, dict):
            raise ValueError("intent must be an object")
        payload = data.get("payload", {})
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("intent.payload must be an object")
        return cls(
            type=str(data.get("type", "")).strip(),
            priority=_safe_int(data.get("priority"), default=0),
            reason=str(data.get("reason", "")).strip(),
            payload=dict(payload),
            requires_llm=bool(data.get("requires_llm", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "priority": self.priority,
            "reason": self.reason,
            "payload": dict(self.payload),
            "requires_llm": self.requires_llm,
        }


@dataclass
class IntentPlan:
    """Validated intermediate plan.

    It contains only intents and planning metadata. It never contains device
    actions or state patches.
    """

    intents: list[AgentIntent] = field(default_factory=list)
    reasoning: str = ""
    risk_level: str = "low"
    interrupt_user: bool = False
    response_requirements: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: object) -> "IntentPlan":
        if not isinstance(data, dict):
            raise ValueError("intent plan must be an object")
        if "actions" in data or "state_patch" in data:
            raise ValueError("intent plan must not contain actions or state_patch")
        intents_raw = data.get("intents", [])
        if not isinstance(intents_raw, list):
            raise ValueError("intent plan intents must be a list")
        requirements = data.get("response_requirements", {})
        if requirements is None:
            requirements = {}
        if not isinstance(requirements, dict):
            raise ValueError("response_requirements must be an object")
        return cls(
            intents=[AgentIntent.from_dict(item) for item in intents_raw],
            reasoning=str(data.get("reasoning") or data.get("reason") or "").strip(),
            risk_level=str(data.get("risk_level") or "low").strip() or "low",
            interrupt_user=bool(data.get("interrupt_user", False)),
            response_requirements=dict(requirements),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intents": [intent.to_dict() for intent in self.intents],
            "reasoning": self.reasoning,
            "risk_level": self.risk_level,
            "interrupt_user": self.interrupt_user,
            "response_requirements": dict(self.response_requirements),
        }


def no_op_plan(reason: str) -> IntentPlan:
    return IntentPlan(intents=[AgentIntent(type="no_op", reason=reason)], reasoning=reason)


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
