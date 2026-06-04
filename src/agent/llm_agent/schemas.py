"""
LLM Agent schema 模块。

本模块定义四个 LLM 角色之间传递的结构化对象：SituationFrame、
SafetyReview、ResponseDraft 和 AgentRun。上游是各角色的 JSON 输出，下游是
DecisionPipeline 的 validator/guard/realizer。

本模块不调用 LLM、不执行动作、不写状态；它只负责解析、拒绝越界字段并提供
安全 fallback 数据结构。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from src.agent.decision.intent_model import IntentPlan, no_op_plan


RiskLevel = Literal["low", "medium", "high"]
SafetyDecision = Literal["approve", "revise", "reject"]


@dataclass
class SituationFrame:
    """LLM-produced understanding frame.

    The frame describes the situation only. It is not allowed to contain
    actions or state patches.
    """

    summary: str
    user_intent: str = ""
    current_state: str = ""
    risks: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    should_respond: bool = False
    risk_level: str = "low"

    @classmethod
    def from_dict(cls, data: object) -> "SituationFrame":
        if not isinstance(data, dict):
            raise ValueError("situation frame must be an object")
        _reject_control_fields(data, "situation frame")
        return cls(
            summary=str(data.get("summary", "")).strip(),
            user_intent=str(data.get("user_intent", "")).strip(),
            current_state=str(data.get("current_state", "")).strip(),
            risks=_string_list(data.get("risks")),
            uncertainties=_string_list(data.get("uncertainties")),
            should_respond=bool(data.get("should_respond", False)),
            risk_level=_risk_level(data.get("risk_level")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "user_intent": self.user_intent,
            "current_state": self.current_state,
            "risks": list(self.risks),
            "uncertainties": list(self.uncertainties),
            "should_respond": self.should_respond,
            "risk_level": self.risk_level,
        }


@dataclass
class SafetyReview:
    """SafetyCritic 的结构化审查结果。

    decision 可为 approve/revise/reject；若 revise，必须附带完整 revised_plan。
    """

    decision: str = "approve"
    reason: str = ""
    revised_plan: IntentPlan | None = None

    @classmethod
    def from_dict(cls, data: object) -> "SafetyReview":
        if not isinstance(data, dict):
            raise ValueError("safety review must be an object")
        decision = str(data.get("decision", "approve")).strip().lower()
        if decision not in {"approve", "revise", "reject"}:
            raise ValueError(f"unknown safety decision: {decision}")
        revised_plan = None
        if decision == "revise" and data.get("revised_plan") is not None:
            revised_plan = IntentPlan.from_dict(data.get("revised_plan"))
        return cls(
            decision=decision,
            reason=str(data.get("reason", "")).strip(),
            revised_plan=revised_plan,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "revised_plan": self.revised_plan.to_dict() if self.revised_plan else None,
        }


@dataclass
class ResponseDraft:
    """ResponseWriter 生成的表达草稿。

    它只包含 speak/display 文本和 tone，不承担行为决策职责。
    """

    speak_text: str = ""
    display_text: str = ""
    tone: str = "calm"
    already_spoken: bool = False

    @classmethod
    def from_dict(cls, data: object) -> "ResponseDraft":
        if not isinstance(data, dict):
            raise ValueError("response draft must be an object")
        _reject_control_fields(data, "response draft")
        return cls(
            speak_text=str(data.get("speak_text", "")).strip(),
            display_text=str(data.get("display_text", "")).strip(),
            tone=str(data.get("tone", "calm")).strip() or "calm",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "speak_text": self.speak_text,
            "display_text": self.display_text,
            "tone": self.tone,
        }


@dataclass
class AgentRun:
    """四角色 LLM 编排的一轮完整输出，供 DecisionPipeline 继续校验。"""

    situation: SituationFrame
    plan: IntentPlan
    safety_review: SafetyReview
    response: ResponseDraft
    used_llm: bool = True
    fallback_reason: str | None = None
    stage_metadata: dict[str, Any] = field(default_factory=dict)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object, accepting common fenced LLM output."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object")
    return data


def fallback_situation(event_type: str, user_text: str = "") -> SituationFrame:
    summary = f"Received event {event_type}."
    should_respond = event_type in {"user_text_input", "speech_recognized", "timer_finished"}
    if user_text:
        summary = "The user sent a message that needs a direct response."
    return SituationFrame(
        summary=summary,
        user_intent="unknown",
        current_state="available context was used, but the LLM frame was unavailable",
        uncertainties=["LLM situation analysis failed"],
        should_respond=should_respond,
        risk_level="low",
    )


def fallback_plan_for_event(event_type: str, user_text: str = "") -> IntentPlan:
    if event_type == "focus_start_requested":
        return IntentPlan(
            intents=[],
            reasoning="Explicit focus-start event; no semantic inference required.",
        )
    if event_type == "focus_stop_requested":
        return IntentPlan(
            intents=[],
            reasoning="Explicit focus-stop event; no semantic inference required.",
        )
    if event_type == "timer_finished":
        return IntentPlan(
            intents=[],
            reasoning="Explicit timer-finished event; no semantic inference required.",
        )
    if event_type in {"user_text_input", "speech_recognized"} and user_text:
        from src.agent.decision.intent_model import AgentIntent

        return IntentPlan(
            intents=[
                AgentIntent(
                    type="answer_user",
                    priority=50,
                    reason="LLM fallback keeps the user-facing dialogue alive.",
                    payload={"response_mode": "dialogue"},
                    requires_llm=True,
                )
            ],
            reasoning="Fallback response for direct user text.",
        )
    return no_op_plan("No safe fallback intent for this event.")


def _reject_control_fields(data: dict[str, Any], label: str) -> None:
    forbidden = {"actions", "state_patch", "device_command"}
    found = forbidden & set(data)
    if found:
        raise ValueError(f"{label} must not contain control fields: {sorted(found)}")


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value).strip()] if str(value).strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _risk_level(value: object) -> str:
    level = str(value or "low").strip().lower()
    return level if level in {"low", "medium", "high"} else "low"
