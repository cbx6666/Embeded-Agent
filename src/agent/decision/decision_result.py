"""
DecisionResult 结果模型模块。

本模块定义 LLM-centered 决策链路的一轮输出。上游是 DecisionPipeline，下游是
AgentCore、AgentLoop trace 和测试断言。

本模块不执行动作、不修改状态、不调用 LLM；它只保存 intents、actions、guard
结果、fallback 原因和阶段 metadata，方便调试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.action import Action
from src.agent.decision.guard import GuardFinding
from src.agent.decision.intent_model import AgentIntent
from src.agent.llm_agent.schemas import ResponseDraft, SafetyReview, SituationFrame


@dataclass
class DecisionResult:
    """一轮决策的可解释结果。"""

    intents: list[AgentIntent] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    blocked_intents: list[AgentIntent] = field(default_factory=list)
    guard_results: list[GuardFinding] = field(default_factory=list)
    used_llm: bool = False
    fallback_reason: str | None = None
    decision_reason: str = ""
    situation: SituationFrame | None = None
    safety_review: SafetyReview | None = None
    response: ResponseDraft | None = None
    stage_metadata: dict[str, Any] = field(default_factory=dict)

    def trace_summary(self) -> dict[str, Any]:
        """生成短 trace 摘要，避免日志输出完整上下文。"""

        return {
            "used_llm": self.used_llm,
            "fallback_reason": self.fallback_reason,
            "decision_reason": self.decision_reason,
            "intent_count": len(self.intents),
            "blocked_count": len(self.blocked_intents),
            "action_count": len(self.actions),
            "stages": sorted(self.stage_metadata),
        }
