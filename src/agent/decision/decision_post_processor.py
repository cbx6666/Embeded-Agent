from __future__ import annotations

"""Rule 与 LLM 决策共用的确定性后处理链。

上游只需要提供 IntentPlan 和可选 ResponseDraft；本模块统一执行 Validator、
Guard、ActionRealizer，并组装 DecisionResult。无论计划来自规则还是 LLM，都不能
绕过这条边界直接生成设备 Action。
"""

from typing import Any

from src.agent.decision.action_realizer import ActionRealizer
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.decision.decision_result import DecisionResult
from src.agent.decision.guard import DeterministicGuard
from src.agent.decision.intent_model import IntentPlan, no_op_plan
from src.agent.decision.validator import IntentPlanValidator
from src.agent.execution.trace import RuntimeTrace
from src.agent.llm_agent.schemas import ResponseDraft, SafetyReview, SituationFrame


class DecisionPostProcessor:
    """把任意来源的 IntentPlan 收敛成可执行 DecisionResult。"""

    def __init__(
        self,
        *,
        validator: IntentPlanValidator | None = None,
        guard: DeterministicGuard | None = None,
        action_realizer: ActionRealizer | None = None,
    ) -> None:
        self.validator = validator or IntentPlanValidator()
        self.guard = guard or DeterministicGuard()
        self.action_realizer = action_realizer or ActionRealizer()

    def finalize(
        self,
        *,
        plan: IntentPlan,
        context: AgentContext,
        decision_source: str,
        used_llm: bool,
        response: ResponseDraft | None = None,
        situation: SituationFrame | None = None,
        safety_review: SafetyReview | None = None,
        fallback_reason: str | None = None,
        source_metadata: dict[str, Any] | None = None,
        trace: RuntimeTrace | None = None,
    ) -> DecisionResult:
        """执行稳定后处理，并把每个阶段写入 trace。"""

        runtime_trace = trace or RuntimeTrace()
        draft = response or ResponseDraft()

        validation = self.validator.validate(plan)
        validation_errors = list(validation.errors)
        runtime_trace.add(
            "validator",
            "intent_plan",
            ok=not validation_errors,
            errors=validation_errors,
            plan=plan.to_dict(),
        )
        validated_plan = plan
        if validation_errors:
            validated_plan = no_op_plan("Intent plan failed validation.")

        guard_decision = self.guard.filter(validated_plan, context)
        runtime_trace.add(
            "guard",
            "filtered",
            findings=[finding.to_dict() for finding in guard_decision.findings],
            blocked_intents=[intent.to_dict() for intent in guard_decision.blocked_intents],
            allowed_intents=[intent.to_dict() for intent in guard_decision.allowed_intents],
        )

        actions = self.action_realizer.realize(
            guard_decision.plan,
            response=draft,
            context=context,
        )
        runtime_trace.add("action_realizer", "realized", action_count=len(actions))
        runtime_trace.add(
            "action",
            "planned",
            actions=[_action_to_dict(action) for action in actions],
        )

        effective_fallback = fallback_reason
        if validation_errors:
            validation_reason = "validation:" + ";".join(validation_errors)
            effective_fallback = (
                f"{effective_fallback};{validation_reason}"
                if effective_fallback
                else validation_reason
            )

        metadata = dict(source_metadata or {})
        metadata.update(
            {
                "decision_source": decision_source,
                "used_llm": used_llm,
                "context": context.to_prompt_dict(),
                "validator": {
                    "ok": not validation_errors,
                    "errors": validation_errors,
                },
                "guard": [
                    finding.to_dict()
                    for finding in guard_decision.findings
                ],
                "action_realizer": {"action_count": len(actions)},
                "trace": runtime_trace.to_dict(),
            }
        )
        return DecisionResult(
            intents=guard_decision.plan.intents,
            actions=actions,
            blocked_intents=guard_decision.blocked_intents,
            guard_results=guard_decision.findings,
            used_llm=used_llm,
            fallback_reason=effective_fallback,
            decision_reason=guard_decision.plan.reasoning,
            situation=situation,
            safety_review=safety_review,
            response=draft,
            stage_metadata=metadata,
        )


def _action_to_dict(action: object) -> dict[str, object]:
    return {
        "type": getattr(action, "type", ""),
        "payload": dict(getattr(action, "payload", {}) or {}),
    }
