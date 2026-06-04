"""
IntentPlan 校验模块。

本模块位于 LLM Agent 输出之后、DeterministicGuard 之前，负责验证模型生成的
IntentPlan 是否符合系统白名单和结构边界。上游输入是 IntentPlanner 或
SafetyCritic 产生的 IntentPlan，下游输出是 IntentPlanValidation。

本模块不判断用户语义、不生成 Action、不修改 AgentState。这里拒绝未注册
intent、LLM 夹带 action/state_patch 等越界输出，防止 prompt 注入绕过设备边界。
"""

from __future__ import annotations
from dataclasses import dataclass, field

from src.agent.decision.intent_model import IntentPlan, REGISTERED_INTENT_TYPES


@dataclass
class IntentPlanValidation:
    """IntentPlan 校验结果。

    `errors` 为空代表可以继续进入 DeterministicGuard；非空时调用方应降级为
    no_op 或其他安全 fallback。
    """

    plan: IntentPlan
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class IntentPlanValidator:
    """LLM IntentPlan 的确定性 schema 边界。

    输入是模型生成的 IntentPlan，输出是结构化校验结果。它只验证边界，不做
    业务策略判断，也不替 LLM 重新规划。
    """

    def validate(self, plan: IntentPlan) -> IntentPlanValidation:
        """校验注册 intent、payload 类型和禁止字段。

        如果 LLM 输出 action 或 state_patch，说明模型试图越过 Intent 层直接
        控制系统，必须拒绝。
        """

        errors: list[str] = []
        for index, intent in enumerate(plan.intents):
            if intent.type not in REGISTERED_INTENT_TYPES:
                errors.append(f"intent[{index}] has unregistered type: {intent.type}")
            if not isinstance(intent.payload, dict):
                errors.append(f"intent[{index}] payload must be an object")
            if "action" in intent.payload or "actions" in intent.payload:
                errors.append(f"intent[{index}] payload must not contain actions")
            if "state_patch" in intent.payload:
                errors.append(f"intent[{index}] payload must not contain state_patch")
        if plan.risk_level not in {"low", "medium", "high"}:
            errors.append(f"unknown plan risk_level: {plan.risk_level}")
        return IntentPlanValidation(plan=plan, errors=errors)
