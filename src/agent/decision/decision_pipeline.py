from __future__ import annotations

"""DecisionPipeline 决策流水线。

它是什么：
DecisionPipeline 是 Agent 高层决策入口。它只接收 Event、AgentState、PersonalContext
和 LLMService，输出 DecisionResult。

它不是什么：
它不读取 LongTermMemoryStore，不读取 UserProfileStore，不写 RuntimeHistory，不写
LongTermMemory，也不直接执行设备动作。

为什么存在：
决策层必须只消费已经构建好的 PersonalContext，保证数据来源集中在
PersonalContextBuilder，避免“决策层到处找记忆”的架构熵增。

边界：
内部顺序是 AgentContextBuilder -> LLMAgentOrchestrator -> IntentPlanValidator ->
DeterministicGuard -> ActionRealizer。任何 LLM 输出都必须经过 validator/guard 后才能变成 Action。
"""

from src.agent.config.policy_config import DecisionPolicyConfig
from src.agent.user.personal_context import PersonalContext
from src.agent.decision.action_realizer import ActionRealizer
from src.agent.decision.agent_context_builder import AgentContextBuilder
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.decision.decision_result import DecisionResult
from src.agent.decision.guard import DeterministicGuard
from src.agent.decision.intent_model import no_op_plan
from src.agent.decision.validator import IntentPlanValidator
from src.agent.event import Event
from src.agent.llm_agent import LLMAgentOrchestrator
from src.agent.state import AgentState
from src.services.llm_service import LLMService


class DecisionPipeline:
    """LLM-centered 决策入口，只消费 PersonalContext。"""

    def __init__(
        self,
        *,
        context_builder: AgentContextBuilder | None = None,
        orchestrator: LLMAgentOrchestrator | None = None,
        validator: IntentPlanValidator | None = None,
        guard: DeterministicGuard | None = None,
        action_realizer: ActionRealizer | None = None,
        decision_policy: DecisionPolicyConfig | None = None,
    ) -> None:
        self.context_builder = context_builder or AgentContextBuilder()
        self.orchestrator = orchestrator or LLMAgentOrchestrator()
        self.validator = validator or IntentPlanValidator()
        self.guard = guard or DeterministicGuard()
        self.action_realizer = action_realizer or ActionRealizer()
        self.decision_policy = decision_policy or DecisionPolicyConfig()
        self.last_result: DecisionResult | None = None

    def decide(
        self,
        *,
        previous_state: AgentState | None,
        current_state: AgentState,
        event: Event,
        llm_service: LLMService,
        personal_context: PersonalContext | None = None,
        personalized_policy: object | None = None,
    ) -> DecisionResult:
        """执行一轮 Event -> Intent -> Action 决策。"""

        del personalized_policy
        context = self.context_builder.build(
            previous_state=previous_state,
            current_state=current_state,
            event=event,
            personal_context=personal_context,
        )
        ignored_reason = self._ignored_system_trigger_reason(event)
        if ignored_reason:
            result = _ignored_decision_result(context, ignored_reason)
            self.last_result = result
            return result

        agent_run = self.orchestrator.decide(context, llm_service)
        validation = self.validator.validate(agent_run.plan)
        validation_errors = list(validation.errors)
        plan = agent_run.plan
        if validation_errors:
            plan = no_op_plan("LLM intent plan failed validation.")

        guard_decision = self.guard.filter(plan, context)
        actions = self.action_realizer.realize(
            guard_decision.plan,
            response=agent_run.response,
            context=context,
        )

        fallback_reason = agent_run.fallback_reason
        if validation_errors:
            suffix = "validation:" + ";".join(validation_errors)
            fallback_reason = f"{fallback_reason};{suffix}" if fallback_reason else suffix

        result = DecisionResult(
            intents=guard_decision.plan.intents,
            actions=actions,
            blocked_intents=guard_decision.blocked_intents,
            guard_results=guard_decision.findings,
            used_llm=agent_run.used_llm,
            fallback_reason=fallback_reason,
            decision_reason=guard_decision.plan.reasoning,
            situation=agent_run.situation,
            safety_review=agent_run.safety_review,
            response=agent_run.response,
            stage_metadata={
                "context": context.to_prompt_dict(),
                "llm_roles": agent_run.stage_metadata,
                "validator": {"ok": not validation_errors, "errors": validation_errors},
                "guard": [finding.to_dict() for finding in guard_decision.findings],
                "action_realizer": {"action_count": len(actions)},
            },
        )
        self.last_result = result
        return result

    def _ignored_system_trigger_reason(self, event: Event) -> str | None:
        if event.type != "system_triggered":
            return None
        trigger = str(event.payload.get("trigger", "")).strip()
        source = str(event.payload.get("source", "")).strip()
        if source == self.decision_policy.action_result_source:
            return self.decision_policy.ignored_system_trigger_reason
        if trigger in self.decision_policy.internal_system_triggers:
            return self.decision_policy.ignored_system_trigger_reason
        if trigger not in self.decision_policy.allowed_autonomous_triggers:
            return self.decision_policy.ignored_system_trigger_reason
        return None


def _ignored_decision_result(context: AgentContext, reason: str) -> DecisionResult:
    plan = no_op_plan(reason)
    return DecisionResult(
        intents=plan.intents,
        actions=[],
        used_llm=False,
        fallback_reason=reason,
        decision_reason=reason,
        stage_metadata={
            "context": context.to_prompt_dict(),
            "decision_policy": {
                "ignored": True,
                "reason": reason,
            },
            "action_realizer": {"action_count": 0},
        },
    )
