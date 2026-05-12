"""
LLM-centered 决策流水线模块。

本模块是 Agent 高层决策的唯一入口。它输入 Reducer 更新后的 AgentState、
当前 Event、ProfileSnapshot 和 LLMService，输出 DecisionResult。内部顺序是：
AgentContextBuilder -> LLMAgentOrchestrator -> IntentPlanValidator ->
DeterministicGuard -> ActionRealizer。

本模块不做关键词理解、不读取旧策略配置、不直接执行设备、不写入长期记忆。
它只负责串联认知链路和确定性边界，并把每个阶段的结果写入 trace metadata。
"""

from __future__ import annotations

"""
LLM-centered 决策流水线模块。

本模块是 Agent 高层决策的唯一入口。它输入 Reducer 更新后的 AgentState、
当前 Event、ProfileSnapshot 和 LLMService，输出 DecisionResult。内部顺序是：
AgentContextBuilder -> LLMAgentOrchestrator -> IntentPlanValidator ->
DeterministicGuard -> ActionRealizer。

本模块不做关键词理解、不读取旧策略配置、不直接执行设备、不写入长期记忆。
它只负责串联认知链路和确定性边界，并把每个阶段的结果写入 trace metadata。
"""

from src.agent.decision.action_realizer import ActionRealizer
from src.agent.decision.decision_result import DecisionResult
from src.agent.decision.guard import DeterministicGuard
from src.agent.decision.intent_model import no_op_plan
from src.agent.decision.validator import IntentPlanValidator
from src.agent.event import Event
from src.agent.llm_agent import AgentContextBuilder, LLMAgentOrchestrator
from src.agent.memory.profile_snapshot_builder import ProfileSnapshot, ProfileSnapshotBuilder
from src.agent.state import AgentState
from src.services.llm_service import LLMService
from src.services.user_profile_service import UserProfileService


class DecisionPipeline:
    """LLM-centered 决策入口。

    输入当前事件和状态，输出意图、动作以及可解释 trace。语义理解由 LLM 角色
    完成；本类只负责编排和边界校验，不承担业务规则推理。
    """

    def __init__(
        self,
        *,
        context_builder: AgentContextBuilder | None = None,
        orchestrator: LLMAgentOrchestrator | None = None,
        validator: IntentPlanValidator | None = None,
        guard: DeterministicGuard | None = None,
        action_realizer: ActionRealizer | None = None,
        profile_snapshot_builder: ProfileSnapshotBuilder | None = None,
    ) -> None:
        self.context_builder = context_builder or AgentContextBuilder()
        self.orchestrator = orchestrator or LLMAgentOrchestrator()
        self.validator = validator or IntentPlanValidator()
        self.guard = guard or DeterministicGuard()
        self.action_realizer = action_realizer or ActionRealizer()
        self.profile_snapshot_builder = profile_snapshot_builder or ProfileSnapshotBuilder()
        self.last_result: DecisionResult | None = None

    def decide(
        self,
        *,
        previous_state: AgentState | None,
        current_state: AgentState,
        event: Event,
        llm_service: LLMService,
        profile_service: UserProfileService | None = None,
        personalized_policy: object | None = None,
        profile_snapshot: ProfileSnapshot | None = None,
    ) -> DecisionResult:
        """执行一轮 Event -> Intent -> Action 决策。

        LLM 输出非法 schema 时降级为 no_op；Guard 拦截后仍会保留被拦截原因。
        这样外部调试可以看到失败发生在 LLM、validator、guard 还是 realizer。
        """

        del personalized_policy

        snapshot = profile_snapshot or self.profile_snapshot_builder.build(
            user_id=current_state.current_user_id,
            state=current_state,
            event=event,
            profile_service=profile_service,
        )
        context = self.context_builder.build(
            previous_state=previous_state,
            current_state=current_state,
            event=event,
            profile_snapshot=snapshot,
        )

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
