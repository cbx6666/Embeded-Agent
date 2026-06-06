from __future__ import annotations

"""DecisionPipeline 决策流水线。

它是什么：
DecisionPipeline 是 Agent 高层决策入口。它接收 Event、AgentState，以及 LLM 路径
可选的 PersonalContext/LLMService，输出 DecisionResult。

它不是什么：
它不读取 LongTermMemoryStore，不读取 UserProfileStore，不写 RuntimeHistory，不写
LongTermMemory，也不直接执行设备动作。

为什么存在：
决策层必须只消费已经构建好的 PersonalContext，保证数据来源集中在
PersonalContextBuilder，避免“决策层到处找记忆”的架构熵增。

边界：
计划来源可以是 RuleIntentBuilder、UnifiedPlanner 或完整 Orchestrator；所有来源最终
统一进入 DecisionPostProcessor，经过 Validator、Guard、ActionRealizer 后才能变成 Action。
"""

from src.agent.config.policy_config import DecisionPolicyConfig
from src.agent.user.personal_context import PersonalContext
from src.agent.decision.action_realizer import ActionRealizer
from src.agent.decision.agent_context_builder import AgentContextBuilder
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.decision.decision_post_processor import DecisionPostProcessor
from src.agent.decision.decision_result import DecisionResult
from src.agent.decision.guard import DeterministicGuard
from src.agent.decision.intent_model import IntentPlan, no_op_plan
from src.agent.decision.rule_intent_builder import RuleIntentBuilder
from src.agent.decision.validator import IntentPlanValidator
from src.agent.event import Event
from src.agent.execution.trace import RuntimeTrace
from src.agent.llm_agent import LLMAgentOrchestrator
from src.agent.llm_agent.schemas import ResponseDraft
from src.agent.state import AgentState
from src.services.llm_service import LLMService


class DecisionPipeline:
    """统一承接 Rule/LLM 计划来源的决策入口。"""

    def __init__(
        self,
        *,
        context_builder: AgentContextBuilder | None = None,
        orchestrator: LLMAgentOrchestrator | None = None,
        validator: IntentPlanValidator | None = None,
        guard: DeterministicGuard | None = None,
        action_realizer: ActionRealizer | None = None,
        rule_intent_builder: RuleIntentBuilder | None = None,
        decision_policy: DecisionPolicyConfig | None = None,
    ) -> None:
        self.context_builder = context_builder or AgentContextBuilder()
        self.orchestrator = orchestrator or LLMAgentOrchestrator()
        self.validator = validator or IntentPlanValidator()
        self.guard = guard or DeterministicGuard()
        self.action_realizer = action_realizer or ActionRealizer()
        self.post_processor = DecisionPostProcessor(
            validator=self.validator,
            guard=self.guard,
            action_realizer=self.action_realizer,
        )
        self.rule_intent_builder = rule_intent_builder or RuleIntentBuilder()
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
        """执行开放语义事件的 LLM 决策。"""

        del personalized_policy
        context = self.context_builder.build(
            previous_state=previous_state,
            current_state=current_state,
            event=event,
            personal_context=personal_context,
        )
        trace = RuntimeTrace()
        trace.add("agent_context", "built", context=context.to_prompt_dict())
        ignored_reason = self._ignored_llm_reason(event)
        if ignored_reason:
            trace.add(
                "validator",
                "skipped",
                ok=True,
                errors=[],
                reason="decision policy ignored the event before LLM planning",
            )
            trace.add("guard", "skipped", findings=[])
            trace.add("action_realizer", "realized", action_count=0)
            trace.add("action", "planned", actions=[])
            result = _ignored_decision_result(context, ignored_reason, trace=trace)
            self.last_result = result
            return result

        agent_run = self.orchestrator.decide(
            context,
            llm_service,
            llm_mode=self.decision_policy.llm_mode,
        )
        for role_name, meta in agent_run.stage_metadata.items():
            if not isinstance(meta, dict):
                continue
            if "prompt" in meta:
                trace.add("prompt", str(role_name), prompt=meta.get("prompt"))
            trace.add(
                "llm_output",
                str(role_name),
                raw=meta.get("raw"),
                fallback=bool(meta.get("fallback", False)),
                error=meta.get("error"),
                skipped=bool(meta.get("skipped", False)),
            )
        roles_called = _called_llm_roles(agent_run.stage_metadata)
        result = self.post_processor.finalize(
            plan=agent_run.plan,
            context=context,
            decision_source="orchestrator",
            used_llm=agent_run.used_llm,
            response=agent_run.response,
            situation=agent_run.situation,
            safety_review=agent_run.safety_review,
            fallback_reason=agent_run.fallback_reason,
            source_metadata={
                "llm_roles": agent_run.stage_metadata,
                "llm_mode": self.decision_policy.llm_mode,
                "llm_roles_called": roles_called,
                "llm_call_count": len(roles_called),
            },
            trace=trace,
        )
        self.last_result = result
        return result

    def decide_structured(
        self,
        *,
        previous_state: AgentState,
        current_state: AgentState,
        event: Event,
    ) -> DecisionResult:
        """让 P0B 结构化事件走 0 LLM 规则链。"""

        context = self.context_builder.build(
            previous_state=previous_state,
            current_state=current_state,
            event=event,
            personal_context=None,
        )
        trace = RuntimeTrace()
        trace.add("agent_context", "built", context=context.to_prompt_dict())

        plan = self.rule_intent_builder.build(
            event=event,
            previous_state=previous_state,
            current_state=current_state,
            context=context,
        )
        if plan is None:
            plan = no_op_plan(f"RuleIntentBuilder does not support event: {event.type}")

        rule_reason = plan.reasoning
        trace.add(
            "rule_intent_builder",
            "plan_built",
            decision_source="rule_intent_builder",
            structured_decision=True,
            used_llm=False,
            rule_event_type=event.type,
            rule_reason=rule_reason,
            plan=plan.to_dict(),
        )

        response = ResponseDraft()
        result = self.post_processor.finalize(
            plan=plan,
            context=context,
            decision_source="rule_intent_builder",
            used_llm=False,
            response=response,
            source_metadata={
                "structured_decision": True,
                "rule_event_type": str(event.type),
                "rule_reason": rule_reason,
                "llm_mode": "none",
                "llm_roles_called": [],
                "llm_call_count": 0,
            },
            trace=trace,
        )
        self.last_result = result
        return result

    def decide_prebuilt(
        self,
        *,
        previous_state: AgentState,
        current_state: AgentState,
        event: Event,
        plan: IntentPlan,
        decision_source: str,
        source_metadata: dict[str, object] | None = None,
    ) -> DecisionResult:
        """让 P1 rule 等预构造计划复用统一后处理链。"""

        context = self.context_builder.build(
            previous_state=previous_state,
            current_state=current_state,
            event=event,
            personal_context=None,
        )
        trace = RuntimeTrace()
        trace.add("agent_context", "built", context=context.to_prompt_dict())
        trace.add(
            decision_source,
            "plan_built",
            decision_source=decision_source,
            used_llm=False,
            plan=plan.to_dict(),
        )
        result = self.post_processor.finalize(
            plan=plan,
            context=context,
            decision_source=decision_source,
            used_llm=False,
            response=ResponseDraft(),
            source_metadata={
                **dict(source_metadata or {}),
                "llm_mode": "none",
                "llm_roles_called": [],
                "llm_call_count": 0,
            },
            trace=trace,
        )
        self.last_result = result
        return result

    def _ignored_llm_reason(self, event: Event) -> str | None:
        if event.type in self.decision_policy.llm_skipped_event_types:
            return f"voice pipeline event skipped before dialogue: {event.type}"
        return self._ignored_system_trigger_reason(event)

    def _ignored_system_trigger_reason(self, event: Event) -> str | None:
        if event.type != "system_triggered":
            return None
        trigger = str(event.payload.get("trigger", "")).strip()
        source = str(event.payload.get("source", "")).strip()
        if not self.decision_policy.is_allowed_trigger(trigger, source):
            return self.decision_policy.ignored_system_trigger_reason
        return None


def _ignored_decision_result(context: AgentContext, reason: str, *, trace: RuntimeTrace) -> DecisionResult:
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
            "trace": trace.to_dict(),
        },
    )


def _called_llm_roles(stage_metadata: dict[str, object]) -> list[str]:
    """提取真实发起过调用的角色，跳过条件关闭的阶段。"""

    return [
        str(role)
        for role, metadata in stage_metadata.items()
        if not (isinstance(metadata, dict) and metadata.get("skipped"))
    ]
