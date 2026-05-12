"""
确定性 Guard 模块。

本模块位于 IntentPlanValidator 之后、ActionRealizer 之前，负责执行不能交给
LLM 的硬边界：高风险计划阻断、用户离场时禁止打扰、提醒冷却、以及非用户
触发场景禁止自主 LLM 回复。上游输入是已通过 schema 校验的 IntentPlan 和
AgentContext，下游输出是过滤后的 IntentPlan 与拦截原因。

本模块不理解自然语言、不做业务规划、不生成 Action，也不修改 AgentState。
"""

from __future__ import annotations
from dataclasses import dataclass, field

from src.agent.decision.intent_model import AgentIntent, IntentPlan, no_op_plan
from src.agent.decision.agent_context_builder import AgentContext


INTERRUPTIVE_INTENTS = {
    "suggest_rest",
    "remind_distraction",
    "adjust_environment_feedback",
    "voice_interaction",
}

COOLDOWN_REASONS = {
    "suggest_rest": "rest_reminder",
    "remind_distraction": "distraction_reminder",
    "adjust_environment_feedback": "environment_warning",
}


@dataclass
class GuardFinding:
    """单个 intent 的 Guard 结果，用于 trace 和测试解释。"""

    intent_type: str
    allowed: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "intent_type": self.intent_type,
            "allowed": self.allowed,
            "reason": self.reason,
        }


@dataclass
class GuardDecision:
    """Guard 过滤后的计划结果。

    `plan` 只包含允许继续落地的 intent；`blocked_intents` 保留被拦截项，便于
    trace 说明为什么没有执行。
    """

    plan: IntentPlan
    blocked_intents: list[AgentIntent] = field(default_factory=list)
    findings: list[GuardFinding] = field(default_factory=list)

    @property
    def allowed_intents(self) -> list[AgentIntent]:
        return list(self.plan.intents)


class DeterministicGuard:
    """LLM 计划之后的确定性安全过滤器。

    输入 IntentPlan 和 AgentContext，输出 GuardDecision。它不负责语义判断；
    只执行稳定、安全、可审计的系统边界。
    """

    def __init__(self, *, reminder_cooldown_sec: int = 300) -> None:
        self.reminder_cooldown_sec = reminder_cooldown_sec

    def filter(self, plan: IntentPlan, context: AgentContext) -> GuardDecision:
        """过滤不可执行或不应打扰用户的 intent。

        高风险计划直接降级为 no_op；部分 intent 被拦截时保留其余安全 intent。
        这样 LLM 可以负责认知，代码仍然掌握最终执行边界。
        """

        if plan.risk_level == "high":
            finding = GuardFinding("*", False, "high-risk plans require deterministic blocking")
            return GuardDecision(
                plan=no_op_plan("Blocked high-risk LLM plan."),
                blocked_intents=list(plan.intents),
                findings=[finding],
            )

        allowed: list[AgentIntent] = []
        blocked: list[AgentIntent] = []
        findings: list[GuardFinding] = []

        for intent in plan.intents:
            reason = self._block_reason(intent, context)
            if reason:
                blocked.append(intent)
                findings.append(GuardFinding(intent.type, False, reason))
                continue
            allowed.append(intent)
            findings.append(GuardFinding(intent.type, True, "allowed"))

        if not allowed and blocked:
            guarded = no_op_plan("All intents were blocked by deterministic guard.")
        else:
            guarded = IntentPlan(
                intents=allowed,
                reasoning=plan.reasoning,
                risk_level=plan.risk_level,
                interrupt_user=plan.interrupt_user,
                response_requirements=dict(plan.response_requirements),
            )
        return GuardDecision(plan=guarded, blocked_intents=blocked, findings=findings)

    def _block_reason(self, intent: AgentIntent, context: AgentContext) -> str | None:
        """返回单个 intent 的拦截原因。

        这里的判断只使用结构化状态和冷却记录，不做关键词理解，避免 Guard
        重新变成隐藏的业务决策层。
        """

        state = context.state_summary
        user = state.get("user", {}) if isinstance(state, dict) else {}
        if intent.type in INTERRUPTIVE_INTENTS and user.get("presence") == "away":
            return "presence safety blocked an interruptive intent while user is away"

        if intent.type in COOLDOWN_REASONS:
            cooldowns = state.get("cooldowns", {}) if isinstance(state, dict) else {}
            reason = str(intent.payload.get("reason") or COOLDOWN_REASONS[intent.type])
            last_ts = cooldowns.get(reason) if isinstance(cooldowns, dict) else None
            if last_ts is not None:
                try:
                    elapsed = int(context.timestamp) - int(last_ts)
                except (TypeError, ValueError):
                    elapsed = self.reminder_cooldown_sec
                if elapsed < self.reminder_cooldown_sec:
                    return f"cooldown active for {reason}"

        if (
            intent.type == "answer_user"
            and intent.requires_llm
            and context.event_type not in {"user_text_input", "speech_recognized"}
        ):
            return "autonomous LLM reply is not allowed without a user message"
        return None
