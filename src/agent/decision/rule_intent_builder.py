from __future__ import annotations

"""把明确的结构化事件确定性映射为 IntentPlan。

本模块只表达“系统想做什么”，不直接生成 Action。这样规则计划与 LLM 计划可以共享
Validator、Guard 和 ActionRealizer，设备执行边界不会因 fast path 而被绕过。
"""

from src.agent.config.policy_config import RuleIntentPolicyConfig
from src.agent.decision.agent_context_builder import AgentContext
from src.agent.decision.intent_model import AgentIntent, IntentPlan
from src.agent.event import Event
from src.agent.state import AgentState


class RuleIntentBuilder:
    """为少量意图唯一、参数明确的事件构造计划。"""

    def __init__(self, policy_config: RuleIntentPolicyConfig | None = None) -> None:
        self.config = policy_config or RuleIntentPolicyConfig()

    def build(
        self,
        *,
        event: Event,
        previous_state: AgentState,
        current_state: AgentState,
        context: AgentContext | None = None,
    ) -> IntentPlan | None:
        """根据 Reducer 前后状态判断事件是否真正造成了业务状态跃迁。"""

        del context
        if event.type not in self.config.intent_by_event:
            return None
        if event.type == "focus_start_requested":
            return self._build_focus_start(event, previous_state, current_state)
        if event.type == "focus_stop_requested":
            return self._build_focus_stop(previous_state, current_state)
        if event.type == "timer_finished":
            return self._build_timer_finished(previous_state, current_state)
        return None

    def _build_focus_start(
        self,
        event: Event,
        previous_state: AgentState,
        current_state: AgentState,
    ) -> IntentPlan:
        if not previous_state.focus.active and current_state.focus.active:
            payload = {}
            if "duration_sec" in event.payload:
                payload["duration_sec"] = event.payload.get("duration_sec")
            return _single_intent_plan(
                intent_type="start_focus",
                priority=self.config.action_priority,
                reason="explicit focus_start_requested event",
                payload=payload,
                reasoning="RuleIntentBuilder mapped focus_start_requested to start_focus.",
            )
        return _single_intent_plan(
            intent_type="no_op",
            priority=self.config.no_op_priority,
            reason="focus_start_requested ignored because focus was already active before event",
            payload={},
            reasoning="RuleIntentBuilder ignored duplicate focus_start_requested.",
        )

    def _build_focus_stop(
        self,
        previous_state: AgentState,
        current_state: AgentState,
    ) -> IntentPlan:
        if previous_state.focus.active and not current_state.focus.active:
            return _single_intent_plan(
                intent_type="stop_focus",
                priority=self.config.action_priority,
                reason="explicit focus_stop_requested event",
                payload={},
                reasoning="RuleIntentBuilder mapped focus_stop_requested to stop_focus.",
            )
        return _single_intent_plan(
            intent_type="no_op",
            priority=self.config.no_op_priority,
            reason="focus_stop_requested ignored because focus was not active before event",
            payload={},
            reasoning="RuleIntentBuilder ignored inactive focus_stop_requested.",
        )

    def _build_timer_finished(
        self,
        previous_state: AgentState,
        current_state: AgentState,
    ) -> IntentPlan:
        if previous_state.focus.active and not current_state.focus.active:
            return _single_intent_plan(
                intent_type="complete_focus",
                priority=self.config.action_priority,
                reason="focus timer finished",
                payload={},
                reasoning="RuleIntentBuilder mapped timer_finished to complete_focus.",
            )
        return _single_intent_plan(
            intent_type="no_op",
            priority=self.config.no_op_priority,
            reason="timer_finished ignored because no focus session was active before event",
            payload={},
            reasoning="RuleIntentBuilder ignored stale timer_finished.",
        )


def _single_intent_plan(
    *,
    intent_type: str,
    priority: int,
    reason: str,
    payload: dict[str, object],
    reasoning: str,
) -> IntentPlan:
    return IntentPlan(
        intents=[
            AgentIntent(
                type=intent_type,
                priority=priority,
                reason=reason,
                payload=payload,
                requires_llm=False,
            )
        ],
        reasoning=reasoning,
        risk_level="low",
        interrupt_user=False,
        response_requirements={},
    )
