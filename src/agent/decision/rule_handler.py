from __future__ import annotations

"""结构化控制事件的规则处理器（0 LLM）。

处理 focus_start_requested / focus_stop_requested / timer_finished。意图由事件类型
和 Reducer 前后状态唯一确定，不进入 LLM。
"""

from src.agent.action.realizer import ActionRealizer
from src.agent.core.models import DecisionResult, Event, Intent
from src.agent.state.agent_state import AgentState


class RuleHandler:
    def __init__(self, realizer: ActionRealizer | None = None) -> None:
        self.realizer = realizer or ActionRealizer()

    def decide(
        self,
        *,
        event: Event,
        previous_state: AgentState,
        current_state: AgentState,
    ) -> DecisionResult:
        intent = self._intent_for(event, previous_state, current_state)
        intents = [intent]
        actions = self.realizer.realize(intents)
        return DecisionResult(
            intents=intents,
            actions=actions,
            used_llm=False,
            source="rule",
            reason=intent.reason,
        )

    def _intent_for(
        self,
        event: Event,
        previous_state: AgentState,
        current_state: AgentState,
    ) -> Intent:
        if event.type == "focus_start_requested":
            if not previous_state.focus.active and current_state.focus.active:
                payload = {}
                if "duration_sec" in event.payload:
                    payload["duration_sec"] = event.payload.get("duration_sec")
                return Intent("start_focus", "explicit focus_start_requested", payload)
            return Intent("no_op", "focus already active before focus_start_requested")

        if event.type == "focus_stop_requested":
            if previous_state.focus.active and not current_state.focus.active:
                return Intent("stop_focus", "explicit focus_stop_requested")
            return Intent("no_op", "focus not active before focus_stop_requested")

        if event.type == "timer_finished":
            if previous_state.focus.active and not current_state.focus.active:
                return Intent("complete_focus", "focus timer finished")
            return Intent("no_op", "no active focus session for timer_finished")

        return Intent("no_op", f"rule handler does not support event: {event.type}")
