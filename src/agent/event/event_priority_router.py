from __future__ import annotations

"""Classify events before the decision pipeline without changing state."""

from dataclasses import dataclass
from typing import Literal

from src.agent.config.policy_config import EventRoutingPolicyConfig
from src.agent.event.event_model import Event

EventPriority = Literal["P0", "P1", "P2", "P3", "P4"]
EventHandling = Literal[
    "orchestrator",
    "rule_intent_builder",
    "low_frequency_check",
    "state_only",
    "profile_handler",
    "settings_handler",
    "feedback_signal",
    "internal_only",
]


@dataclass(frozen=True)
class EventRoute:
    priority: EventPriority
    handling: EventHandling
    should_enter_decision: bool
    should_allow_llm: bool
    reason: str


class EventPriorityRouter:
    """在状态归约之后判断事件时效和处理机制。

    Router 不改变状态，也不做业务决策。P0 只表示需要立即处理；是否调用 LLM
    由 handling 和 should_allow_llm 单独表达。
    """

    def __init__(self, policy_config: EventRoutingPolicyConfig | None = None) -> None:
        self.policy_config = policy_config or EventRoutingPolicyConfig()

    def classify(self, event: Event) -> EventRoute:
        """返回标准事件的稳定路由结果。"""

        event_type = str(event.type)
        if event_type in self.policy_config.open_semantic_events:
            return EventRoute(
                priority="P0",
                handling="orchestrator",
                should_enter_decision=True,
                should_allow_llm=True,
                reason=f"open_semantic_event:{event_type}",
            )
        if event_type in self.policy_config.structured_decision_events:
            return EventRoute(
                priority="P0",
                handling="rule_intent_builder",
                should_enter_decision=True,
                should_allow_llm=False,
                reason=f"structured_immediate_decision:{event_type}",
            )
        if event_type == "system_triggered":
            trigger = str(event.payload.get("trigger", "")).strip()
            if trigger in self.policy_config.low_frequency_triggers:
                return EventRoute(
                    priority="P1",
                    handling="low_frequency_check",
                    should_enter_decision=True,
                    should_allow_llm=True,
                    reason=f"low_frequency_check:{trigger}",
                )
            return EventRoute(
                priority="P4",
                handling="internal_only",
                should_enter_decision=False,
                should_allow_llm=False,
                reason=f"internal_system_trigger:{trigger or 'unspecified'}",
            )
        if event_type in self.policy_config.user_state_events:
            # P2 是事实输入：先沉淀当前值和趋势，等待低频 P1 汇总判断。
            return EventRoute(
                priority="P2",
                handling="state_only",
                should_enter_decision=False,
                should_allow_llm=False,
                reason=f"state_input_only:{event_type}",
            )
        if event_type in self.policy_config.telemetry_events:
            # P3 可能高频出现，只更新计时、环境或生命周期状态。
            return EventRoute(
                priority="P3",
                handling="state_only",
                should_enter_decision=False,
                should_allow_llm=False,
                reason=f"telemetry_or_lifecycle:{event_type}",
            )
        dedicated_handling = self.policy_config.p4_handling_by_event.get(event_type)
        if dedicated_handling is not None:
            return EventRoute(
                priority="P4",
                handling=dedicated_handling,  # type: ignore[arg-type]
                should_enter_decision=False,
                should_allow_llm=False,
                reason=f"{dedicated_handling}:{event_type}",
            )
        return EventRoute(
            priority="P4",
            handling="internal_only",
            should_enter_decision=False,
            should_allow_llm=False,
            reason=f"unclassified_event:{event_type}",
        )
