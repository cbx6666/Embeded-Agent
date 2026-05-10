from __future__ import annotations

"""策略入口层。"""

from src.agent.action import Action
from src.agent.decision.intent import AgentIntent
from src.agent.decision.planner import plan_intents
from src.agent.decision.realizer import realize_actions
from src.agent.event import Event
from src.agent.state import AgentState
from src.services.llm_service import LLMService


def decide_actions_with_intents(
    previous_state: AgentState,
    current_state: AgentState,
    event: Event,
    llm_service: LLMService,
) -> tuple[list[AgentIntent], list[Action]]:
    """先规划意图，再将意图落成动作。"""
    intents = plan_intents(previous_state, current_state, event, llm_service=llm_service)
    actions = realize_actions(intents, current_state, event, llm_service)
    return intents, actions


def decide_actions(
    previous_state: AgentState,
    current_state: AgentState,
    event: Event,
    llm_service: LLMService,
) -> list[Action]:
    """兼容只需要动作列表的旧入口。"""
    _, actions = decide_actions_with_intents(
        previous_state=previous_state,
        current_state=current_state,
        event=event,
        llm_service=llm_service,
    )
    return actions
