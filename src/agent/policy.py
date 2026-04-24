from __future__ import annotations

"""Policy compatibility entrypoints."""

from src.agent.action import Action
from src.agent.event import Event
from src.agent.intent import AgentIntent
from src.agent.planner import plan_intents
from src.agent.realizer import realize_actions
from src.agent.state import AgentState
from src.services.llm_service import LLMService


def decide_actions_with_intents(
    previous_state: AgentState,
    current_state: AgentState,
    event: Event,
    llm_service: LLMService,
) -> tuple[list[AgentIntent], list[Action]]:
    """Plan intents first, then realize actions."""
    intents = plan_intents(previous_state, current_state, event)
    actions = realize_actions(intents, current_state, event, llm_service)
    return intents, actions
