from dataclasses import dataclass

from src.agent.state.types import DialogueState, Mode


@dataclass
class InteractionState:
    """交互状态：反映模式、对话阶段和最近一次交互时间。"""

    mode: Mode = "normal"
    in_conversation: bool = False
    dialogue_state: DialogueState = "idle"
    last_user_time: int | None = None
    last_agent_response_time: int | None = None
