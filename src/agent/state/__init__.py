"""状态模型包。

本包只负责定义 Agent 内部状态，不混放事件和动作。
状态被拆成多个子模块，便于后续按领域逐步扩展。
"""

from src.agent.state.agent_state import AgentState
from src.agent.state.cooldown_state import CooldownState
from src.agent.state.environment_state import EnvironmentState
from src.agent.state.focus_state import FocusState
from src.agent.state.interaction_state import InteractionState
from src.agent.history.runtime_history import RuntimeHistory
from src.agent.state.types import (
    DialogueState,
    LightState,
    Mode,
    UserAttention,
    UserBehavior,
    UserEmotion,
    UserFatigueLevel,
    UserPresence,
)
from src.agent.state.user_state import UserState
from src.agent.user.user_profile import (
    UserInfo,
    UserPreference,
    UserProfile,
)

__all__ = [
    "AgentState",
    "CooldownState",
    "DialogueState",
    "EnvironmentState",
    "FocusState",
    "InteractionState",
    "LightState",
    "RuntimeHistory",
    "Mode",
    "UserAttention",
    "UserBehavior",
    "UserEmotion",
    "UserFatigueLevel",
    "UserInfo",
    "UserPresence",
    "UserPreference",
    "UserProfile",
    "UserState",
]
