from __future__ import annotations

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
    # 关怀语音里"兴趣点缀"的轮换游标：每次带兴趣的关怀/回复后 +1，
    # 让连续多次关怀依次轮换不同兴趣（讲笑话 -> 打篮球 -> 听相声 …），而不是每次同一个。
    care_rotation_index: int = 0
