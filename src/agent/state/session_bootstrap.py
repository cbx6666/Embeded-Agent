from __future__ import annotations

"""冷启动时会话态清理：丢弃上次未结束的专注与对话中间态。"""

from src.agent.state.agent_state import AgentState
from src.agent.state.focus_state import FocusState


def reset_ephemeral_session_on_cold_start(state: AgentState) -> bool:
    """进程重启后清空专注计时与会话中间态，保留用户感知/环境/历史统计。

    返回 True 表示发生了重置（便于启动日志提示）。
    """
    changed = False

    if (
        state.focus.active
        or state.focus.start_ts is not None
        or state.focus.remaining_sec not in (None, 0)
    ):
        last_end = state.focus.last_focus_end_ts
        state.focus = FocusState(last_focus_end_ts=last_end)
        changed = True

    interaction = state.interaction
    if (
        interaction.in_conversation
        or interaction.dialogue_state != "idle"
        or interaction.mode != "normal"
    ):
        interaction.in_conversation = False
        interaction.dialogue_state = "idle"
        interaction.mode = "normal"
        changed = True

    return changed
