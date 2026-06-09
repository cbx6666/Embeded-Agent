from __future__ import annotations

"""运行态落盘策略：高频感知事件节流，关键决策立即持久化。"""

import time

from src.agent.core.models import DecisionResult
from src.agent.event.event_model import Event

# 感知 / 环境 / 语音生命周期等高频事件：内存与 runtime_history 仍每条更新，落盘最多 1 次/秒。
_RUNTIME_THROTTLE_SEC = 1.0

# 冷却、专注、用户语音、自主检查等：每次处理完立即写盘。
_PERSIST_IMMEDIATE_EVENT_TYPES = frozenset(
    {
        "speech_recognized",
        "system_triggered",
        "focus_start_requested",
        "focus_stop_requested",
        "timer_finished",
        "tts_finished",
    }
)


def should_persist_runtime_state(
    event: Event,
    *,
    decision: DecisionResult,
    last_persist_mono: float,
    min_interval_sec: float = _RUNTIME_THROTTLE_SEC,
) -> bool:
    """是否应将当前 AgentState 写入 runtime_store.json。"""
    if decision.actions:
        return True
    event_type = str(event.type)
    if event_type == "system_triggered":
        # 自检 no_op / deferred 与高频感知一样节流，有 action 时上面已立即落盘。
        return (time.monotonic() - last_persist_mono) >= max(0.0, float(min_interval_sec))
    if event_type in _PERSIST_IMMEDIATE_EVENT_TYPES:
        return True
    return (time.monotonic() - last_persist_mono) >= max(0.0, float(min_interval_sec))
