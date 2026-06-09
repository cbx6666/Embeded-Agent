from __future__ import annotations

"""自主检查 defer / admission / revert 元数据：Handler 与 AgentCore 共用，避免 defer 被当成普通 no_op。"""

from typing import Any

# final_action_reason：因语音会话 / 媒体 / TTS 保护区延后，本轮检查尚未消费。
DEFERRED_OUTCOMES = frozenset(
    {
        "voice_session_active_deferred",
        "voice_session_deferred",
        "tts_speaking_deferred",
    }
)

# 强触发：defer 时回退调度周期，下一轮可尽快重试。
STRONG_REVERT_TRIGGERS = frozenset(
    {
        "behavior_distraction_check",
        "wellness_care_check",
        "sensor_status_report",
    }
)


def apply_defer_metadata(
    log: dict[str, Any],
    *,
    outcome: str,
    defer_reason: str,
    trigger: str,
) -> None:
    """在 handler log_fields 上标记 defer，供 AgentCore 跳过准入冷却并按需 revert。"""
    log["final_action_reason"] = outcome
    log["guard_result"] = "deferred"
    log["deferred"] = True
    log["defer_reason"] = defer_reason
    log["should_mark_admitted"] = False
    # environment_care：defer 不回退 60s 周期，仅跳过准入冷却，避免语音保护区高频重试刷屏。
    log["should_revert_schedule"] = trigger in STRONG_REVERT_TRIGGERS


def is_deferred_decision(decision: Any) -> bool:
    log = dict(getattr(decision, "log_fields", {}) or {})
    if log.get("deferred"):
        return True
    return str(log.get("final_action_reason") or "") in DEFERRED_OUTCOMES
