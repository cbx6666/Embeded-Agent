from __future__ import annotations

"""TTS 任务类型与播放策略：优先级、可打断性、过期、合并键、媒体/会话约束。"""

from dataclasses import dataclass
from enum import Enum, IntEnum


class TTSJobKind(str, Enum):
    WAKE_ACK = "wake_ack"
    USER_REPLY = "user_reply"
    MEDIA_ACK = "media_ack"
    AUTONOMOUS_DISTRACTION = "autonomous_distraction"
    AUTONOMOUS_CARE = "autonomous_care"
    AUTONOMOUS_ENV = "autonomous_env"
    AUTONOMOUS_SENSOR = "autonomous_sensor"
    AUTONOMOUS_SUGGESTION = "autonomous_suggestion"


class TTSJobPriority(IntEnum):
    """数值越小越优先。"""

    WAKE_ACK = 0
    USER_REPLY = 1
    MEDIA_ACK = 1
    AUTONOMOUS_DISTRACTION = 2
    AUTONOMOUS_CARE = 3
    AUTONOMOUS_ENV = 4
    AUTONOMOUS_SENSOR = 5
    AUTONOMOUS_SUGGESTION = 6


@dataclass(frozen=True)
class TTSJobSpec:
    kind: TTSJobKind
    priority: int
    interruptible: bool
    expire_seconds: float | None
    allow_during_media: bool
    coalesce_key: str | None
    user_session_protected: bool
    is_autonomous: bool


_REASON_TO_KIND: dict[str, TTSJobKind] = {
    "distraction_reminder": TTSJobKind.AUTONOMOUS_DISTRACTION,
    "rest_reminder": TTSJobKind.AUTONOMOUS_CARE,
    "emotion_reminder": TTSJobKind.AUTONOMOUS_CARE,
    "posture_reminder": TTSJobKind.AUTONOMOUS_CARE,
    "environment_warning": TTSJobKind.AUTONOMOUS_ENV,
    "status_report": TTSJobKind.AUTONOMOUS_SENSOR,
    "media_suggestion": TTSJobKind.AUTONOMOUS_SUGGESTION,
    "joke_reminder": TTSJobKind.AUTONOMOUS_SUGGESTION,
}

_KIND_SPECS: dict[TTSJobKind, TTSJobSpec] = {
    TTSJobKind.WAKE_ACK: TTSJobSpec(
        kind=TTSJobKind.WAKE_ACK,
        priority=int(TTSJobPriority.WAKE_ACK),
        interruptible=False,
        expire_seconds=None,
        allow_during_media=True,
        coalesce_key=None,
        user_session_protected=False,
        is_autonomous=False,
    ),
    TTSJobKind.USER_REPLY: TTSJobSpec(
        kind=TTSJobKind.USER_REPLY,
        priority=int(TTSJobPriority.USER_REPLY),
        interruptible=True,
        expire_seconds=60.0,
        allow_during_media=False,
        coalesce_key=None,
        user_session_protected=False,
        is_autonomous=False,
    ),
    TTSJobKind.MEDIA_ACK: TTSJobSpec(
        kind=TTSJobKind.MEDIA_ACK,
        priority=int(TTSJobPriority.MEDIA_ACK),
        interruptible=True,
        expire_seconds=45.0,
        allow_during_media=True,
        coalesce_key=None,
        user_session_protected=False,
        is_autonomous=False,
    ),
    TTSJobKind.AUTONOMOUS_DISTRACTION: TTSJobSpec(
        kind=TTSJobKind.AUTONOMOUS_DISTRACTION,
        priority=int(TTSJobPriority.AUTONOMOUS_DISTRACTION),
        interruptible=True,
        expire_seconds=90.0,
        allow_during_media=False,
        coalesce_key="distraction",
        user_session_protected=True,
        is_autonomous=True,
    ),
    TTSJobKind.AUTONOMOUS_CARE: TTSJobSpec(
        kind=TTSJobKind.AUTONOMOUS_CARE,
        priority=int(TTSJobPriority.AUTONOMOUS_CARE),
        interruptible=True,
        expire_seconds=120.0,
        allow_during_media=False,
        coalesce_key="care",
        user_session_protected=True,
        is_autonomous=True,
    ),
    TTSJobKind.AUTONOMOUS_ENV: TTSJobSpec(
        kind=TTSJobKind.AUTONOMOUS_ENV,
        priority=int(TTSJobPriority.AUTONOMOUS_ENV),
        interruptible=True,
        expire_seconds=90.0,
        allow_during_media=False,
        coalesce_key="environment",
        user_session_protected=True,
        is_autonomous=True,
    ),
    TTSJobKind.AUTONOMOUS_SENSOR: TTSJobSpec(
        kind=TTSJobKind.AUTONOMOUS_SENSOR,
        priority=int(TTSJobPriority.AUTONOMOUS_SENSOR),
        interruptible=True,
        expire_seconds=180.0,
        allow_during_media=False,
        coalesce_key="sensor",
        user_session_protected=True,
        is_autonomous=True,
    ),
    TTSJobKind.AUTONOMOUS_SUGGESTION: TTSJobSpec(
        kind=TTSJobKind.AUTONOMOUS_SUGGESTION,
        priority=int(TTSJobPriority.AUTONOMOUS_SUGGESTION),
        interruptible=True,
        expire_seconds=120.0,
        allow_during_media=False,
        coalesce_key="media_suggestion",
        user_session_protected=True,
        is_autonomous=True,
    ),
}


def resolve_job_spec(*, source: str, reason: str = "", kind: str = "") -> TTSJobSpec:
    """根据 speak payload 的 source/reason 解析 TTS 任务策略。"""
    if source == "wake_ack":
        return _KIND_SPECS[TTSJobKind.WAKE_ACK]
    if source == "media_ack" or reason == "media_play_ack":
        return _KIND_SPECS[TTSJobKind.MEDIA_ACK]
    if kind == "status_report" or reason == "status_report" or source == "status_report":
        return _KIND_SPECS[TTSJobKind.AUTONOMOUS_SENSOR]
    mapped = _REASON_TO_KIND.get(reason) or _REASON_TO_KIND.get(source)
    if mapped is not None:
        return _KIND_SPECS[mapped]
    if source in _REASON_TO_KIND:
        return _KIND_SPECS[_REASON_TO_KIND[source]]
    if kind == "notification":
        return _KIND_SPECS[TTSJobKind.AUTONOMOUS_CARE]
    return _KIND_SPECS[TTSJobKind.USER_REPLY]


def is_autonomous_spec(spec: TTSJobSpec) -> bool:
    return spec.is_autonomous
