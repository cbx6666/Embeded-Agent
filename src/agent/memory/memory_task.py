from __future__ import annotations

"""后台长期记忆任务的数据模型与稳定标识生成逻辑。

本模块只负责把运行时对象转换为可排队、可重试、可去重的不可变任务，
不负责判断内容是否值得记忆，也不直接执行 Memory Pipeline。
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

from src.agent.action import Action
from src.agent.event import Event
from src.agent.execution.action_result import ActionResult
from src.agent.state import AgentState


MemoryTaskType = Literal["event", "action"]


@dataclass(frozen=True)
class MemoryTask:
    """一条不可变的长期记忆后台任务。

    ``task_id`` 由任务的业务身份生成，不依赖对象地址和创建时间，因此同一
    业务任务重复提交时会得到相同标识。当前 Worker 只在进程内保存去重状态，
    重启后的跨进程幂等不由本模型保证。
    """

    task_id: str
    user_id: str
    task_type: MemoryTaskType
    source_event_id: str | None
    created_at: float
    priority: int
    retry_count: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class MemorySubmitResult:
    """任务提交阶段的同步结果。

    该结果只表示任务是否被 Worker 接收，不表示 Memory Pipeline 已执行成功。
    调用方可以立即将它写入 trace，而不必等待后台任务完成。
    """

    accepted: bool
    task_id: str | None
    reason: str
    queue_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "task_id": self.task_id,
            "reason": self.reason,
            "queue_size": self.queue_size,
        }


@dataclass
class MemoryWorkerMetrics:
    """Memory Worker 的进程内运行指标快照。"""

    submitted_count: int = 0
    enqueued_count: int = 0
    duplicate_count: int = 0
    dropped_count: int = 0
    processed_count: int = 0
    failed_count: int = 0
    retried_count: int = 0
    dead_letter_count: int = 0
    queue_size: int = 0
    max_queue_size: int = 100
    average_process_time_ms: float = 0.0
    remaining_queue_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_event_memory_task(
    *,
    user_id: str,
    event: Event,
    state: AgentState,
    priority: int | None = None,
    created_at: float | None = None,
) -> MemoryTask:
    """根据事件和状态快照构造事件记忆任务。

    状态快照会随任务传递给后台 Pipeline，但不参与 ``task_id`` 计算。这样即使
    AgentState 在重复提交之间发生变化，同一来源事件仍会被识别为同一业务任务。
    """

    event_payload = _event_to_dict(event)
    source_event_id = _source_event_id(event_payload)
    payload = {
        "event": event_payload,
        "state": state.to_dict(),
    }
    return _build_task(
        user_id=user_id,
        task_type="event",
        source_event_id=source_event_id,
        created_at=created_at,
        priority=_event_priority(event) if priority is None else int(priority),
        payload=payload,
        identity_payload=event_payload,
    )


def build_action_memory_task(
    *,
    user_id: str,
    actions: list[Action],
    timestamp: int,
    action_results: list[ActionResult],
    source_event: Event,
    state: AgentState,
    priority: int = 30,
    created_at: float | None = None,
) -> MemoryTask:
    """根据动作及其执行结果构造动作记忆任务。

    任务身份由来源事件、动作内容和执行结果共同决定；状态仅作为执行上下文，
    不影响幂等键。
    """

    source_event_payload = _event_to_dict(source_event)
    payload = {
        "actions": [_action_to_dict(action) for action in actions],
        "timestamp": int(timestamp),
        "action_results": [_action_result_to_dict(result) for result in action_results],
        "source_event": source_event_payload,
        "state": state.to_dict(),
    }
    return _build_task(
        user_id=user_id,
        task_type="action",
        source_event_id=_source_event_id(source_event_payload),
        created_at=created_at,
        priority=int(priority),
        payload=payload,
        identity_payload={
            "actions": payload["actions"],
            "timestamp": payload["timestamp"],
            "action_results": payload["action_results"],
            "source_event": source_event_payload,
        },
    )


def _build_task(
    *,
    user_id: str,
    task_type: MemoryTaskType,
    source_event_id: str | None,
    created_at: float | None,
    priority: int,
    payload: dict[str, Any],
    identity_payload: dict[str, Any],
) -> MemoryTask:
    """规范化任务数据，并生成与运行时对象身份无关的稳定任务 ID。"""

    normalized_user_id = str(user_id).strip()
    if not normalized_user_id:
        raise ValueError("memory task user_id is required")
    normalized_payload = _canonical_value(payload)
    normalized_identity_payload = _canonical_value(identity_payload)
    if not isinstance(normalized_payload, dict):
        raise ValueError("memory task payload must be an object")
    if not isinstance(normalized_identity_payload, dict):
        raise ValueError("memory task identity payload must be an object")
    task_id = _stable_hash(
        {
            "user_id": normalized_user_id,
            "task_type": task_type,
            "source_event_id": source_event_id,
            "payload": normalized_identity_payload,
        },
        prefix="mem",
    )
    return MemoryTask(
        task_id=task_id,
        user_id=normalized_user_id,
        task_type=task_type,
        source_event_id=source_event_id,
        created_at=time.time() if created_at is None else float(created_at),
        priority=max(0, min(100, int(priority))),
        retry_count=0,
        payload=normalized_payload,
    )


def _source_event_id(event: dict[str, Any]) -> str:
    """优先使用上游 event_id；缺失时根据完整事件内容生成稳定标识。"""

    payload = event.get("payload", {})
    if isinstance(payload, dict):
        explicit = payload.get("event_id")
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
    return _stable_hash(event, prefix="evt")


def _event_priority(event: Event) -> int:
    """根据用户表达的持久性信号估算任务优先级。"""

    if event.type in {"break_suggestion_accepted", "break_suggestion_rejected"}:
        return 80
    text = str(event.payload.get("text", "")).strip().lower()
    durable_markers = (
        "以后",
        "记住",
        "我喜欢",
        "我不喜欢",
        "我更喜欢",
        "我讨厌",
        "我习惯",
        "从现在开始",
        "默认",
        "不要再",
        "每次",
        "remember",
        "from now on",
        "i prefer",
        "i like",
        "i dislike",
        "by default",
    )
    return 100 if any(marker in text for marker in durable_markers) else 50


def _event_to_dict(event: Event) -> dict[str, Any]:
    return {
        "type": str(event.type),
        "timestamp": int(event.timestamp),
        "payload": _canonical_value(event.payload),
    }


def _action_to_dict(action: Action) -> dict[str, Any]:
    return {
        "type": str(action.type),
        "payload": _canonical_value(action.payload),
    }


def _action_result_to_dict(result: ActionResult) -> dict[str, Any]:
    return {
        "action_type": str(result.action_type),
        "success": bool(result.success),
        "timestamp": int(result.timestamp),
        "reason": str(result.reason),
        "payload": _canonical_value(result.payload),
    }


def _stable_hash(value: Any, *, prefix: str) -> str:
    """对规范化 JSON 计算稳定摘要，避免字典顺序影响标识。"""

    raw = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:24]}"


def _canonical_value(value: Any) -> Any:
    """把任意任务输入递归转换为可稳定序列化的基础数据结构。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical_value(value.to_dict())
    return {"unsupported_type": f"{type(value).__module__}.{type(value).__qualname__}"}
