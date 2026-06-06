from __future__ import annotations

"""长期记忆任务的可靠后台执行器。

Worker 使用单线程和有界 FIFO 队列，将耗时的 Memory Pipeline 移出
``AgentCore.handle_event`` 热路径，同时提供进程内去重、失败重试、死信记录、
运行指标和限时关闭能力。
"""

import copy
import logging
import queue
import threading
import time
from dataclasses import replace
from typing import Any

from src.agent.action import Action
from src.agent.event import Event
from src.agent.execution.action_result import ActionResult
from src.agent.memory.long_term_memory_pipeline import LongTermMemoryPipeline
from src.agent.memory.memory_task import (
    MemorySubmitResult,
    MemoryTask,
    MemoryWorkerMetrics,
    build_action_memory_task,
    build_event_memory_task,
)
from src.agent.state import AgentState
from src.services.llm_service import LLMService


logger = logging.getLogger(__name__)


class MemoryBackgroundWorker:
    """使用有界 FIFO 队列串行执行长期记忆任务。

    单 Worker 线程会保持全局提交顺序，因此也自然保持同一用户的任务顺序，
    并避免多个后台线程并发写入 LongTermMemoryStore。若未来扩展为多 Worker，
    必须保证同一 ``user_id`` 始终路由到同一串行队列。
    """

    def __init__(
        self,
        pipeline: LongTermMemoryPipeline,
        llm_service: LLMService,
        *,
        max_queue_size: int = 100,
        max_retries: int = 2,
        retry_backoff_sec: tuple[float, ...] = (1.0, 3.0, 10.0),
        processed_id_limit: int = 10_000,
    ) -> None:
        self.pipeline = pipeline
        self.llm_service = llm_service
        self.max_queue_size = max(1, int(max_queue_size))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_sec = tuple(max(0.0, float(value)) for value in retry_backoff_sec)
        self.processed_id_limit = max(1, int(processed_id_limit))

        self._queue: queue.Queue[MemoryTask] = queue.Queue(maxsize=self.max_queue_size)
        self._state_lock = threading.RLock()
        self._idle_condition = threading.Condition(self._state_lock)
        self._accepting = True
        self._force_stop = threading.Event()
        self._pending_task_ids: set[str] = set()
        self._processed_task_ids: set[str] = set()
        self._processed_task_order: list[str] = []
        self._unfinished_count = 0
        self._active_task_id: str | None = None
        self._dead_letter_tasks: list[dict[str, Any]] = []
        self._metrics = MemoryWorkerMetrics(max_queue_size=self.max_queue_size)
        self._total_process_time_ms = 0.0
        self._timed_task_count = 0

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="memory-worker",
            daemon=True,
        )
        self._thread.start()

    def submit_event_memory(
        self,
        *,
        user_id: str,
        event: Event,
        state: AgentState,
        priority: int | None = None,
    ) -> MemorySubmitResult:
        """提交事件记忆任务，只等待入队结果，不等待 Pipeline 执行。"""

        try:
            task = build_event_memory_task(
                user_id=user_id,
                event=copy.deepcopy(event),
                state=AgentState.from_dict(copy.deepcopy(state.to_dict())),
                priority=priority,
            )
        except Exception as exc:
            return self._invalid_submit(str(exc))
        return self._submit(task)

    def submit_action_memory(
        self,
        *,
        user_id: str,
        actions: list[Action],
        timestamp: int,
        action_results: list[ActionResult],
        source_event: Event,
        state: AgentState,
        priority: int = 30,
    ) -> MemorySubmitResult:
        """提交动作记忆任务，只等待入队结果，不等待 Pipeline 执行。"""

        try:
            task = build_action_memory_task(
                user_id=user_id,
                actions=copy.deepcopy(actions),
                timestamp=int(timestamp),
                action_results=copy.deepcopy(action_results),
                source_event=copy.deepcopy(source_event),
                state=AgentState.from_dict(copy.deepcopy(state.to_dict())),
                priority=priority,
            )
        except Exception as exc:
            return self._invalid_submit(str(exc))
        return self._submit(task)

    def get_metrics(self) -> dict[str, Any]:
        """返回线程安全的 Worker 指标快照。"""

        with self._state_lock:
            self._refresh_queue_metrics_locked()
            return self._metrics.to_dict()

    def stats(self) -> dict[str, Any]:
        """兼容已有调试代码的指标查询别名。"""

        return self.get_metrics()

    def get_dead_letters(self) -> list[dict[str, Any]]:
        """返回死信记录副本，防止调用方修改 Worker 内部状态。"""

        with self._state_lock:
            return copy.deepcopy(self._dead_letter_tasks)

    def wait_for_idle(self, *, timeout: float | None = None) -> bool:
        """等待已接收任务处理完毕。

        该方法仅供测试、回放和关闭流程使用；``AgentCore.handle_event`` 不得调用，
        否则会重新把长期记忆处理变成当前响应的同步阻塞步骤。
        """

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._idle_condition:
            while self._unfinished_count:
                if deadline is None:
                    self._idle_condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle_condition.wait(timeout=remaining)
            return True

    def shutdown(self, timeout: float = 5.0) -> dict[str, Any]:
        """停止接收新任务，并在给定时间内尽量排空已接收任务。

        超时后会通知 Worker 停止后续处理并立即返回状态。执行中的外部 LLM 调用
        无法被 Python 线程强制中断，因此工作线程使用 daemon 模式，避免进程退出
        被永久阻塞。
        """

        timeout = max(0.0, float(timeout))
        deadline = time.monotonic() + timeout
        with self._idle_condition:
            self._accepting = False
            self._idle_condition.notify_all()
            while self._unfinished_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._idle_condition.wait(timeout=remaining)
            timed_out = self._unfinished_count > 0
            remaining_count = self._unfinished_count
            self._metrics.remaining_queue_size = remaining_count

        if timed_out:
            self._force_stop.set()
        # 至少留出一个队列轮询周期，让后台线程观察到“已排空”或“强制停止”信号；
        # 这只会小幅延长 join，不会继续无限等待外部 LLM 调用。
        join_timeout = max(0.1, deadline - time.monotonic())
        self._thread.join(timeout=join_timeout)
        metrics = self.get_metrics()
        logger.info(
            "memory worker stopped: timed_out=%s remaining_queue_size=%d metrics=%s",
            timed_out,
            remaining_count,
            metrics,
        )
        return {
            "timed_out": timed_out,
            "remaining_queue_size": remaining_count,
            "worker_alive": self._thread.is_alive(),
            "metrics": metrics,
        }

    def _submit(self, task: MemoryTask) -> MemorySubmitResult:
        """原子完成去重检查、容量检查和任务入队。"""

        with self._state_lock:
            self._metrics.submitted_count += 1
            if not self._accepting:
                return self._submit_result(False, task.task_id, "worker_shutdown")
            if task.task_id in self._pending_task_ids or task.task_id in self._processed_task_ids:
                self._metrics.duplicate_count += 1
                return self._submit_result(False, task.task_id, "duplicate_task")

            self._pending_task_ids.add(task.task_id)
            try:
                self._queue.put_nowait(task)
            except queue.Full:
                self._pending_task_ids.discard(task.task_id)
                self._metrics.dropped_count += 1
                logger.warning(
                    "memory task dropped: reason=queue_full task_id=%s priority=%d",
                    task.task_id,
                    task.priority,
                )
                return self._submit_result(False, task.task_id, "queue_full")

            self._unfinished_count += 1
            self._metrics.enqueued_count += 1
            self._refresh_queue_metrics_locked()
            self._idle_condition.notify_all()
            return self._submit_result(True, task.task_id, "enqueued")

    def _invalid_submit(self, error: str) -> MemorySubmitResult:
        with self._state_lock:
            self._metrics.submitted_count += 1
            logger.warning("invalid memory task: %s", error)
            return self._submit_result(False, None, "invalid_task")

    def _submit_result(
        self,
        accepted: bool,
        task_id: str | None,
        reason: str,
    ) -> MemorySubmitResult:
        self._refresh_queue_metrics_locked()
        return MemorySubmitResult(
            accepted=accepted,
            task_id=task_id,
            reason=reason,
            queue_size=self._queue.qsize(),
        )

    def _worker_loop(self) -> None:
        """串行消费任务；单线程执行是 FIFO 和同用户顺序保证的基础。"""

        while True:
            if self._force_stop.is_set():
                return
            try:
                task = self._queue.get(timeout=0.1)
            except queue.Empty:
                with self._state_lock:
                    if not self._accepting and self._unfinished_count == 0:
                        return
                continue

            with self._state_lock:
                self._active_task_id = task.task_id
                self._refresh_queue_metrics_locked()

            started_at = time.monotonic()
            terminal_success = self._process_with_retries(task)
            elapsed_ms = (time.monotonic() - started_at) * 1000.0

            with self._idle_condition:
                if terminal_success:
                    self._metrics.processed_count += 1
                else:
                    self._metrics.failed_count += 1
                self._record_process_time_locked(elapsed_ms)
                self._pending_task_ids.discard(task.task_id)
                self._remember_processed_id_locked(task.task_id)
                self._unfinished_count = max(0, self._unfinished_count - 1)
                self._active_task_id = None
                self._queue.task_done()
                self._refresh_queue_metrics_locked()
                self._idle_condition.notify_all()

    def _process_with_retries(self, task: MemoryTask) -> bool:
        """在当前队列位置内完成重试，避免失败任务被后续任务越过。"""

        current = task
        while True:
            logger.debug(
                "%s memory task started: task_id=%s user_id=%s retry_count=%d priority=%d",
                current.task_type,
                current.task_id,
                current.user_id,
                current.retry_count,
                current.priority,
            )
            try:
                self._execute_task(current)
            except Exception as exc:
                if current.retry_count < self.max_retries and not self._force_stop.is_set():
                    delay = self._retry_delay(current.retry_count)
                    with self._state_lock:
                        self._metrics.retried_count += 1
                    logger.warning(
                        "%s memory task retrying: task_id=%s retry_count=%d delay_sec=%.3f error=%s",
                        current.task_type,
                        current.task_id,
                        current.retry_count + 1,
                        delay,
                        exc,
                    )
                    if self._force_stop.wait(delay):
                        self._dead_letter(current, exc)
                        return False
                    current = replace(current, retry_count=current.retry_count + 1)
                    continue
                self._dead_letter(current, exc)
                return False

            logger.debug(
                "%s memory task finished: task_id=%s user_id=%s retry_count=%d",
                current.task_type,
                current.task_id,
                current.user_id,
                current.retry_count,
            )
            return True

    def _execute_task(self, task: MemoryTask) -> None:
        """从稳定载荷恢复领域对象，并调用对应的 Memory Pipeline 入口。"""

        if task.task_type == "event":
            event = _event_from_dict(task.payload["event"])
            state = AgentState.from_dict(copy.deepcopy(task.payload["state"]))
            self.pipeline.process_event(task.user_id, event, state, self.llm_service)
            return
        if task.task_type == "action":
            actions = [_action_from_dict(item) for item in task.payload["actions"]]
            results = [_action_result_from_dict(item) for item in task.payload["action_results"]]
            source_event = _event_from_dict(task.payload["source_event"])
            state = AgentState.from_dict(copy.deepcopy(task.payload["state"]))
            self.pipeline.process_actions(
                task.user_id,
                actions,
                int(task.payload["timestamp"]),
                action_results=results,
                source_event=source_event,
                state=state,
                llm_service=self.llm_service,
            )
            return
        raise ValueError(f"unsupported memory task type: {task.task_type}")

    def _dead_letter(self, task: MemoryTask, error: Exception) -> None:
        """记录达到重试上限的任务，不保存完整用户原文。"""

        record = {
            "task_id": task.task_id,
            "user_id": task.user_id,
            "task_type": task.task_type,
            "source_event_id": task.source_event_id,
            "created_at": task.created_at,
            "failed_at": time.time(),
            "retry_count": task.retry_count,
            "last_error": _safe_error_summary(error),
            "payload_summary": _payload_summary(task),
        }
        with self._state_lock:
            self._dead_letter_tasks.append(record)
            self._metrics.dead_letter_count += 1
        logger.error("memory task moved to dead letter: %s", record)

    def _retry_delay(self, retry_count: int) -> float:
        if not self.retry_backoff_sec:
            return 0.0
        index = min(max(0, retry_count), len(self.retry_backoff_sec) - 1)
        return self.retry_backoff_sec[index]

    def _record_process_time_locked(self, elapsed_ms: float) -> None:
        self._total_process_time_ms += max(0.0, elapsed_ms)
        self._timed_task_count += 1
        self._metrics.average_process_time_ms = round(
            self._total_process_time_ms / self._timed_task_count,
            3,
        )

    def _remember_processed_id_locked(self, task_id: str) -> None:
        # 已处理 ID 只保存在当前进程，并通过数量上限控制内存占用；
        # Worker 重启后不会保留这部分幂等状态。
        if task_id in self._processed_task_ids:
            return
        self._processed_task_ids.add(task_id)
        self._processed_task_order.append(task_id)
        while len(self._processed_task_order) > self.processed_id_limit:
            oldest = self._processed_task_order.pop(0)
            self._processed_task_ids.discard(oldest)

    def _refresh_queue_metrics_locked(self) -> None:
        self._metrics.queue_size = self._queue.qsize()


def _event_from_dict(data: dict[str, Any]) -> Event:
    return Event(
        type=str(data.get("type", "")),  # type: ignore[arg-type]
        timestamp=int(data.get("timestamp", 0)),
        payload=copy.deepcopy(data.get("payload", {})),
    )


def _action_from_dict(data: dict[str, Any]) -> Action:
    return Action(
        type=str(data.get("type", "")),  # type: ignore[arg-type]
        payload=copy.deepcopy(data.get("payload", {})),
    )


def _action_result_from_dict(data: dict[str, Any]) -> ActionResult:
    return ActionResult(
        action_type=str(data.get("action_type", "")),
        success=bool(data.get("success", False)),
        timestamp=int(data.get("timestamp", 0)),
        reason=str(data.get("reason", "")),
        payload=copy.deepcopy(data.get("payload", {})),
    )


def _payload_summary(task: MemoryTask) -> dict[str, Any]:
    """生成可观测但不包含完整用户文本的死信载荷摘要。"""

    if task.task_type == "event":
        event = task.payload.get("event", {})
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        text = str(payload.get("text", "")) if isinstance(payload, dict) else ""
        return {
            "event_type": event.get("type") if isinstance(event, dict) else None,
            "event_timestamp": event.get("timestamp") if isinstance(event, dict) else None,
            "text_length": len(text),
        }
    actions = task.payload.get("actions", [])
    results = task.payload.get("action_results", [])
    return {
        "source_event_type": (
            task.payload.get("source_event", {}).get("type")
            if isinstance(task.payload.get("source_event"), dict)
            else None
        ),
        "action_types": [
            str(item.get("type", ""))
            for item in actions
            if isinstance(item, dict)
        ],
        "result_statuses": [
            bool(item.get("success", False))
            for item in results
            if isinstance(item, dict)
        ],
    }


def _safe_error_summary(error: Exception) -> str:
    """压缩并截断异常文本，避免日志携带过长或意外敏感内容。"""

    message = " ".join(str(error).split())
    if len(message) > 200:
        message = f"{message[:197]}..."
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
