from __future__ import annotations

"""系统时间驱动的多任务周期调度器（带防打断与剩余时间保留）。

调度器维护多个 :class:`ScheduledTask`，每个任务有独立周期与优先级：

    behavior_distraction_check : interval=20s, priority=1
    wellness_care_check        : interval=30s, priority=2
    environment_care_check     : interval=60s, priority=3
    sensor_status_report       : interval=300s, priority=4
    (speech_recognized   : priority=0，外部事件，不由本调度器产生)

核心规则：

- 每次 ``run_due`` 按真实流逝时间递减各任务的 ``remaining_sec``；<=0 标记 ``due``。
- 每个 tick 最多发出一个事件：在可运行的 due 任务中选优先级最高（priority 数字最小）。
- 防打断：当有更高优先级任务正在运行（``busy_priority_provider`` 返回更小的优先级数）
  时，低优先级任务的倒计时**冻结**（不递减、不发出、不重置），高优先级结束后从原处继续。
- 任务真正发出后，``remaining_sec`` 从完整周期重新开始。
- 高频 state_only 事件不经过本调度器，不会占用或阻塞它。

调度器不读取 LLM、不生成意图；是否提醒由各 handler + LLM + Guard 决定。
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from src.agent.event.event_model import Event
from src.agent.policy_config import SchedulePolicy
from src.agent.state.agent_state import AgentState

EventSink = Callable[[Event], object]
StateProvider = Callable[[], AgentState]
BusyPriorityProvider = Callable[[], "int | None"]


def build_system_trigger_event(
    state: AgentState,
    now_ts: int,
    *,
    trigger: str,
    source: str = "agent_autonomy",
) -> Event:
    """构造一个 system_triggered 周期事件。"""

    return Event(
        type="system_triggered",
        timestamp=int(now_ts),
        payload={
            "trigger": trigger,
            "source": source,
            "mode": state.interaction.mode,
            "focus_active": state.focus.active,
        },
    )


@dataclass
class ScheduledTask:
    """单个周期任务的运行时状态。"""

    name: str
    trigger: str
    interval_sec: int
    priority: int
    remaining_sec: float
    enabled: bool = True
    due: bool = False
    last_tick_ts: int | None = None
    last_emitted_at: int | None = None
    delayed_by: int | None = None

    def status(self, now: int) -> dict[str, object]:
        return {
            "task_name": self.name,
            "priority": self.priority,
            "remaining_sec": round(self.remaining_sec, 2),
            "due": self.due,
            "delayed_by": self.delayed_by,
            "emitted_at": self.last_emitted_at,
            "now": now,
        }


class AutonomousScheduler:
    """按各任务独立周期产生 system_triggered 事件的可启停后台调度器。"""

    def __init__(
        self,
        *,
        state_provider: StateProvider,
        event_sink: EventSink,
        config: SchedulePolicy | None = None,
        time_fn: Callable[[], float] = time.time,
        busy_priority_provider: BusyPriorityProvider | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.state_provider = state_provider
        self.event_sink = event_sink
        self.config = config or SchedulePolicy()
        self.time_fn = time_fn
        self.busy_priority_provider = busy_priority_provider
        self._log_fn = log_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._baseline_ts: int | None = None
        self.tasks: list[ScheduledTask] = [
            ScheduledTask(
                name=task.name,
                trigger=task.trigger,
                interval_sec=max(1, int(task.interval_sec)),
                priority=int(task.priority),
                remaining_sec=float(max(1, int(task.interval_sec))),
                enabled=bool(task.enabled),
            )
            for task in self.config.tasks
        ]

    # ---- 生命周期 ------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event = threading.Event()
            self._baseline_ts = None
            self._thread = threading.Thread(
                target=self._run, name="agent-autonomous-scheduler", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # ---- 调度核心 ------------------------------------------------------------
    def run_due(self, now_ts: int | None = None) -> list[Event]:
        """同步执行一次到期计算（按真实流逝时间），最多发出一个事件。"""

        if self._stop_event.is_set():
            return []
        now = int(self.time_fn()) if now_ts is None else int(now_ts)

        with self._lock:
            if self._baseline_ts is None:
                self._baseline_ts = now
                for task in self.tasks:
                    task.last_tick_ts = now
                if not self.config.emit_immediately_on_start:
                    return []
            busy = self._busy_priority()
            chosen = self._advance_and_select(now, busy)
            if chosen is None:
                return []
            chosen.remaining_sec = float(chosen.interval_sec)
            chosen.due = False
            chosen.last_emitted_at = now
            chosen.delayed_by = None
            self._log(chosen.status(now))

        state = self.state_provider()
        event = build_system_trigger_event(
            state, now_ts=now, trigger=chosen.trigger, source=self.config.event_source
        )
        if not self._stop_event.is_set():
            self.event_sink(event)
        return [event]

    def revert_emission(self, trigger: str, *, retry_after_sec: float = 0.0) -> bool:
        """Handler 延后播报时撤销本轮 emit 对倒计时的重置，使任务尽快重试。"""

        with self._lock:
            task = next((t for t in self.tasks if t.trigger == trigger), None)
            if task is None:
                return False
            retry = max(0.0, float(retry_after_sec))
            if retry <= 0.0:
                task.due = True
                task.remaining_sec = 0.0
            else:
                task.due = False
                task.remaining_sec = retry
            return True

    def _advance_and_select(self, now: int, busy: int | None) -> ScheduledTask | None:
        """递减倒计时并选出本 tick 要发出的任务（被高优先级冻结的任务不动）。"""

        for task in self.tasks:
            if not task.enabled:
                task.last_tick_ts = now
                continue
            frozen = busy is not None and task.priority > busy
            elapsed = 0 if task.last_tick_ts is None else max(0, now - task.last_tick_ts)
            task.last_tick_ts = now
            if frozen:
                # 冻结：保留 remaining_sec / due，记录被谁延迟。
                task.delayed_by = busy
                continue
            task.delayed_by = None
            task.remaining_sec -= elapsed
            if task.remaining_sec <= 0:
                task.due = True

        runnable = [
            task
            for task in self.tasks
            if task.enabled and task.due and not (busy is not None and task.priority > busy)
        ]
        if not runnable:
            # 标注被延迟的 due 任务，便于观测。
            for task in self.tasks:
                if task.enabled and task.due and busy is not None and task.priority > busy:
                    task.delayed_by = busy
            return None
        runnable.sort(key=lambda t: t.priority)
        return runnable[0]

    def _busy_priority(self) -> int | None:
        if self.busy_priority_provider is None:
            return None
        try:
            value = self.busy_priority_provider()
        except Exception:  # noqa: BLE001 - 防御：busy 探测失败按空闲处理
            return None
        return None if value is None else int(value)

    def _log(self, status: dict[str, object]) -> None:
        if self._log_fn is None:
            return
        try:
            self._log_fn(
                "[scheduler] "
                + " ".join(f"{key}={value}" for key, value in status.items())
            )
        except Exception:  # noqa: BLE001 - 日志不应影响调度
            pass

    def _run(self) -> None:
        wait_sec = max(0.1, float(self.config.poll_interval_sec))
        while not self._stop_event.wait(wait_sec):
            self.run_due()

    # ---- 调试 ----------------------------------------------------------------
    def task_status(self, now_ts: int | None = None) -> list[dict[str, object]]:
        now = int(self.time_fn()) if now_ts is None else int(now_ts)
        with self._lock:
            return [task.status(now) for task in self.tasks]
