from __future__ import annotations

"""基于系统时间的低频自主检查调度器。

Scheduler 不读取 LLM、不生成 Intent，也不判断用户是否真的需要提醒。它只按配置
产生 `system_triggered` 检查事件，后续是否 skip/rule/llm 由
AutonomousCheckPolicy 决定。
"""

import threading
import time
from collections.abc import Callable

from src.agent.config.policy_config import AutonomousScheduleConfig
from src.agent.event import Event
from src.agent.state import AgentState


EventSink = Callable[[Event], object]
StateProvider = Callable[[], AgentState]


def build_autonomous_check_event(
    state: AgentState,
    now_ts: int,
    reason: str = "periodic_check",
    *,
    source: str = "agent_autonomy",
) -> Event:
    """构造统一的 P1 自主检查事件。"""

    return Event(
        type="system_triggered",
        timestamp=int(now_ts),
        payload={
            "trigger": reason,
            "source": source,
            "mode": state.interaction.mode,
            "focus_active": state.focus.active,
        },
    )


class AutonomousScheduler:
    """按可配置间隔产生 P1 检查事件的可启停后台调度器。"""

    def __init__(
        self,
        *,
        state_provider: StateProvider,
        event_sink: EventSink,
        config: AutonomousScheduleConfig | None = None,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.state_provider = state_provider
        self.event_sink = event_sink
        self.config = config or AutonomousScheduleConfig()
        self.time_fn = time_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._baseline_ts: int | None = None
        self._last_emitted_at: dict[str, int] = {}

    def start(self) -> None:
        """启动单个守护线程；重复调用不会创建第二个调度循环。"""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event = threading.Event()
            self._baseline_ts = int(self.time_fn())
            if self.config.emit_immediately_on_start:
                self._baseline_ts = None
            self._thread = threading.Thread(
                target=self._run,
                name="agent-autonomous-scheduler",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """停止调度并等待后台线程退出。"""

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

    def run_due(self, now_ts: int | None = None) -> list[Event]:
        """同步执行一次到期计算，便于测试和无后台线程环境复用。"""

        if self._stop_event.is_set():
            return []
        now = int(self.time_fn()) if now_ts is None else int(now_ts)
        with self._lock:
            if self._baseline_ts is None:
                self._baseline_ts = now
                if not self.config.emit_immediately_on_start:
                    return []
            baseline = self._baseline_ts
            due: list[str] = []
            for trigger, raw_interval in self.config.intervals_sec.items():
                if trigger in self.config.disabled_triggers:
                    continue
                interval = max(1, int(raw_interval))
                last_ts = self._last_emitted_at.get(trigger, baseline)
                if now - last_ts >= interval:
                    self._last_emitted_at[trigger] = now
                    due.append(trigger)

        if not due:
            return []
        state = self.state_provider()
        events = [
            build_autonomous_check_event(
                state,
                now_ts=now,
                reason=trigger,
                source=self.config.event_source,
            )
            for trigger in due
        ]
        for event in events:
            if self._stop_event.is_set():
                break
            self.event_sink(event)
        return events

    def _run(self) -> None:
        """低频轮询系统时间；业务 gate 保持在调度线程之外。"""

        wait_sec = max(0.1, float(self.config.poll_interval_sec))
        while not self._stop_event.wait(wait_sec):
            self.run_due()
