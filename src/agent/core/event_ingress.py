from __future__ import annotations

"""Agent 事件入站：后台线程串行处理，高频感知事件合并，避免多线程抢锁。"""

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from src.agent.event.event_model import Event

# 同类型只保留最新一条（约 1Hz 感知足够驱动状态与 UI）。
_COALESCE_EVENT_TYPES = frozenset(
    {
        "user_fatigue_updated",
        "user_emotion_updated",
        "user_presence_updated",
        "user_attention_updated",
        "user_posture_updated",
        "user_activity_updated",
        "timer_ticked",
    }
)

# 入队时提高优先级（数值越小越优先）。
_PRIORITY_EVENT_TYPES = frozenset(
    {
        "speech_recognized",
        "system_triggered",
        "tts_finished",
        "tts_started",
        "focus_start_requested",
        "focus_stop_requested",
        "timer_finished",
        "voice_wake_detected",
    }
)

EventProcessor = Callable[[Event], tuple[list, list]]


@dataclass(order=True)
class _HeapItem:
    priority: int
    seq: int
    event: Event = field(compare=False)


class AgentEventIngress:
    """将多生产者事件收敛到单 worker 线程，降低 AgentCore RLock 争用。"""

    def __init__(self, processor: EventProcessor) -> None:
        self._processor = processor
        self._coalesce: dict[str, Event] = {}
        self._coalesce_lock = threading.Lock()
        self._heap: list[_HeapItem] = []
        self._heap_lock = threading.Lock()
        self._seq = 0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def worker_thread(self) -> threading.Thread | None:
        return self._thread

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="agent-event-worker", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout)))
            self._thread = None

    def submit(
        self,
        event: Event,
        *,
        llm_priority: int | None = None,
    ) -> tuple[list, list]:
        """非 worker 线程调用：合并或入队后立即返回空结果。"""
        event_type = str(event.type)
        if event_type in _COALESCE_EVENT_TYPES:
            with self._coalesce_lock:
                self._coalesce[event_type] = event
            self._wake.set()
            return [], []

        priority = 0
        if llm_priority is not None:
            priority = int(llm_priority)
        elif event_type not in _PRIORITY_EVENT_TYPES:
            priority = 8

        with self._heap_lock:
            self._seq += 1
            heapq.heappush(self._heap, _HeapItem(priority, self._seq, event))
        self._wake.set()
        return [], []

    def _drain_coalesce(self) -> list[Event]:
        with self._coalesce_lock:
            batch = list(self._coalesce.values())
            self._coalesce.clear()
        return batch

    def _pop_heap(self) -> Event | None:
        with self._heap_lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap).event

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.05)
            self._wake.clear()
            for event in self._drain_coalesce():
                if self._stop.is_set():
                    break
                self._processor(event)
            while not self._stop.is_set():
                event = self._pop_heap()
                if event is None:
                    break
                self._processor(event)
            if self._stop.is_set():
                break
            # 空闲时避免忙等
            with self._coalesce_lock:
                has_coalesce = bool(self._coalesce)
            with self._heap_lock:
                has_heap = bool(self._heap)
            if not has_coalesce and not has_heap:
                time.sleep(0.01)
