from __future__ import annotations

"""专注计时服务模块。

TimerService 只负责产生倒计时 tick；
tick 之后要更新状态、触发提醒还是结束专注，由 AgentCore 再包装成标准事件处理。
"""

import threading
import time
from collections.abc import Callable


class TimerService:
    """提供简单倒计时能力。"""

    def __init__(self, background: bool = True) -> None:
        # background=False 主要用于测试场景，此时可以进入计时状态，
        # 但不真正创建后台 tick 线程。
        self.background = background
        # 运行中的后台计时线程；测试模式下保持为 None。
        self._thread: threading.Thread | None = None
        # stop_event 用于通知旧线程退出；每次 start 会重建一个新的事件对象。
        self._stop_event = threading.Event()
        self._active = False
        # 保护 _active、_thread、_stop_event 的并发读写。
        self._lock = threading.Lock()

    def start(self, duration_sec: int, callback: Callable[[int], None]) -> None:
        """启动一轮新的倒计时。"""
        # 新计时开始前先停止旧计时，保证同一时间只有一个 focus timer。
        self.stop()
        if duration_sec <= 0:
            # 非正时长视为立即结束，仍然通过 callback 走统一事件链路。
            callback(0)
            return

        with self._lock:
            self._active = True
            self._stop_event = threading.Event()
            if not self.background:
                return

            # 使用守护线程运行计时器，这样 CLI 退出时不需要等待
            # 一轮长时间专注任务自然结束。
            self._thread = threading.Thread(
                target=self._run,
                args=(duration_sec, callback, self._stop_event),
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """停止当前倒计时。"""
        with self._lock:
            self._active = False
            # 后台线程会在下一轮循环看到 stop_event 后退出。
            self._stop_event.set()

    def is_active(self) -> bool:
        """返回当前是否仍处于活跃计时状态。"""
        with self._lock:
            return self._active

    def _run(
        self,
        duration_sec: int,
        callback: Callable[[int], None],
        stop_event: threading.Event,
    ) -> None:
        """后台线程循环，每秒回传一次剩余时间。"""
        started_at = time.time()
        while not stop_event.is_set():
            elapsed = int(time.time() - started_at)
            remaining_sec = max(duration_sec - elapsed, 0)
            # 将当前剩余时间回传给 AgentCore；
            # 是否只更新状态，还是进一步触发提醒或结束动作，由上层决定。
            callback(remaining_sec)
            if remaining_sec <= 0:
                # 倒计时自然结束后也标记 inactive。
                self.stop()
                break
            time.sleep(1)
