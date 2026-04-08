from __future__ import annotations

"""专注计时服务模块。"""

import threading
import time
from collections.abc import Callable


class TimerService:
    """提供简单倒计时能力。"""

    def __init__(self, background: bool = True) -> None:
        # background=False 主要用于测试场景，此时可以进入计时状态，
        # 但不真正创建后台 tick 线程。
        self.background = background
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._active = False
        self._lock = threading.Lock()

    def start(self, duration_sec: int, callback: Callable[[int], None]) -> None:
        """启动一轮新的倒计时。"""
        self.stop()
        if duration_sec <= 0:
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
                self.stop()
                break
            time.sleep(1)
