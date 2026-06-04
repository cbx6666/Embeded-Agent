from __future__ import annotations

"""单麦克风仲裁：避免多个 arecord 并发占用同一 ALSA 设备。"""

import threading
from contextlib import contextmanager


_MIC_LOCK = threading.RLock()


@contextmanager
def mic_capture_lock(*, timeout_sec: float | None = None):
    """串行化同一进程内对单麦克风的占用（唤醒检测 vs 用户录音）。"""
    if timeout_sec is None:
        acquired = _MIC_LOCK.acquire(blocking=True)
    else:
        acquired = _MIC_LOCK.acquire(timeout=float(timeout_sec))
    if not acquired:
        raise TimeoutError("麦克风正被其它语音任务占用")
    try:
        yield
    finally:
        _MIC_LOCK.release()
