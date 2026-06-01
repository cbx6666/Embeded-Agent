from __future__ import annotations

"""单麦克风仲裁：避免多个 arecord 并发占用同一 ALSA 设备。"""

import threading
from contextlib import contextmanager


_MIC_LOCK = threading.RLock()


@contextmanager
def mic_capture_lock(*, timeout_sec: float | None = None):
    acquired = _MIC_LOCK.acquire(timeout=None if timeout_sec is None else float(timeout_sec))
    try:
        if not acquired:
            return
        yield
    finally:
        if acquired:
            _MIC_LOCK.release()
