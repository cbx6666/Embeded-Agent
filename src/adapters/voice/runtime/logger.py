from __future__ import annotations

"""语音子系统统一中文日志。"""

import sys
from typing import Callable

_log_hook: Callable[[str], None] | None = None


def set_voice_log_hook(hook: Callable[[str], None] | None) -> None:
    global _log_hook
    _log_hook = hook


def voice_log(message: str) -> None:
    line = f"[语音] {message}"
    if _log_hook is not None:
        _log_hook(line)
    else:
        print(line, flush=True)


def voice_debug(message: str) -> None:
    if _log_hook is not None:
        _log_hook(f"[语音调试] {message}")
