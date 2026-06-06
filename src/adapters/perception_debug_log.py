from __future__ import annotations

"""摄像头感知调试日志：行为 / 情绪 / 疲劳检测输出。"""

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_PERCEPTION_DEBUG_DIR = Path("data/perception_debug")
LOG_FILENAME = "perception.log"


class PerceptionDebugLog:
    """进程内感知调试日志（追加写入 perception.log，同时可选打印到终端）。"""

    def __init__(
        self,
        *,
        log_dir: str | Path = DEFAULT_PERCEPTION_DEBUG_DIR,
        enabled: bool = False,
        console: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.console = bool(console)
        self.log_dir = Path(log_dir).expanduser()
        self.log_path = self.log_dir / LOG_FILENAME
        self._lock = threading.Lock()
        env_dir = os.environ.get("EMBED_PERCEPTION_DEBUG_DIR", "").strip()
        if env_dir:
            self.log_dir = Path(env_dir).expanduser()
            self.log_path = self.log_dir / LOG_FILENAME
        env_on = os.environ.get("EMBED_PERCEPTION_DEBUG", "").strip().lower()
        if env_on in {"1", "true", "yes", "on"}:
            self.enabled = True
        if os.environ.get("EMBED_PERCEPTION_DEBUG_CONSOLE", "").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            self.console = False

    def start_session(self, *, note: str = "") -> None:
        if not self.enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        header = f"===== perception debug session {datetime.now().isoformat(timespec='seconds')} ====="
        if note:
            header = f"{header} {note}"
        with self.log_path.open("w", encoding="utf-8") as handle:
            handle.write(header + "\n")

    def log(self, module: str, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        payload = " ".join(f"{key}={value!r}" for key, value in fields.items())
        line = f"[{ts}] [{module}] {event}" + (f" {payload}" if payload else "")
        with self._lock:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self.console:
            print(f"[PerceptionDebug] {line}", flush=True)


_manager: PerceptionDebugLog | None = None


def configure_perception_debug(
    *,
    enabled: bool,
    log_dir: str | Path = DEFAULT_PERCEPTION_DEBUG_DIR,
    console: bool = True,
    session_note: str = "",
) -> PerceptionDebugLog:
    """配置进程内感知调试日志，并在启用时重置日志文件。"""
    global _manager
    _manager = PerceptionDebugLog(log_dir=log_dir, enabled=enabled, console=console)
    if _manager.enabled:
        _manager.start_session(note=session_note)
    return _manager


def perception_debug() -> PerceptionDebugLog:
    global _manager
    if _manager is None:
        _manager = PerceptionDebugLog()
    return _manager
