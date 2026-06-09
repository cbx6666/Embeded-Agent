from __future__ import annotations

"""控制台输出适配器模块。"""

import threading
from datetime import datetime
from pathlib import Path
from typing import TextIO

from src.agent.action import Action


class ConsoleOutput:
    """将系统动作转换成控制台或文件输出文本。"""

    def __init__(
        self,
        stream: TextIO | None = None,
        silent: bool = False,
        *,
        log_path: str | Path | None = None,
        log_console: bool = True,
    ) -> None:
        self.stream = stream
        self.silent = silent
        self.log_console = log_console
        self._lock = threading.Lock()
        self._log_file: TextIO | None = None
        if log_path is not None:
            path = Path(log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                handle.write(
                    f"===== session {datetime.now().isoformat(timespec='seconds')} =====\n"
                )
            self._log_file = path.open("a", encoding="utf-8")

    def execute(self, action: Action) -> None:
        if action.type == "set_tts_volume":
            self.show_text(f"[TTS] volume={action.payload.get('volume')}")
            return
        if action.type == "play_media":
            title = str(action.payload.get("title", "")).strip()
            path = str(action.payload.get("path", "")).strip()
            self.show_text(f"[Media] 开始播放：{title or path}")
            return
        if action.type == "stop_media":
            self.show_text("[Media] 停止播放")
            return
        text = str(action.payload.get("text", "")).strip()
        if not text:
            return
        if action.type == "speak":
            self.show_text(f"[Agent] {text}")
        elif action.type == "display":
            self.show_text(f"[Display] {text}")

    def show_text(self, text: str) -> None:
        if self.silent:
            return
        with self._lock:
            if self._log_file is not None:
                self._log_file.write(text + "\n")
                self._log_file.flush()
            if not self.log_console and self._log_file is not None:
                return
            if self.stream is not None:
                self.stream.write(text + "\n")
                self.stream.flush()
            else:
                print(text, flush=True)

    def close(self) -> None:
        with self._lock:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
