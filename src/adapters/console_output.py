from __future__ import annotations

"""控制台输出适配器模块。"""

import threading
from typing import TextIO

from src.agent.action import Action


class ConsoleOutput:
    """将系统动作转换成控制台输出文本。"""

    def __init__(self, stream: TextIO | None = None, silent: bool = False) -> None:
        self.stream = stream
        self.silent = silent
        self._lock = threading.Lock()

    def execute(self, action: Action) -> None:
        text = str(action.payload.get("text", "")).strip()
        if not text:
            if action.type == "render_pet_expression":
                expression = str(action.payload.get("expression", "")).strip()
                if expression:
                    self.show_text(f"[Pet] expression={expression}")
            elif action.type == "set_light_state":
                state = str(action.payload.get("state", "")).strip()
                if state:
                    self.show_text(f"[Light] state={state}")
            return
        if action.type == "speak":
            self.show_text(f"[Agent] {text}")
        elif action.type == "display":
            self.show_text(f"[Display] {text}")

    def show_text(self, text: str) -> None:
        if self.silent:
            return
        with self._lock:
            if self.stream is not None:
                self.stream.write(text + "\n")
                self.stream.flush()
            else:
                print(text, flush=True)
