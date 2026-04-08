from __future__ import annotations

"""JSON 持久化模块。"""

import json
import time
from pathlib import Path
from typing import Any

from src.agent.state import AgentState


class JsonStore:
    """使用 JSON 文件保存和恢复运行时状态。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_state_dict(self) -> dict[str, Any] | None:
        """读取状态字典。

        如果文件不存在、读取失败或格式不合法，则返回 None。
        """
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        state_data = data.get("state")
        if isinstance(state_data, dict):
            return state_data
        return None

    def save_state(self, state: AgentState) -> None:
        """将当前状态快照写入 JSON 文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": int(time.time()),
            "state": state.to_dict(),
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
