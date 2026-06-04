from __future__ import annotations

"""运行时状态 JSON 持久化模块。

JsonStore 保存的是 AgentState 快照，属于运行时状态；
显式用户画像由 UserProfileStore 单独保存。
"""

import json
import time
from pathlib import Path
from typing import Any

from src.agent.state import AgentState


class JsonStore:
    """使用 JSON 文件保存和恢复运行时状态。"""

    def __init__(self, path: str | Path) -> None:
        # 调用方决定具体路径；测试通常传入临时目录。
        self.path = Path(path)

    def load_state_dict(self) -> dict[str, Any] | None:
        """读取状态字典。

        如果文件不存在、读取失败或格式不合法，则返回 None。
        """
        # 返回 None 让 AgentState.from_dict 自行回退到默认状态。
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 状态文件损坏不应阻止 Agent 启动，后续会以默认状态覆盖。
            return None
        state_data = data.get("state")
        if isinstance(state_data, dict):
            return state_data
        return None

    def save_state(self, state: AgentState) -> None:
        """将当前状态快照写入 JSON 文件。"""
        # 确保存储目录存在，便于 data/ 目录首次运行时自动创建。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": int(time.time()),
            # AgentState 自己负责把嵌套 dataclass 转成 dict。
            "state": state.to_dict(),
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
