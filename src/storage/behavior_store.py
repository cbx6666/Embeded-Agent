from __future__ import annotations

"""长期行为统计 JSON 持久化模块。

BehaviorStore 只负责读取和写入原始 JSON 字典，不理解“夜间学习”“音乐休息”
等业务语义。BehaviorStats 的恢复、统计含义和画像抽取由 agent/memory 层负责。
"""

import json
import time
from pathlib import Path
from typing import Any


class BehaviorStore:
    """从 JSON 文件保存和恢复用户长期行为统计原始数据。"""

    def __init__(self, path: str | Path = "data/behavior_stats.json") -> None:
        # 行为统计独立于 runtime_store 和 user_profiles，避免短期状态与长期模型互相污染。
        self.path = Path(path)

    def load_stats(self) -> dict[str, dict[str, Any]]:
        """读取所有用户的行为统计原始字典；文件缺失或损坏时返回空字典。"""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        raw_stats = data.get("stats")
        if not isinstance(raw_stats, dict):
            return {}

        # 这里只做最外层类型过滤，不修复字段含义，保持 storage 层边界干净。
        return {
            str(user_id): dict(raw_stat)
            for user_id, raw_stat in raw_stats.items()
            if isinstance(raw_stat, dict)
        }

    def save_stats(self, stats: dict[str, dict[str, Any]]) -> None:
        """把所有用户的行为统计原始字典写入 JSON 文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "updated_at": int(time.time()),
            "stats": {
                user_id: stat
                for user_id, stat in sorted(stats.items())
            },
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
