from __future__ import annotations

"""长期用户画像 JSON 持久化模块。

ProfileStore 只关心 JSON 文件读写和最外层结构校验，
不理解“提醒风格”“偏好内容类型”等业务语义。
原始字典到 UserProfile dataclass 的转换由 UserProfileService 负责。
"""

import json
import time
from pathlib import Path
from typing import Any


class ProfileStore:
    """从 JSON 文件保存和恢复用户画像原始数据。"""

    def __init__(self, path: str | Path = "data/user_profiles.json") -> None:
        # 默认路径和运行时状态分开，避免长期 profile 随短期状态一起膨胀。
        self.path = Path(path)

    def load_profiles(self) -> dict[str, dict[str, Any]]:
        """读取 profiles 原始字典；文件不存在或格式异常时返回空字典。"""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 文件损坏不阻塞 Agent 启动；服务层会自动补齐 default 用户。
            return {}

        raw_profiles = data.get("profiles")
        if not isinstance(raw_profiles, dict):
            return {}

        # 这里只做最外层类型过滤，不修复字段含义，避免 store 层掺杂业务规则。
        return {
            str(user_id): dict(raw_profile)
            for user_id, raw_profile in raw_profiles.items()
            if isinstance(raw_profile, dict)
        }

    def save_profiles(self, profiles: dict[str, dict[str, Any]]) -> None:
        """把 profiles 原始字典写入 JSON 文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "updated_at": int(time.time()),
            # 排序后写入，减少无意义 diff，也方便人工查看。
            "profiles": {
                user_id: profile
                for user_id, profile in sorted(profiles.items())
            },
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
