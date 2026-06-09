from __future__ import annotations

"""UserProfile 持久化仓库。

它是什么：
UserProfileStore 只负责把显式用户画像读写到 JSON 文件。

它不是什么：
它不是长期记忆仓库，不保存 LLM 提取出的行为模式，也不执行偏好推断。

为什么存在：
显式 profile 是 Authoritative Source，需要独立于运行期状态和长期记忆持久化。

边界：
只有 UserProfileService 应该调用本仓库；其余模块通过 service 读取快照。
"""

import json
import time
from pathlib import Path
from typing import Any


class UserProfileStore:
    """JSON 文件形式的显式用户画像仓库。"""

    def __init__(self, path: str | Path = "data/user/user_profiles.json") -> None:
        self.path = Path(path)

    def load_profiles(self) -> dict[str, dict[str, Any]]:
        """读取 profile 原始字典；文件不存在或格式异常时返回空字典。"""

        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        raw_profiles = data.get("profiles")
        if not isinstance(raw_profiles, dict):
            return {}
        return {
            str(user_id): dict(raw_profile)
            for user_id, raw_profile in raw_profiles.items()
            if isinstance(raw_profile, dict)
        }

    def save_profiles(self, profiles: dict[str, dict[str, Any]], *, compact: bool = True) -> None:
        """把 profiles 原始字典写入 JSON 文件（默认紧凑 JSON）。"""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": int(time.time()),
            "profiles": {
                user_id: profile
                for user_id, profile in sorted(profiles.items())
            },
        }
        if compact:
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        else:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.path.write_text(text, encoding="utf-8")
