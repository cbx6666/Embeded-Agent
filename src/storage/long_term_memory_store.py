from __future__ import annotations

"""LongTermMemory 持久化仓库。

它是什么：
LongTermMemoryStore 保存已经通过 MemoryValidator 的长期记忆，并负责去重、证据合并、
基础衰减字段和冲突标记。

它不是什么：
它不是 LLM 调用器，不判断记忆价值，不保存 UserProfile，不向 DecisionPipeline 暴露直接读取入口。

为什么存在：
长期记忆需要独立持久化和可追溯证据。把仓库与 Pipeline 拆开，可以保证 LLM 永远只能提出
候选，不能直接写状态。

边界：
写入者是 LongTermMemoryPipeline；读取者是 PersonalContextBuilder。
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.agent.memory.long_term_memory import LongTermMemory
from src.agent.memory.memory_candidate import MemoryCandidate


class LongTermMemoryStore:
    """JSON 文件形式的长期记忆仓库。"""

    def __init__(self, path: str | Path = "data/long_term_memory.json") -> None:
        self.path = Path(path)

    def list(self, user_id: str | None = None) -> list[LongTermMemory]:
        """读取全部或指定用户的 active 长期记忆。"""

        memories = [item for item in self._load() if item.status == "active"]
        if user_id is None:
            return memories
        return [item for item in memories if item.user_id == user_id]

    def upsert_candidate(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        *,
        timestamp: int | None = None,
    ) -> LongTermMemory:
        """写入或合并一条通过验证的候选记忆。"""

        now = int(time.time()) if timestamp is None else int(timestamp)
        memories = self._load()
        memory_id = _memory_id(user_id, candidate.memory_type, candidate.content)
        for item in memories:
            if item.id == memory_id:
                item.confidence = max(item.confidence, candidate.confidence)
                item.evidence = _merge_evidence(item.evidence, candidate.evidence)
                item.metadata.update(candidate.metadata)
                item.updated_at = now
                item.status = "active"
                self._save(memories)
                return item

        stored = LongTermMemory(
            id=memory_id,
            user_id=user_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
            confidence=candidate.confidence,
            evidence=list(candidate.evidence),
            source=candidate.source,
            metadata=dict(candidate.metadata),
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )
        memories.append(stored)
        self._save(memories)
        return stored

    def replace_user_memories(self, user_id: str, memories: list[LongTermMemory]) -> None:
        """替换指定用户的长期记忆集合，供维护/迁移工具使用。"""

        all_memories = [item for item in self._load() if item.user_id != user_id]
        all_memories.extend(memories)
        self._save(all_memories)

    def _load(self) -> list[LongTermMemory]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        raw = data.get("memories", [])
        if not isinstance(raw, list):
            return []
        memories: list[LongTermMemory] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                memories.append(LongTermMemory.from_dict(item))
            except (TypeError, ValueError):
                continue
        return memories

    def _save(self, memories: list[LongTermMemory]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": int(time.time()),
            "memories": [
                item.to_dict()
                for item in sorted(memories, key=lambda x: (x.user_id, x.memory_type, x.content))
            ],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _memory_id(user_id: str, memory_type: str, content: str) -> str:
    raw = f"{user_id}\n{memory_type}\n{content.strip().lower()}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _merge_evidence(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(left)
    seen = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in merged}
    for item in right:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            merged.append(dict(item))
            seen.add(key)
    return merged[-20:]
