"""
MemoryStore 持久化模块。

本模块负责保存已经通过 MemoryValidator 的长期记忆。上游输入是
LLMMemoryManager 写入的 MemoryCandidate，下游输出是 ProfileSnapshotBuilder
读取的 StoredMemory 列表。

本模块不调用 LLM、不判断语义价值、不生成用户画像结论；它只负责 JSON 存取、
去重 upsert 和 evidence 合并。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agent.memory.schemas import MemoryCandidate


@dataclass
class StoredMemory:
    """已通过校验并持久化的长期记忆记录。"""

    id: str
    user_id: str
    memory_type: str
    content: str
    confidence: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    source: str = "llm"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    updated_at: int = 0
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "source": self.source,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredMemory":
        return cls(
            id=str(data.get("id", "")),
            user_id=str(data.get("user_id", "default")),
            memory_type=str(data.get("memory_type", "uncertain")),
            content=str(data.get("content", "")),
            confidence=float(data.get("confidence", 0.5)),
            evidence=[dict(item) for item in data.get("evidence", []) if isinstance(item, dict)],
            source=str(data.get("source", "llm")),
            metadata=dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), dict) else {},
            created_at=int(data.get("created_at", 0)),
            updated_at=int(data.get("updated_at", 0)),
            status=str(data.get("status", "active")),
        )


class MemoryStore:
    """JSON 文件记忆仓库。

    LLM 只提出候选记忆，本仓库只保存调用方已经验证过的记录。相同用户、类型
    和内容会合并 evidence，避免重复膨胀。
    """

    def __init__(self, path: str | Path = "data/memory_store.json") -> None:
        self.path = Path(path)

    def list(self, user_id: str | None = None) -> list[StoredMemory]:
        """读取全部或指定用户的长期记忆。"""

        memories = self._load()
        if user_id is None:
            return memories
        return [item for item in memories if item.user_id == user_id]

    def upsert_candidate(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        *,
        timestamp: int | None = None,
    ) -> StoredMemory:
        """写入或合并一条候选记忆。

        这里不再做 schema 判断，调用方必须先经过 MemoryValidator；store 只负责
        稳定 ID、证据合并和持久化。
        """

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

        stored = StoredMemory(
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
        )
        memories.append(stored)
        self._save(memories)
        return stored

    def replace_user_memories(self, user_id: str, memories: list[StoredMemory]) -> None:
        all_memories = [item for item in self._load() if item.user_id != user_id]
        all_memories.extend(memories)
        self._save(all_memories)

    def _load(self) -> list[StoredMemory]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        raw = data.get("memories", [])
        if not isinstance(raw, list):
            return []
        memories: list[StoredMemory] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                memories.append(StoredMemory.from_dict(item))
            except (TypeError, ValueError):
                continue
        return memories

    def _save(self, memories: list[StoredMemory]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": int(time.time()),
            "memories": [item.to_dict() for item in sorted(memories, key=lambda x: (x.user_id, x.memory_type, x.content))],
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
