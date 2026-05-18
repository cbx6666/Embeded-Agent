from __future__ import annotations

"""长期记忆实体模型。

它是什么：
LongTermMemory 是系统从长期交互中沉淀出的可证据化记忆，带有 confidence、evidence、
decay 和 contradiction metadata。

它不是什么：
它不是 UserProfile，不保存用户明确声明的权威资料；也不是 RuntimeHistory，不保存完整
会话窗口。

为什么存在：
系统需要从重复行为和行动结果中学习，但这些学习必须可追溯、可衰减、可处理冲突。

边界：
只有 LongTermMemoryStore 能持久化该模型；只有 LongTermMemoryPipeline 在通过
MemoryValidator 后才能写入。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LongTermMemory:
    """已经验证并持久化的长期记忆记录。"""

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
    last_accessed_at: int = 0
    decay: float = 1.0
    status: str = "active"
    contradiction_of: str | None = None

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
            "last_accessed_at": self.last_accessed_at,
            "decay": self.decay,
            "status": self.status,
            "contradiction_of": self.contradiction_of,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LongTermMemory":
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
            last_accessed_at=int(data.get("last_accessed_at", 0)),
            decay=float(data.get("decay", 1.0)),
            status=str(data.get("status", "active")),
            contradiction_of=data.get("contradiction_of"),
        )
