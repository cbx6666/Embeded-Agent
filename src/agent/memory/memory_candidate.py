from __future__ import annotations

"""长期记忆候选模型。

它是什么：
MemoryCandidate 是 LLM 或确定性提取器提出的“可能值得沉淀为长期记忆”的候选。

它不是什么：
它不是已经写入的 LongTermMemory，也不是 UserProfile。候选必须经过 critic、
consolidator 和 MemoryValidator 后才允许进入 LongTermMemoryStore。

为什么存在：
LLM 可以参与观察和提取，但不能直接写 store。候选对象就是 LLM 输出与确定性写入边界
之间的缓冲层。

边界：
MemoryCandidate 只允许描述来自 event、dialogue、action outcome 或 repeated behaviors
的可证据化内容；显式 profile 字段不应在这里重复保存。
"""

from dataclasses import dataclass, field
from typing import Any


ALLOWED_LONG_TERM_MEMORY_TYPES = {
    "behavior_preference",
    "behavior_pattern",
    "interaction_style",
    "active_constraint",
    "uncertain",
}

@dataclass
class MemoryCandidate:
    """尚未写入长期记忆仓库的候选记忆。"""

    memory_type: str
    content: str
    confidence: float = 0.5
    evidence: list[dict[str, Any]] = field(default_factory=list)
    source: str = "llm"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: object) -> "MemoryCandidate":
        """从 LLM JSON 恢复候选。

        这里不做旧类型兼容转换；不属于长期记忆语义边界的类型会交给
        MemoryValidator 拒绝，避免旧概念继续污染 LongTermMemory。
        """

        if not isinstance(data, dict):
            raise ValueError("memory candidate must be an object")
        evidence = data.get("evidence", [])
        if evidence is None:
            evidence = []
        if not isinstance(evidence, list):
            raise ValueError("memory evidence must be a list")
        metadata = data.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("memory metadata must be an object")
        return cls(
            memory_type=str(data.get("memory_type", "")).strip(),
            content=str(data.get("content", "")).strip(),
            confidence=_clamp(data.get("confidence", 0.5)),
            evidence=[dict(item) for item in evidence if isinstance(item, dict)],
            source=str(data.get("source", "llm")).strip() or "llm",
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可用于 prompt、trace 和 store 的稳定字典。"""

        return {
            "memory_type": self.memory_type,
            "content": self.content,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


def _clamp(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return max(0.0, min(1.0, number))
