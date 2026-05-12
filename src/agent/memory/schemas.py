from __future__ import annotations

"""
LLM 记忆数据结构模块。

本文件定义 LLM-managed Memory 链路中跨角色传递的结构化数据。上游是
`LLMMemoryManager` 中的 MemoryExtractor/MemoryCritic/MemoryConsolidator，
下游是 `MemoryStore` 和 `ProfileSnapshotBuilder`。

本模块不调用 LLM、不写入磁盘、不修改用户画像，也不参与行为决策。它只负责
把模型输出收敛为可验证的 Python 对象，为后续确定性校验提供边界。
"""

from dataclasses import dataclass, field
from typing import Any


ALLOWED_MEMORY_TYPES = {
    "explicit_preference",
    "behavior_pattern",
    "interaction_style",
    "active_constraint",
    "recent_context",
    "uncertain",
}


@dataclass
class MemoryCandidate:
    """LLM 提出的候选记忆。

    输入来自 MemoryExtractor 或 MemoryConsolidator 的 JSON 输出，输出给
    MemoryValidator/MemoryStore。它不代表已经写入长期记忆，只有通过确定性
    校验后才可以进入 store。
    """

    memory_type: str
    content: str
    confidence: float = 0.5
    evidence: list[dict[str, Any]] = field(default_factory=list)
    source: str = "llm"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: object) -> "MemoryCandidate":
        """从 LLM JSON 恢复候选记忆，并拒绝明显畸形的字段结构。

        这里不判断业务是否值得记忆；价值判断由 LLM 角色完成，安全写入由
        MemoryValidator 完成。解析失败会抛出 ValueError，让上层进入可解释
        fallback。
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
        """转成可持久化字典，供 prompt、trace 和 store 复用。"""

        return {
            "memory_type": self.memory_type,
            "content": self.content,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


def _clamp(value: object) -> float:
    """把 LLM 给出的置信度裁剪到 0 到 1，避免畸形输出污染后续排序。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return max(0.0, min(1.0, number))
