from __future__ import annotations

"""长期画像候选与记忆策略决策模型。

InsightExtractor 先生成 MemoryCandidate，MemoryPolicy 再决定是否允许写入长期画像。
这样“抽象出什么”和“能不能长期保存”是两步，不会混在一个类里。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryCandidate:
    """一个待写入长期画像的候选结论。"""

    insight_type: str
    content: str
    confidence: float
    evidence_count: int
    source: str
    explanation: str
    contradiction_group: str | None = None


@dataclass
class MemoryPolicyDecision:
    """MemoryPolicy 对候选画像的生命周期判断。"""

    candidate: MemoryCandidate
    allow_write: bool
    reason: str
    requires_confirmation: bool = False
    contradicted_contents: list[str] = field(default_factory=list)
