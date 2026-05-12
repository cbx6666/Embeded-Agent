from __future__ import annotations

"""长期记忆写入校验器。

它是什么：
MemoryValidator 是 LongTermMemoryStore 之前的确定性边界，校验候选类型、内容、证据、
置信度和来源。

它不是什么：
它不是 LLM critic，不判断“用户到底是不是这样的人”，不写 store，也不写 profile。

为什么存在：
LLM 可以参与提取，但最终写入必须由确定性代码把关，防止无证据、低质量或越权的内容进入
长期记忆。

边界：
所有 LongTermMemory 写入都必须先经过本类；UserProfile 字段不允许从这里写入。
"""

from src.agent.memory.memory_candidate import ALLOWED_LONG_TERM_MEMORY_TYPES, MemoryCandidate


class MemoryValidator:
    """候选长期记忆的确定性写入边界。"""

    def validate(self, candidate: MemoryCandidate) -> str | None:
        """返回 None 表示可写入，否则返回拒绝原因。"""

        if candidate.memory_type not in ALLOWED_LONG_TERM_MEMORY_TYPES:
            return f"invalid memory_type: {candidate.memory_type}"
        if not candidate.content:
            return "memory content is empty"
        if not candidate.evidence:
            return "memory evidence is required"
        if candidate.confidence < 0.0 or candidate.confidence > 1.0:
            return "memory confidence out of range"
        if candidate.memory_type == "behavior_preference" and candidate.source == "profile":
            return "explicit profile data must stay in UserProfile"
        return None
