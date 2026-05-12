from __future__ import annotations

"""长期记忆合并与冲突处理。

它是什么：
MemoryConsolidator 负责把 critic 通过的候选与已有 LongTermMemory 做合并前整理，
并保留 LLM consolidate 阶段的 metadata。

它不是什么：
它不是 store，不直接持久化；也不是 UserProfile 合并器，不把行为偏好升级为显式资料。

为什么存在：
长期记忆会随着反复交互变多。合并阶段让系统有地方处理重复、冲突和表达归一化，避免 store
变成不可解释的流水账。

边界：
Consolidator 可以读取已有 LongTermMemory 摘要并调用 LLM，但输出仍然只是 MemoryCandidate，
最终写入必须经过 MemoryValidator。
"""

import json
from typing import Any

from src.agent.memory.long_term_memory import LongTermMemory
from src.agent.memory.memory_candidate import MemoryCandidate
from src.services.llm_service import LLMService


class MemoryConsolidator:
    """封装长期记忆 consolidate 阶段。"""

    def consolidate(
        self,
        *,
        existing: list[LongTermMemory],
        candidates: list[MemoryCandidate],
        llm_service: LLMService,
    ) -> tuple[list[MemoryCandidate], dict[str, Any]]:
        """合并候选；LLM 失败时返回原候选并记录 fallback。"""

        if not candidates:
            return [], {"skipped": True}
        prompt = (
            "Consolidate durable long-term memory candidates with existing memories. "
            "Return JSON: {\"candidates\":[...]} using the same candidate schema. "
            "Do not create UserProfile fields. Keep explicit profile facts out of memory.\n"
            + json.dumps(
                {
                    "existing": [item.to_dict() for item in existing[-20:]],
                    "new": [candidate.to_dict() for candidate in candidates],
                },
                ensure_ascii=False,
            )
        )
        try:
            data = json.loads(llm_service.complete_json("memory_consolidator", prompt))
            raw = data.get("candidates", []) if isinstance(data, dict) else []
            consolidated = [MemoryCandidate.from_dict(item) for item in raw]
            return consolidated, {"fallback": False, "count": len(consolidated)}
        except Exception as exc:
            return candidates, {"fallback": True, "error": str(exc)}
