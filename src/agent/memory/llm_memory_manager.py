"""
LLM-managed Memory 编排模块。

本模块负责把事件、交互结果和状态摘要交给 MemoryObserver、MemoryExtractor、
MemoryCritic、MemoryConsolidator 四个 LLM 记忆角色，生成可验证的
MemoryCandidate。上游输入是 MemoryContext，下游输出是写入 MemoryStore 的
StoredMemory 以及 MemoryRunResult trace。

本模块不允许 LLM 直接写 store 或 profile；所有候选记忆必须先通过
MemoryValidator。它也不参与当前行为决策，DecisionPipeline 只能读取
ProfileSnapshot。
"""

from __future__ import annotations

"""
LLM-managed Memory 编排模块。

本模块负责把事件、交互结果和状态摘要交给 MemoryObserver、MemoryExtractor、
MemoryCritic、MemoryConsolidator 四个 LLM 记忆角色，生成可验证的
MemoryCandidate。上游输入是 MemoryContext，下游输出是写入 MemoryStore 的
StoredMemory 以及 MemoryRunResult trace。

本模块不允许 LLM 直接写 store 或 profile；所有候选记忆必须先通过
MemoryValidator。它也不参与当前行为决策，DecisionPipeline 只能读取
ProfileSnapshot。
"""

import json
from dataclasses import dataclass, field
from typing import Any

from src.agent.event import Event
from src.agent.memory.schemas import ALLOWED_MEMORY_TYPES, MemoryCandidate
from src.agent.memory.memory_store import MemoryStore, StoredMemory
from src.agent.state import AgentState
from src.services.llm_service import LLMService


@dataclass
class MemoryContext:
    """传给记忆 LLM 角色的紧凑上下文。

    它只包含用户、事件、状态摘要和可选 outcome，避免把完整历史写入记忆
    prompt。
    """
    user_id: str
    event: dict[str, Any]
    state_summary: dict[str, Any]
    outcome: dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """转成 JSON prompt 块，供四个 memory role 复用。"""

        return json.dumps(
            {
                "user_id": self.user_id,
                "event": self.event,
                "state_summary": self.state_summary,
                "outcome": self.outcome,
            },
            ensure_ascii=False,
            indent=2,
        )


@dataclass
class MemoryRunResult:
    """一轮记忆处理的可解释结果。

    `candidates` 是 LLM 提出的候选，`stored` 是最终通过确定性校验并写入的
    记录，`rejected` 保存拒绝原因。
    """
    candidates: list[MemoryCandidate] = field(default_factory=list)
    stored: list[StoredMemory] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    stage_metadata: dict[str, Any] = field(default_factory=dict)


class MemoryContextBuilder:
    """记忆上下文构建器。

    输入 Event、AgentState 和可选 outcome，输出 MemoryContext。它只摘要
    与长期记忆相关的状态，不把短期噪声直接写入 profile。
    """

    def build(
        self,
        *,
        user_id: str,
        event: Event,
        state: AgentState,
        outcome: dict[str, Any] | None = None,
    ) -> MemoryContext:
        """构造 LLM-managed memory 所需的最小上下文。"""

        return MemoryContext(
            user_id=user_id,
            event={"type": event.type, "timestamp": event.timestamp, "payload": dict(event.payload)},
            state_summary={
                "focus_active": state.focus.active,
                "focus_elapsed_sec": state.focus.elapsed_sec,
                "user_presence": state.user.presence,
                "user_attention": state.user.attention,
                "user_emotion": state.user.emotion,
                "user_fatigue": state.user.fatigue_level,
            },
            outcome=dict(outcome or {}),
        )


class LLMMemoryManager:
    """LLM-managed Memory 主编排器。

    输入 MemoryContext 和 LLMService，输出 MemoryRunResult。它负责调用四个
    memory role，但最终写入必须经过 MemoryValidator 和 MemoryStore。
    """

    def __init__(self, store: MemoryStore, validator: "MemoryValidator | None" = None) -> None:
        self.store = store
        self.validator = validator or MemoryValidator()

    def process(self, context: MemoryContext, llm_service: LLMService) -> MemoryRunResult:
        """处理一次事件或交互结果，产出候选记忆并安全写入。

        Observer 判断是否值得记忆；Extractor 提取候选；Critic 审查候选；
        Consolidator 合并候选。任何阶段失败都会记录 metadata，并走安全 fallback。
        """

        result = MemoryRunResult()
        observer = self._observe(context, llm_service)
        result.stage_metadata["memory_observer"] = observer
        if not observer.get("worth_remembering", False):
            return result

        candidates, extractor_meta = self._extract(context, llm_service)
        result.stage_metadata["memory_extractor"] = extractor_meta
        result.candidates = candidates

        approved, rejected, critic_meta = self._critic(candidates, llm_service)
        result.stage_metadata["memory_critic"] = critic_meta
        result.rejected.extend(rejected)

        consolidated, consolidation_meta = self._consolidate(context.user_id, approved, llm_service)
        result.stage_metadata["memory_consolidator"] = consolidation_meta

        for candidate in consolidated:
            error = self.validator.validate(candidate)
            if error:
                result.rejected.append(error)
                continue
            stored = self.store.upsert_candidate(
                context.user_id,
                candidate,
                timestamp=int(context.event.get("timestamp", 0)),
            )
            result.stored.append(stored)
        return result

    def update(self, context: MemoryContext, llm_service: LLMService) -> MemoryRunResult:
        """处理带 outcome 的记忆更新。

        `process` 和 `update` 使用同一条 LLM-managed memory 链路；命名区分只是
        为了让调用方表达“事件观察”和“交互结果反馈”两种上游来源。两者都不会
        让 LLM 直接写 store，最终仍经过 `_validate_candidate`。
        """

        return self.process(context, llm_service)

    def _observe(self, context: MemoryContext, llm_service: LLMService) -> dict[str, Any]:
        """判断当前上下文是否值得进入长期记忆链路。"""

        prompt = (
            "Decide whether this interaction may contain durable user memory. "
            "Return JSON: {\"worth_remembering\": true|false, \"reason\": \"...\"}.\n"
            + context.to_prompt()
        )
        try:
            data = json.loads(llm_service.complete_json("memory_observer", prompt))
            return data if isinstance(data, dict) else {"worth_remembering": False}
        except Exception as exc:
            return {"worth_remembering": False, "fallback": True, "error": str(exc)}

    def _extract(self, context: MemoryContext, llm_service: LLMService) -> tuple[list[MemoryCandidate], dict[str, Any]]:
        """让 LLM 提取候选记忆；解析失败时返回空候选并记录 fallback。"""

        prompt = (
            "Extract durable memory candidates. Return JSON: "
            "{\"candidates\":[{\"memory_type\":\"explicit_preference\","
            "\"content\":\"...\",\"confidence\":0.8,\"evidence\":[{}]}]}.\n"
            + context.to_prompt()
        )
        try:
            data = json.loads(llm_service.complete_json("memory_extractor", prompt))
            raw = data.get("candidates", []) if isinstance(data, dict) else []
            candidates = [MemoryCandidate.from_dict(item) for item in raw]
            return candidates, {"fallback": False, "count": len(candidates)}
        except Exception as exc:
            return [], {"fallback": True, "error": str(exc)}

    def _critic(
        self,
        candidates: list[MemoryCandidate],
        llm_service: LLMService,
    ) -> tuple[list[MemoryCandidate], list[str], dict[str, Any]]:
        """让 LLM 审查候选记忆。

        LLM 审查失败时保留原候选继续进入确定性 validator，避免因为服务波动
        丢掉所有记忆机会。
        """

        if not candidates:
            return [], [], {"skipped": True}
        prompt = (
            "Review memory candidates. Return JSON: "
            "{\"approved_indexes\":[0],\"rejected_reasons\":[]}.\n"
            + json.dumps([candidate.to_dict() for candidate in candidates], ensure_ascii=False)
        )
        try:
            data = json.loads(llm_service.complete_json("memory_critic", prompt))
            indexes = data.get("approved_indexes", []) if isinstance(data, dict) else []
            approved = [
                candidates[int(index)]
                for index in indexes
                if isinstance(index, int) or str(index).isdigit()
                if 0 <= int(index) < len(candidates)
            ]
            rejected = data.get("rejected_reasons", []) if isinstance(data, dict) else []
            return approved, [str(item) for item in rejected], {"fallback": False}
        except Exception as exc:
            return candidates, [], {"fallback": True, "error": str(exc)}

    def _consolidate(
        self,
        user_id: str,
        candidates: list[MemoryCandidate],
        llm_service: LLMService,
    ) -> tuple[list[MemoryCandidate], dict[str, Any]]:
        """让 LLM 合并新旧记忆。

        Consolidator 失败时回退到原候选，最终仍由 MemoryStore 的 upsert 去重。
        """

        if not candidates:
            return [], {"skipped": True}
        existing = [item.to_dict() for item in self.store.list(user_id)[-20:]]
        prompt = (
            "Consolidate new memory candidates with existing memories. Return JSON: "
            "{\"candidates\":[...]} using the same candidate schema.\n"
            + json.dumps({"existing": existing, "new": [c.to_dict() for c in candidates]}, ensure_ascii=False)
        )
        try:
            data = json.loads(llm_service.complete_json("memory_consolidator", prompt))
            raw = data.get("candidates", []) if isinstance(data, dict) else []
            consolidated = [MemoryCandidate.from_dict(item) for item in raw]
            return consolidated, {"fallback": False, "count": len(consolidated)}
        except Exception as exc:
            return candidates, {"fallback": True, "error": str(exc)}


class MemoryValidator:
    """候选记忆的确定性写入边界。

    输入是 LLM 生成并经过 MemoryCritic/MemoryConsolidator 处理后的
    `MemoryCandidate`，输出是 `None` 或可解释拒绝原因。它不判断“用户是不是
    真的这样的人”，只验证类型、内容、证据和置信度边界，防止无证据记忆污染
    长期画像。
    """

    def validate(self, candidate: MemoryCandidate) -> str | None:
        """校验候选记忆是否允许写入 MemoryStore。"""

        if candidate.memory_type not in ALLOWED_MEMORY_TYPES:
            return f"invalid memory_type: {candidate.memory_type}"
        if not candidate.content:
            return "memory content is empty"
        if not candidate.evidence:
            return "memory evidence is required"
        if candidate.confidence < 0.0 or candidate.confidence > 1.0:
            return "memory confidence out of range"
        return None
