from __future__ import annotations

"""LongTermMemory 提取管线。

它是什么：
LongTermMemoryPipeline 是长期记忆的唯一写入编排入口，执行
observe -> extract -> critic -> consolidate -> validate -> store。

它不是什么：
它不是 RuntimeHistoryService，不保存短期窗口；不是 UserProfileService，不写显式资料；
也不是 DecisionPipeline，不参与当前动作决策。

为什么存在：
系统学习用户必须可证据化、可审查、可合并。将 LLM 参与阶段包在统一管线里，能明确
“LLM 只能提出候选，不能直接写 state/store/profile”。

边界：
上游读取 Event、Action outcome 和 AgentState 摘要；下游只能写 LongTermMemoryStore。
PersonalContextBuilder 后续读取 store 生成决策上下文。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agent.prompt_io import prompt_path, read_prompt

from src.agent.action import Action
from src.agent.event import Event
from src.agent.memory.long_term_memory import LongTermMemory
from src.agent.memory.memory_candidate import MemoryCandidate
from src.agent.memory.memory_consolidator import MemoryConsolidator
from src.agent.memory.memory_validator import MemoryValidator
from src.agent.execution.action_result import ActionResult
from src.agent.state import AgentState
from src.services.llm_service import LLMService
from src.storage.long_term_memory_store import LongTermMemoryStore


@dataclass
class LongTermMemoryContext:
    """传给长期记忆 LLM 角色的最小观察上下文。"""

    user_id: str
    event: dict[str, Any]
    state_summary: dict[str, Any]
    outcome: dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
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
class LongTermMemoryRunResult:
    """一轮长期记忆处理的可解释结果。"""

    candidates: list[MemoryCandidate] = field(default_factory=list)
    stored: list[LongTermMemory] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    stage_metadata: dict[str, Any] = field(default_factory=dict)


class LongTermMemoryContextBuilder:
    """从事件、动作结果和运行状态中构造长期记忆观察上下文。"""

    def build(
        self,
        *,
        user_id: str,
        event: Event,
        state: AgentState,
        outcome: dict[str, Any] | None = None,
    ) -> LongTermMemoryContext:
        """只提取与长期学习相关的状态摘要。"""

        return LongTermMemoryContext(
            user_id=user_id,
            event={"type": event.type, "timestamp": event.timestamp, "payload": dict(event.payload)},
            state_summary={
                "focus_active": state.focus.active,
                "focus_elapsed_sec": state.focus.elapsed_sec,
                "user_presence": state.user.presence,
                "user_attention": state.user.attention,
                "user_emotion": state.user.emotion,
                "user_fatigue": state.user.fatigue_level,
                "recent_message_count": len(state.runtime_history.recent_messages),
            },
            outcome=dict(outcome or {}),
        )


class LongTermMemoryPipeline:
    """长期记忆学习主入口。"""

    def __init__(
        self,
        store: LongTermMemoryStore | None = None,
        *,
        context_builder: LongTermMemoryContextBuilder | None = None,
        validator: MemoryValidator | None = None,
        consolidator: MemoryConsolidator | None = None,
        observer_prompt_path: Path | None = None,
        extractor_prompt_path: Path | None = None,
        critic_prompt_path: Path | None = None,
    ) -> None:
        self.store = store or LongTermMemoryStore()
        self.context_builder = context_builder or LongTermMemoryContextBuilder()
        self.validator = validator or MemoryValidator()
        self.consolidator = consolidator or MemoryConsolidator()
        _prompts = Path(__file__).resolve().parent / "prompts"
        self.observer_prompt_path = observer_prompt_path or prompt_path(_prompts, "memory_observer.md")
        self.extractor_prompt_path = extractor_prompt_path or prompt_path(_prompts, "memory_extractor.md")
        self.critic_prompt_path = critic_prompt_path or prompt_path(_prompts, "memory_critic.md")
        self.last_result: LongTermMemoryRunResult | None = None

    def process_event(
        self,
        user_id: str,
        event: Event,
        state: AgentState,
        llm_service: LLMService,
    ) -> LongTermMemoryRunResult:
        """从当前事件中观察长期记忆候选。"""

        context = self.context_builder.build(user_id=user_id, event=event, state=state)
        self.last_result = self._process(context, llm_service)
        return self.last_result

    def process_actions(
        self,
        user_id: str,
        actions: list[Action],
        timestamp: int,
        *,
        action_results: list[ActionResult] | None = None,
        source_event: Event | None = None,
        state: AgentState | None = None,
        llm_service: LLMService | None = None,
    ) -> LongTermMemoryRunResult | None:
        """从动作 outcome 中观察长期记忆候选。"""

        if llm_service is None or source_event is None or state is None:
            return None
        outcome = {
            "actions": [{"type": action.type, "payload": dict(action.payload)} for action in actions],
            "action_results": [_action_result_to_dict(result) for result in action_results or []],
            "timestamp": timestamp,
        }
        context = self.context_builder.build(
            user_id=user_id,
            event=source_event,
            state=state,
            outcome=outcome,
        )
        self.last_result = self._process(context, llm_service)
        return self.last_result

    def _process(self, context: LongTermMemoryContext, llm_service: LLMService) -> LongTermMemoryRunResult:
        result = LongTermMemoryRunResult()
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

        consolidated, consolidation_meta = self.consolidator.consolidate(
            existing=self.store.list(context.user_id),
            candidates=approved,
            llm_service=llm_service,
        )
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

    def _observe(self, context: LongTermMemoryContext, llm_service: LLMService) -> dict[str, Any]:
        prompt = f"{read_prompt(self.observer_prompt_path)}\n\nMemoryContext JSON:\n{context.to_prompt()}"
        try:
            raw = llm_service.complete_json("memory_observer", prompt)
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {"worth_remembering": False, "prompt": prompt, "raw": raw, "fallback": False}
            return {**data, "prompt": prompt, "raw": raw, "fallback": False}
        except Exception as exc:
            return {"worth_remembering": False, "prompt": prompt, "fallback": True, "error": str(exc)}

    def _extract(
        self,
        context: LongTermMemoryContext,
        llm_service: LLMService,
    ) -> tuple[list[MemoryCandidate], dict[str, Any]]:
        prompt = f"{read_prompt(self.extractor_prompt_path)}\n\nMemoryContext JSON:\n{context.to_prompt()}"
        try:
            raw_output = llm_service.complete_json("memory_extractor", prompt)
            data = json.loads(raw_output)
            raw = data.get("candidates", []) if isinstance(data, dict) else []
            candidates = [MemoryCandidate.from_dict(item) for item in raw]
            return candidates, {"prompt": prompt, "raw": raw_output, "fallback": False, "count": len(candidates)}
        except Exception as exc:
            return [], {"prompt": prompt, "fallback": True, "error": str(exc)}

    def _critic(
        self,
        candidates: list[MemoryCandidate],
        llm_service: LLMService,
    ) -> tuple[list[MemoryCandidate], list[str], dict[str, Any]]:
        if not candidates:
            return [], [], {"skipped": True}
        prompt = (
            f"{read_prompt(self.critic_prompt_path)}\n\n"
            f"Candidates JSON:\n{json.dumps([candidate.to_dict() for candidate in candidates], ensure_ascii=False)}"
        )
        try:
            raw_output = llm_service.complete_json("memory_critic", prompt)
            data = json.loads(raw_output)
            indexes = data.get("approved_indexes", []) if isinstance(data, dict) else []
            approved = [
                candidates[int(index)]
                for index in indexes
                if isinstance(index, int) or str(index).isdigit()
                if 0 <= int(index) < len(candidates)
            ]
            rejected = data.get("rejected_reasons", []) if isinstance(data, dict) else []
            return approved, [str(item) for item in rejected], {"prompt": prompt, "raw": raw_output, "fallback": False}
        except Exception as exc:
            return candidates, [], {"prompt": prompt, "fallback": True, "error": str(exc)}



def _action_result_to_dict(result: ActionResult) -> dict[str, Any]:
    return {
        "action_type": result.action_type,
        "success": bool(result.success),
        "timestamp": int(result.timestamp),
        "reason": result.reason,
        "payload": dict(result.payload),
    }
