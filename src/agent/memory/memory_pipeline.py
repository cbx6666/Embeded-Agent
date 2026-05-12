"""
MemoryPipeline 门面模块。

本模块是 AgentCore 调用 LLM-managed Memory 的入口。上游输入是当前 Event、
Action outcome 和 AgentState，下游输出是 MemoryRunResult 或 ProfileSnapshot。

本模块不做记忆价值判断、不直接写 profile，也不参与当前动作决策；它只把 Core
与 LLMMemoryManager、ProfileSnapshotBuilder 连接起来。
"""

from __future__ import annotations

from src.agent.action import Action
from src.agent.event import Event
from src.agent.memory.llm_memory_manager import LLMMemoryManager, MemoryContextBuilder, MemoryRunResult
from src.agent.memory.memory_store import MemoryStore
from src.agent.memory.profile_snapshot_builder import ProfileSnapshot, ProfileSnapshotBuilder
from src.agent.state import AgentState
from src.services.llm_service import LLMService
from src.services.user_profile_service import UserProfileService


class MemoryPipeline:
    """LLM-managed Memory 的主入口。

    输入事件或动作结果，输出记忆处理结果；同时提供 build_profile_snapshot，
    让 DecisionPipeline 只消费快照而不是直接读取 store。
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        context_builder: MemoryContextBuilder | None = None,
        manager: LLMMemoryManager | None = None,
        profile_snapshot_builder: ProfileSnapshotBuilder | None = None,
    ) -> None:
        self.store = store or MemoryStore()
        self.context_builder = context_builder or MemoryContextBuilder()
        self.manager = manager or LLMMemoryManager(self.store)
        self.profile_snapshot_builder = profile_snapshot_builder or ProfileSnapshotBuilder(self.store)
        self.last_result: MemoryRunResult | None = None

    def process_event(
        self,
        user_id: str,
        event: Event,
        state: AgentState,
        llm_service: LLMService,
    ) -> MemoryRunResult:
        """处理事件侧记忆观察。"""

        context = self.context_builder.build(user_id=user_id, event=event, state=state)
        self.last_result = self.manager.update(context, llm_service)
        return self.last_result

    def process_actions(
        self,
        user_id: str,
        actions: list[Action],
        timestamp: int,
        *,
        source_event: Event | None = None,
        state: AgentState | None = None,
        llm_service: LLMService | None = None,
    ) -> MemoryRunResult | None:
        """处理动作结果反馈。

        缺少 source_event、state 或 llm_service 时直接跳过，避免不完整 outcome
        写入长期记忆。
        """

        if llm_service is None or source_event is None or state is None:
            return None
        outcome = {
            "actions": [{"type": action.type, "payload": dict(action.payload)} for action in actions],
            "timestamp": timestamp,
        }
        context = self.context_builder.build(
            user_id=user_id,
            event=source_event,
            state=state,
            outcome=outcome,
        )
        self.last_result = self.manager.process(context, llm_service)
        return self.last_result

    def build_profile_snapshot(
        self,
        user_id: str,
        state: AgentState,
        event: Event | None = None,
        profile_service: UserProfileService | None = None,
    ) -> ProfileSnapshot:
        """为决策链路生成 ProfileSnapshot。"""

        return self.profile_snapshot_builder.build(
            user_id=user_id,
            state=state,
            event=event,
            profile_service=profile_service,
        )
