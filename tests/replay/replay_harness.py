from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.adapters.console_output import ConsoleOutput
from src.agent.core import AgentCore
from src.agent.event import Event
from src.agent.memory.long_term_memory_pipeline import LongTermMemoryPipeline
from src.agent.user.personal_context_builder import PersonalContextBuilder
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.services.user_profile_service import UserProfileService
from src.storage.json_store import JsonStore
from src.storage.long_term_memory_store import LongTermMemoryStore
from src.storage.user_profile_store import UserProfileStore
from tests.fakes.fake_llm_service import FakeLLMService


@dataclass(frozen=True)
class ReplayResult:
    actions_by_event: tuple[tuple[dict[str, Any], ...], ...]
    trace_json_by_event: tuple[str, ...]


def replay_event_log(
    events: Iterable[Event],
    *,
    llm_factory: Callable[[], FakeLLMService] = FakeLLMService,
) -> ReplayResult:
    """Replay an event log in a fresh runtime and return deterministic action snapshots."""

    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        memory_store = LongTermMemoryStore(root / "memory.json")
        profile_service = UserProfileService(UserProfileStore(root / "profiles.json"), now_fn=lambda: 0)
        core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=llm_factory(),
            store=JsonStore(root / "runtime.json"),
            long_term_memory_pipeline=LongTermMemoryPipeline(memory_store),
            personal_context_builder=PersonalContextBuilder(
                long_term_memory_store=memory_store,
                user_profile_service=profile_service,
            ),
        )
        try:
            actions_by_event: list[tuple[dict[str, Any], ...]] = []
            trace_json_by_event: list[str] = []
            for event in events:
                actions, _ = core.handle_event(event)
                actions_by_event.append(tuple(_action_snapshot(action) for action in actions))
                trace_json_by_event.append(core.last_runtime_trace.to_json(indent=None) if core.last_runtime_trace else "")
            return ReplayResult(tuple(actions_by_event), tuple(trace_json_by_event))
        finally:
            core.shutdown()


def _action_snapshot(action: object) -> dict[str, Any]:
    return {
        "type": getattr(action, "type", ""),
        "payload": dict(getattr(action, "payload", {}) or {}),
    }
