from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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


class RealAgentBehaviorScenarioTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.memory_store = LongTermMemoryStore(self.root / "memory.json")
        self.profile_service = UserProfileService(UserProfileStore(self.root / "profiles.json"))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_multi_turn_preference_focus_absence_and_trace(self) -> None:
        user_text = "我喜欢温和提醒，不喜欢频繁打断；接下来我想专注学习 40 分钟。"
        candidate = _preference_candidate(
            content="用户偏好温和、低频率的提醒方式。",
            timestamp=100,
            user_text=user_text,
            key="reminder_style",
            value="gentle",
        )
        llm = FakeLLMService(
            {
                "memory_observer": [
                    _json({"worth_remembering": True, "reason": "explicit durable preference"}),
                    _json({"worth_remembering": False, "reason": "action outcome only"}),
                ],
                "memory_extractor": _json({"candidates": [candidate]}),
                "memory_critic": _json({"approved_indexes": [0], "rejected_reasons": []}),
                "memory_consolidator": _json({"candidates": [candidate]}),
            }
        )
        core = self._build_core(llm)
        try:
            core.handle_event(Event(type="user_text_input", timestamp=100, payload={"text": user_text}))

            memories = self.memory_store.list("default")
            self.assertEqual(len(memories), 1)
            self.assertIn("温和", memories[0].content)

            focus_actions, _ = core.handle_event(
                Event(type="focus_start_requested", timestamp=120, payload={"duration_sec": 2400, "source": "user"})
            )
            self.assertEqual(
                [action.payload["duration_sec"] for action in focus_actions if action.type == "start_timer"],
                [2400],
            )

            core.handle_event(Event(type="user_fatigue_updated", timestamp=130, payload={"fatigue_level": "high"}))
            core.handle_event(Event(type="user_presence_updated", timestamp=140, payload={"presence": "away"}))
            blocked_actions, _ = core.handle_event(
                Event(type="system_triggered", timestamp=150, payload={"trigger": "focus_health_check"})
            )

            self.assertFalse(any(action.payload.get("reason") == "rest_reminder" for action in blocked_actions))
            self.assertTrue(core.last_runtime_trace)
            trace = core.last_runtime_trace
            self.assertTrue(_has_stage_order(trace.stages(), ["event", "reducer", "personal_context", "agent_context", "prompt", "llm_output", "validator", "guard", "action_realizer", "action"]))
            self.assertIn("温和", trace.to_json())
            self.assertIn("presence safety blocked", trace.to_json())
            self.assertIn("prompt", trace.find("prompt", "response_writer")[0].payload)
        finally:
            core.shutdown()

    def test_later_opposite_preference_contradicts_old_memory(self) -> None:
        first = _preference_candidate(
            content="用户偏好低频率提醒。",
            timestamp=200,
            user_text="不要频繁提醒我。",
            key="reminder_frequency",
            value="low",
        )
        second = _preference_candidate(
            content="用户现在希望疲惫时更主动地提醒。",
            timestamp=300,
            user_text="我改主意了，疲惫时你可以更主动提醒我。",
            key="reminder_frequency",
            value="high",
        )

        stored_first = self.memory_store.upsert_candidate("u1", _candidate_from_dict(first), timestamp=200)
        stored_second = self.memory_store.upsert_candidate("u1", _candidate_from_dict(second), timestamp=300)

        active = self.memory_store.list("u1")
        all_items = self.memory_store.list("u1", include_inactive=True)
        self.assertEqual([item.id for item in active], [stored_second.id])
        self.assertEqual([item.id for item in all_items if item.status == "contradicted"], [stored_first.id])
        self.assertEqual(active[0].metadata["contradicts"], [stored_first.id])

    def _build_core(self, llm: FakeLLMService) -> AgentCore:
        return AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=llm,
            store=JsonStore(self.root / "runtime.json"),
            long_term_memory_pipeline=LongTermMemoryPipeline(self.memory_store),
            personal_context_builder=PersonalContextBuilder(
                long_term_memory_store=self.memory_store,
                user_profile_service=self.profile_service,
            ),
        )


def _preference_candidate(*, content: str, timestamp: int, user_text: str, key: str, value: str) -> dict[str, object]:
    return {
        "memory_type": "behavior_preference",
        "content": content,
        "confidence": 0.86,
        "evidence": [
            {
                "source_event_type": "user_text_input",
                "timestamp": timestamp,
                "source": "dialogue",
                "user_text": user_text,
            }
        ],
        "metadata": {"preference_key": key, "preference_value": value},
    }


def _candidate_from_dict(data: dict[str, object]):
    from src.agent.memory.memory_candidate import MemoryCandidate

    return MemoryCandidate.from_dict(data)


def _json(data: dict[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _has_stage_order(stages: tuple[str, ...], expected: list[str]) -> bool:
    position = 0
    for stage in stages:
        if position < len(expected) and stage == expected[position]:
            position += 1
    return position == len(expected)


if __name__ == "__main__":
    unittest.main()
