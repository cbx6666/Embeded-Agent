from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.core import AgentCore
from src.agent.event import Event
from src.agent.memory.long_term_memory_pipeline import LongTermMemoryPipeline
from src.agent.memory.memory_candidate import MemoryCandidate
from src.agent.user.personal_context_builder import PersonalContextBuilder
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.services.user_profile_service import UserProfileService
from src.storage.json_store import JsonStore
from src.storage.long_term_memory_store import LongTermMemoryStore
from src.storage.user_profile_store import UserProfileStore
from tests.fakes.fake_llm_service import FakeLLMService


class RuntimeStabilityScenarioTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.memory_store = LongTermMemoryStore(self.root / "memory.json")
        self.profile_service = UserProfileService(UserProfileStore(self.root / "profiles.json"))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_100_plus_events_keep_context_bounded_and_guard_effective(self) -> None:
        core = self._build_core(FakeLLMService())
        try:
            core.state.focus.active = True
            core.state.focus.start_ts = 1000
            core.state.focus.target_duration_sec = 3600
            core.state.focus.remaining_sec = 3600
            core.state.user.presence = "present"
            core.state.user.fatigue_level = "high"

            rest_reminders = 0
            for index in range(105):
                timestamp = 1000 + index * 10
                if index % 5 == 0:
                    event = Event(type="timer_ticked", timestamp=timestamp, payload={"remaining_sec": 3600 - index * 10})
                else:
                    event = Event(type="system_triggered", timestamp=timestamp, payload={"trigger": "focus_health_check"})
                actions, _ = core.handle_event(event)
                rest_reminders += int(any(action.payload.get("reason") == "rest_reminder" for action in actions))

            self.assertLessEqual(rest_reminders, 4)
            self.assertLessEqual(len(core.state.runtime_history.recent_events), 20)
            self.assertLessEqual(len(core.state.runtime_history.recent_actions), 20)
            self.assertTrue(core.last_runtime_trace)
            self.assertIn("cooldown active", core.last_runtime_trace.to_json())

            personal_context = core.personal_context_builder.build(
                user_id=core.state.current_user_id,
                state=core.state,
                event=Event(type="system_triggered", timestamp=3000, payload={"trigger": "focus_health_check"}),
            )
            recent_types = {event["type"] for event in personal_context.runtime_history["recent_events"]}
            self.assertNotIn("timer_ticked", recent_types)
        finally:
            core.shutdown()

    def test_memory_compression_and_retrieval_remain_bounded_as_store_grows(self) -> None:
        for index in range(30):
            content = f"Preference {index}: user likes stable reminders."
            if index == 25:
                content = "Needle: user likes gentle reminders during focus."
            self.memory_store.upsert_candidate(
                "u1",
                MemoryCandidate(
                    memory_type="behavior_preference",
                    content=content,
                    confidence=0.55 + (index / 100),
                    evidence=[
                        {
                            "source_event_type": "user_text_input",
                            "timestamp": 100 + index,
                            "source": "dialogue",
                            "user_text": content,
                        }
                    ],
                    metadata={"preference_key": f"pref_{index}", "preference_value": "on"},
                ),
                timestamp=100 + index,
            )

        personal_context = PersonalContextBuilder(long_term_memory_store=self.memory_store).build(
            user_id="u1",
            state=self._state_for_user("u1"),
            event=Event(type="system_triggered", timestamp=500, payload={"trigger": "focus_health_check"}),
        )
        relevant = personal_context.retrieve_relevant(event_type="system_triggered", text="gentle focus", limit=8)

        self.assertLessEqual(len(personal_context.behavior_preferences), 6)
        self.assertLessEqual(len(relevant), 8)
        self.assertTrue(any("Needle" in item["content"] for item in relevant))
        self.assertGreater(personal_context.compression["input_counts"]["behavior_preference"], 6)

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

    def _state_for_user(self, user_id: str):
        from src.agent.state import AgentState

        return AgentState(current_user_id=user_id)


if __name__ == "__main__":
    unittest.main()
