from __future__ import annotations

import json
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


class PersonalizationConsistencyScenarioTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.memory_store = LongTermMemoryStore(self.root / "memory.json")
        self.profile_service = UserProfileService(UserProfileStore(self.root / "profiles.json"))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_same_user_prompt_keeps_preference_and_different_users_are_isolated(self) -> None:
        self._store_preference("alice", "Alice prefers gentle visual reminders.", "reminder_style", "gentle", 10)
        self._store_preference("bob", "Bob prefers concise direct reminders.", "reminder_style", "direct", 11)

        alice_prompt = self._run_user_message("alice", "Can you remind me later?", 20)
        bob_prompt = self._run_user_message("bob", "Can you remind me later?", 30)

        self.assertIn("gentle", alice_prompt)
        self.assertNotIn("Bob prefers", alice_prompt)
        self.assertIn("direct", bob_prompt)
        self.assertNotIn("Alice prefers", bob_prompt)

    def test_runtime_history_does_not_become_long_term_preference_without_evidence(self) -> None:
        self._store_preference("alice", "Alice prefers gentle reminders.", "reminder_style", "gentle", 40)
        llm = FakeLLMService(
            {
                "memory_observer": json.dumps({"worth_remembering": False, "reason": "short-lived runtime note"}),
                "response_writer": json.dumps({"speak_text": "好的，这次我会直接一点。", "display_text": "好的，这次我会直接一点。", "tone": "calm"}),
            }
        )
        core = self._build_core(llm, user_id="alice")
        try:
            core.handle_event(Event(type="user_text_input", timestamp=50, payload={"text": "这一次可以直接一点，不代表长期偏好。"}))
            memories = self.memory_store.list("alice")
            self.assertEqual(len(memories), 1)
            self.assertIn("gentle", memories[0].content)
            self.assertTrue(
                any("这一次可以直接一点" in message["text"] for message in core.state.runtime_history.recent_messages)
            )
        finally:
            core.shutdown()

    def _run_user_message(self, user_id: str, text: str, timestamp: int) -> str:
        llm = FakeLLMService({"memory_observer": json.dumps({"worth_remembering": False})})
        core = self._build_core(llm, user_id=user_id)
        try:
            core.handle_event(Event(type="user_text_input", timestamp=timestamp, payload={"text": text}))
            prompts = [prompt for role, prompt in llm.prompts if role == "response_writer"]
            self.assertTrue(prompts)
            return prompts[-1]
        finally:
            core.shutdown()

    def _build_core(self, llm: FakeLLMService, *, user_id: str) -> AgentCore:
        core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=llm,
            store=JsonStore(self.root / f"runtime-{user_id}.json"),
            long_term_memory_pipeline=LongTermMemoryPipeline(self.memory_store),
            personal_context_builder=PersonalContextBuilder(
                long_term_memory_store=self.memory_store,
                user_profile_service=self.profile_service,
            ),
        )
        core.switch_user(user_id, timestamp=1)
        return core

    def _store_preference(self, user_id: str, content: str, key: str, value: str, timestamp: int) -> None:
        self.memory_store.upsert_candidate(
            user_id,
            MemoryCandidate(
                memory_type="behavior_preference",
                content=content,
                confidence=0.9,
                evidence=[
                    {
                        "source_event_type": "user_text_input",
                        "timestamp": timestamp,
                        "source": "dialogue",
                        "user_text": content,
                    }
                ],
                metadata={"preference_key": key, "preference_value": value},
            ),
            timestamp=timestamp,
        )


if __name__ == "__main__":
    unittest.main()
