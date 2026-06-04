from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.event import Event
from src.agent.memory.long_term_memory_pipeline import LongTermMemoryPipeline
from src.agent.memory.memory_candidate import MemoryCandidate
from src.agent.memory.memory_validator import MemoryValidator
from src.agent.state import AgentState
from src.storage.long_term_memory_store import LongTermMemoryStore
from tests.fakes.fake_llm_service import FakeLLMService


class MemoryPollutionScenarioTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = LongTermMemoryStore(self.root / "memory.json")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_hallucinated_or_weak_memory_never_enters_long_term_store(self) -> None:
        hallucinated = {
            "memory_type": "behavior_preference",
            "content": "User loves loud hourly reminders.",
            "confidence": 0.9,
            "evidence": [{"source": "llm"}],
            "metadata": {"preference_key": "reminder_style", "preference_value": "loud"},
        }
        llm = FakeLLMService(
            {
                "memory_observer": json.dumps({"worth_remembering": True}),
                "memory_extractor": json.dumps({"candidates": [hallucinated]}),
                "memory_critic": json.dumps({"approved_indexes": [0], "rejected_reasons": []}),
                "memory_consolidator": json.dumps({"candidates": [hallucinated]}),
            }
        )

        result = LongTermMemoryPipeline(self.store).process_event(
            "u1",
            Event(type="user_text_input", timestamp=10, payload={"text": "hello"}),
            AgentState(current_user_id="u1"),
            llm,
        )

        self.assertEqual(self.store.list("u1"), [])
        self.assertIn("memory evidence is not grounded", result.rejected[0])
        self.assertTrue(result.stage_metadata["memory_extractor"]["raw"])

    def test_invalid_memory_type_and_dialogue_free_preference_are_rejected(self) -> None:
        validator = MemoryValidator()
        invalid_type = MemoryCandidate(
            memory_type="personality_fact",
            content="User is always angry.",
            confidence=0.8,
            evidence=[{"source_event_type": "user_text_input", "timestamp": 20, "source": "dialogue", "user_text": "not this"}],
        )
        missing_dialogue_text = MemoryCandidate(
            memory_type="behavior_preference",
            content="User prefers proactive reminders.",
            confidence=0.8,
            evidence=[{"source_event_type": "user_text_input", "timestamp": 21, "source": "dialogue"}],
            metadata={"preference_key": "reminder_frequency", "preference_value": "high"},
        )

        self.assertEqual(validator.validate(invalid_type), "invalid memory_type: personality_fact")
        self.assertEqual(
            validator.validate(missing_dialogue_text),
            "preference memory requires user_text_input/speech_recognized evidence with timestamp, source, and user_text/snippet",
        )

    def test_repeated_memory_merges_caps_evidence_and_decay_is_deterministic(self) -> None:
        for index in range(35):
            self.store.upsert_candidate(
                "u1",
                MemoryCandidate(
                    memory_type="behavior_preference",
                    content="User prefers gentle reminders.",
                    confidence=0.7,
                    evidence=[
                        {
                            "source_event_type": "user_text_input",
                            "timestamp": 100 + index,
                            "source": "dialogue",
                            "user_text": f"gentle reminder please {index}",
                        }
                    ],
                    metadata={"preference_key": "reminder_style", "preference_value": "gentle"},
                ),
                timestamp=100 + index,
            )

        memories = self.store.list("u1")
        self.assertEqual(len(memories), 1)
        self.assertLessEqual(len(memories[0].evidence), 20)

        self.store.apply_decay(now=100 + 120 * 86400)
        decayed = self.store.list("u1")[0]
        self.assertLess(decayed.decay, 1.0)
        self.assertGreaterEqual(decayed.decay, 0.1)


if __name__ == "__main__":
    unittest.main()
