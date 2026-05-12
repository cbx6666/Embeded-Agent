from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.core import AgentCore
from src.agent.event import Event
from src.services.llm_service import LLMService
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore


class SpyLLMService(LLMService):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_json(self, role: str, prompt: str) -> str:  # type: ignore[override]
        self.calls.append(role)
        return super()._mock_complete_json(role, prompt)

    def generate_reply(self, text: str, state=None) -> str:  # type: ignore[override]
        return "fallback reply"


class AgentCoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.llm = SpyLLMService()
        self.core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=self.llm,
            store=JsonStore(self.root / "runtime_store.json"),
        )

    def tearDown(self) -> None:
        self.core.shutdown()
        self.temp_dir.cleanup()

    def test_focus_start_updates_state_and_realizes_timer_action(self) -> None:
        actions, results = self.core.handle_event(
            Event(type="focus_start_requested", timestamp=1000, payload={"duration_sec": 600, "source": "test"})
        )

        self.assertTrue(self.core.state.focus.active)
        self.assertEqual(self.core.state.focus.target_duration_sec, 600)
        self.assertIn("start_timer", {action.type for action in actions})
        self.assertTrue(all(result.success for result in results))

    def test_user_text_runs_llm_roles_and_records_messages(self) -> None:
        actions, _ = self.core.handle_event(
            Event(type="user_text_input", timestamp=2000, payload={"text": "hello", "source": "test"})
        )

        self.assertIn("situation_analyst", self.llm.calls)
        self.assertIn("intent_planner", self.llm.calls)
        self.assertIn("safety_critic", self.llm.calls)
        self.assertIn("response_writer", self.llm.calls)
        self.assertIn("speak", {action.type for action in actions})
        self.assertTrue(any(message["role"] == "user" for message in self.core.state.runtime_history.recent_messages))

    def test_timer_finished_generates_completion_feedback(self) -> None:
        self.core.handle_event(
            Event(type="focus_start_requested", timestamp=1000, payload={"duration_sec": 1500, "source": "test"})
        )

        actions, _ = self.core.handle_event(
            Event(type="timer_finished", timestamp=2500, payload={"timer": "focus"})
        )

        self.assertFalse(self.core.state.focus.active)
        self.assertIn("stop_timer", {action.type for action in actions})
        self.assertIn("speak", {action.type for action in actions})

    def test_guard_cooldown_blocks_repeated_rest_notification(self) -> None:
        self.core.state.focus.active = True
        self.core.state.focus.elapsed_sec = 600
        self.core.state.focus.remaining_sec = 900
        self.core.state.user.presence = "present"
        self.core.state.user.fatigue_level = "high"

        first_actions, _ = self.core.handle_event(
            Event(type="system_triggered", timestamp=3000, payload={"trigger": "focus_health_check"})
        )
        second_actions, _ = self.core.handle_event(
            Event(type="system_triggered", timestamp=3020, payload={"trigger": "focus_health_check"})
        )

        self.assertTrue(any(action.payload.get("reason") == "rest_reminder" for action in first_actions))
        self.assertFalse(any(action.payload.get("reason") == "rest_reminder" for action in second_actions))


if __name__ == "__main__":
    unittest.main()
