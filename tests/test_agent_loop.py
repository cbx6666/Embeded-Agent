from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.adapters.console_output import ConsoleOutput
from src.agent.action import Action
from src.agent.core import AgentCore
from src.agent.event import Event
from src.agent.runtime.action_result import ActionResult
from src.agent.runtime.autonomy import build_autonomous_check_event
from src.agent.runtime.internal_events import build_internal_events_from_results
from src.agent.runtime.loop import AgentLoop
from src.services.llm_service import LLMService
from src.services.memory_service import MemoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore


class StubLLMService(LLMService):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_json(self, role: str, prompt: str) -> str:  # type: ignore[override]
        self.calls.append(role)
        return super()._mock_complete_json(role, prompt)

    def generate_reply(self, text: str, state=None) -> str:  # type: ignore[override]
        return "fallback reply"


class AgentLoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.llm = StubLLMService()
        self.core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            memory_service=MemoryService(),
            llm_service=self.llm,
            store=JsonStore(Path(self.temp_dir.name) / "runtime_store.json"),
        )
        self.loop = AgentLoop(self.core, max_steps=3)

    def tearDown(self) -> None:
        self.core.shutdown()
        self.temp_dir.cleanup()

    def test_run_once_handles_user_text_and_records_trace(self) -> None:
        actions = self.loop.run_once(
            Event(type="user_text_input", timestamp=1000, payload={"text": "hello", "source": "test"})
        )

        self.assertIn("speak", {action.type for action in actions})
        self.assertGreaterEqual(len(self.loop.recent_traces), 1)
        self.assertIn("answer_user", {intent["type"] for intent in self.loop.recent_traces[0].intents})

    def test_focus_start_closed_loop_has_step_limit(self) -> None:
        actions = self.loop.run_once(
            Event(type="focus_start_requested", timestamp=2000, payload={"duration_sec": 1500, "source": "test"})
        )

        self.assertIn("start_timer", {action.type for action in actions})
        self.assertTrue(self.core.state.focus.active)
        self.assertLessEqual(len(self.loop.recent_traces), 3)

    def test_action_results_can_become_internal_events(self) -> None:
        event = Event(type="user_text_input", timestamp=3000, payload={"text": "hello"})
        actions = [
            Action(type="speak", payload={"text": "ok"}),
            Action(type="set_light_state", payload={"state": "alert"}),
        ]
        results = [
            ActionResult(action_type="speak", success=True, timestamp=3000),
            ActionResult(action_type="set_light_state", success=False, timestamp=3000, reason="hardware_offline"),
        ]

        internal_events = build_internal_events_from_results(self.core.state, event, actions, results)
        triggers = {str(item.payload.get("trigger")) for item in internal_events}

        self.assertIn("agent_response_completed", triggers)
        self.assertIn("action_failed", triggers)

    def test_periodic_check_does_not_speak_when_user_is_away(self) -> None:
        self.core.state.user.presence = "away"

        actions = self.loop.run_once(
            build_autonomous_check_event(self.core.state, now_ts=5000, reason="periodic_check")
        )

        self.assertFalse(any(action.type == "speak" for action in actions))

    def test_focus_health_check_triggers_rest_reminder_and_cooldown(self) -> None:
        self.core.state.focus.active = True
        self.core.state.focus.elapsed_sec = 600
        self.core.state.focus.remaining_sec = 900
        self.core.state.user.presence = "present"
        self.core.state.user.fatigue_level = "high"

        first = self.loop.run_once(
            build_autonomous_check_event(self.core.state, now_ts=6000, reason="focus_health_check")
        )
        second = self.loop.run_once(
            build_autonomous_check_event(self.core.state, now_ts=6020, reason="focus_health_check")
        )

        self.assertTrue(any(action.payload.get("reason") == "rest_reminder" for action in first))
        self.assertFalse(any(action.payload.get("reason") == "rest_reminder" for action in second))

    def test_max_steps_prevents_infinite_internal_loop(self) -> None:
        loop = AgentLoop(self.core, max_steps=2)
        forced_event = Event(
            type="system_triggered",
            timestamp=10000,
            payload={"trigger": "agent_response_completed", "source": "test"},
        )

        with patch("src.agent.runtime.loop.build_internal_events_from_results", return_value=[forced_event]):
            actions = loop.run_once(
                Event(type="user_text_input", timestamp=10000, payload={"text": "hello", "source": "test"})
            )

        self.assertLessEqual(len(loop.recent_traces), 2)
        self.assertTrue(any(action.type in {"speak", "display"} for action in actions))


if __name__ == "__main__":
    unittest.main()
