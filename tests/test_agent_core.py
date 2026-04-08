from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.core import AgentCore
from src.agent.event import Event
from src.services.llm_service import LLMService
from src.services.memory_service import MemoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore


class AgentCoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "runtime_store.json"
        self.core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            memory_service=MemoryService(),
            llm_service=LLMService(),
            store=JsonStore(self.store_path),
        )

    def tearDown(self) -> None:
        self.core.shutdown()
        self.temp_dir.cleanup()

    def test_start_focus_updates_state(self) -> None:
        actions = self.core.handle_event(
            Event(
                type="focus_start_requested",
                timestamp=1000,
                payload={"duration_sec": 1500, "source": "test"},
            )
        )
        self.assertTrue(self.core.state.focus.active)
        self.assertEqual(self.core.state.interaction.mode, "focus")
        self.assertEqual(self.core.state.focus.target_duration_sec, 1500)
        self.assertEqual(self.core.state.focus.remaining_sec, 1500)
        self.assertTrue(any(action.type == "start_timer" for action in actions))

    def test_end_focus_records_session(self) -> None:
        self.core.handle_event(
            Event(
                type="focus_start_requested",
                timestamp=1000,
                payload={"duration_sec": 1500, "source": "test"},
            )
        )
        actions = self.core.handle_event(
            Event(type="focus_stop_requested", timestamp=1120, payload={"source": "test"})
        )
        self.assertFalse(self.core.state.focus.active)
        self.assertEqual(len(self.core.state.memory.focus_sessions), 1)
        session = self.core.state.memory.focus_sessions[-1]
        self.assertEqual(session["actual_duration_sec"], 120)
        self.assertEqual(session["reason"], "manual_stop")
        self.assertTrue(any(action.type == "stop_timer" for action in actions))

    def test_timer_expiry_stops_focus_and_speaks(self) -> None:
        self.core.handle_event(
            Event(
                type="focus_start_requested",
                timestamp=1000,
                payload={"duration_sec": 1500, "source": "test"},
            )
        )
        actions = self.core.handle_event(
            Event(type="timer_finished", timestamp=2500, payload={"timer": "focus"})
        )
        self.assertFalse(self.core.state.focus.active)
        self.assertEqual(self.core.state.memory.focus_sessions[-1]["reason"], "timer_complete")
        self.assertTrue(any(action.type == "stop_timer" for action in actions))
        self.assertTrue(
            any("专注时间到了" in action.payload.get("text", "") for action in actions),
            msg="timer 到期后应产生完成提醒",
        )

    def test_mock_state_update_changes_global_state(self) -> None:
        self.core.handle_event(
            Event(
                type="user_emotion_updated",
                timestamp=1000,
                payload={"emotion": "tired", "source": "mock"},
            )
        )
        self.assertEqual(self.core.state.user.emotion, "tired")

    def test_rest_reminder_has_cooldown(self) -> None:
        self.core.handle_event(
            Event(
                type="focus_start_requested",
                timestamp=0,
                payload={"duration_sec": 1500, "source": "test"},
            )
        )
        self.core.handle_event(
            Event(
                type="user_attention_updated",
                timestamp=1,
                payload={"attention": "focused", "source": "mock"},
            )
        )
        self.core.handle_event(
            Event(
                type="user_emotion_updated",
                timestamp=2,
                payload={"emotion": "tired", "source": "mock"},
            )
        )

        first_actions = self.core.handle_event(
            Event(type="timer_ticked", timestamp=601, payload={"remaining_sec": 899})
        )
        self.assertTrue(
            any(action.payload.get("kind") == "rest_reminder" for action in first_actions)
        )
        second_actions = self.core.handle_event(
            Event(type="timer_ticked", timestamp=620, payload={"remaining_sec": 880})
        )
        self.assertFalse(
            any(action.payload.get("kind") == "rest_reminder" for action in second_actions)
        )


if __name__ == "__main__":
    unittest.main()
