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


class SpyLLMService(LLMService):
    def __init__(self) -> None:
        self.call_count = 0
        self.last_text = ""

    def generate_reply(self, text: str, state) -> str:  # type: ignore[override]
        self.call_count += 1
        self.last_text = text
        return "这是来自 LLM 的回复。"


class AgentCoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "runtime_store.json"
        self.spy_llm = SpyLLMService()
        self.core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            memory_service=MemoryService(),
            llm_service=self.spy_llm,
            store=JsonStore(self.store_path),
        )

    def tearDown(self) -> None:
        self.core.shutdown()
        self.temp_dir.cleanup()

    def test_handle_event_with_results_focus_start_updates_state_and_actions(self) -> None:
        actions, _ = self.core.handle_event_with_results(
            Event(
                type="focus_start_requested",
                timestamp=1000,
                payload={"duration_sec": 1500, "source": "test"},
            )
        )

        self.assertTrue(self.core.state.focus.active)
        self.assertEqual(self.core.state.interaction.mode, "focus")
        self.assertEqual(self.core.state.focus.target_duration_sec, 1500)
        self.assertIn("start_timer", {action.type for action in actions})

    def test_handle_event_with_results_returns_action_results(self) -> None:
        actions, results = self.core.handle_event_with_results(
            Event(
                type="focus_start_requested",
                timestamp=1010,
                payload={"duration_sec": 600, "source": "test"},
            )
        )

        self.assertEqual(len(actions), len(results))
        self.assertTrue(all(result.success for result in results))
        self.assertIn("start_timer", {result.action_type for result in results})

    def test_handle_event_focus_stop_records_session(self) -> None:
        self.core.handle_event_with_results(
            Event(
                type="focus_start_requested",
                timestamp=1000,
                payload={"duration_sec": 1500, "source": "test"},
            )
        )

        actions, _ = self.core.handle_event_with_results(
            Event(type="focus_stop_requested", timestamp=1120, payload={"source": "test"})
        )

        self.assertFalse(self.core.state.focus.active)
        self.assertEqual(len(self.core.state.memory.focus_sessions), 1)
        self.assertEqual(self.core.state.memory.focus_sessions[-1]["actual_duration_sec"], 120)
        self.assertIn("stop_timer", {action.type for action in actions})

    def test_timer_finished_generates_completion_feedback(self) -> None:
        self.core.handle_event_with_results(
            Event(
                type="focus_start_requested",
                timestamp=1000,
                payload={"duration_sec": 1500, "source": "test"},
            )
        )

        actions, _ = self.core.handle_event_with_results(
            Event(type="timer_finished", timestamp=2500, payload={"timer": "focus"})
        )

        self.assertFalse(self.core.state.focus.active)
        self.assertTrue(any(action.type == "stop_timer" for action in actions))
        self.assertTrue(any("专注时间到了" in str(action.payload.get("text", "")) for action in actions))

    def test_status_query_uses_rules_without_llm(self) -> None:
        self.core.handle_event_with_results(
            Event(
                type="user_emotion_updated",
                timestamp=2000,
                payload={"emotion": "happy", "confidence": 0.88, "source": "camera"},
            )
        )

        actions, _ = self.core.handle_event_with_results(
            Event(
                type="user_text_input",
                timestamp=2001,
                payload={"text": "现在状态如何", "source": "test"},
            )
        )

        self.assertEqual(self.spy_llm.call_count, 0)
        self.assertTrue(any("主导情绪" in str(action.payload.get("text", "")) for action in actions))

    def test_user_text_dialogue_calls_llm_and_returns_speak_display(self) -> None:
        actions, _ = self.core.handle_event_with_results(
            Event(type="user_text_input", timestamp=3000, payload={"text": "你好呀", "source": "test"})
        )

        self.assertEqual(self.spy_llm.call_count, 1)
        self.assertIn("speak", {action.type for action in actions})
        self.assertIn("display", {action.type for action in actions})

    def test_rest_reminder_has_cooldown(self) -> None:
        self.core.handle_event_with_results(
            Event(type="focus_start_requested", timestamp=0, payload={"duration_sec": 1500, "source": "test"})
        )
        self.core.handle_event_with_results(
            Event(
                type="user_attention_updated",
                timestamp=1,
                payload={"attention": "focused", "source": "mock"},
            )
        )
        self.core.handle_event_with_results(
            Event(
                type="user_emotion_updated",
                timestamp=2,
                payload={"emotion": "tired", "source": "mock"},
            )
        )

        first_actions, _ = self.core.handle_event_with_results(
            Event(type="timer_ticked", timestamp=601, payload={"remaining_sec": 899, "timer": "focus"})
        )
        second_actions, _ = self.core.handle_event_with_results(
            Event(type="timer_ticked", timestamp=620, payload={"remaining_sec": 880, "timer": "focus"})
        )

        self.assertTrue(any(action.payload.get("reason") == "rest_reminder" for action in first_actions))
        self.assertFalse(any(action.payload.get("reason") == "rest_reminder" for action in second_actions))

    def test_speech_and_text_share_same_status_logic(self) -> None:
        text_actions, _ = self.core.handle_event_with_results(
            Event(type="user_text_input", timestamp=6000, payload={"text": "当前状态", "source": "test"})
        )
        speech_actions, _ = self.core.handle_event_with_results(
            Event(type="speech_recognized", timestamp=6001, payload={"text": "当前状态", "source": "asr"})
        )

        text_reasons = {action.payload.get("reason") for action in text_actions}
        speech_reasons = {action.payload.get("reason") for action in speech_actions}
        self.assertIn("status_query", text_reasons)
        self.assertIn("status_query", speech_reasons)


if __name__ == "__main__":
    unittest.main()
