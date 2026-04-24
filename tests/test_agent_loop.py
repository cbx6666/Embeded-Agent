from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.adapters.console_output import ConsoleOutput
from src.agent.action import Action
from src.agent.action_result import ActionResult
from src.agent.autonomy import build_autonomous_check_event
from src.agent.core import AgentCore
from src.agent.event import Event
from src.agent.internal_events import build_internal_events_from_results
from src.agent.loop import AgentLoop
from src.services.llm_service import LLMService
from src.services.memory_service import MemoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore


class SpyLLMService(LLMService):
    def __init__(self) -> None:
        self.call_count = 0

    def generate_reply(self, text: str, state) -> str:  # type: ignore[override]
        self.call_count += 1
        return "这是来自 LLM 的回复。"


class AgentLoopTestCase(unittest.TestCase):
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
        self.loop = AgentLoop(self.core, max_steps=3)

    def tearDown(self) -> None:
        self.core.shutdown()
        self.temp_dir.cleanup()

    def test_run_once_handles_user_text_and_returns_speak_display(self) -> None:
        actions = self.loop.run_once(
            Event(type="user_text_input", timestamp=1000, payload={"text": "你好呀", "source": "test"})
        )

        self.assertIn("speak", {action.type for action in actions})
        self.assertIn("display", {action.type for action in actions})
        self.assertEqual(self.spy_llm.call_count, 1)
        self.assertGreaterEqual(len(self.loop.recent_traces), 1)

    def test_focus_start_runs_closed_loop_without_dead_loop(self) -> None:
        actions = self.loop.run_once(
            Event(
                type="focus_start_requested",
                timestamp=2000,
                payload={"duration_sec": 1500, "source": "test"},
            )
        )

        self.assertIn("start_timer", {action.type for action in actions})
        self.assertTrue(self.core.state.focus.active)
        self.assertEqual(len(self.loop.recent_traces), 3)

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

    def test_action_failed_generates_rule_based_error_feedback(self) -> None:
        actions = self.loop.run_once(
            Event(
                type="system_triggered",
                timestamp=4000,
                payload={
                    "trigger": "action_failed",
                    "source": "test",
                    "action_type": "start_timer",
                    "reason": "mock failure",
                },
            )
        )

        feedback_texts = [
            str(action.payload.get("text", ""))
            for action in actions
            if action.type in {"speak", "display"}
        ]
        self.assertTrue(any("动作执行失败" in text for text in feedback_texts))

    def test_periodic_check_does_not_speak_when_user_is_away(self) -> None:
        self.core.state.user.presence = "away"

        actions = self.loop.run_once(
            build_autonomous_check_event(self.core.state, now_ts=5000, reason="periodic_check")
        )

        self.assertFalse(any(action.type == "speak" for action in actions))

    def test_focus_health_check_triggers_rest_reminder_when_fatigued(self) -> None:
        self._activate_focus()
        self.core.state.user.presence = "present"
        self.core.state.user.fatigue_level = "high"

        actions = self.loop.run_once(
            build_autonomous_check_event(self.core.state, now_ts=6000, reason="focus_health_check")
        )

        self.assertTrue(any(action.payload.get("reason") == "rest_reminder" for action in actions))

    def test_focus_health_check_respects_cooldown(self) -> None:
        self._activate_focus()
        self.core.state.user.presence = "present"
        self.core.state.user.fatigue_level = "high"

        first = self.loop.run_once(
            build_autonomous_check_event(self.core.state, now_ts=7000, reason="focus_health_check")
        )
        second = self.loop.run_once(
            build_autonomous_check_event(self.core.state, now_ts=7020, reason="focus_health_check")
        )

        self.assertTrue(any(action.payload.get("reason") == "rest_reminder" for action in first))
        self.assertFalse(any(action.payload.get("reason") == "rest_reminder" for action in second))

    def test_silent_mode_does_not_generate_speak(self) -> None:
        self.core.state.interaction.mode = "silent"

        actions = self.loop.run_once(
            Event(type="user_text_input", timestamp=8000, payload={"text": "你好", "source": "test"})
        )

        self.assertFalse(any(action.type == "speak" for action in actions))
        self.assertTrue(any(action.type == "display" for action in actions))

    def test_speaking_state_does_not_repeat_speak(self) -> None:
        self._activate_focus()
        self.core.state.user.presence = "present"
        self.core.state.user.fatigue_level = "high"
        self.core.state.interaction.dialogue_state = "speaking"

        actions = self.loop.run_once(
            build_autonomous_check_event(self.core.state, now_ts=9000, reason="focus_health_check")
        )

        self.assertFalse(any(action.type == "speak" for action in actions))
        self.assertTrue(any(action.type == "display" for action in actions))

    def test_max_steps_prevents_infinite_internal_loop(self) -> None:
        loop = AgentLoop(self.core, max_steps=2)
        forced_event = Event(
            type="system_triggered",
            timestamp=10000,
            payload={"trigger": "agent_response_completed", "source": "test"},
        )

        with patch("src.agent.loop.build_internal_events_from_results", return_value=[forced_event]):
            actions = loop.run_once(
                Event(type="user_text_input", timestamp=10000, payload={"text": "你好", "source": "test"})
            )

        self.assertLessEqual(len(loop.recent_traces), 2)
        self.assertTrue(any(action.type in {"speak", "display"} for action in actions))

    def _activate_focus(self) -> None:
        self.core.state.focus.active = True
        self.core.state.focus.start_ts = 0
        self.core.state.focus.target_duration_sec = 1500
        self.core.state.focus.elapsed_sec = 600
        self.core.state.focus.remaining_sec = 900
        self.core.state.interaction.mode = "focus"


if __name__ == "__main__":
    unittest.main()
