from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.adapters.console_output import ConsoleOutput
from src.agent.action import Action
from src.agent.core import AgentCore
from src.agent.event import Event
from src.agent.execution.action_result import ActionResult
from src.agent.execution.internal_events import build_internal_events_from_results
from src.agent.execution.loop import AgentLoop
from src.agent.scheduling import build_autonomous_check_event
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore
from tests.fakes.fake_llm_service import FakeLLMService


class AgentLoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.llm = FakeLLMService(reply_text="fallback reply")
        self.core = AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
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

    def test_focus_start_generates_single_start_timer(self) -> None:
        actions = self.loop.run_once(
            Event(type="focus_start_requested", timestamp=2100, payload={"duration_sec": 1500, "source": "test"})
        )

        self.assertEqual(1, sum(1 for action in actions if action.type == "start_timer"))
        self.assertEqual(1, sum(1 for action in actions if action.type == "display" and action.payload.get("reason") == "focus_start"))

    def test_continue_focus_restores_focus_from_timer_action_result(self) -> None:
        self.llm.responses.update(
            {
                "intent_planner": [
                    '{"intents":[{"type":"continue_focus","priority":90,"reason":"continue","payload":{"duration_minutes":20},"requires_llm":false}],"reasoning":"continue","risk_level":"low","interrupt_user":false}'
                ]
            }
        )

        actions = self.loop.run_once(
            Event(type="user_text_input", timestamp=2150, payload={"text": "continue focus", "source": "test"})
        )

        self.assertIn("start_timer", {action.type for action in actions})
        self.assertTrue(self.core.state.focus.active)
        self.assertEqual(self.core.state.focus.target_duration_sec, 1200)

    def test_internal_system_trigger_does_not_replan_start_focus(self) -> None:
        self.core.handle_event(
            Event(type="focus_start_requested", timestamp=2200, payload={"duration_sec": 1500, "source": "test"})
        )
        self.llm.calls.clear()

        actions, _ = self.core.handle_event(
            Event(
                type="system_triggered",
                timestamp=2200,
                payload={"trigger": "focus_timer_started", "source": "agent_action_result"},
            )
        )

        self.assertEqual(actions, [])
        self.assertEqual([intent.type for intent in self.core.last_intents], ["no_op"])
        self.assertNotIn("intent_planner", self.llm.calls)

    def test_agent_response_completed_does_not_trigger_visible_intents(self) -> None:
        actions, _ = self.core.handle_event(
            Event(
                type="system_triggered",
                timestamp=2300,
                payload={"trigger": "agent_response_completed", "source": "agent_action_result"},
            )
        )

        self.assertEqual(actions, [])
        self.assertEqual([intent.type for intent in self.core.last_intents], ["no_op"])
        self.assertNotIn("answer_user", {intent.type for intent in self.core.last_intents})
        self.assertNotIn("suggest_rest", {intent.type for intent in self.core.last_intents})
        self.assertNotIn("start_focus", {intent.type for intent in self.core.last_intents})

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
        self._prepare_sustained_focus_fatigue()

        first = self.loop.run_once(
            build_autonomous_check_event(self.core.state, now_ts=6000, reason="focus_health_check")
        )
        second = self.loop.run_once(
            build_autonomous_check_event(self.core.state, now_ts=6020, reason="focus_health_check")
        )

        self.assertTrue(any(action.payload.get("reason") == "rest_reminder" for action in first))
        self.assertFalse(any(action.payload.get("reason") == "rest_reminder" for action in second))

    def test_allowed_focus_health_check_still_triggers_rest(self) -> None:
        self._prepare_sustained_focus_fatigue()

        actions = self.loop.run_once(
            Event(type="system_triggered", timestamp=6100, payload={"trigger": "focus_health_check", "source": "agent_autonomy"})
        )

        self.assertTrue(any(action.payload.get("reason") == "rest_reminder" for action in actions))

    def test_allowed_environment_check_still_triggers_environment_feedback(self) -> None:
        self.core.state.focus.active = True
        self.core.state.user.presence = "present"
        for timestamp in (6100, 6101, 6102):
            self.core.handle_event(
                Event(
                    type="noise_level_updated",
                    timestamp=timestamp,
                    payload={"level": "high", "noise_db": 80, "confidence": 0.9},
                )
            )

        actions = self.loop.run_once(
            Event(type="system_triggered", timestamp=6200, payload={"trigger": "environment_check", "source": "agent_autonomy"})
        )

        self.assertTrue(any(action.type == "set_light_state" for action in actions))
        self.assertTrue(any(action.payload.get("reason") == "environment_warning" for action in actions))

    def test_last_effective_decision_not_overwritten_by_internal_noop(self) -> None:
        self._prepare_sustained_focus_fatigue()

        self.loop.run_once(
            Event(type="system_triggered", timestamp=6300, payload={"trigger": "focus_health_check", "source": "agent_autonomy"})
        )

        self.assertIsNotNone(self.core.last_effective_decision_result)
        self.assertIn("suggest_rest", {intent.type for intent in self.core.last_effective_decision_result.intents})  # type: ignore[union-attr]
        self.assertEqual([intent.type for intent in self.core.last_decision_result.intents], ["no_op"])  # type: ignore[union-attr]

    def _prepare_sustained_focus_fatigue(self) -> None:
        self.core.state.focus.active = True
        self.core.state.focus.elapsed_sec = 600
        self.core.state.focus.remaining_sec = 900
        self.core.state.user.presence = "present"
        for timestamp in (5900, 5901, 5902):
            self.core.handle_event(
                Event(
                    type="user_fatigue_updated",
                    timestamp=timestamp,
                    payload={"fatigue_level": "high", "confidence": 0.9},
                )
            )

    def test_max_steps_prevents_infinite_internal_loop(self) -> None:
        loop = AgentLoop(self.core, max_steps=2)
        forced_event = Event(
            type="system_triggered",
            timestamp=10000,
            payload={"trigger": "agent_response_completed", "source": "test"},
        )

        with patch("src.agent.execution.loop.build_internal_events_from_results", return_value=[forced_event]):
            actions = loop.run_once(
                Event(type="user_text_input", timestamp=10000, payload={"text": "hello", "source": "test"})
            )

        self.assertLessEqual(len(loop.recent_traces), 2)
        self.assertTrue(any(action.type in {"speak", "display"} for action in actions))


if __name__ == "__main__":
    unittest.main()
