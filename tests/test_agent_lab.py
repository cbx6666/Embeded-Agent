from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.adapters.mock_input import parse_mock_command
from src.agent.core import AgentCore
from src.agent.decision.decision_result import DecisionResult
from src.agent.decision.intent_model import AgentIntent
from src.agent.execution.action_result import ActionResult
from src.agent_lab import _format_trace, _is_unknown_slash_command, _show_last_decision, build_scenario_events
from src.agent.execution.trace import AgentDecisionTrace
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.storage.json_store import JsonStore
from tests.fakes.fake_llm_service import FakeLLMService


class AgentLabTestCase(unittest.TestCase):
    def test_focus_fatigue_rest_scenario_contains_health_check(self) -> None:
        events = build_scenario_events("focus_fatigue_rest", start_ts=1000)

        self.assertEqual(events[0][1].type, "focus_start_requested")
        self.assertEqual(events[-1][1].type, "system_triggered")
        self.assertEqual(events[-1][1].payload.get("trigger"), "focus_health_check")

    def test_away_periodic_scenario_contains_periodic_check(self) -> None:
        events = build_scenario_events("away_periodic", start_ts=2000)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[1][1].payload.get("trigger"), "periodic_check")

    def test_unknown_scenario_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            build_scenario_events("unknown", start_ts=3000)

    def test_format_trace_marks_llm_selected_intent(self) -> None:
        trace = AgentDecisionTrace(
            event_type="user_text_input",
            timestamp=4000,
            state_summary={},
            intents=[
                {
                    "type": "answer_user",
                    "reason": "user_dialogue",
                    "payload": {"llm_selected": True},
                    "requires_llm": True,
                }
            ],
            actions=[{"type": "speak", "payload": {"text": "hi"}}],
            results=[{"action_type": "speak", "success": True}],
            loop_step=1,
        )

        formatted = _format_trace(trace)

        self.assertIn("answer_user[llm_selected][requires_llm]@user_dialogue", formatted)
        self.assertIn("actions=['speak']", formatted)

    def test_last_effective_decision_not_overwritten_by_internal_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            core = AgentCore(
                output=ConsoleOutput(silent=True),
                timer_service=TimerService(background=False),
                runtime_history_service=RuntimeHistoryService(),
                llm_service=FakeLLMService(),
                store=JsonStore(Path(temp_dir) / "runtime.json"),
            )
            try:
                core.last_effective_decision_result = DecisionResult(
                    intents=[AgentIntent(type="suggest_rest", reason="fatigue")],
                )
                core.last_effective_action_results = [
                    ActionResult(action_type="speak", success=True, timestamp=1),
                ]
                core.last_decision_result = DecisionResult(
                    intents=[AgentIntent(type="no_op", reason="internal")],
                )
                stream = io.StringIO()

                _show_last_decision(ConsoleOutput(stream=stream), core)

                rendered = stream.getvalue()
                self.assertIn("suggest_rest", rendered)
                self.assertNotIn("no_op", rendered)
            finally:
                core.shutdown()

    def test_unknown_slash_command_is_not_sent_to_llm(self) -> None:
        self.assertTrue(_is_unknown_slash_command("/history/reset"))
        self.assertFalse(_is_unknown_slash_command("我想开始专注"))
        self.assertFalse(_is_unknown_slash_command("/mock presence present"))

    def test_mock_behavior_does_not_overwrite_attention(self) -> None:
        event = parse_mock_command("/mock behavior working")

        self.assertIsNotNone(event)
        self.assertEqual(event.payload.get("behavior"), "working")  # type: ignore[union-attr]
        self.assertNotIn("attention", event.payload)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
