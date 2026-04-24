from __future__ import annotations

import unittest

from src.agent_lab import _format_trace, build_scenario_events
from src.agent.runtime.trace import AgentDecisionTrace


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


if __name__ == "__main__":
    unittest.main()
