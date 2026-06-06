from __future__ import annotations

import unittest

from src.agent.action import Action
from src.agent.event import Event
from src.agent.execution.action_result import ActionResult
from src.agent.memory.memory_gate import should_process_action_memory, should_process_event_memory


class MemoryGateTestCase(unittest.TestCase):
    def test_trivial_greeting_is_skipped(self) -> None:
        allowed, reason = should_process_event_memory(
            Event(type="user_text_input", timestamp=1, payload={"text": "你好"})
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "skipped_trivial_user_text")

    def test_explicit_long_term_preference_is_allowed(self) -> None:
        allowed, reason = should_process_event_memory(
            Event(
                type="user_text_input",
                timestamp=1,
                payload={"text": "以后我更喜欢 1 小时专注，不喜欢 25 分钟"},
            )
        )

        self.assertTrue(allowed)
        self.assertEqual(reason, "allowed_explicit_long_term_signal")

    def test_sensor_and_focus_events_are_skipped(self) -> None:
        sensor_allowed, sensor_reason = should_process_event_memory(
            Event(type="light_level_updated", timestamp=1, payload={"light_lux": 100})
        )
        focus_allowed, focus_reason = should_process_event_memory(
            Event(type="focus_start_requested", timestamp=2, payload={"duration_sec": 3600})
        )

        self.assertFalse(sensor_allowed)
        self.assertEqual(sensor_reason, "skipped_sensor_event")
        self.assertFalse(focus_allowed)
        self.assertEqual(focus_reason, "skipped_focus_or_timer_event")

    def test_internal_action_result_event_is_skipped(self) -> None:
        event = Event(
            type="system_triggered",
            timestamp=1,
            payload={"trigger": "agent_response_completed", "source": "agent_action_result"},
        )

        event_allowed, event_reason = should_process_event_memory(event)
        action_allowed, action_reason = should_process_action_memory(
            actions=[Action(type="start_timer", payload={"duration_sec": 60})],
            results=[ActionResult(action_type="start_timer", success=True, timestamp=1)],
            source_event=event,
        )

        self.assertFalse(event_allowed)
        self.assertEqual(event_reason, "skipped_internal_event")
        self.assertFalse(action_allowed)
        self.assertEqual(action_reason, "skipped_internal_event")

    def test_empty_and_speak_display_actions_are_skipped(self) -> None:
        source_event = Event(type="user_text_input", timestamp=1, payload={"text": "hello there"})

        empty_allowed, empty_reason = should_process_action_memory([], [], source_event)
        visible_allowed, visible_reason = should_process_action_memory(
            actions=[
                Action(type="speak", payload={"text": "ok"}),
                Action(type="display", payload={"text": "ok"}),
            ],
            results=[
                ActionResult(action_type="speak", success=True, timestamp=1),
                ActionResult(action_type="display", success=True, timestamp=1),
            ],
            source_event=source_event,
        )

        self.assertFalse(empty_allowed)
        self.assertEqual(empty_reason, "skipped_empty_actions")
        self.assertFalse(visible_allowed)
        self.assertEqual(visible_reason, "skipped_speak_display_only")

    def test_non_presentation_action_outcome_is_allowed(self) -> None:
        allowed, reason = should_process_action_memory(
            actions=[Action(type="start_timer", payload={"duration_sec": 3600})],
            results=[ActionResult(action_type="start_timer", success=True, timestamp=1)],
            source_event=Event(type="user_text_input", timestamp=1, payload={"text": "开始一小时专注"}),
        )

        self.assertTrue(allowed)
        self.assertEqual(reason, "allowed_action_outcome")


if __name__ == "__main__":
    unittest.main()
