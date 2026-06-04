from __future__ import annotations

import unittest

from src.agent.event import Event
from tests.replay.replay_harness import replay_event_log


class DeterministicReplayTestCase(unittest.TestCase):
    def test_same_event_log_replays_same_actions_and_trace(self) -> None:
        events = (
            Event(type="user_text_input", timestamp=100, payload={"text": "Please remind me less and keep it gentle."}),
            Event(type="focus_start_requested", timestamp=120, payload={"duration_sec": 600, "source": "user"}),
            Event(type="user_fatigue_updated", timestamp=200, payload={"fatigue_level": "high"}),
            Event(type="system_triggered", timestamp=220, payload={"trigger": "focus_health_check"}),
            Event(type="system_triggered", timestamp=240, payload={"trigger": "focus_health_check"}),
        )

        first = replay_event_log(events)
        second = replay_event_log(events)

        self.assertEqual(first.actions_by_event, second.actions_by_event)
        self.assertEqual(first.trace_json_by_event, second.trace_json_by_event)
        self.assertTrue(any("cooldown active" in trace for trace in first.trace_json_by_event))

    def test_replay_changes_when_input_changes(self) -> None:
        baseline = replay_event_log(
            (
                Event(type="focus_start_requested", timestamp=10, payload={"duration_sec": 300, "source": "user"}),
            )
        )
        changed = replay_event_log(
            (
                Event(type="focus_start_requested", timestamp=10, payload={"duration_sec": 900, "source": "user"}),
            )
        )

        self.assertNotEqual(baseline.actions_by_event, changed.actions_by_event)


if __name__ == "__main__":
    unittest.main()
