from __future__ import annotations

import time
import unittest

from src.agent.core.models import DecisionResult, Intent
from src.agent.event.event_model import Event
from src.agent.state.state_persistence import should_persist_runtime_state


class StatePersistencePolicyTest(unittest.TestCase):
    def test_immediate_for_speech(self) -> None:
        ok = should_persist_runtime_state(
            Event(type="speech_recognized", timestamp=1, payload={"text": "hi"}),
            decision=DecisionResult(intents=[Intent("no_op", "")]),
            last_persist_mono=time.monotonic(),
        )
        self.assertTrue(ok)

    def test_throttle_high_frequency_perception(self) -> None:
        now = time.monotonic()
        event = Event(type="user_fatigue_updated", timestamp=1, payload={})
        decision = DecisionResult(intents=[Intent("no_op", "")])
        self.assertTrue(
            should_persist_runtime_state(event, decision=decision, last_persist_mono=now - 2.0)
        )
        self.assertFalse(
            should_persist_runtime_state(event, decision=decision, last_persist_mono=now - 0.1)
        )

    def test_system_triggered_no_op_is_throttled(self) -> None:
        now = time.monotonic()
        event = Event(
            type="system_triggered",
            timestamp=1,
            payload={"trigger": "wellness_care_check", "source": "agent_autonomy"},
        )
        decision = DecisionResult(intents=[Intent("no_op", "")])
        self.assertFalse(
            should_persist_runtime_state(event, decision=decision, last_persist_mono=now - 0.1)
        )
        self.assertTrue(
            should_persist_runtime_state(event, decision=decision, last_persist_mono=now - 2.0)
        )

    def test_immediate_when_actions_generated(self) -> None:
        ok = should_persist_runtime_state(
            Event(type="user_emotion_updated", timestamp=1, payload={}),
            decision=DecisionResult(
                intents=[Intent("answer_user", "")],
                actions=[{"type": "speak"}],  # type: ignore[list-item]
            ),
            last_persist_mono=time.monotonic(),
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
