import time
import unittest

from src.adapters.posture_adapter import PostureAdapter
from src.adapters.console_output import ConsoleOutput
from src.agent.core import build_default_core


class TestPostureAdapter(unittest.TestCase):
    def test_publish_and_debounce_and_confidence(self):
        core = build_default_core(output=ConsoleOutput(silent=True))
        adapter = PostureAdapter(core, min_confidence=0.6, debounce_seconds=0.1, summary_threshold_seconds=1.0)

        # high confidence -> publish
        ok = adapter.publish_posture("slouch", confidence=0.9, frame_id=1)
        self.assertTrue(ok)
        self.assertTrue(core.state.memory.recent_events)
        last = core.state.memory.recent_events[-1]
        self.assertEqual(last["type"], "user_posture_updated")
        self.assertEqual(last["payload"].get("posture"), "slouch")

        # immediate same posture with high confidence -> suppressed by debounce
        ok2 = adapter.publish_posture("slouch", confidence=0.95, frame_id=2)
        self.assertFalse(ok2)

        # different posture -> should publish
        ok3 = adapter.publish_posture("upright", confidence=0.95, frame_id=3)
        self.assertTrue(ok3)
        last2 = core.state.memory.recent_events[-1]
        self.assertEqual(last2["payload"].get("posture"), "upright")

        # low confidence -> not published
        ok4 = adapter.publish_posture("slouch", confidence=0.4, frame_id=4)
        self.assertFalse(ok4)

    def test_summary_event_emitted_when_accumulated(self):
        core = build_default_core(output=ConsoleOutput(silent=True))
        adapter = PostureAdapter(core, min_confidence=0.5, debounce_seconds=0.0, summary_threshold_seconds=0.5)

        # simulate accumulation by setting internal accumulator near threshold
        adapter._bad_posture_accum["slouch"] = 0.6
        ok = adapter.publish_posture("slouch", confidence=0.8, frame_id=10)
        self.assertTrue(ok)
        # find summary event in recent_events
        types = [e["type"] for e in core.state.memory.recent_events]
        self.assertIn("user_posture_summary", types)


if __name__ == "__main__":
    unittest.main()
