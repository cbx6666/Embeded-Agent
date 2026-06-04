from __future__ import annotations

import unittest

from src.adapters.voice.local_wake_ack import LocalWakeAckPlayer, missing_ack_files
from src.agent.config.policy_config import DecisionPolicyConfig
from src.agent.decision.decision_pipeline import DecisionPipeline
from src.agent.event.event_model import Event


class LocalWakeAckTestCase(unittest.TestCase):
    def test_preload_default_clips(self) -> None:
        if missing_ack_files():
            self.skipTest("run scripts/generate_wake_ack_audio.py first")
        player = LocalWakeAckPlayer(ack_dir="assets/voice/wake_ack")
        count = player.preload()
        self.assertGreaterEqual(count, 4)


class VoiceWakeLlmSkipTestCase(unittest.TestCase):
    def test_voice_wake_skips_llm(self) -> None:
        pipeline = DecisionPipeline(decision_policy=DecisionPolicyConfig())
        event = Event(type="voice_wake_detected", timestamp=1, payload={"keyword": "小助"})
        reason = pipeline._ignored_llm_reason(event)
        self.assertIsNotNone(reason)
        self.assertIn("voice_wake_detected", reason or "")


if __name__ == "__main__":
    unittest.main()
