from __future__ import annotations

import unittest

from src.adapters.voice.wake.local_wake_ack import (
    DEFAULT_WAKE_ACK_TEXT,
    LocalWakeAckPlayer,
    missing_ack_files,
)
from src.adapters.voice.runtime.voice_runtime import VoiceRuntime
from src.agent.event.event_model import Event
from src.agent.event.router import EventRouter


class LocalWakeAckTestCase(unittest.TestCase):
    def test_default_wake_ack_text_is_authoritative(self) -> None:
        self.assertEqual(DEFAULT_WAKE_ACK_TEXT, "我在，请说。")
        runtime = VoiceRuntime(sink=None)
        self.assertEqual(runtime._wake_ack_text, DEFAULT_WAKE_ACK_TEXT)

    def test_preload_default_clips(self) -> None:
        if missing_ack_files():
            self.skipTest("run scripts/generate_wake_ack_audio.py first")
        player = LocalWakeAckPlayer(ack_dir="assets/voice/wake_ack")
        count = player.preload()
        self.assertGreaterEqual(count, 4)


class VoiceWakeLlmSkipTestCase(unittest.TestCase):
    def test_voice_wake_skips_llm(self) -> None:
        router = EventRouter()
        event = Event(type="voice_wake_detected", timestamp=1, payload={"keyword": "小助"})
        decision = router.classify(event)
        self.assertEqual(decision.kind, "state_only")
        self.assertFalse(decision.uses_llm)


if __name__ == "__main__":
    unittest.main()
