from __future__ import annotations

"""语音链路仲裁、TTS 队列、唤醒抢占、缓冲合并测试。"""

import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.adapters.voice.arbitration.reminder_buffer import ReminderBuffer
from src.adapters.voice.arbitration.session_probe import VoiceSessionProbe, should_defer_autonomous_speak
from src.adapters.voice.arbitration.tts_job_policy import TTSJobKind, resolve_job_spec
from src.adapters.voice.arbitration.voice_arbiter import ArbiterAction, VoiceInteractionArbiter
from src.adapters.voice.runtime.state_machine import VoiceSessionStateMachine, VoiceState
from src.adapters.voice.tts.playback_manager import TTSPlaybackManager


class _FakeTTSBackend:
    def speak(self, text: str, **kwargs) -> None:
        time.sleep(0.05)


class VoiceArbiterTest(unittest.TestCase):
    def setUp(self) -> None:
        VoiceSessionProbe._instance = None  # noqa: SLF001
        self.probe = VoiceSessionProbe.global_probe()
        self.probe.reset_for_tests()
        self.state = VoiceSessionStateMachine()
        self.probe.bind_state_view(self.state)
        self.probe.bind_media_playing(lambda: False)
        self.arbiter = VoiceInteractionArbiter(probe=self.probe)

    def test_defer_in_ack_playing(self) -> None:
        self.state.transition(VoiceState.ACK_PLAYING, "test")
        spec = resolve_job_spec(source="rest_reminder", reason="rest_reminder")
        decision = self.arbiter.decide_enqueue(
            spec,
            text="休息一下",
            source="rest_reminder",
            reason="rest_reminder",
            priority=spec.priority,
            payload={},
        )
        self.assertEqual(decision.action, ArbiterAction.BUFFER)

    def test_wake_ack_always_play(self) -> None:
        self.state.transition(VoiceState.LISTENING, "test")
        spec = resolve_job_spec(source="wake_ack")
        decision = self.arbiter.decide_pre_play(spec, created_at=time.time(), now=time.time())
        self.assertEqual(decision.action, ArbiterAction.PLAY)

    def test_autonomous_blocked_during_listening_pre_play(self) -> None:
        self.state.transition(VoiceState.LISTENING, "test")
        spec = resolve_job_spec(source="distraction_reminder", reason="distraction_reminder")
        decision = self.arbiter.decide_pre_play(spec, created_at=time.time(), now=time.time())
        self.assertEqual(decision.action, ArbiterAction.BUFFER)

    def test_media_playing_buffers_autonomous(self) -> None:
        self.probe.bind_media_playing(lambda: True)
        spec = resolve_job_spec(source="environment_warning", reason="environment_warning")
        decision = self.arbiter.decide_enqueue(
            spec,
            text="光线偏暗",
            source="environment_warning",
            reason="environment_warning",
            priority=spec.priority,
            payload={},
        )
        self.assertEqual(decision.action, ArbiterAction.BUFFER)
        self.assertEqual(len(self.arbiter.reminder_buffer), 1)


class ReminderBufferTest(unittest.TestCase):
    def test_coalesce_same_key(self) -> None:
        buf = ReminderBuffer()
        spec = resolve_job_spec(source="rest_reminder", reason="rest_reminder")
        buf.offer(text="a", source="rest_reminder", reason="rest_reminder", priority=3, payload={}, spec=spec)
        buf.offer(text="b", source="rest_reminder", reason="rest_reminder", priority=3, payload={}, spec=spec)
        self.assertEqual(len(buf), 1)
        item = buf.pop_best()
        assert item is not None
        self.assertEqual(item.text, "b")

    def test_pop_best_prefers_higher_priority(self) -> None:
        buf = ReminderBuffer()
        spec_env = resolve_job_spec(source="environment_warning", reason="environment_warning")
        spec_dist = resolve_job_spec(source="distraction_reminder", reason="distraction_reminder")
        buf.offer(
            text="env",
            source="environment_warning",
            reason="environment_warning",
            priority=spec_env.priority,
            payload={},
            spec=spec_env,
        )
        buf.offer(
            text="dist",
            source="distraction_reminder",
            reason="distraction_reminder",
            priority=spec_dist.priority,
            payload={},
            spec=spec_dist,
        )
        first = buf.pop_best()
        assert first is not None
        self.assertEqual(first.text, "dist")


class TTSPlaybackManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        VoiceSessionProbe._instance = None  # noqa: SLF001
        self.probe = VoiceSessionProbe.global_probe()
        self.probe.reset_for_tests()
        self.state = VoiceSessionStateMachine()
        self.probe.bind_state_view(self.state)
        self.probe.bind_media_playing(lambda: False)
        self.arbiter = VoiceInteractionArbiter(probe=self.probe)
        self.finished: list[str] = []
        self._lock = threading.Lock()
        self.tts = TTSPlaybackManager(
            tts_backend=_FakeTTSBackend(),
            arbiter=self.arbiter,
            on_finished=lambda _jid, text, *_rest: self._record_finished(text),
        )
        self.tts.start()

    def tearDown(self) -> None:
        self.tts.stop()

    def _record_finished(self, text: str) -> None:
        with self._lock:
            self.finished.append(text)

    def test_wake_ack_finishes_on_cancel(self) -> None:
        done = threading.Event()

        def _on_done() -> None:
            done.set()

        self.tts.enqueue("我在", priority=0, source="wake_ack", on_finished=_on_done)
        time.sleep(0.02)
        self.tts.cancel_current_interruptible("test_cancel")
        time.sleep(0.15)
        self.assertTrue(done.wait(timeout=2.0), "cancel 后 on_finished 应触发")

    def test_prepare_for_wake_purges_autonomous(self) -> None:
        self.tts.enqueue("分心提醒", source="distraction_reminder", reason="distraction_reminder")
        self.tts.prepare_for_wake()
        with self.tts._heap_lock:  # noqa: SLF001
            sources = [j.source for j in self.tts._heap]
        self.assertNotIn("distraction_reminder", sources)

    def test_wake_ack_plays_before_autonomous(self) -> None:
        self.tts.enqueue("环境提醒", source="environment_warning", reason="environment_warning")
        self.tts.enqueue("我在", priority=0, source="wake_ack")
        time.sleep(0.3)
        with self._lock:
            self.assertIn("我在", self.finished)


class SessionProbeHandlerTest(unittest.TestCase):
    def test_should_defer_with_probe_listening(self) -> None:
        VoiceSessionProbe._instance = None  # noqa: SLF001
        probe = VoiceSessionProbe.global_probe()
        sm = VoiceSessionStateMachine()
        probe.bind_state_view(sm)
        sm.transition(VoiceState.LISTENING, "t")
        self.assertTrue(should_defer_autonomous_speak(dialogue_state="idle"))


if __name__ == "__main__":
    unittest.main()
