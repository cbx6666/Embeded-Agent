from __future__ import annotations

"""本地音乐/相声陪伴功能测试。"""

import tempfile
import time
import unittest
import wave
from pathlib import Path

from src.agent.core.models import Event, Intent
from src.agent.decision.speech_llm_handler import SpeechLLMHandler
from src.agent.decision.wellness_care_handler import WellnessCareHandler
from src.agent.media.intent_parser import parse_media_control
from src.agent.media.media_controller import MediaController
from src.agent.media.media_library import build_library_catalog, scan_media_library
from src.agent.media.media_models import MediaRequest, MediaSource
from src.agent.media.media_policy import MediaCarePolicy, WELLNESS_CARES_BETWEEN_MEDIA_ASK
from src.agent.policy_config import MediaPolicy, WellnessCareCheckPolicy
from src.agent.state.agent_state import AgentState


def _write_silent_wav(path: Path, *, duration_sec: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 16000
    frames = int(rate * duration_sec)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)


def _music_tree(root: Path) -> None:
    _write_silent_wav(root / "music" / "light" / "a.wav")
    _write_silent_wav(root / "music" / "relaxing" / "b.wav")
    _write_silent_wav(root / "xiangsheng" / "short" / "c.wav")
    _write_silent_wav(root / "opera" / "jingju" / "d.wav")


def _present_state() -> AgentState:
    state = AgentState()
    state.user.presence = "present"
    return state


def _set_fatigue(state: AgentState) -> None:
    state.user.fatigue_level = "high"
    state.user.fatigue_confidence = 0.9


def _set_recent(state: AgentState, signal: str, samples: list[tuple[int, str, float]]) -> None:
    state.runtime_history.signal_trends[signal] = {
        "current": samples[-1][1] if samples else None,
        "recent_values": [
            {"timestamp": ts, "value": value, "confidence": conf} for ts, value, conf in samples
        ],
    }


def _set_fatigue_trend(state: AgentState) -> None:
    _set_recent(
        state,
        "fatigue",
        [
            (1970, "high", 0.9),
            (1980, "high", 0.9),
            (1990, "high", 0.9),
        ],
    )


class MediaPolicyTest(unittest.TestCase):
    def test_first_wellness_can_ask_media_with_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _music_tree(root)
            library = scan_media_library(root)
            policy = MediaCarePolicy()
            choice = policy.try_media_suggestion(care_focus="fatigue", library=library)
            self.assertIsNotNone(choice)
            assert choice is not None
            self.assertEqual(choice.strategy, "media_suggestion")

    def test_need_two_wellness_cares_before_reask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _music_tree(root)
            library = scan_media_library(root)
            policy = MediaCarePolicy()
            self.assertIsNone(
                policy.try_media_suggestion(
                    care_focus="fatigue",
                    media_suggestion_ever_asked=True,
                    wellness_cares_since_media_ask=0,
                    library=library,
                )
            )
            self.assertIsNone(
                policy.try_media_suggestion(
                    care_focus="fatigue",
                    media_suggestion_ever_asked=True,
                    wellness_cares_since_media_ask=1,
                    library=library,
                )
            )
            self.assertIsNotNone(
                policy.try_media_suggestion(
                    care_focus="fatigue",
                    media_suggestion_ever_asked=True,
                    wellness_cares_since_media_ask=WELLNESS_CARES_BETWEEN_MEDIA_ASK,
                    library=library,
                )
            )

    def test_unknown_focus_no_default_light(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _music_tree(root)
            library = scan_media_library(root)
            policy = MediaCarePolicy()
            self.assertIsNone(
                policy.try_media_suggestion(care_focus="unknown_focus", library=library)
            )

    def test_empty_library_skips_suggestion(self) -> None:
        from src.agent.media.media_models import MediaLibraryIndex

        policy = MediaCarePolicy()
        empty = MediaLibraryIndex(root=".", tracks=[])
        self.assertIsNone(policy.try_media_suggestion(care_focus="fatigue", library=empty))


class SpeechMediaLLMTest(unittest.TestCase):
    def test_play_music_requires_llm_media_control(self) -> None:
        """规则层不再抢先；LLM 未返回 media_control 时不应播放。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _music_tree(root)
            mc = MediaController(music_root=root, mock_play_sec=0.1)
            from src.agent.action.realizer import ActionRealizer

            handler = SpeechLLMHandler(realizer=ActionRealizer(), media_controller=mc)
            state = _present_state()
            decision = handler.decide(
                state=state,
                event=Event(type="speech_recognized", timestamp=100, payload={"text": "放点音乐"}),
                llm_client=_FakeLLM({"intent": "answer_user", "reply": "好的。"}),
                user_context={},
            )
            self.assertFalse(any(a.type == "play_media" for a in decision.actions))
            self.assertTrue(decision.used_llm)


class MediaPlaybackIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _music_tree(self.root)
        self.mc = MediaController(music_root=self.root, mock_play_sec=0.1)
        from src.agent.action.realizer import ActionRealizer

        self.handler = SpeechLLMHandler(realizer=ActionRealizer(), media_controller=self.mc)

    def tearDown(self) -> None:
        self.mc.stop_by_user()
        self._tmp.cleanup()

    def test_user_play_music_flow(self) -> None:
        state = _present_state()
        music_track = next(t for t in self.mc.selector.index.tracks if t.media_type == "music")
        decision = self.handler.decide(
            state=state,
            event=Event(type="speech_recognized", timestamp=100, payload={"text": "放点音乐"}),
            llm_client=_FakeLLM(
                {
                    "intent": "media_control",
                    "action": "play_media",
                    "media_type": "music",
                    "track_id": music_track.id,
                    "reply": "好，给你放点轻松的音乐。",
                }
            ),
            user_context={"memories": {"hobby": "用户喜欢听轻音乐"}},
        )
        speak_actions = [a for a in decision.actions if a.type == "speak"]
        play_actions = [a for a in decision.actions if a.type == "play_media"]
        self.assertTrue(speak_actions)
        self.assertTrue(play_actions)
        self.assertTrue(play_actions[0].payload.get("defer_after_speak"))
        self.assertTrue(decision.used_llm)

    def test_wake_word_stops_playback(self) -> None:
        state = _present_state()
        ctx = self.mc.build_selection_context(agent_state=state)
        track = self.mc.select_track(
            MediaRequest(action="play_media", media_type="music", source=MediaSource.USER_EXPLICIT),
            ctx,
        )
        assert track is not None
        self.mc.play_track(track)
        time.sleep(0.02)
        self.assertTrue(self.mc.is_playing())
        self.mc.stop_for_wake_word()
        time.sleep(0.02)
        self.assertFalse(self.mc.is_playing())

    def test_stop_media_intent(self) -> None:
        state = _present_state()
        ctx = self.mc.build_selection_context(agent_state=state)
        track = self.mc.select_track(
            MediaRequest(action="play_media", media_type="music", source=MediaSource.USER_EXPLICIT),
            ctx,
        )
        assert track is not None
        self.mc.play_track(track)
        time.sleep(0.05)
        decision = self.handler.decide(
            state=state,
            event=Event(type="speech_recognized", timestamp=200, payload={"text": "别放了"}),
            llm_client=_FakeLLM(
                {
                    "intent": "media_control",
                    "action": "stop_media",
                    "reply": "好的，先不放了。",
                }
            ),
            user_context={},
        )
        stop_actions = [a for a in decision.actions if a.type == "stop_media"]
        self.assertTrue(stop_actions)
        self.mc.stop_by_user()


class MediaCounterSyncTest(unittest.TestCase):
    def test_tts_finished_increments_once_per_wellness_speak(self) -> None:
        from src.agent.core import build_default_core
        from src.agent.core.models import Event

        core = build_default_core(timer_background=False)
        core.state.cooldown.media_suggestion_ever_asked = True
        core.state.cooldown.wellness_cares_since_media_ask = 0

        core.handle_event(
            Event(
                type="tts_finished",
                timestamp=100,
                payload={
                    "text": "坐直一点吧",
                    "kind": "notification",
                    "reason": "posture_reminder",
                },
            )
        )
        self.assertEqual(core.state.cooldown.wellness_cares_since_media_ask, 1)

        core.handle_event(
            Event(
                type="tts_finished",
                timestamp=101,
                payload={
                    "text": "坐直一点吧",
                    "kind": "notification",
                    "reason": "posture_reminder",
                    "cancelled": True,
                },
            )
        )
        self.assertEqual(core.state.cooldown.wellness_cares_since_media_ask, 1)

    def test_tts_finished_media_suggestion_resets_counter(self) -> None:
        from src.agent.core import build_default_core
        from src.agent.core.models import Event

        core = build_default_core(timer_background=False)
        core.state.cooldown.media_suggestion_ever_asked = True
        core.state.cooldown.wellness_cares_since_media_ask = 2

        core.handle_event(
            Event(
                type="tts_finished",
                timestamp=200,
                payload={
                    "text": "要不要听点音乐？",
                    "kind": "notification",
                    "reason": "media_suggestion",
                },
            )
        )
        self.assertTrue(core.state.cooldown.media_suggestion_ever_asked)
        self.assertEqual(core.state.cooldown.wellness_cares_since_media_ask, 0)


class WellnessMediaCareTest(unittest.TestCase):
    def setUp(self) -> None:
        from src.adapters.voice.arbitration.session_probe import VoiceSessionProbe

        VoiceSessionProbe.global_probe().reset_for_tests()

    def test_after_media_ask_uses_wellness_until_two_cares(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _music_tree(root)
            mc = MediaController(music_root=root, mock_play_sec=0.1)
            from src.agent.action.realizer import ActionRealizer

            handler = WellnessCareHandler(
                realizer=ActionRealizer(),
                media_controller=mc,
            )
            state = _present_state()
            _set_fatigue(state)
            _set_fatigue_trend(state)
            state.cooldown.media_suggestion_ever_asked = True
            state.cooldown.wellness_cares_since_media_ask = 0
            result = handler.decide(
                state=state,
                event=Event(type="system_triggered", timestamp=2000, payload={}),
                llm_client=_FakeLLM({"reply": "感觉你有点累了，歇会儿吧"}),
                user_context={},
            )
            self.assertEqual(result.intents[0].type, "suggest_rest")
            self.assertEqual(result.log_fields.get("care_strategy"), "wellness_fatigue")
            self.assertTrue(result.log_fields.get("media_unavailable"))

    def test_wellness_suggests_media_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _music_tree(root)
            mc = MediaController(music_root=root, mock_play_sec=0.1)
            from src.agent.action.realizer import ActionRealizer

            handler = WellnessCareHandler(
                realizer=ActionRealizer(),
                media_controller=mc,
                check_policy=WellnessCareCheckPolicy(),
            )
            state = _present_state()
            _set_fatigue(state)
            _set_fatigue_trend(state)
            llm = _FakeLLM({"reply": "感觉你有点累了，要不要听点轻音乐放松一下？"})
            decision = handler.decide(
                state=state,
                event=Event(type="system_triggered", timestamp=1020, payload={"trigger": "wellness_care_check"}),
                llm_client=llm,
                user_context={"memories": {"hobby": "用户喜欢听轻音乐"}},
            )
            intent_types = [i.type for i in decision.intents]
            self.assertIn("suggest_media", intent_types)
            self.assertTrue(decision.used_llm)
            self.assertIn("轻音乐", decision.reply_text)


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete_json(self, role: str, prompt: str, *, temperature=None) -> dict:
        return dict(self.payload)


if __name__ == "__main__":
    unittest.main()
