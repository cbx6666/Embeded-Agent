from __future__ import annotations

import unittest

from src.adapters.screen.expression_styles import resolve_expression
from src.adapters.screen.pet_display_context import PetDisplayContext
from src.adapters.screen.pet_renderer import render_pet_png_bytes
from src.adapters.screen.screen_adapter import ScreenDisplayAdapter


class ExpressionMappingTest(unittest.TestCase):
    def test_dialogue_states(self) -> None:
        self.assertEqual(resolve_expression(agent_state="listening"), "happy")
        self.assertEqual(resolve_expression(agent_state="thinking"), "neutral")
        self.assertEqual(resolve_expression(agent_state="speaking"), "happy")
        self.assertEqual(resolve_expression(agent_state="focus_mode"), "sleepy")

    def test_idle_uses_emotion(self) -> None:
        self.assertEqual(resolve_expression(agent_state="idle", user_emotion="stressed"), "angry")
        self.assertEqual(resolve_expression(agent_state="idle", user_emotion="neutral"), "idle")


class ScreenAdapterSpeechTest(unittest.TestCase):
    def test_feed_voice_events(self) -> None:
        updates: list[PetDisplayContext] = []

        class _Hw:
            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def update(self, ctx: PetDisplayContext) -> None:
                updates.append(ctx)

        adapter = ScreenDisplayAdapter(hardware=_Hw())

        class _Ev:
            def __init__(self, etype: str, payload: dict) -> None:
                self.type = etype
                self.payload = payload

        adapter.feed_event(_Ev("voice_input_started", {}))
        adapter.sync_visual_state(dialogue_state="listening")
        self.assertEqual(updates[-1].speech_mode, "listening")

        adapter.feed_event(_Ev("speech_recognized", {"text": "你好"}))
        adapter.sync_visual_state(dialogue_state="thinking")
        self.assertEqual(updates[-1].user_speech_text, "你好")

        adapter.feed_event(_Ev("tts_started", {"text": "我在呢"}))
        adapter.sync_visual_state(dialogue_state="speaking")
        self.assertEqual(updates[-1].agent_speech_text, "我在呢")
        self.assertEqual(updates[-1].speech_mode, "agent")


class HeadlessRenderTest(unittest.TestCase):
    def test_render_dashboard_png(self) -> None:
        ctx = PetDisplayContext(
            agent_state="listening",
            expression="happy",
            temperature_c=26.5,
            humidity_pct=62.0,
            emotion_pie={"happy": 40, "neutral": 20},
            fatigue_pie={"none": 50, "mild": 10},
            speech_mode="listening",
        )
        png = render_pet_png_bytes(agent_state="listening", context=ctx, size=(640, 360))
        self.assertTrue(png.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
