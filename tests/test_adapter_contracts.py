from __future__ import annotations

import unittest

from src.adapters.pet_display import PetDisplayAdapter
from src.adapters.voice_adapter import VoiceAdapter
from src.agent.action import Action, play_voice, render_pet_expression
from src.agent.event import display_sensor_updated, voice_input_captured


class _Sink:
    def __init__(self) -> None:
        self.events = []

    def handle_event(self, event) -> None:
        self.events.append(event)


class _DisplayHardware:
    def __init__(self) -> None:
        self.renders = []
        self.snapshot = {
            "brightness": 72,
            "fps": 30,
            "sensor_values": {"touch": True},
        }

    def render_expression(self, expression: str, payload: dict) -> None:
        self.renders.append((expression, payload))

    def read_sensor_snapshot(self) -> dict:
        return self.snapshot


class _VoiceBackend:
    def __init__(self) -> None:
        self.calls = []

    def speak(self, text: str, payload: dict) -> None:
        self.calls.append((text, payload))


class AdapterContractTestCase(unittest.TestCase):
    def test_display_sensor_factory_builds_event(self) -> None:
        event = display_sensor_updated(
            timestamp=100,
            expression="happy",
            source="pet_display",
            brightness=80,
            fps=24,
            sensor_values={"touch": False},
            screen_id="screen-a",
        )
        self.assertEqual(event.type, "display_sensor_updated")
        self.assertEqual(event.payload["expression"], "happy")
        self.assertEqual(event.payload["brightness"], 80)
        self.assertEqual(event.payload["screen_id"], "screen-a")

    def test_voice_input_factory_builds_event(self) -> None:
        event = voice_input_captured(
            timestamp=200,
            text="你好",
            source="microphone",
            confidence=1.2,
            language="zh-CN",
            is_final=False,
            audio_id="utt-1",
        )
        self.assertEqual(event.type, "voice_input_captured")
        self.assertEqual(event.payload["text"], "你好")
        self.assertEqual(event.payload["confidence"], 1.0)
        self.assertFalse(event.payload["is_final"])

    def test_pet_display_adapter_executes_render_expression(self) -> None:
        hardware = _DisplayHardware()
        adapter = PetDisplayAdapter(hardware=hardware)
        adapter.execute(render_pet_expression("sleepy", style="blink", intensity=0.8))
        self.assertEqual(len(hardware.renders), 1)
        expression, payload = hardware.renders[0]
        self.assertEqual(expression, "sleepy")
        self.assertEqual(payload["style"], "blink")

    def test_pet_display_adapter_emits_sensor_snapshot(self) -> None:
        sink = _Sink()
        hardware = _DisplayHardware()
        adapter = PetDisplayAdapter(hardware=hardware, sink=sink)
        adapter.poll_and_emit_sensor_snapshot(expression="idle", screen_id="main")
        self.assertEqual(len(sink.events), 1)
        event = sink.events[0]
        self.assertEqual(event.type, "display_sensor_updated")
        self.assertEqual(event.payload["fps"], 30)
        self.assertEqual(event.payload["sensor_values"]["touch"], True)

    def test_voice_adapter_executes_play_voice(self) -> None:
        backend = _VoiceBackend()
        adapter = VoiceAdapter(output_backend=backend)
        adapter.execute(play_voice(text="欢迎回来", voice="xiaoyu", emotion="happy"))
        self.assertEqual(len(backend.calls), 1)
        text, payload = backend.calls[0]
        self.assertEqual(text, "欢迎回来")
        self.assertEqual(payload["voice"], "xiaoyu")

    def test_voice_adapter_accepts_legacy_speak_action(self) -> None:
        backend = _VoiceBackend()
        adapter = VoiceAdapter(output_backend=backend)
        adapter.execute(Action(type="speak", payload={"text": "你好", "kind": "greeting"}))
        self.assertEqual(len(backend.calls), 1)
        text, payload = backend.calls[0]
        self.assertEqual(text, "你好")
        self.assertEqual(payload["kind"], "greeting")

    def test_voice_adapter_emits_input_event(self) -> None:
        sink = _Sink()
        adapter = VoiceAdapter(sink=sink)
        adapter.emit_voice_input(text="开始专注", confidence=0.91, language="zh-CN")
        self.assertEqual(len(sink.events), 1)
        event = sink.events[0]
        self.assertEqual(event.type, "voice_input_captured")
        self.assertEqual(event.payload["text"], "开始专注")
        self.assertEqual(event.payload["language"], "zh-CN")


if __name__ == "__main__":
    unittest.main()
