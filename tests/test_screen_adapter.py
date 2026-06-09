from __future__ import annotations

import unittest

from src.adapters.screen.pet_display_context import PetDisplayContext
from src.adapters.screen.screen_adapter import ScreenDisplayAdapter


class _FakeHardware:
    def __init__(self) -> None:
        self.updates: list[PetDisplayContext] = []

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def update(self, context: PetDisplayContext) -> None:
        self.updates.append(context)


class ScreenAdapterSyncTest(unittest.TestCase):
    def test_sync_listening_state(self) -> None:
        hw = _FakeHardware()
        adapter = ScreenDisplayAdapter(hardware=hw)
        adapter.sync_visual_state(dialogue_state="listening")
        self.assertEqual(hw.updates[-1].agent_state, "listening")

    def test_sync_speaking_state(self) -> None:
        hw = _FakeHardware()
        adapter = ScreenDisplayAdapter(hardware=hw)
        adapter.sync_visual_state(dialogue_state="speaking")
        self.assertEqual(hw.updates[-1].agent_state, "speaking")

    def test_focus_active_overrides_dialogue(self) -> None:
        hw = _FakeHardware()
        adapter = ScreenDisplayAdapter(hardware=hw)
        adapter.sync_visual_state(
            dialogue_state="speaking",
            focus_active=True,
            focus_remaining=120,
            focus_duration=1500,
        )
        last = hw.updates[-1]
        self.assertEqual(last.agent_state, "focus_mode")
        self.assertEqual(last.focus_remaining, 120)
        self.assertEqual(last.focus_duration, 1500)

    def test_sync_throttles_hardware_push_to_once_per_second(self) -> None:
        hw = _FakeHardware()
        adapter = ScreenDisplayAdapter(hardware=hw)
        adapter.sync_visual_state(dialogue_state="idle", emotion="happy")
        adapter.sync_visual_state(dialogue_state="idle", emotion="stressed")
        self.assertEqual(len(hw.updates), 1)
        self.assertEqual(hw.updates[-1].emotion, "happy")
        adapter.sync_visual_state(dialogue_state="speaking", emotion="stressed", force=True)
        self.assertEqual(len(hw.updates), 2)
        self.assertEqual(hw.updates[-1].agent_state, "speaking")
        self.assertEqual(hw.updates[-1].emotion, "stressed")


if __name__ == "__main__":
    unittest.main()
