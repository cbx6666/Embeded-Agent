from __future__ import annotations

import unittest


class VoiceAdapterImportTestCase(unittest.TestCase):
    def test_voice_package_exports(self) -> None:
        from src.adapters.voice import (
            BaiduShortASRBackend,
            BaiduTTSBackend,
            BoardVoiceAdapter,
            build_wake_word_detector,
        )

        self.assertTrue(callable(build_wake_word_detector))
        self.assertTrue(hasattr(BaiduShortASRBackend, "recognize_file"))
        self.assertTrue(hasattr(BaiduTTSBackend, "speak"))
        self.assertTrue(hasattr(BoardVoiceAdapter, "start"))


if __name__ == "__main__":
    unittest.main()
