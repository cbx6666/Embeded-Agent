from __future__ import annotations

import struct
import unittest
from pathlib import Path

from src.adapters.voice.sherpa_kws import (
    build_keywords_file,
    ensure_keywords_file,
    resolve_sherpa_kws_dir,
    resolve_transducer_paths,
)


class SherpaKwsSetupTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_dir = resolve_sherpa_kws_dir(
            "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
        )

    def test_resolve_transducer_paths(self) -> None:
        paths = resolve_transducer_paths(self.model_dir, use_int8=True)
        for key in ("tokens", "encoder", "decoder", "joiner"):
            self.assertTrue(paths[key].is_file(), paths[key])

    def test_build_keywords_file(self) -> None:
        out = Path("data/test_sherpa_keywords.txt")
        out.unlink(missing_ok=True)
        build_keywords_file(
            model_dir=self.model_dir,
            phrases=["小助"],
            keywords_file=out,
            keywords_threshold=0.25,
            keywords_score=2.0,
        )
        text = out.read_text(encoding="utf-8")
        self.assertIn("@小助", text)
        self.assertIn("x iǎo", text)
        out.unlink(missing_ok=True)

    def test_sherpa_detector_silent_audio(self) -> None:
        try:
            from src.adapters.voice.wake_word_detector import SherpaOnnxWakeWordDetector
        except ImportError:
            self.skipTest("sherpa-onnx not installed")

        keywords = ensure_keywords_file(
            model_dir=self.model_dir,
            phrases=["小助"],
            keywords_file="data/test_sherpa_kw_runtime.txt",
            force=True,
        )
        detector = SherpaOnnxWakeWordDetector(
            model_dir=self.model_dir,
            keywords_file=keywords,
            wake_word="小助",
            sink=None,
            alsa_device="plughw:1,0",
        )
        silent = struct.pack("<" + "h" * 1600, *([0] * 1600))
        self.assertIsNone(detector.detect_once(silent))
        detector.stop()
        Path("data/test_sherpa_kw_runtime.txt").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
