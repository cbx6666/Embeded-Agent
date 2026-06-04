from __future__ import annotations

import struct
import unittest

from src.adapters.voice.vad_recorder import VadConfig, detect_end_frame_index, frames_contain_speech, pcm_rms
from src.adapters.voice.voice_streaming import SentenceChunker


class VadRecorderTestCase(unittest.TestCase):
    def test_pcm_rms_detects_silence_and_speech(self) -> None:
        silence = struct.pack("<100h", *([0] * 100))
        speech = struct.pack("<100h", *([5000] * 100))
        self.assertLess(pcm_rms(silence), 100)
        self.assertGreater(pcm_rms(speech), 1000)

    def test_detect_end_after_trailing_silence(self) -> None:
        cfg = VadConfig(frame_ms=30, silence_duration_sec=0.6, speech_energy_threshold=500.0)
        frame_bytes = int(cfg.sample_rate * 2 * cfg.frame_ms / 1000)
        silence_frame = b"\x00\x00" * (frame_bytes // 2)
        speech_frame = struct.pack(f"<{frame_bytes // 2}h", *([6000] * (frame_bytes // 2)))

        frames = [speech_frame] * 15 + [silence_frame] * 25
        end = detect_end_frame_index(frames, config=cfg)
        self.assertIsNotNone(end)
        assert end is not None
        self.assertLess(end, len(frames) - 1)

    def test_frames_contain_speech(self) -> None:
        cfg = VadConfig(frame_ms=30, min_speech_duration_sec=0.35, speech_energy_threshold=500.0)
        frame_bytes = int(cfg.sample_rate * 2 * cfg.frame_ms / 1000)
        silence = b"\x00\x00" * (frame_bytes // 2)
        speech = struct.pack(f"<{frame_bytes // 2}h", *([6000] * (frame_bytes // 2)))
        self.assertFalse(frames_contain_speech([silence] * 20, config=cfg))
        self.assertTrue(frames_contain_speech([speech] * 15, config=cfg))


class SentenceChunkerTestCase(unittest.TestCase):
    def test_splits_chinese_sentences(self) -> None:
        chunker = SentenceChunker()
        ready = chunker.feed("你好。")
        self.assertEqual(ready, ["你好。"])
        ready = chunker.feed("我是小助")
        self.assertEqual(ready, [])
        ready = chunker.feed("。")
        self.assertEqual(ready, ["我是小助。"])


if __name__ == "__main__":
    unittest.main()
