from __future__ import annotations

import unittest

from src.adapters.voice.wake.detector import _pcm_chunk_bytes


class WakePcmBytesTestCase(unittest.TestCase):
    def test_memoryview_to_bytes(self) -> None:
        raw = b"\x00\x01" * 4
        self.assertEqual(_pcm_chunk_bytes(memoryview(raw)), raw)

    def test_bytes_passthrough(self) -> None:
        raw = b"\xff\xfe" * 2
        self.assertEqual(_pcm_chunk_bytes(raw), raw)


if __name__ == "__main__":
    unittest.main()
