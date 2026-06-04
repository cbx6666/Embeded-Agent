from __future__ import annotations

import unittest

from src.adapters.voice.mic_arbitration import mic_capture_lock


class MicArbitrationTestCase(unittest.TestCase):
    def test_mic_capture_lock_acquires_without_timeout_error(self) -> None:
        with mic_capture_lock():
            pass


if __name__ == "__main__":
    unittest.main()
