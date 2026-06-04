import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.adapters.voice.board_voice_adapter import BoardVoiceAdapter
from src.adapters.voice.persistent_mic import PersistentMicCapture


class PersistentMicTests(unittest.TestCase):
    def test_record_seconds_writes_wav_from_tap(self) -> None:
        mic = PersistentMicCapture(alsa_device="plughw:0,0", sample_rate=16000, frame_ms=30)
        fake_pcm = b"\x01\x00" * 8000
        mic._running = True
        mic._proc = mock.Mock(poll=mock.Mock(return_value=None))
        mic._ring.append(fake_pcm[:960])

        def _fake_wait(timeout: float) -> bool:
            mic._tap_chunks = [fake_pcm]
            return True

        with mock.patch.object(mic._tap_done, "wait", side_effect=_fake_wait):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "tap.wav"
                path, dur = mic.record_seconds(out, 0.5, pre_roll_sec=0.03)
                self.assertIsNotNone(path)
                assert path is not None
                self.assertTrue(path.is_file())
                self.assertGreater(dur, 0.0)

    def test_wake_shares_capture_device_dual_mic(self) -> None:
        adapter = BoardVoiceAdapter(
            alsa_device="plughw:0,0",
            wake_alsa_device="plughw:1,0",
            persistent_capture=False,
        )
        adapter._detector = mock.Mock(_alsa_device="plughw:1,0")
        self.assertFalse(adapter._wake_shares_capture_device())

    def test_wake_shares_capture_device_same_card(self) -> None:
        adapter = BoardVoiceAdapter(
            alsa_device="plughw:0,0",
            wake_alsa_device="plughw:0,0",
            persistent_capture=False,
        )
        adapter._detector = mock.Mock(_alsa_device="plughw:0,0")
        self.assertTrue(adapter._wake_shares_capture_device())



if __name__ == "__main__":
    unittest.main()
