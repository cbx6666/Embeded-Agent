from __future__ import annotations

import unittest
from unittest.mock import patch

from src.adapters.voice.alsa_audio_devices import (
    card_from_alsa_device,
    format_alsa_device,
    get_cached_playback_device,
    invalidate_playback_device_cache,
    parse_aplay_list_output,
    parse_arecord_list_output,
    parse_alsa_device,
    playback_device_for_tts,
    prepare_playback_device,
    resolve_capture_device,
    resolve_playback_device,
    resolve_voice_pipeline_devices,
)


SAMPLE_APLAY_L = """
**** List of PLAYBACK Hardware Devices ****
card 1: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
"""

SAMPLE_ARECORD_L = """
**** List of CAPTURE Hardware Devices ****
card 0: C920 [HD Pro Webcam C920], device 0: USB Audio [USB Audio]
card 1: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
"""


class AlsaAudioDevicesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        invalidate_playback_device_cache()

    def test_parse_aplay_list_output(self) -> None:
        devices = parse_aplay_list_output(SAMPLE_APLAY_L)
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["card"], 1)
        self.assertEqual(devices[0]["device"], 0)
        self.assertEqual(devices[0]["alsa_device"], "plughw:1,0")

    def test_parse_arecord_list_output(self) -> None:
        devices = parse_arecord_list_output(SAMPLE_ARECORD_L)
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["alsa_device"], "plughw:0,0")

    def test_parse_alsa_device(self) -> None:
        self.assertEqual(parse_alsa_device("plughw:1,0"), (1, 0))
        self.assertEqual(parse_alsa_device("hw:0,0"), (0, 0))
        self.assertIsNone(parse_alsa_device("default"))

    def test_format_alsa_device(self) -> None:
        self.assertEqual(format_alsa_device(1, 0), "plughw:1,0")

    @patch("src.adapters.voice.alsa_audio_devices.list_playback_devices")
    def test_resolve_prefers_same_card_as_capture(self, mock_list) -> None:
        mock_list.return_value = parse_aplay_list_output(SAMPLE_APLAY_L)
        device = resolve_playback_device(prefer_card=1)
        self.assertEqual(device, "plughw:1,0")

    @patch("src.adapters.voice.alsa_audio_devices.list_playback_devices")
    def test_resolve_auto_picks_first_playback_device(self, mock_list) -> None:
        mock_list.return_value = parse_aplay_list_output(SAMPLE_APLAY_L)
        device = resolve_playback_device()
        self.assertEqual(device, "plughw:1,0")

    @patch("src.adapters.voice.alsa_audio_devices.list_playback_devices")
    def test_resolve_excludes_capture_card_when_split(self, mock_list) -> None:
        mock_list.return_value = parse_aplay_list_output(SAMPLE_APLAY_L)
        device = resolve_playback_device(exclude_card=0)
        self.assertEqual(device, "plughw:1,0")

    @patch("src.adapters.voice.alsa_audio_devices.list_playback_devices")
    def test_explicit_device_overrides_auto(self, mock_list) -> None:
        mock_list.return_value = parse_aplay_list_output(SAMPLE_APLAY_L)
        device = resolve_playback_device(explicit="plughw:9,0")
        self.assertEqual(device, "plughw:9,0")
        mock_list.assert_not_called()

    @patch("src.adapters.voice.alsa_audio_devices.list_capture_devices")
    def test_resolve_capture_prefers_camera(self, mock_list) -> None:
        mock_list.return_value = parse_arecord_list_output(SAMPLE_ARECORD_L)
        device = resolve_capture_device(exclude_card=1)
        self.assertEqual(device, "plughw:0,0")

    @patch("src.adapters.voice.alsa_audio_devices.list_playback_devices")
    def test_playback_device_for_tts_split_excludes_mic_card(self, mock_list) -> None:
        mock_list.return_value = parse_aplay_list_output(SAMPLE_APLAY_L)
        device = playback_device_for_tts(prefer_capture_device="plughw:0,0")
        self.assertEqual(device, "plughw:1,0")

    @patch("src.adapters.voice.alsa_audio_devices.list_capture_devices")
    @patch("src.adapters.voice.alsa_audio_devices.list_playback_devices")
    def test_resolve_voice_pipeline_devices(self, mock_play, mock_capture) -> None:
        mock_capture.return_value = parse_arecord_list_output(SAMPLE_ARECORD_L)
        mock_play.return_value = parse_aplay_list_output(SAMPLE_APLAY_L)
        capture, playback = resolve_voice_pipeline_devices(
            playback_explicit="plughw:1,0",
        )
        self.assertEqual(capture, "plughw:0,0")
        self.assertEqual(playback, "plughw:1,0")

    @patch("src.adapters.voice.alsa_audio_devices.subprocess.run")
    def test_prepare_playback_device_boosts_volume(self, mock_run) -> None:
        device = prepare_playback_device("plughw:1,0")
        self.assertEqual(device, "plughw:1,0")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[:4], ["amixer", "-c", "1", "set"])

    def test_card_from_alsa_device(self) -> None:
        self.assertEqual(card_from_alsa_device("plughw:1,0"), 1)

    def test_cache_reuses_result(self) -> None:
        with patch(
            "src.adapters.voice.alsa_audio_devices.resolve_playback_device",
            return_value="plughw:1,0",
        ) as mock_resolve:
            first = get_cached_playback_device(None, 1, None)
            second = get_cached_playback_device(None, 1, None)
            self.assertEqual(first, "plughw:1,0")
            self.assertEqual(second, "plughw:1,0")
            mock_resolve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
