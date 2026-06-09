"""BoardVoiceAdapter：语音子系统对外 facade，业务逻辑在 VoiceRuntime。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.adapters.voice.runtime.voice_debug_log import configure_voice_debug
from src.adapters.voice.runtime.voice_runtime import VoiceRuntime
from src.adapters.voice.wake.local_wake_ack import DEFAULT_WAKE_ACK_DIR, DEFAULT_WAKE_ACK_TEXT


class BoardVoiceAdapter:
    """板级语音适配器 facade：委托 VoiceRuntime 处理唤醒/录音/TTS。"""

    WAKE_LATEST_RAW_WAV = VoiceRuntime.WAKE_LATEST_RAW_WAV
    WAKE_LATEST_ASR_WAV = "wake_latest_asr.wav"

    def __init__(
        self,
        *,
        sink: Any | None = None,
        detector: Any | None = None,
        recognizer: Any | None = None,
        tts_backend: Any | None = None,
        alsa_device: str = "plughw:0,0",
        wake_alsa_device: str | None = None,
        sample_rate: int = 16000,
        post_ack_user_window_sec: float = 2.5,
        max_capture_duration_sec: float = 15.0,
        silence_duration_sec: float = 0.8,
        wake_ack_text: str | None = None,
        wake_ack_mode: str = "local",
        wake_ack_dir: str | Path | None = None,
        audio_dir: str | Path = "data/",
        debug: bool = False,
        voice_debug_dir: str | Path = "data/voice_debug",
        voice_debug_log: bool = True,
        voice_debug_console: bool = False,
    ) -> None:
        self._debug = debug
        self.last_recording_path: Path | None = None
        self._voice_debug_manager = configure_voice_debug(
            log_dir=voice_debug_dir,
            enabled=bool(voice_debug_log),
            verbose=bool(debug),
            console=bool(voice_debug_console),
        )

        def _log_hook(line: str) -> None:
            if self._voice_debug_manager.enabled:
                self._voice_debug_manager.log_line(line, event="board_voice")
            else:
                print(line, flush=True)

        self._runtime = VoiceRuntime(
            sink=sink,
            wake_detector=detector,
            recognizer=recognizer,
            tts_backend=tts_backend,
            alsa_device=alsa_device,
            wake_alsa_device=wake_alsa_device,
            sample_rate=sample_rate,
            audio_dir=audio_dir,
            wake_ack_text=wake_ack_text,
            wake_ack_mode=wake_ack_mode,
            wake_ack_dir=wake_ack_dir,
            max_capture_duration_sec=max_capture_duration_sec,
            silence_duration_sec=silence_duration_sec,
            post_ack_user_window_sec=post_ack_user_window_sec,
            voice_debug_manager=self._voice_debug_manager,
            log_hook=_log_hook,
        )

    @property
    def _sink(self) -> Any:
        return self._runtime._bridge._sink  # noqa: SLF001

    @_sink.setter
    def _sink(self, value: Any) -> None:
        self._runtime.bind_sink(value)
        media = getattr(value, "media_controller", None)
        if media is not None:
            self._runtime.bind_media_controller(media)

    def start(self) -> None:
        self._runtime.start()

    def stop(self) -> None:
        self._runtime.stop()

    def is_running(self) -> bool:
        return self._runtime.is_running()

    def is_tts_running(self) -> bool:
        return self._runtime.is_tts_running()

    def preload_wake_ack(self) -> int:
        return self._runtime.preload_wake_ack()

    def execute(self, action: Any) -> None:
        self._runtime.execute(action)

    def on_speak(self, callback: Callable[[str], None]) -> None:
        del callback

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = bool(value)

    def start_background_loop(self, *, interval_sec: float = 30.0) -> None:
        from src.adapters.voice.runtime.logger import voice_log

        voice_log(f"voice_loop 已弃用（interval={interval_sec}s），请使用唤醒词交互")

    def stop_background_loop(self) -> None:
        pass

    def run_recognize_once(self) -> Any:
        from src.agent.event.event_model import Event
        import time

        if not self._runtime.is_running():
            self._runtime.start()
        path = self._runtime._audio_dir / self.WAKE_LATEST_RAW_WAV
        audio_path, _dur, _reason = self._runtime.audio_input.record_until_silence(path)
        self.last_recording_path = audio_path
        if audio_path is None or self._runtime._recognizer is None:
            return None
        result = self._runtime._recognizer.recognize_file(audio_path)
        text = str(getattr(result, "text", result) or "").strip()
        if not text:
            return None
        return Event(
            type="speech_recognized",
            timestamp=int(time.time()),
            payload={
                "text": text,
                "source": "board_voice",
                "confidence": 0.9,
                "language": "zh",
                "is_final": True,
            },
        )

    def replay_last_recording(self) -> bool:
        path = self.last_recording_path
        if path is None or not path.is_file():
            return False
        from src.adapters.voice.tts.audio_playback import play_wav_file

        playback = getattr(self._runtime._tts_backend, "alsa_playback_device", None)
        play_wav_file(
            path,
            alsa_playback_device=playback,
            prefer_capture_device=self._runtime.audio_input.alsa_device,
        )
        return True
