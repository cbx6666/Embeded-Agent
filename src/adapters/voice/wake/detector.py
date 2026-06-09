"""唤醒词检测：仅通过 AudioInputManager 的 PCM feed_audio 喂入，不独立打开 arecord。"""

from __future__ import annotations

import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


class EventSink(Protocol):
    def handle_event(self, event: Any) -> Any: ...


@dataclass
class WakeWordEvent:
    keyword: str
    timestamp: int
    source: str = "microphone"
    confidence: float = 1.0


def _pcm_chunk_bytes(indata: Any) -> bytes:
    """sounddevice RawInputStream 回调里 indata 可能是 memoryview/cffi buffer。"""
    if hasattr(indata, "tobytes"):
        return indata.tobytes()
    return bytes(indata)


class WakeWordDetector(ABC):
    """唤醒词检测器：由 VoiceRuntime 通过 feed_audio 喂入 PCM，禁止 start()+独立 arecord。"""

    def __init__(
        self,
        sink: EventSink | None = None,
        source: str = "microphone",
        *,
        alsa_device: str = "plughw:0,0",
    ) -> None:
        self._sink = sink
        self._source = source
        self._alsa_device = alsa_device
        self._on_wake: list[Callable[[WakeWordEvent], None]] = []
        self._media_playback_competing = False

    def set_media_playback_competing(self, active: bool) -> None:
        """媒体播放时扬声器灌麦：缩短冷却并降低 KWS 阈值。"""
        self._media_playback_competing = bool(active)
        if hasattr(self, "_spotter") and hasattr(self, "_base_keywords_threshold"):
            threshold = (
                self._media_keywords_threshold
                if self._media_playback_competing
                else self._base_keywords_threshold
            )
            try:
                self._spotter.keywords_threshold = threshold
            except Exception:
                pass

    @abstractmethod
    def detect_once(self, audio_chunk: bytes) -> str | None:
        """分析一段音频 chunk，返回命中的唤醒词（None = 未命中）。"""

    def feed_audio(self, audio_chunk: bytes) -> bool:
        keyword = self.detect_once(audio_chunk)
        if keyword is None:
            return False
        self._emit(keyword)
        return True

    def on_wake(self, callback: Callable[[WakeWordEvent], None]) -> None:
        self._on_wake.append(callback)

    def start(self) -> None:
        raise RuntimeError(
            "WakeWordDetector.start() 已移除：请使用 AudioInputManager 单路 capture，"
            "并通过 feed_audio() 喂入检测器。"
        )

    def stop(self) -> None:
        """兼容旧调用；feed 模式下无需停止独立采集线程。"""

    def reset_stream(self) -> None:
        """子类可覆盖：录音暂停后清空 KWS 流状态。"""

    def _emit(self, keyword: str) -> None:
        event = WakeWordEvent(
            keyword=keyword,
            timestamp=int(time.time()),
            source=self._source,
        )
        for cb in self._on_wake:
            cb(event)
        if self._sink is not None:
            from src.agent.event.event_model import Event

            self._sink.handle_event(
                Event(
                    type="voice_wake_detected",
                    timestamp=event.timestamp,
                    payload={
                        "keyword": keyword,
                        "source": self._source,
                        "confidence": event.confidence,
                    },
                )
            )


class PorcupineWakeWordDetector(WakeWordDetector):
    def __init__(
        self,
        *,
        model_path: str | Path,
        keyword_path: str | Path,
        sensitivities: list[float] | None = None,
        sample_rate: int = 16000,
        sink: EventSink | None = None,
        source: str = "microphone",
        alsa_device: str = "plughw:0,0",
    ) -> None:
        super().__init__(sink=sink, source=source, alsa_device=alsa_device)
        self._model_path = Path(model_path)
        self._keyword_path = Path(keyword_path)
        self._sensitivities = sensitivities or [0.5]
        self._porcupine = self._init_porcupine()
        self._keyword = self._keyword_path.stem

    def _init_porcupine(self) -> Any:
        try:
            import porcupine
        except ImportError as exc:
            raise ImportError(
                "请先安装 porcupine：pip install porcupine"
            ) from exc
        return porcupine.create(
            model_path=str(self._model_path),
            keyword_paths=[str(self._keyword_path)],
            sensitivities=self._sensitivities,
        )

    def detect_once(self, audio_chunk: bytes) -> str | None:
        pcm = list(struct.unpack(f"<{len(audio_chunk)//2}h", audio_chunk))
        if self._porcupine.process(pcm) >= 0:
            return self._keyword
        return None


class EnergyBasedWakeWordDetector(WakeWordDetector):
    def __init__(
        self,
        *,
        wake_word: str = "小助",
        energy_threshold: float = 0.02,
        min_trigger_frames: int = 3,
        silence_cooldown_sec: float = 3.0,
        sink: EventSink | None = None,
        source: str = "microphone",
        alsa_device: str = "plughw:0,0",
    ) -> None:
        super().__init__(sink=sink, source=source, alsa_device=alsa_device)
        self._wake_word = wake_word
        self._energy_threshold = float(energy_threshold)
        self._min_trigger_frames = int(min_trigger_frames)
        self._silence_cooldown = float(silence_cooldown_sec)
        self._trigger_count = 0
        self._last_trigger_time = 0.0

    def detect_once(self, audio_chunk: bytes) -> str | None:
        import math

        if len(audio_chunk) < 2:
            return None
        samples = struct.unpack(f"<{len(audio_chunk)//2}h", audio_chunk)
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        energy = min(1.0, rms / 32768.0)
        now = time.time()
        if energy > self._energy_threshold:
            self._trigger_count += 1
            if (
                self._trigger_count >= self._min_trigger_frames
                and now - self._last_trigger_time > self._silence_cooldown
            ):
                self._last_trigger_time = now
                self._trigger_count = 0
                return self._wake_word
        else:
            self._trigger_count = 0
        return None


class MockWakeWordDetector(WakeWordDetector):
    def __init__(
        self,
        *,
        wake_word: str = "小助",
        sink: EventSink | None = None,
        source: str = "mock",
        trigger_after_sec: float = 0.0,
    ) -> None:
        super().__init__(sink=sink, source=source, alsa_device="mock")
        self._wake_word = wake_word
        self._trigger_after_sec = float(trigger_after_sec)
        self._start_time: float | None = None
        self._has_triggered = False

    def detect_once(self, audio_chunk: bytes) -> str | None:
        del audio_chunk
        if self._start_time is None:
            self._start_time = time.time()
        if not self._has_triggered and time.time() - self._start_time >= self._trigger_after_sec:
            self._has_triggered = True
            return self._wake_word
        return None


class SherpaOnnxWakeWordDetector(WakeWordDetector):
    def __init__(
        self,
        *,
        model_dir: str | Path,
        keywords_file: str | Path,
        wake_word: str = "小助",
        keywords_threshold: float = 0.25,
        keywords_score: float = 2.0,
        num_threads: int = 2,
        use_int8: bool = True,
        sample_rate: int = 16000,
        silence_cooldown_sec: float = 0.8,
        sink: EventSink | None = None,
        source: str = "microphone",
        alsa_device: str = "plughw:0,0",
    ) -> None:
        from src.adapters.voice.wake.sherpa_kws import create_keyword_spotter, resolve_sherpa_kws_dir

        super().__init__(sink=sink, source=source, alsa_device=alsa_device)
        self._wake_word = wake_word.strip() or "小助"
        self._sample_rate = int(sample_rate)
        self._silence_cooldown = float(silence_cooldown_sec)
        self._media_silence_cooldown = 0.25
        self._base_keywords_threshold = float(keywords_threshold)
        self._media_keywords_threshold = max(0.08, self._base_keywords_threshold * 0.55)
        self._last_trigger_time = 0.0
        self._detect_lock = threading.Lock()
        self._spotter = create_keyword_spotter(
            model_dir=resolve_sherpa_kws_dir(model_dir),
            keywords_file=Path(keywords_file).expanduser().resolve(),
            keywords_threshold=keywords_threshold,
            keywords_score=keywords_score,
            num_threads=num_threads,
            use_int8=use_int8,
        )
        self._stream = self._spotter.create_stream()

    def reset_stream(self) -> None:
        with self._detect_lock:
            try:
                self._spotter.reset_stream(self._stream)
            except Exception:
                pass

    def detect_once(self, audio_chunk: bytes) -> str | None:
        import numpy as np

        if len(audio_chunk) < 2:
            return None
        with self._detect_lock:
            samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            self._stream.accept_waveform(self._sample_rate, samples)
            while self._spotter.is_ready(self._stream):
                self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream).strip()
            if not result:
                return None
            now = time.time()
            cooldown = (
                self._media_silence_cooldown
                if self._media_playback_competing
                else self._silence_cooldown
            )
            if now - self._last_trigger_time < cooldown:
                self._spotter.reset_stream(self._stream)
                return None
            self._last_trigger_time = now
            self._spotter.reset_stream(self._stream)
            return self._wake_word or result


def build_wake_word_detector(
    *,
    backend: str = "energy",
    sink: EventSink | None = None,
    source: str = "microphone",
    **kwargs: Any,
) -> WakeWordDetector:
    backend = backend.lower().strip()
    if backend in {"sherpa-onnx", "sherpa", "kws", "sherpa_onnx"}:
        return SherpaOnnxWakeWordDetector(sink=sink, source=source, **kwargs)
    if backend in {"porcupine", "picovoice"}:
        return PorcupineWakeWordDetector(sink=sink, source=source, **kwargs)
    if backend in {"energy", "ste", "simple"}:
        return EnergyBasedWakeWordDetector(sink=sink, source=source, **kwargs)
    if backend in {"mock", "dummy", "test"}:
        return MockWakeWordDetector(sink=sink, source=source, **kwargs)
    raise ValueError(
        f"未知的唤醒词检测器 backend：{backend}，可选：sherpa-onnx / porcupine / energy / mock"
    )
