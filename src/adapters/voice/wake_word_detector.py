"""唤醒词检测模块。

持续监听麦克风，当检测到唤醒词时生成 `voice_wake_detected` 事件并注入 AgentCore。
支持多种检测策略：
- **Porcupine**（Picovoice）：生产级高精度，需 `.pv` 模型文件
- **Sherpa-ONNX KWS**：开源中文关键词唤醒，无需注册（默认推荐）
- **EnergyBased**：基于音频能量的简单检测，支持 Windows/macOS/Linux
- **Mock**：始终返回 True，用于本地调试

跨平台音频支持：
- Linux：使用 `arecord`（alsa-utils）
- Windows/macOS：使用 `sounddevice`（pip install sounddevice）
- 备选：`pyaudio`
"""

from __future__ import annotations

import io
import os
import struct
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

# ---------------------------------------------------------------------------
# 类型别名 / 协议
# ---------------------------------------------------------------------------

class EventSink(Protocol):
    """可以接收 Agent 标准事件的对象（与 AgentCore 兼容）。"""
    def handle_event_with_results(self, event: Any) -> Any: ...


@dataclass
class WakeWordEvent:
    """唤醒词命中时的标准事件结构（内部使用）。"""
    keyword: str
    timestamp: int
    source: str = "microphone"
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# 跨平台音频工具
# ---------------------------------------------------------------------------

def _pcm_chunk_bytes(indata: Any) -> bytes:
    """sounddevice RawInputStream 回调里 indata 可能是 memoryview/cffi buffer。"""
    if hasattr(indata, "tobytes"):
        return indata.tobytes()
    return bytes(indata)


def _get_audio_recorder(
    alsa_device: str = "plughw:0,0",
    sample_rate: int = 16000,
    chunk_size: int | None = None,
) -> Iterator[bytes]:
    """跨平台音频 chunk 迭代器，按顺序尝试各音频后端。

    优先级（每层失败后自动降级）：
    1. sounddevice（跨平台）
    2. pyaudio（跨平台）
    3. arecord（Linux alsa-utils）—— Atlas 200i DK 走这里
    4. 空迭代器（所有后端均不可用）
    """
    import platform
    system = platform.system()
    cs = chunk_size or 1024

    # ---- 1. sounddevice（跨平台） ----
    try:
        yield from _sounddevice_recorder(alsa_device, sample_rate, cs)
        return
    except Exception as exc:
        # 可能是 ImportError（未安装），也可能是 RuntimeError（无设备/无麦克风）
        print(f"[Audio] sounddevice 不可用: {exc}", flush=True)

    # ---- 2. pyaudio（跨平台） ----
    try:
        yield from _pyaudio_recorder(alsa_device, sample_rate, cs)
        return
    except Exception as exc:
        print(f"[Audio] pyaudio 不可用: {exc}", flush=True)

    # ---- 3. arecord（Linux） ----
    if system == "Linux":
        try:
            yield from _arecord_recorder(alsa_device, sample_rate, cs)
            return
        except Exception as exc:
            print(f"[Audio] arecord 不可用: {exc}", flush=True)

    # ---- 4. sox（macOS 备选） ----
    if system == "Darwin":
        try:
            yield from _sox_recorder(alsa_device, sample_rate, "sox", cs)
            return
        except Exception as exc:
            print(f"[Audio] sox 不可用: {exc}", flush=True)

    # 所有后端均失败，打印诊断信息
    print(
        f"[Audio] ⚠️ 所有音频后端均不可用。\n"
        f"  系统: {system} | 设备: {alsa_device} | 采样率: {sample_rate}\n"
        f"  建议:\n"
        f"    pip install sounddevice   # 跨平台推荐\n"
        f"    apt install alsa-utils    # Linux (arecord/aplay)\n"
        f"    apt install sox           # macOS/Linux (sox)",
        flush=True,
    )
    return  # 空迭代器，不阻塞
    yield  # unreachable，抑制 PEP-479


def _sounddevice_recorder(
    device: str | int | None,
    sample_rate: int,
    chunk_size: int,
) -> Iterator[bytes]:
    """使用 sounddevice 库录制音频（跨平台）。"""
    import sounddevice  # type: ignore
    q: list[bytes] = []
    q_lock = threading.Lock()
    stopped = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[WakeWord-sounddevice] {status}", flush=True)
        if stopped.is_set():
            raise sounddevice.CallbackStop
        with q_lock:
            q.append(_pcm_chunk_bytes(indata))

    # device=None 表示默认麦克风
    device_id = device if isinstance(device, int) else None
    stream = sounddevice.RawInputStream(
        samplerate=sample_rate,
        blocksize=chunk_size,
        device=device_id,
        channels=1,
        dtype="int16",
        callback=callback,
    )

    with stream:
        while not stopped.is_set():
            with q_lock:
                if q:
                    yield q.pop(0)
            time.sleep(0.01)
        # 耗尽队列
        with q_lock:
            while q:
                yield q.pop(0)


def _pyaudio_recorder(
    device: str | int | None,
    sample_rate: int,
    chunk_size: int,
) -> Iterator[bytes]:
    """使用 pyaudio 库录制音频（跨平台）。"""
    import pyaudio  # type: ignore
    p = pyaudio.PyAudio()
    device_id = device if isinstance(device, int) else None
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=device_id,
        frames_per_buffer=chunk_size,
    )

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            yield data
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def _arecord_recorder(alsa_device: str, sample_rate: int, chunk_size: int) -> Iterator[bytes]:
    """使用 arecord（Linux alsa-utils）录制音频。"""
    proc = subprocess.Popen(
        [
            "arecord",
            "-D", alsa_device,
            "-f", "S16_LE",
            "-r", str(sample_rate),
            "-c", "1",
            "-q",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        while True:
            data = proc.stdout.read(chunk_size * 2)  # 2 bytes per sample
            if data:
                yield data
    finally:
        proc.terminate()
        proc.wait(timeout=3)


def _sox_recorder(device: str, sample_rate: int, sox_cmd: str, chunk_size: int) -> Iterator[bytes]:
    """使用 sox 录制音频（macOS 备选）。"""
    try:
        proc = subprocess.Popen(
            [sox_cmd, "-d", "-r", str(sample_rate), "-b", "16", "-c", "1", "-t", "raw", "-q", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            while True:
                data = proc.stdout.read(chunk_size * 2)
                if data:
                    yield data
        finally:
            proc.terminate()
            proc.wait(timeout=3)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class WakeWordDetector(ABC):
    """唤醒词检测器抽象基类。"""

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
        self._running = False
        self._thread: threading.Thread | None = None
        self._on_wake: list[Callable[[WakeWordEvent], None]] = []
        self._arecord_proc: Any | None = None

    @abstractmethod
    def detect_once(self, audio_chunk: bytes) -> str | None:
        """分析一段音频 chunk，返回命中的唤醒词（None = 未命中）。"""
        ...

    def on_wake(self, callback: Callable[[WakeWordEvent], None]) -> None:
        """注册唤醒词命中回调。"""
        self._on_wake.append(callback)

    def start(self) -> None:
        """启动后台监听线程。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="WakeWordDetector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止后台监听线程。"""
        self._running = False
        self._terminate_arecord_proc()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _terminate_arecord_proc(self) -> None:
        proc = self._arecord_proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            self._arecord_proc = None

    def _emit(self, keyword: str) -> None:
        """将唤醒事件注入 AgentCore 并触发回调。"""
        event = WakeWordEvent(
            keyword=keyword,
            timestamp=int(time.time()),
            source=self._source,
        )
        for cb in self._on_wake:
            cb(event)
        if self._sink is not None:
            from src.agent.event.event_model import Event
            self._sink.handle_event_with_results(
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

    def _run_loop(self) -> None:
        """后台主循环：持续采集音频并检测唤醒词。

        录音时 pkill -9 arecord 会导致当前音频流中断，
        外层 while 兜底会自动重启检测，无需重新 start()。
        """
        import threading as _th
        import subprocess as _sp
        restart_delay = 1.0  # 重启前等待秒数（给录音进程释放设备的时间）

        import platform as _platform

        prefer_arecord = _platform.system() == "Linux" and os.environ.get(
            "EMBED_WAKE_AUDIO", "arecord"
        ).strip().lower() in {"arecord", "alsa", "linux"}

        while self._running:
            chunk_size = 512  # samples at 16kHz ≈ 32ms
            handled = False

            if prefer_arecord:
                handled = self._run_arecord_loop(chunk_size, restart_delay)
                if handled and not self._running:
                    break
                if self._running:
                    time.sleep(restart_delay)
                continue

            handled = self._run_sounddevice_loop(chunk_size)
            if handled and not self._running:
                break
            if not handled and _platform.system() == "Linux":
                handled = self._run_arecord_loop(chunk_size, restart_delay)

            if self._running:
                print(f"[WakeWord] 等待 {restart_delay}s 后重启唤醒词检测...", flush=True)
                time.sleep(restart_delay)

    def _run_sounddevice_loop(self, chunk_size: int) -> bool:
        """使用 sounddevice 采集；成功跑过一段返回 True。"""
        import threading as _th

        try:
            import sounddevice as _sd
        except ImportError:
            return False

        q: list[bytes] = []
        _stopped = _th.Event()

        def _callback(_indata, _frames, _time_info, _status):
            if _status:
                print(f"[WakeWord-sounddevice] {_status}", flush=True)
            if _stopped.is_set():
                raise _sd.CallbackStop
            q.append(_pcm_chunk_bytes(_indata))

        stream = _sd.RawInputStream(
            samplerate=16000,
            blocksize=chunk_size,
            device=None,
            channels=1,
            dtype="int16",
            callback=_callback,
        )
        try:
            with stream:
                while self._running:
                    time.sleep(0.05)
                    if not q:
                        continue
                    chunk = q.pop(0)
                    keyword = self.detect_once(chunk)
                    if keyword is not None:
                        self._emit(keyword)
            return True
        except Exception as exc:
            if self._running:
                print(f"[WakeWord] sounddevice 异常: {exc}", flush=True)
            return False
        finally:
            _stopped.set()

    def _run_arecord_loop(self, chunk_size: int, restart_delay: float) -> bool:
        """Linux arecord 采集（板载麦推荐 plughw:1,0）。"""
        import subprocess as _sp

        try:
            proc = _sp.Popen(
                [
                    "arecord",
                    "-D",
                    self._alsa_device,
                    "-f",
                    "S16_LE",
                    "-r",
                    "16000",
                    "-c",
                    "1",
                    "-q",
                ],
                stdout=_sp.PIPE,
                stderr=_sp.DEVNULL,
            )
            self._arecord_proc = proc
            try:
                while self._running:
                    data = proc.stdout.read(chunk_size * 2)
                    if not data:
                        break
                    keyword = self.detect_once(data)
                    if keyword is not None:
                        self._emit(keyword)
                    time.sleep(0.01)
                return True
            finally:
                self._arecord_proc = None
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        except Exception as exc:
            if self._running:
                print(f"[WakeWord] arecord 异常: {exc}", flush=True)
            return False

    @abstractmethod
    def _audio_loop(self) -> Iterator[bytes]:
        """生成音频 chunk 的迭代器，子类实现具体音频设备读取。"""
        ...

    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Porcupine（Picovoice）实现 —— 生产级高精度
# ---------------------------------------------------------------------------

class PorcupineWakeWordDetector(WakeWordDetector):
    """基于 Picovoice Porcupine 的唤醒词检测器。

    需要安装 porcupine：pip install porcupine
    并下载 .ppn（唤醒词模型）和 .pv（参数文件）。

    示例：
        detector = PorcupineWakeWordDetector(
            model_path="models/porcupine_params_zh.pv",
            keyword_path="models/hey_assistant_zh.ppn",
            sensitivities=[0.5],
        )
    """

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
        self._sample_rate = int(sample_rate)
        self._alsa_device = alsa_device
        self._porcupine = self._init_porcupine()
        self._keyword = self._keyword_path.stem

    def _init_porcupine(self) -> Any:
        try:
            import porcupine
        except ImportError:
            raise ImportError(
                "请先安装 porcupine：pip install porcupine\n"
                "并从 https://picovoice.ai/porcupine/ 下载中文唤醒词模型。"
            )
        return porcupine.create(
            model_path=str(self._model_path),
            keyword_paths=[str(self._keyword_path)],
            sensitivities=self._sensitivities,
        )

    def detect_once(self, audio_chunk: bytes) -> str | None:
        pcm = self._bytes_to_int16(audio_chunk)
        keyword_index = self._porcupine.process(pcm)
        if keyword_index >= 0:
            return self._keyword
        return None

    def _audio_loop(self) -> Iterator[bytes]:
        yield from _get_audio_recorder(
            alsa_device=self._alsa_device,
            sample_rate=self._sample_rate,
            chunk_size=self._porcupine.frame_length,
        )

    @staticmethod
    def _bytes_to_int16(data: bytes) -> list[int]:
        return list(struct.unpack(f"<{len(data)//2}h", data))


# ---------------------------------------------------------------------------
# 能量检测实现 —— 开发/简单场景用（跨平台）
# ---------------------------------------------------------------------------

class EnergyBasedWakeWordDetector(WakeWordDetector):
    """基于短时能量（STE）的简单唤醒词检测，跨平台支持。

    原理：持续监听麦克风，当检测到连续多帧音频能量超过阈值时触发唤醒。
    支持 Windows/macOS/Linux（自动选择可用音频后端）。
    """

    def __init__(
        self,
        *,
        wake_word: str = "小助",
        energy_threshold: float = 0.02,
        min_trigger_frames: int = 3,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        silence_cooldown_sec: float = 3.0,
        sink: EventSink | None = None,
        source: str = "microphone",
        alsa_device: str = "plughw:0,0",
    ) -> None:
        super().__init__(sink=sink, source=source, alsa_device=alsa_device)
        self._wake_word = wake_word
        self._energy_threshold = float(energy_threshold)
        self._min_trigger_frames = int(min_trigger_frames)
        self._sample_rate = int(sample_rate)
        self._frame_samples = int(sample_rate * frame_ms / 1000)
        self._silence_cooldown = float(silence_cooldown_sec)
        self._alsa_device = alsa_device
        self._trigger_count = 0
        self._last_trigger_time: float = 0.0

    def detect_once(self, audio_chunk: bytes) -> str | None:
        energy = self._compute_energy(audio_chunk)
        now = time.time()

        if energy > self._energy_threshold:
            self._trigger_count += 1
            if self._trigger_count >= self._min_trigger_frames:
                if now - self._last_trigger_time > self._silence_cooldown:
                    self._last_trigger_time = now
                    self._trigger_count = 0
                    print(f"[WakeWord-Energy] 能量触发: energy={energy:.4f} >= {self._energy_threshold}", flush=True)
                    return self._wake_word
        else:
            self._trigger_count = 0

        return None

    def _compute_energy(self, audio_chunk: bytes) -> float:
        """计算归一化短时能量（0~1）。"""
        import math
        if len(audio_chunk) < 2:
            return 0.0
        samples = struct.unpack(f"<{len(audio_chunk)//2}h", audio_chunk)
        sum_sq = sum(s * s for s in samples)
        rms = math.sqrt(sum_sq / len(samples))
        return min(1.0, rms / 32768.0)

    def _audio_loop(self) -> Iterator[bytes]:
        yield from _get_audio_recorder(
            alsa_device=self._alsa_device,
            sample_rate=self._sample_rate,
            chunk_size=self._frame_samples,
        )


# ---------------------------------------------------------------------------
# Mock 实现 —— 用于本地无麦克风调试
# ---------------------------------------------------------------------------

class MockWakeWordDetector(WakeWordDetector):
    """始终触发唤醒词的 Mock 检测器，用于无硬件/无麦克风环境的本地调试。"""

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
        if self._start_time is None:
            self._start_time = time.time()

        if not self._has_triggered:
            elapsed = time.time() - self._start_time
            if elapsed >= self._trigger_after_sec:
                self._has_triggered = True
                print(f"[WakeWord-Mock] Mock 触发唤醒词：{self._wake_word}", flush=True)
                return self._wake_word
        return None

    def _audio_loop(self) -> Iterator[bytes]:
        # Mock 不需要真实音频，返回空迭代器
        while self._running:
            time.sleep(0.1)
        return
        yield  # 使函数成为生成器


# ---------------------------------------------------------------------------
# Sherpa-ONNX KWS —— 开源中文关键词唤醒（无需 Picovoice 注册）
# ---------------------------------------------------------------------------

class SherpaOnnxWakeWordDetector(WakeWordDetector):
    """基于 Sherpa-ONNX 的离线关键词唤醒（如「小助」）。

    需 pip install sherpa-onnx pypinyin，并运行 scripts/setup_sherpa_kws.py。
    """

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
        silence_cooldown_sec: float = 2.0,
        sink: EventSink | None = None,
        source: str = "microphone",
        alsa_device: str = "plughw:0,0",
    ) -> None:
        from src.adapters.voice.sherpa_kws import create_keyword_spotter, resolve_sherpa_kws_dir

        super().__init__(sink=sink, source=source, alsa_device=alsa_device)
        self._model_dir = resolve_sherpa_kws_dir(model_dir)
        self._keywords_file = Path(keywords_file).expanduser().resolve()
        self._wake_word = wake_word.strip() or "小助"
        self._sample_rate = int(sample_rate)
        self._silence_cooldown = float(silence_cooldown_sec)
        self._last_trigger_time = 0.0
        self._detect_lock = threading.Lock()

        self._spotter = create_keyword_spotter(
            model_dir=self._model_dir,
            keywords_file=self._keywords_file,
            keywords_threshold=keywords_threshold,
            keywords_score=keywords_score,
            num_threads=num_threads,
            use_int8=use_int8,
        )
        self._stream = self._spotter.create_stream()

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
            if now - self._last_trigger_time < self._silence_cooldown:
                self._spotter.reset_stream(self._stream)
                return None

            self._last_trigger_time = now
            self._spotter.reset_stream(self._stream)
            keyword = self._wake_word or result
            print(f"[WakeWord-Sherpa] 命中：{result!r} → {keyword!r}", flush=True)
            return keyword

    def stop(self) -> None:
        super().stop()
        with self._detect_lock:
            try:
                self._spotter.reset_stream(self._stream)
            except Exception:
                pass

    def _audio_loop(self) -> Iterator[bytes]:
        yield from ()


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_wake_word_detector(
    *,
    backend: str = "energy",
    sink: EventSink | None = None,
    source: str = "microphone",
    **kwargs: Any,
) -> WakeWordDetector:
    """根据 backend 名称构造对应的唤醒词检测器。

    Args:
        backend: 唤醒引擎，可选 "sherpa-onnx" | "porcupine" | "energy" | "mock"
        sink: 事件下沉对象
        source: 事件来源标识
        **kwargs: 透传给具体检测器
    Returns:
        WakeWordDetector 实例
    """
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


# ---------------------------------------------------------------------------
# 工具函数：列出可用音频设备
# ---------------------------------------------------------------------------

def list_audio_devices() -> None:
    """列出系统所有可用的音频输入设备（跨平台）。"""
    try:
        import sounddevice
        print("可用音频输入设备（sounddevice）：", flush=True)
        sounddevice.query_devices(kind="input")
        return
    except ImportError:
        pass

    try:
        import pyaudio
        p = pyaudio.PyAudio()
        print("可用音频输入设备（pyaudio）：", flush=True)
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']} (channels={info['maxInputChannels']}, rate={info['defaultSampleRate']})", flush=True)
        p.terminate()
        return
    except ImportError:
        pass

    print("无可用音频设备，请安装 sounddevice（pip install sounddevice）或 pyaudio。", flush=True)
