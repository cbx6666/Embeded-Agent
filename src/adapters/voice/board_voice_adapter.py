"""板级语音适配器。

整合三条链路：
1. **唤醒词检测**：后台线程持续监听麦克风，命中唤醒词后触发语音采集
2. **ASR 语音识别**：采集用户语音后调用百度短语音识别生成 `speech_recognized` 事件
3. **TTS 语音播报**：接收 Agent 的 `speak` / `set_tts_*` 动作，执行语音合成与播放

AgentCore 通过 `handle_event_with_results` 接收事件，通过 `execute(action)` 接收动作，
构成完整的双向语音交互闭环。

Usage:
    from src.adapters.voice.baidu_asr_backend import BaiduShortASRBackend
    from src.adapters.voice.baidu_tts_backend import BaiduTTSBackend
    from src.adapters.voice.wake_word_detector import build_wake_word_detector

    asr = BaiduShortASRBackend()
    tts = BaiduTTSBackend()
    detector = build_wake_word_detector(backend="energy")

    adapter = BoardVoiceAdapter(
        sink=core,            # AgentCore
        detector=detector,
        recognizer=asr,
        tts_backend=tts,
    )
    adapter.start()          # 启动后台监听
    # ...
    adapter.stop()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

class EventSink(Protocol):
    """可接收 Agent 标准事件的对象。"""
    def handle_event_with_results(self, event: Any) -> Any: ...


# ---------------------------------------------------------------------------
# 录音会话管理
# ---------------------------------------------------------------------------

@dataclass
class VoiceSession:
    """单次语音采集会话的状态。"""
    session_id: str
    start_time: float
    audio_path: Path | None = None
    is_active: bool = False


# ---------------------------------------------------------------------------
# 主适配器
# ---------------------------------------------------------------------------

class BoardVoiceAdapter:
    """板级语音适配器：整合唤醒词检测、ASR、TTS。

    参数：
        sink: AgentCore 实例，事件将注入此处
        detector: 唤醒词检测器（WakeWordDetector）
        recognizer: 语音识别后端（如 BaiduShortASRBackend）
        tts_backend: 语音合成后端（如 BaiduTTSBackend）
        alsa_device: ALSA 录音设备，默认 plughw:0,0
        sample_rate: 采样率，默认 16000 Hz
        capture_duration_sec: 唤醒后录音时长，默认 10 秒
        audio_dir: 录音文件存放目录，默认 data/
    """

    def __init__(
        self,
        *,
        sink: EventSink | None = None,
        detector: Any | None = None,
        recognizer: Any | None = None,
        tts_backend: Any | None = None,
        alsa_device: str = "plughw:0,0",
        sample_rate: int = 16000,
        capture_duration_sec: int = 10,
        audio_dir: str | Path = "data/",
        debug: bool = False,
    ) -> None:
        self._sink = sink
        self._detector = detector
        self._recognizer = recognizer
        self._tts_backend = tts_backend
        self._alsa_device = alsa_device
        self._sample_rate = int(sample_rate)
        self._capture_duration_sec = int(capture_duration_sec)
        self._audio_dir = Path(audio_dir)
        self._debug = bool(debug)

        self._running = False
        self._thread: threading.Thread | None = None
        self._session_counter = 0
        self._current_session: VoiceSession | None = None
        self._session_lock = threading.Lock()

        # TTS 播报锁：播报时持有，阻止唤醒词检测和录音抢麦克风
        self._tts_lock = threading.Lock()
        self._tts_running = False

        # 动作回调钩子
        self._on_speak: list[Callable[[str], None]] = []

    def is_tts_running(self) -> bool:
        """返回 TTS 是否正在播报。"""
        return self._tts_running

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动后台唤醒词检测线程。"""
        if self._running:
            return
        self._running = True

        if self._detector is not None:
            self._detector.on_wake(self._on_wake_detected)
            self._detector.start()
            self._log("唤醒词检测器已启动。")

        self._thread = threading.Thread(target=self._idle_loop, name="BoardVoiceAdapter", daemon=True)
        self._thread.start()
        self._log("BoardVoiceAdapter 已启动。")

    def stop(self) -> None:
        """停止后台线程和唤醒词检测。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._detector is not None:
            self._detector.stop()
        self._log("BoardVoiceAdapter 已停止。")

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # 唤醒词命中处理
    # ------------------------------------------------------------------

    def _on_wake_detected(self, wake_event: Any) -> None:
        """唤醒词命中后触发语音采集。

        如果 TTS 正在播报（持有 _tts_lock），跳过此次唤醒，
        等待 TTS 结束再继续检测。
        """
        # 非阻塞获取锁：如果 TTS 正在播报，跳过此次唤醒
        acquired = self._tts_lock.acquire(blocking=False)
        if not acquired:
            self._log("唤醒词命中但 TTS 正在播报，跳过本次采集。")
            return
        try:
            self._log(f"唤醒词命中：{wake_event.keyword}，开始采集语音...")
            self._emit_voice_wake_event(wake_event)
            self._start_voice_capture_session()
        finally:
            self._tts_lock.release()

    def _emit_voice_wake_event(self, wake_event: Any) -> None:
        """将 voice_wake_detected 事件注入 AgentCore。"""
        if self._sink is None:
            return
        from src.agent.event.event_model import Event
        self._sink.handle_event_with_results(
            Event(
                type="voice_wake_detected",
                timestamp=int(time.time()),
                payload={
                    "keyword": getattr(wake_event, "keyword", "unknown"),
                    "source": getattr(wake_event, "source", "microphone"),
                    "confidence": getattr(wake_event, "confidence", 1.0),
                },
            )
        )

    def _start_voice_capture_session(self) -> None:
        """启动一轮语音采集会话。"""
        with self._session_lock:
            self._session_counter += 1
            self._current_session = VoiceSession(
                session_id=f"session_{self._session_counter}_{int(time.time())}",
                start_time=time.time(),
                is_active=True,
            )

    # ------------------------------------------------------------------
    # 语音识别
    # ------------------------------------------------------------------

    def _idle_loop(self) -> None:
        """空闲时持续监听，检测到语音时执行识别。"""
        while self._running:
            try:
                session = self._get_active_session()
                if session is not None:
                    self._run_capture_session(session)
                time.sleep(0.5)
            except Exception as exc:
                self._log(f"[ERROR] 语音采集异常：{exc}")
                time.sleep(1.0)

    def _get_active_session(self) -> VoiceSession | None:
        with self._session_lock:
            return self._current_session if self._current_session and self._current_session.is_active else None

    def _run_capture_session(self, session: VoiceSession) -> None:
        """执行一轮完整的语音采集 + 识别流程。"""
        self._emit_voice_input_started(session.session_id)

        audio_path = self._capture_audio(session)
        session.audio_path = audio_path

        self._emit_voice_input_stopped(session.session_id)

        if audio_path is None or not audio_path.is_file():
            self._log("语音采集失败，未生成音频文件。")
            with self._session_lock:
                session.is_active = False
                self._current_session = None
            return

        text = self._recognize_audio(audio_path)
        self._emit_speech_recognized(text, session.session_id)

        # 清理
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass

        with self._session_lock:
            session.is_active = False
            self._current_session = None

    def _capture_audio(self, session: VoiceSession) -> Path | None:
        """跨平台录音，返回音频文件路径。"""
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self._audio_dir / f"wake_{session.session_id}.wav"

        from src.adapters.voice.baidu_asr_backend import record_audio_wav
        try:
            result = record_audio_wav(
                output_path=audio_path,
                duration_sec=self._capture_duration_sec,
                sample_rate=self._sample_rate,
                device=None,
            )
            if result is not None and result.is_file() and result.stat().st_size > 1000:
                self._log(f"录音完成：{result} ({result.stat().st_size} bytes)")
                return result
        except Exception as exc:
            self._log(f"[ERROR] 录音失败：{exc}")
        return None

    def _recognize_audio(self, audio_path: Path) -> str:
        """调用 ASR 后端识别音频。"""
        if self._recognizer is None:
            self._log("[WARN] 未配置 ASR recognizer，跳过识别。")
            return ""

        try:
            if hasattr(self._recognizer, "recognize_file"):
                text = self._recognizer.recognize_file(audio_path)
            elif hasattr(self._recognizer, "recognize"):
                text = self._recognizer.recognize(str(audio_path))
            else:
                self._log("[WARN] recognizer 不支持 recognize_file 或 recognize 方法。")
                return ""
            self._log(f"ASR 识别结果：{text!r}")
            return text.strip()
        except Exception as exc:
            self._log(f"[ERROR] ASR 识别失败：{exc}")
            return ""

    # ------------------------------------------------------------------
    # 事件注入
    # ------------------------------------------------------------------

    def _emit_voice_input_started(self, session_id: str) -> None:
        if self._sink is None:
            return
        from src.agent.event.event_model import Event
        self._sink.handle_event_with_results(
            Event(
                type="voice_input_started",
                timestamp=int(time.time()),
                payload={"source": "board_voice", "session_id": session_id},
            )
        )

    def _emit_voice_input_stopped(self, session_id: str) -> None:
        if self._sink is None:
            return
        from src.agent.event.event_model import Event
        self._sink.handle_event_with_results(
            Event(
                type="voice_input_stopped",
                timestamp=int(time.time()),
                payload={"source": "board_voice", "session_id": session_id},
            )
        )

    def _emit_speech_recognized(self, text: str, session_id: str) -> None:
        if self._sink is None or not text:
            return
        from src.agent.event.event_model import Event
        self._sink.handle_event_with_results(
            Event(
                type="speech_recognized",
                timestamp=int(time.time()),
                payload={
                    "text": text,
                    "source": "board_voice",
                    "session_id": session_id,
                    "confidence": 0.9,
                    "language": "zh",
                },
            )
        )

    # ------------------------------------------------------------------
    # 动作执行（TTS / 参数设置）
    # ------------------------------------------------------------------

    def execute(self, action: Any) -> None:
        """执行 Agent 生成的语音相关动作。

        支持的动作类型：
        - speak: 调用 TTS 合成并播放
        - set_tts_voice: 切换 TTS 音色
        - set_tts_volume: 设置 TTS 音量
        - set_tts_speed: 设置 TTS 语速
        - tts_started / tts_finished: 语音播报状态事件
        """
        action_type = str(getattr(action, "type", ""))
        payload = dict(getattr(action, "payload", {}))

        if action_type == "speak":
            self._do_speak(payload)
        elif action_type == "set_tts_voice":
            voice_id = str(payload.get("voice_id", "0"))
            if self._tts_backend is not None:
                self._tts_backend.set_voice(voice_id)
            self._log(f"TTS 音色切换为：{voice_id}")
        elif action_type == "set_tts_volume":
            volume = int(payload.get("volume", 5))
            if self._tts_backend is not None:
                self._tts_backend.set_volume(volume)
            self._log(f"TTS 音量设置为：{volume}")
        elif action_type == "set_tts_speed":
            speed = float(payload.get("speed", 5))
            if self._tts_backend is not None:
                self._tts_backend.set_speed(speed)
            self._log(f"TTS 语速设置为：{speed}")

    def _do_speak(self, payload: dict[str, Any]) -> None:
        """执行一次完整的 TTS 播报：注入 tts_started → 合成播放 → 注入 tts_finished。

        播报期间持有 _tts_lock，唤醒词检测和录音都会等待此锁，
        确保 TTS 播放时麦克风不被抢占。
        """
        text = str(payload.get("text", "")).strip()
        if not text:
            return

        voice = payload.get("voice")
        volume = payload.get("volume")
        speed = payload.get("speed")

        with self._tts_lock:
            self._tts_running = True
            try:
                # 注入 tts_started
                self._emit_tts_event("tts_started", text)

                for cb in self._on_speak:
                    cb(text)

                if self._tts_backend is not None:
                    try:
                        self._tts_backend.speak(text, voice=voice, volume=volume, speed=speed)
                    except Exception as exc:
                        self._log(f"[ERROR] TTS 播报失败：{exc}")
                else:
                    self._log(f"[TTS] {text}")

                # 注入 tts_finished
                self._emit_tts_event("tts_finished", text)
            finally:
                self._tts_running = False

    def _emit_tts_event(self, event_type: str, text: str) -> None:
        if self._sink is None:
            return
        from src.agent.event.event_model import Event
        self._sink.handle_event_with_results(
            Event(
                type=event_type,
                timestamp=int(time.time()),
                payload={"text": text, "source": "board_voice"},
            )
        )

    # ------------------------------------------------------------------
    # 一次性格式（不启动后台线程）
    # ------------------------------------------------------------------

    def run_recognize_once(self) -> Any | None:
        """立即执行一次语音识别（不依赖后台唤醒），返回 speech_recognized 事件或 None。

        如果 TTS 正在播报，等待其结束后再录音。
        """
        # 等待 TTS 播报结束
        if self._tts_running:
            self._log("TTS 正在播报，等待结束后再录音...")
            while self._tts_running:
                time.sleep(0.1)

        from src.adapters.voice.baidu_asr_backend import record_audio_wav
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self._audio_dir / "voice_once.wav"

        try:
            result = record_audio_wav(
                output_path=audio_path,
                duration_sec=self._capture_duration_sec,
                sample_rate=self._sample_rate,
                device=None,
            )
            if result is None or not result.is_file() or result.stat().st_size < 1000:
                self._log("录音文件无效。")
                return None
        except Exception as exc:
            self._log(f"[ERROR] 录音失败：{exc}")
            return None

        text = self._recognize_audio(result)
        if not text:
            return None

        from src.agent.event.event_model import Event
        return Event(
            type="speech_recognized",
            timestamp=int(time.time()),
            payload={
                "text": text,
                "source": "board_voice_once",
                "confidence": 0.9,
                "language": "zh",
            },
        )

    # ------------------------------------------------------------------
    # 后台循环支持（可选，不依赖唤醒词时持续采集）
    # ------------------------------------------------------------------

    def start_background_loop(self, interval_sec: float = 1.0) -> None:
        """启动后台循环：每隔 interval_sec 执行一次语音识别并注入 Agent。"""
        self._running = True
        self._thread = threading.Thread(
            target=self._background_loop,
            args=(interval_sec,),
            name="BoardVoiceAdapter-bg",
            daemon=True,
        )
        self._thread.start()
        self._log(f"后台语音循环已启动，间隔 {interval_sec}s。")

    def stop_background_loop(self) -> None:
        self._running = False

    def _background_loop(self, interval_sec: float) -> None:
        while self._running:
            event = self.run_recognize_once()
            if event is not None and self._sink is not None:
                self._sink.handle_event_with_results(event)
            time.sleep(interval_sec)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._debug:
            print(f"[BoardVoiceAdapter] {msg}", flush=True)

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = bool(value)
