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

import queue
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


@dataclass
class _TTSJob:
    text: str
    voice: Any = None
    volume: Any = None
    speed: Any = None
    done: threading.Event | None = None


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

    WAKE_LATEST_RAW_WAV = "wake_latest.wav"
    WAKE_LATEST_ASR_WAV = "wake_latest_asr.wav"

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
        post_wake_capture_sec: int | None = None,
        post_ack_listen_delay_sec: float = 0.5,
        post_ack_user_window_sec: float = 2.5,
        capture_mode: str = "vad",
        max_capture_duration_sec: float = 15.0,
        silence_duration_sec: float = 0.8,
        cloud_streaming: bool = True,
        keep_voice_recordings: bool = False,
        playback_recording: bool = False,
        voice_record_dir: str | Path = "data/voice_recordings",
        wake_record_timing: str = "sync",
        wake_ack_text: str = "我在，请说。",
        wake_ack_mode: str = "local",
        wake_ack_dir: str | Path = "assets/voice/wake_ack",
        audio_dir: str | Path = "data/",
        debug: bool = False,
        voice_debug_dir: str | Path = "data/voice_debug",
        voice_debug_log: bool = True,
        wake_alsa_device: str | None = None,
        persistent_capture: bool = True,
        wake_echo_trim: bool = False,
    ) -> None:
        self._sink = sink
        self._detector = detector
        self._recognizer = recognizer
        self._tts_backend = tts_backend
        self._alsa_device = alsa_device
        self._sample_rate = int(sample_rate)
        self._capture_duration_sec = int(capture_duration_sec)
        self._post_wake_capture_sec = (
            int(post_wake_capture_sec)
            if post_wake_capture_sec is not None
            else min(8, max(4, self._capture_duration_sec))
        )
        self._post_ack_listen_delay_sec = max(0.0, float(post_ack_listen_delay_sec))
        self._post_ack_user_window_sec = max(1.0, float(post_ack_user_window_sec))
        self._capture_mode = str(capture_mode or "vad").strip().lower()
        self._max_capture_duration_sec = max(2.0, float(max_capture_duration_sec))
        self._silence_duration_sec = max(0.3, float(silence_duration_sec))
        self._cloud_streaming = bool(cloud_streaming)
        self._keep_voice_recordings = bool(keep_voice_recordings)
        self._playback_recording = bool(playback_recording)
        self._voice_record_dir = Path(voice_record_dir)
        self._wake_record_timing = str(wake_record_timing or "sync").strip().lower()
        self.last_recording_path: Path | None = None
        self._wake_ack_text = str(wake_ack_text).strip() or "我在，请说。"
        self._wake_ack_mode = str(wake_ack_mode).strip().lower() or "local"
        self._wake_ack_dir = Path(wake_ack_dir)
        self._audio_dir = Path(audio_dir)
        self._debug = bool(debug)
        self._local_wake_ack = None
        from src.adapters.voice.voice_debug_log import configure_voice_debug

        self._voice_debug_manager = configure_voice_debug(
            log_dir=voice_debug_dir,
            enabled=bool(voice_debug_log),
            verbose=bool(debug),
        )
        self._session_debug: Any | None = None
        self._wake_alsa_device = (wake_alsa_device or alsa_device or "").strip()
        self._persistent_capture_enabled = bool(persistent_capture)
        self._wake_echo_trim = bool(wake_echo_trim)
        self._persistent_mic = None
        if self._persistent_capture_enabled:
            from src.adapters.voice.persistent_mic import PersistentMicCapture

            self._persistent_mic = PersistentMicCapture(
                alsa_device=self._alsa_device,
                sample_rate=self._sample_rate,
            )

        self._running = False
        self._thread: threading.Thread | None = None
        self._wake_flow_thread: threading.Thread | None = None
        self._session_counter = 0
        self._current_session: VoiceSession | None = None
        self._session_lock = threading.Lock()

        # TTS 播报队列：流式逐句按顺序播放，避免下一句打断上一句
        self._tts_queue: queue.Queue[_TTSJob | None] = queue.Queue()
        self._tts_worker: threading.Thread | None = None
        self._tts_running = False

        # 动作回调钩子
        self._on_speak: list[Callable[[str], None]] = []

    def _wake_raw_audio_path(self) -> Path:
        return self._audio_dir / self.WAKE_LATEST_RAW_WAV

    def _wake_asr_audio_path(self) -> Path:
        return self._audio_dir / self.WAKE_LATEST_ASR_WAV

    def _cleanup_stale_wake_artifacts(self) -> None:
        """删除旧版按 session 命名的录音，只保留 wake_latest*。"""
        from src.adapters.voice.voice_debug_log import cleanup_legacy_voice_debug

        cleanup_legacy_voice_debug(self._voice_debug_manager.log_dir)
        patterns = (
            "wake_session_*.wav",
            "wake_session_*.trim.wav",
            "wake_*.trim.wav",
        )
        for pattern in patterns:
            for path in self._audio_dir.glob(pattern):
                if path.name in {self.WAKE_LATEST_RAW_WAV, self.WAKE_LATEST_ASR_WAV}:
                    continue
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        if self._keep_voice_recordings:
            for path in self._voice_record_dir.glob("wake_*.wav"):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def is_tts_running(self) -> bool:
        """返回 TTS 是否正在播报或队列中仍有待播句子。"""
        return self._tts_running or not self._tts_queue.empty()

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动后台唤醒词检测线程。"""
        if self._running:
            return
        self._running = True
        self._cleanup_stale_wake_artifacts()
        self._ensure_tts_worker()

        if self._detector is not None:
            self._detector.on_wake(self._on_wake_detected)
            self._detector.start()
            self._log("唤醒词检测器已启动。")

        self.preload_wake_ack()
        if self._persistent_mic is not None:
            if self._persistent_mic.start():
                self._info(
                    f"摄像头麦常驻采集已启动（{self._alsa_device}），唤醒后可立即录音"
                )
            else:
                self._info(
                    f"常驻采集启动失败，将回退普通 arecord：{self._persistent_mic.last_error}"
                )
        self._log("BoardVoiceAdapter 已启动。")

    def preload_wake_ack(self) -> int:
        """启动时预加载本地唤醒应答 WAV 到内存。"""
        if self._wake_ack_mode != "local":
            return 0
        from src.adapters.voice.local_wake_ack import LocalWakeAckPlayer, missing_ack_files

        tts_playback = None
        if self._tts_backend is not None:
            tts_playback = getattr(self._tts_backend, "alsa_playback_device", None)

        self._local_wake_ack = LocalWakeAckPlayer(
            ack_dir=self._wake_ack_dir,
            alsa_playback_device=tts_playback,
            prefer_capture_device=self._alsa_device,
        )
        count = self._local_wake_ack.preload()
        if count == 0:
            missing = missing_ack_files(self._wake_ack_dir)
            print(
                "[BoardVoiceAdapter] 本地唤醒应答未就绪，将回退云端 TTS。\n"
                f"  缺失：{', '.join(missing) or 'manifest/目录为空'}\n"
                "  请运行：python scripts/generate_wake_ack_audio.py",
                flush=True,
            )
        else:
            print(f"[BoardVoiceAdapter] 已预加载 {count} 条本地唤醒应答。", flush=True)
        return count

    def stop(self) -> None:
        """停止后台线程和唤醒词检测。"""
        self._running = False
        if self._tts_worker is not None and self._tts_worker.is_alive():
            self._tts_queue.put(None)
            self._tts_worker.join(timeout=30.0)
            self._tts_worker = None
        if self._wake_flow_thread is not None:
            self._wake_flow_thread.join(timeout=15.0)
            self._wake_flow_thread = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._persistent_mic is not None:
            self._persistent_mic.stop()
        if self._detector is not None:
            self._detector.stop()
        self._log("BoardVoiceAdapter 已停止。")

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # 唤醒词命中处理
    # ------------------------------------------------------------------

    def _ensure_tts_worker(self) -> None:
        if self._tts_worker is not None and self._tts_worker.is_alive():
            return
        self._tts_worker = threading.Thread(
            target=self._tts_worker_loop,
            name="BoardVoiceTTS",
            daemon=True,
        )
        self._tts_worker.start()

    def _tts_worker_loop(self) -> None:
        while True:
            job = self._tts_queue.get()
            try:
                if job is None:
                    break
                self._tts_running = True
                try:
                    self._synthesize_and_play(
                        job.text,
                        voice=job.voice,
                        volume=job.volume,
                        speed=job.speed,
                    )
                finally:
                    self._tts_running = False
                    if job.done is not None:
                        job.done.set()
            finally:
                self._tts_queue.task_done()

    def _enqueue_tts(
        self,
        text: str,
        *,
        voice: Any = None,
        volume: Any = None,
        speed: Any = None,
        wait: bool = False,
        timeout_sec: float = 300.0,
    ) -> None:
        self._ensure_tts_worker()
        done = threading.Event() if wait else None
        self._tts_queue.put(
            _TTSJob(text=text, voice=voice, volume=volume, speed=speed, done=done)
        )
        if wait and done is not None:
            if not done.wait(timeout=timeout_sec):
                self._info("[WARN] TTS 播报等待超时。")

    def _wake_shares_capture_device(self) -> bool:
        """唤醒监听是否与用户录音同一张声卡（同卡才需停检测）。"""
        if self._detector is None:
            return False
        wake_dev = (self._wake_alsa_device or getattr(self._detector, "_alsa_device", "") or "").strip()
        cap_dev = (self._alsa_device or "").strip()
        return bool(wake_dev and cap_dev and wake_dev == cap_dev)

    def _on_wake_detected(self, wake_event: Any) -> None:
        """唤醒命中：即时播报 → 录下用户后续话语 → 送 ASR/Agent。"""
        if self._wake_flow_thread is not None and self._wake_flow_thread.is_alive():
            self._log("正在录音/处理上一轮唤醒，忽略重复触发。")
            return
        self._wake_flow_thread = threading.Thread(
            target=self._handle_wake_flow,
            args=(wake_event,),
            name="BoardVoiceWakeFlow",
            daemon=True,
        )
        self._wake_flow_thread.start()

    def _handle_wake_flow(self, wake_event: Any) -> None:
        """唤醒命中：立即从常驻麦/普通麦采集 → 后台 ASR/Agent/TTS。"""
        detector_was_running = self._detector is not None and self._detector.is_running()
        stop_detector = detector_was_running and self._wake_shares_capture_device()
        session: VoiceSession | None = None
        audio_path: Path | None = None
        raw_audio_path: Path | None = None
        session_debug: Any | None = None
        flow_status = "failed"
        try:
            if stop_detector and self._detector is not None:
                self._detector.stop()
            self._info(f"唤醒词命中：{wake_event.keyword}")

            with self._session_lock:
                self._session_counter += 1
                session = VoiceSession(
                    session_id=f"session_{self._session_counter}_{int(time.time())}",
                    start_time=time.time(),
                    is_active=True,
                )

            session_debug = self._voice_debug_manager.open_session(session.session_id)
            self._session_debug = session_debug
            session_debug.save_json(
                "pipeline_config.json",
                {
                    "alsa_device": self._alsa_device,
                    "wake_record_timing": self._wake_record_timing,
                    "capture_mode": self._capture_mode,
                    "post_wake_capture_sec": self._post_wake_capture_sec,
                    "post_ack_delay_sec": self._post_ack_listen_delay_sec,
                    "wake_ack_mode": self._wake_ack_mode,
                    "wake_echo_trim": self._wake_echo_trim,
                },
            )
            session_debug.info(
                "wake_hit",
                keyword=getattr(wake_event, "keyword", ""),
                detector_stopped=stop_detector,
                persistent_mic=self._persistent_mic is not None and self._persistent_mic.is_running,
            )

            audio_path, raw_audio_path, flow_status = self._run_wake_capture(session, session_debug)
            session.audio_path = audio_path
            self._emit_voice_input_stopped(session.session_id)
        except Exception as exc:
            if session_debug is not None:
                session_debug.exception("wake_flow_exception", exc)
            self._info(f"[ERROR] 唤醒流程异常：{exc}")
            raise
        finally:
            if stop_detector and self._detector is not None and self._running:
                self._detector.start()
            self._wake_flow_thread = None

        if session is None:
            return

        worker = threading.Thread(
            target=self._process_wake_recognition,
            args=(session, audio_path, raw_audio_path, session_debug, flow_status),
            name=f"WakeASR-{session.session_id}",
            daemon=True,
        )
        worker.start()

    def _build_wake_vad_config(self, ack_trim_sec: float) -> Any:
        from src.adapters.voice.vad_recorder import VadConfig

        max_sec = float(max(self._post_wake_capture_sec, self._max_capture_duration_sec))
        guard = max(1.0, float(ack_trim_sec) + 0.5)
        return VadConfig(
            sample_rate=self._sample_rate,
            silence_duration_sec=self._silence_duration_sec,
            max_duration_sec=max_sec,
            initial_timeout_sec=max(12.0, max_sec + 2.0),
            pre_roll_ms=450,
            min_elapsed_before_end_sec=guard,
        )

    def _run_wake_capture(
        self,
        session: VoiceSession,
        session_debug: Any,
    ) -> tuple[Path | None, Path | None, str]:
        """低延迟采集：常驻麦 + VAD 说完停录 + 应答并行播放。"""
        ack_thread, ack_trim_sec = self._start_wake_ack_parallel()
        ack_started_at = time.monotonic()
        vad_cfg = self._build_wake_vad_config(ack_trim_sec)
        use_persistent = self._ensure_persistent_mic()

        if use_persistent:
            self._info(
                f"麦克风已在听（{self._alsa_device}），说完自动停录"
                f"（最长 {vad_cfg.max_duration_sec:.0f}s，静音 {vad_cfg.silence_duration_sec:.1f}s）；"
                f"应答并行播放，请对着摄像头说话"
            )
        else:
            self._prepare_mic_for_capture(detector_already_stopped=True)
            self._info(
                f"开始 VAD 录音（{self._alsa_device}），说完自动停录…"
            )

        self._emit_voice_input_started(session.session_id)
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self._wake_raw_audio_path()

        if use_persistent:
            result, recorded, stop_reason = self._persistent_mic.record_until_silence(
                raw_path,
                config=vad_cfg,
                pre_roll_sec=0.45,
            )
            if session_debug is not None:
                session_debug.info(
                    "persistent_vad_ok" if result else "persistent_vad_failed",
                    duration_sec=round(recorded, 2),
                    stop_reason=stop_reason,
                    error=getattr(self._persistent_mic, "last_error", ""),
                )
            raw_audio_path = result
        else:
            from src.adapters.voice.mic_arbitration import mic_capture_lock
            from src.adapters.voice.vad_recorder import record_audio_vad_wav

            try:
                with mic_capture_lock():
                    result, recorded, stop_reason = record_audio_vad_wav(
                        raw_path,
                        alsa_device=self._alsa_device,
                        config=vad_cfg,
                        prepare_device=False,
                        debug=session_debug,
                    )
                if session_debug is not None:
                    session_debug.info(
                        "vad_capture_ok" if result else "vad_capture_failed",
                        duration_sec=round(recorded, 2),
                        stop_reason=stop_reason,
                    )
                raw_audio_path = result
            except TimeoutError:
                self._info("[ERROR] 麦克风被占用，VAD 录音未开始。")
                raw_audio_path = None

        if self._wake_echo_trim and ack_thread is not None:
            ack_thread.join(timeout=8.0)
        measured_ack = time.monotonic() - ack_started_at if ack_started_at else ack_trim_sec

        if raw_audio_path is None or not Path(raw_audio_path).is_file():
            return None, None, "failed"

        if self._wake_echo_trim and (use_persistent or self._wake_record_timing == "sync"):
            audio_path, raw_audio_path = self._prepare_asr_audio(raw_audio_path, measured_ack)
        else:
            audio_path = raw_audio_path
            if session_debug is not None and raw_audio_path is not None:
                from src.adapters.voice.vad_recorder import wav_duration_sec, wav_peak_rms

                session_debug.info(
                    "asr_use_raw_no_trim",
                    duration_sec=round(wav_duration_sec(raw_audio_path), 2),
                    peak_rms=round(wav_peak_rms(raw_audio_path), 1),
                )

        status = "capture_ok" if audio_path is not None else "failed"
        return audio_path, raw_audio_path, status

    def _ensure_persistent_mic(self) -> bool:
        """确保摄像头麦常驻流可用（失败则回退普通 arecord）。"""
        if self._persistent_mic is None or not self._persistent_capture_enabled:
            return False
        if not self._persistent_mic.is_running:
            if not self._persistent_mic.start():
                self._info(
                    f"常驻麦重启失败，回退普通录音：{self._persistent_mic.last_error}"
                )
                return False
        return True

    def _process_wake_recognition(
        self,
        session: VoiceSession,
        audio_path: Path | None,
        raw_audio_path: Path | None,
        session_debug: Any | None,
        initial_status: str,
    ) -> None:
        """后台 ASR + Agent（不阻塞下一轮唤醒监听）。"""
        flow_status = initial_status
        try:
            if audio_path is None or not audio_path.is_file():
                if session_debug is not None:
                    session_debug.error("capture_failed", reason="no_audio_file")
                self._info("语音采集失败，未生成音频文件。")
                return

            self._finalize_capture_audio(
                audio_path,
                session.session_id,
                raw_path=raw_audio_path,
            )

            text = self._recognize_audio(audio_path, debug=session_debug)
            if text:
                flow_status = "asr_ok"
                self._info(f"ASR 识别：{text!r}")
                self._emit_speech_recognized(text, session.session_id)
            else:
                flow_status = "asr_empty"
                self._info("未识别到有效语音；请连着唤醒词或应答后尽快说问题。")
        finally:
            if session_debug is not None:
                session_debug.finish(
                    status=flow_status,
                    audio_path=str(audio_path) if audio_path else "",
                )
                self._info(f"语音调试日志：{session_debug.dir / 'voice.log'}")
            self._session_debug = None
            if audio_path is not None and audio_path.is_file():
                self.last_recording_path = audio_path.resolve()
            with self._session_lock:
                session.is_active = False

    def _prepare_asr_audio(
        self,
        raw_path: Path | None,
        measured_ack_sec: float,
    ) -> tuple[Path | None, Path | None]:
        """分析原始录音，裁掉应答回声，返回 (asr_path, raw_path)。"""
        if raw_path is None or not raw_path.is_file():
            return None, None

        from src.adapters.voice.vad_recorder import (
            trim_wav_leading,
            wav_duration_sec,
            wav_peak_rms,
        )

        duration = wav_duration_sec(raw_path)
        peak = wav_peak_rms(raw_path)
        self._info(
            f"原始录音 {duration:.1f}s，峰值RMS={peak:.0f}（阈值约450；过低请检查 --voice-alsa-device）"
        )
        if peak < 120:
            self._info("⚠ 原始录音几乎无声，请运行 --list-audio-devices 确认摄像头麦克风设备。")

        trim_sec = min(measured_ack_sec + 0.05, duration * 0.5)
        if trim_sec < 0.15 or duration - trim_sec < 0.4:
            return raw_path, raw_path

        trimmed = self._wake_asr_audio_path()
        trim_wav_leading(raw_path, trim_sec, output_path=trimmed)
        trim_peak = wav_peak_rms(trimmed)
        trim_dur = wav_duration_sec(trimmed)
        self._info(
            f"ASR 用音频 {trim_dur:.1f}s（裁掉应答约 {trim_sec:.1f}s），峰值RMS={trim_peak:.0f}"
        )
        return trimmed, raw_path

    def _finalize_capture_audio(
        self,
        audio_path: Path,
        session_id: str,
        *,
        raw_path: Path | None = None,
    ) -> Path | None:
        """录音后：可选回放；保留目录下仅 latest*.wav（覆盖写入）。"""
        asr_path = Path(audio_path)
        raw = Path(raw_path) if raw_path is not None else asr_path
        if not raw.is_file() and not asr_path.is_file():
            return None

        self.last_recording_path = (
            raw.resolve() if raw.is_file() else asr_path.resolve()
        )

        playback_target = raw if raw.is_file() else asr_path
        if self._playback_recording:
            self._info("回放原始录音（含应答回声，便于排查）")
            self._playback_recording_file(playback_target)

        if self._keep_voice_recordings or self._playback_recording:
            self._voice_record_dir.mkdir(parents=True, exist_ok=True)
            if raw.is_file() and raw.resolve() != asr_path.resolve():
                latest_raw = self._voice_record_dir / "latest_raw.wav"
                latest_raw.write_bytes(raw.read_bytes())
                self._info(f"原始录音：{latest_raw.resolve()}")
            latest = self._voice_record_dir / "latest.wav"
            latest.write_bytes(asr_path.read_bytes())
            self.last_recording_path = (
                (self._voice_record_dir / "latest_raw.wav").resolve()
                if (self._voice_record_dir / "latest_raw.wav").is_file()
                else latest.resolve()
            )
            self._info(
                f"ASR 用录音：{latest.resolve()}；原始："
                f"{self._voice_record_dir / 'latest_raw.wav'}"
            )
            self._print_replay_hint(self.last_recording_path)

        return asr_path

    def _playback_recording_file(self, path: Path) -> None:
        from src.adapters.voice.audio_playback import play_wav_file

        tts_playback = None
        if self._tts_backend is not None:
            tts_playback = getattr(self._tts_backend, "alsa_playback_device", None)
        try:
            self._info(f"回放刚才的录音：{path.name}")
            play_wav_file(
                path,
                alsa_playback_device=tts_playback,
                prefer_capture_device=self._alsa_device,
            )
        except Exception as exc:
            self._info(f"录音回放失败：{exc}")
            self._print_replay_hint(path)

    def _print_replay_hint(self, path: Path) -> None:
        playback = None
        if self._tts_backend is not None:
            playback = getattr(self._tts_backend, "alsa_playback_device", None)
        if not playback:
            from src.adapters.voice.alsa_audio_devices import playback_device_for_tts

            playback = playback_device_for_tts(
                prefer_capture_device=self._alsa_device,
                split_input_output=True,
            )
        device = playback or "default"
        self._info(f"手动回放：aplay -D {device} '{path.resolve()}'")

    def replay_last_recording(self) -> bool:
        """CLI /voice_replay：回放最近一次保留或刚录制的音频。"""
        path = self.last_recording_path
        if path is None or not path.is_file():
            return False
        self._playback_recording_file(path)
        return True

    def _start_wake_ack_parallel(self) -> tuple[threading.Thread | None, float]:
        """与录音并行播放唤醒应答，返回 (线程, 用于裁切回声的秒数)。"""
        if self._wake_ack_mode == "off":
            return None, 0.0

        if self._wake_ack_mode == "local" and self._local_wake_ack is not None:
            thread, duration, text = self._local_wake_ack.play_async(strategy="random")
            if thread is not None:
                self._log(f"并行本地应答：{text!r} (~{duration:.1f}s)")
                return thread, duration

        def _cloud_ack() -> None:
            self._tts_running = True
            try:
                if self._tts_backend is None:
                    print(f"[BoardVoiceAdapter] {self._wake_ack_text}", flush=True)
                    return
                self._tts_backend.speak(
                    self._wake_ack_text,
                    voice=None,
                    volume=None,
                    speed=None,
                )
            finally:
                self._tts_running = False

        self._log(f"并行云端应答：{self._wake_ack_text!r}")
        thread = threading.Thread(target=_cloud_ack, name="WakeAckCloud", daemon=True)
        thread.start()
        return thread, max(1.5, len(self._wake_ack_text) * 0.15)

    def _prepare_mic_for_capture(self, *, detector_already_stopped: bool = False) -> None:
        """停掉唤醒检测占用的 arecord，并释放目标麦克风（常驻麦模式下不 kill 全局 arecord）。"""
        if self._persistent_mic is not None and self._persistent_mic.is_running:
            if (
                not detector_already_stopped
                and self._detector is not None
                and self._detector.is_running()
                and self._wake_shares_capture_device()
            ):
                self._detector.stop()
            if self._session_debug is not None:
                self._session_debug.step(
                    "skip_release_persistent_mic",
                    device=self._alsa_device,
                    important=True,
                )
            return

        if (
            not detector_already_stopped
            and self._detector is not None
            and self._detector.is_running()
        ):
            self._detector.stop()
        from src.adapters.voice.baidu_asr_backend import release_capture_device

        if self._session_debug is not None:
            self._session_debug.step(
                "release_capture_device",
                device=self._alsa_device,
                important=True,
            )
        release_capture_device(self._alsa_device, settle_ms=250, allow_global_pkill=True)

    def _play_wake_ack(self) -> None:
        """唤醒后立刻播放本地预加载短句；失败时再走云端 TTS。"""
        if self._wake_ack_mode == "off":
            return

        self._tts_running = True
        try:
            if self._wake_ack_mode == "local" and self._local_wake_ack is not None:
                played_text = self._local_wake_ack.play(strategy="random")
                if played_text:
                    self._log(f"本地唤醒应答：{played_text!r}")
                    return

            self._log(f"云端唤醒应答：{self._wake_ack_text!r}")
            if self._tts_backend is None:
                print(f"[BoardVoiceAdapter] {self._wake_ack_text}", flush=True)
                return
            self._tts_backend.speak(
                self._wake_ack_text,
                voice=None,
                volume=None,
                speed=None,
            )
        except Exception as exc:
            self._log(f"[ERROR] 唤醒应答失败：{exc}")
            print(f"[BoardVoiceAdapter] {self._wake_ack_text}", flush=True)
        finally:
            self._tts_running = False

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

    # ------------------------------------------------------------------
    # 语音识别
    # ------------------------------------------------------------------

    def _run_capture_session(self, session: VoiceSession, *, duration_sec: int | None = None) -> None:
        """执行一轮完整的语音采集 + 识别流程。"""
        self._emit_voice_input_started(session.session_id)

        capture_sec = duration_sec if duration_sec is not None else self._capture_duration_sec
        audio_path = self._capture_audio(session, duration_sec=capture_sec)
        session.audio_path = audio_path

        self._emit_voice_input_stopped(session.session_id)

        if audio_path is None or not audio_path.is_file():
            self._log("语音采集失败，未生成音频文件。")
            with self._session_lock:
                session.is_active = False
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

    def _capture_audio(
        self,
        session: VoiceSession,
        *,
        duration_sec: int | None = None,
        vad_min_elapsed_before_end_sec: float = 0.0,
        vad_pre_roll_ms: int = 200,
        vad_initial_timeout_sec: float | None = None,
        skip_device_release: bool = False,
        on_capture_started: Callable[[], None] | None = None,
        force_vad: bool = False,
        vad_max_duration_sec: float | None = None,
        vad_silence_duration_sec: float | None = None,
    ) -> Path | None:
        """跨平台录音，返回音频文件路径。"""
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self._wake_raw_audio_path()

        fixed_sec = duration_sec if duration_sec is not None else self._capture_duration_sec
        use_vad = self._capture_mode == "vad" or force_vad
        try:
            from src.adapters.voice.mic_arbitration import mic_capture_lock

            with mic_capture_lock():
                if use_vad:
                    from src.adapters.voice.vad_recorder import VadConfig, record_audio_vad_wav

                    initial_timeout = (
                        float(vad_initial_timeout_sec)
                        if vad_initial_timeout_sec is not None
                        else max(6.0, float(fixed_sec))
                    )
                    max_duration = (
                        float(vad_max_duration_sec)
                        if vad_max_duration_sec is not None
                        else max(float(fixed_sec), self._max_capture_duration_sec)
                    )
                    silence_sec = (
                        float(vad_silence_duration_sec)
                        if vad_silence_duration_sec is not None
                        else self._silence_duration_sec
                    )
                    result, recorded_sec, stop_reason = record_audio_vad_wav(
                        audio_path,
                        alsa_device=self._alsa_device,
                        prepare_device=not skip_device_release,
                        on_first_frame=on_capture_started,
                        debug=self._session_debug,
                        config=VadConfig(
                            sample_rate=self._sample_rate,
                            silence_duration_sec=silence_sec,
                            max_duration_sec=max_duration,
                            initial_timeout_sec=initial_timeout,
                            pre_roll_ms=int(vad_pre_roll_ms),
                            min_elapsed_before_end_sec=max(0.0, float(vad_min_elapsed_before_end_sec)),
                        ),
                    )
                    if result is not None and result.is_file() and result.stat().st_size > 1000:
                        self._info(
                            f"录音完成（VAD {recorded_sec:.1f}s，{stop_reason}）："
                            f"{result.stat().st_size} bytes"
                        )
                        if recorded_sec >= max_duration - 0.15:
                            self._info(
                                f"已达最长 {max_duration:.0f}s；"
                                f"若未识别完整，请缩短句子并在句末稍停"
                            )
                        return result
                    if self._session_debug is not None:
                        self._session_debug.warn(
                            "vad_capture_failed",
                            stop_reason=stop_reason,
                            recorded_sec=recorded_sec,
                        )
                    return None

                from src.adapters.voice.baidu_asr_backend import record_audio_wav

                result = record_audio_wav(
                    output_path=audio_path,
                    duration_sec=fixed_sec,
                    sample_rate=self._sample_rate,
                    alsa_device=self._alsa_device,
                    prepare_device=not skip_device_release,
                    debug=self._session_debug,
                )
                if result is not None and result.is_file() and result.stat().st_size > 1000:
                    self._info(f"录音完成（固定 {fixed_sec}s）：{result.stat().st_size} bytes")
                    return result
        except TimeoutError:
            self._info("[ERROR] 麦克风被占用，录音未开始。")
        except Exception as exc:
            self._info(f"[ERROR] 录音失败：{exc}")
        return None

    def _capture_fixed_audio(
        self,
        session: VoiceSession,
        *,
        duration_sec: int | None = None,
        skip_device_release: bool = False,
        debug: Any | None = None,
    ) -> Path | None:
        """固定时长录音（after_ack 主路径 / VAD 失败兜底）。"""
        fixed_sec = duration_sec if duration_sec is not None else self._post_wake_capture_sec
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self._wake_raw_audio_path()
        dbg = debug if debug is not None else self._session_debug
        try:
            from src.adapters.voice.mic_arbitration import mic_capture_lock

            with mic_capture_lock():
                from src.adapters.voice.baidu_asr_backend import record_audio_wav

                result = record_audio_wav(
                    output_path=audio_path,
                    duration_sec=fixed_sec,
                    sample_rate=self._sample_rate,
                    alsa_device=self._alsa_device,
                    prepare_device=not skip_device_release,
                    debug=dbg,
                )
                if result is not None and result.is_file() and result.stat().st_size > 1000:
                    from src.adapters.voice.vad_recorder import wav_duration_sec, wav_peak_rms

                    peak = wav_peak_rms(result)
                    dur = wav_duration_sec(result)
                    self._info(
                        f"录音完成（固定 {fixed_sec}s）：{result.stat().st_size} bytes，"
                        f"时长 {dur:.1f}s，峰值RMS={peak:.0f}"
                    )
                    if dbg is not None:
                        dbg.info(
                            "fixed_capture_ok",
                            bytes=result.stat().st_size,
                            duration_sec=round(dur, 2),
                            peak_rms=round(peak, 1),
                        )
                    return result
                if dbg is not None:
                    dbg.error("fixed_capture_empty", path=str(audio_path))
        except TimeoutError:
            self._info("[ERROR] 麦克风被占用，录音未开始。")
            if dbg is not None:
                dbg.error("mic_capture_lock_timeout")
        except Exception as exc:
            self._info(f"[ERROR] 录音失败：{exc}")
            if dbg is not None:
                dbg.exception("fixed_capture_exception", exc)
        return None

    def _recognize_audio(self, audio_path: Path, *, debug: Any | None = None) -> str:
        """调用 ASR 后端识别音频。"""
        dbg = debug if debug is not None else self._session_debug
        if self._recognizer is None:
            self._log("[WARN] 未配置 ASR recognizer，跳过识别。")
            if dbg is not None:
                dbg.warn("asr_skipped", reason="no_recognizer")
            return ""

        if dbg is not None:
            dbg.step("asr_start", path=str(audio_path.resolve()), important=True)
        try:
            if hasattr(self._recognizer, "recognize_file"):
                text = self._recognizer.recognize_file(audio_path)
            elif hasattr(self._recognizer, "recognize"):
                text = self._recognizer.recognize(str(audio_path))
            else:
                self._log("[WARN] recognizer 不支持 recognize_file 或 recognize 方法。")
                if dbg is not None:
                    dbg.warn("asr_skipped", reason="unsupported_recognizer")
                return ""
            self._log(f"ASR 识别结果：{text!r}")
            if dbg is not None:
                dbg.info("asr_result", text=text)
            return text.strip()
        except Exception as exc:
            self._info(f"[ERROR] ASR 识别失败：{exc}")
            if dbg is not None:
                dbg.exception("asr_failed", exc)
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

        stream_sink = None
        llm_service = getattr(self._sink, "llm_service", None)
        if self._cloud_streaming and llm_service is not None:
            stream_sink = _StreamingTTSSink(self)
            llm_service.voice_stream_sink = stream_sink
        try:
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
        finally:
            if llm_service is not None:
                llm_service.voice_stream_sink = None

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
        """执行一次完整的 TTS 播报：注入 tts_started → 排队合成播放 → 注入 tts_finished。"""
        text = str(payload.get("text", "")).strip()
        if not text:
            return

        voice = payload.get("voice")
        volume = payload.get("volume")
        speed = payload.get("speed")

        self._info(f"TTS 播报：{text[:80]}{'…' if len(text) > 80 else ''}")
        self._emit_tts_event("tts_started", text)

        for cb in self._on_speak:
            cb(text)

        self._enqueue_tts(text, voice=voice, volume=volume, speed=speed, wait=True)
        self._emit_tts_event("tts_finished", text)

    def _speak_streaming_sentence(self, text: str) -> None:
        """流式逐句播报：入队顺序播放，等上一句播完再播下一句。"""
        self._info(f"流式 TTS：{text[:80]}{'…' if len(text) > 80 else ''}")
        self._enqueue_tts(text, wait=False)

    def _synthesize_and_play(
        self,
        text: str,
        *,
        voice: Any,
        volume: Any,
        speed: Any,
    ) -> None:
        if self._tts_backend is not None:
            try:
                self._tts_backend.speak(text, voice=voice, volume=volume, speed=speed)
            except Exception as exc:
                self._log(f"[ERROR] TTS 播报失败：{exc}")
                print(f"[BoardVoiceAdapter] TTS 播报失败：{exc}", flush=True)
        else:
            self._log(f"[TTS] {text}")

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
                alsa_device=self._alsa_device,
            )
            if result is None or not result.is_file() or result.stat().st_size < 1000:
                self._log("录音文件无效。")
                return None
        except Exception as exc:
            self._log(f"[ERROR] 录音失败：{exc}")
            return None

        self._finalize_capture_audio(result, f"voice_once_{int(time.time())}")
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

    def _info(self, msg: str) -> None:
        """关键语音里程碑：默认也打印，便于板端联调。"""
        print(f"[BoardVoiceAdapter] {msg}", flush=True)
        if self._session_debug is not None:
            self._session_debug.info("board_voice", message=msg)

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = bool(value)


class _StreamingTTSSink:
    """LLM 流式输出时逐句 TTS。"""

    def __init__(self, adapter: BoardVoiceAdapter) -> None:
        self._adapter = adapter
        self.spoke_any = False

    def on_sentence(self, sentence: str) -> None:
        text = str(sentence).strip()
        if not text:
            return
        self._adapter._speak_streaming_sentence(text)
        self.spoke_any = True
