from __future__ import annotations

"""VoiceRuntime：语音子系统编排器（单输入流 + 单播放队列 + 状态机）。"""

import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from src.adapters.voice.arbitration.session_probe import VoiceSessionProbe
from src.adapters.voice.arbitration.tts_job_policy import TTSJobPriority, resolve_job_spec
from src.adapters.voice.arbitration.voice_arbiter import ArbiterAction, VoiceInteractionArbiter
from src.adapters.voice.bridge.agent_bridge import AgentBridge
from src.adapters.voice.input.audio_input_manager import AudioInputManager
from src.adapters.voice.runtime.logger import set_voice_log_hook, voice_log
from src.adapters.voice.runtime.session import VoiceSession
from src.adapters.voice.runtime.state_machine import VoiceSessionStateMachine, VoiceState
from src.adapters.voice.tts.playback_manager import TTSPlaybackManager
from src.adapters.voice.vad.recorder import VadConfig
from src.adapters.voice.wake.local_wake_ack import DEFAULT_WAKE_ACK_DIR, DEFAULT_WAKE_ACK_TEXT

_WAKE_ACK_ECHO_NORMALIZED = frozenset(
    {"我在请说", "在的请说", "嗯你说", "我在", "在的", "嗯"}
)

_AUTONOMOUS_REASONS = frozenset(
    {
        "distraction_reminder",
        "rest_reminder",
        "emotion_reminder",
        "posture_reminder",
        "environment_warning",
        "status_report",
        "media_suggestion",
        "joke_reminder",
    }
)


class VoiceRuntime:
    """整合 AudioInputManager、唤醒、VAD、ASR、TTSPlaybackManager 与 Agent 桥接。"""

    WAKE_LATEST_RAW_WAV = "wake_latest.wav"

    def __init__(
        self,
        *,
        sink: Any | None = None,
        wake_detector: Any | None = None,
        recognizer: Any | None = None,
        tts_backend: Any | None = None,
        alsa_device: str = "plughw:0,0",
        wake_alsa_device: str | None = None,
        sample_rate: int = 16000,
        audio_dir: str | Path = "data/",
        wake_ack_text: str | None = None,
        wake_ack_mode: str = "local",
        wake_ack_dir: str | Path | None = None,
        max_capture_duration_sec: float = 15.0,
        silence_duration_sec: float = 0.8,
        post_ack_user_window_sec: float = 2.5,
        voice_debug_manager: Any | None = None,
        log_hook: Callable[[str], None] | None = None,
    ) -> None:
        if log_hook is not None:
            set_voice_log_hook(log_hook)

        self._wake_detector = wake_detector
        self._recognizer = recognizer
        self._tts_backend = tts_backend
        self._audio_dir = Path(audio_dir)
        self._wake_ack_text = (wake_ack_text or "").strip() or DEFAULT_WAKE_ACK_TEXT
        self._wake_ack_mode = (wake_ack_mode or "local").strip().lower()
        self._wake_ack_dir = Path(wake_ack_dir or DEFAULT_WAKE_ACK_DIR)
        self._local_wake_ack = None
        self._voice_debug = voice_debug_manager

        self._max_capture_sec = max_capture_duration_sec
        self._silence_sec = silence_duration_sec
        self._post_ack_window = post_ack_user_window_sec

        self._user_alsa = alsa_device
        wake_dev = (wake_alsa_device or "").strip() or alsa_device
        self._wake_alsa = wake_dev
        self._dual_capture = wake_dev != alsa_device
        self._input = AudioInputManager(alsa_device=alsa_device, sample_rate=sample_rate)
        self._wake_input = (
            AudioInputManager(alsa_device=wake_dev, sample_rate=sample_rate)
            if self._dual_capture
            else None
        )
        self._state = VoiceSessionStateMachine()
        self._probe = VoiceSessionProbe.global_probe()
        self._probe.bind_state_view(self._state)
        self._arbiter = VoiceInteractionArbiter(probe=self._probe)
        self._bridge = AgentBridge(sink)
        self._media_controller: Any | None = None
        self._user_speak_active = False
        self._active_speak_is_user = False
        self._session_lock = threading.Lock()
        self._session_counter = 0
        self._wake_flow_lock = threading.Lock()
        self._wake_flow_active = False
        self._last_wake_hint = 0.0
        self._running = False
        self._current_text = ""
        self._deferred_play_media: Any | None = None
        self._deferred_play_lock = threading.Lock()
        self._mic_watch_stop = threading.Event()
        self._mic_watch_thread: threading.Thread | None = None

        playback_dev = getattr(tts_backend, "alsa_playback_device", None) if tts_backend else None
        self._tts = TTSPlaybackManager(
            tts_backend=tts_backend,
            alsa_playback_device=playback_dev,
            prefer_capture_device=alsa_device,
            arbiter=self._arbiter,
            on_started=self._on_tts_started,
            on_finished=self._on_tts_finished,
            on_cancelled=self._on_tts_cancelled,
        )

        if wake_detector is not None:
            wake_detector.on_wake(self._on_wake_detected)

    def bind_sink(self, sink: Any) -> None:
        self._bridge.bind(sink)

    def bind_media_controller(self, media_controller: Any | None) -> None:
        """绑定媒体控制器：播放独占仲裁 + 唤醒词打断。"""
        self._media_controller = media_controller
        if media_controller is None:
            return
        prior = getattr(media_controller, "_on_state_changed", None)

        def _chain(state: Any) -> None:
            if callable(prior):
                prior(state)
            playing = bool(getattr(state, "is_playing", False))
            self._sync_wake_competing_audio()
            if not playing:
                self._tts.set_media_idle_check(self._media_idle_check)
                self._signal_tts_worker()

        media_controller._on_state_changed = _chain  # noqa: SLF001
        self._probe.bind_media_playing(lambda: media_controller.is_playing())
        self._tts.set_media_idle_check(self._media_idle_check)

    @property
    def state_machine(self) -> VoiceSessionStateMachine:
        return self._state

    @property
    def audio_input(self) -> AudioInputManager:
        return self._input

    def _sync_wake_competing_audio(self) -> None:
        """TTS 或媒体经扬声器播放时都会灌麦，需同步降低 KWS 阈值。"""
        detector = self._wake_detector
        if detector is None or not hasattr(detector, "set_media_playback_competing"):
            return
        mc = self._media_controller
        media_playing = mc is not None and bool(mc.is_playing())
        tts_busy = self._tts.is_busy() or self._state.is_speaking()
        detector.set_media_playback_competing(media_playing or tts_busy)

    def _reset_wake_kws_stream(self) -> None:
        """TTS 结束后清空 KWS 流状态，避免扬声器回声污染后续检测。"""
        detector = self._wake_detector
        reset = getattr(detector, "reset_stream", None) if detector is not None else None
        if callable(reset):
            reset()

    def preload_wake_ack(self) -> int:
        if self._wake_ack_mode != "local":
            return 0
        from src.adapters.voice.wake.local_wake_ack import LocalWakeAckPlayer, missing_ack_files

        playback = getattr(self._tts_backend, "alsa_playback_device", None)
        self._local_wake_ack = LocalWakeAckPlayer(
            ack_dir=self._wake_ack_dir,
            alsa_playback_device=playback,
            prefer_capture_device=self._input.alsa_device,
        )
        count = self._local_wake_ack.preload()
        if count == 0:
            missing = missing_ack_files(self._wake_ack_dir)
            voice_log(f"本地唤醒应答未就绪，将回退云端 TTS；缺失：{', '.join(missing) or '无'}")
        else:
            voice_log(f"已预加载 {count} 条本地唤醒应答")
        return count

    def _start_mic_watchdog(self) -> None:
        self._mic_watch_stop.clear()
        if self._mic_watch_thread is not None and self._mic_watch_thread.is_alive():
            return
        self._mic_watch_thread = threading.Thread(
            target=self._mic_watchdog_loop,
            name="VoiceRuntimeMicWatch",
            daemon=True,
        )
        self._mic_watch_thread.start()

    def _mic_watchdog_loop(self) -> None:
        while not self._mic_watch_stop.wait(5.0):
            if not self._running:
                continue
            restarted = False
            if not self._input.is_running and self._input.restart_capture():
                restarted = True
            wake_input = self._wake_input
            if wake_input is not None and not wake_input.is_running and wake_input.restart_capture():
                restarted = True
            if restarted:
                self._reset_wake_kws_stream()
                voice_log("capture 已重启，KWS 流已重置")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        if not self._input.start():
            voice_log(f"用户录音流启动失败：{self._input.last_error}")
        if self._wake_input is not None:
            self._wake_input.add_pcm_listener(self._on_pcm_chunk)
            if not self._wake_input.start():
                voice_log(f"唤醒监听流启动失败：{self._wake_input.last_error}")
        else:
            self._input.add_pcm_listener(self._on_pcm_chunk)
        self._reset_wake_kws_stream()
        self._tts.start()
        self.preload_wake_ack()
        self._start_mic_watchdog()
        threshold = getattr(self._wake_detector, "_base_keywords_threshold", None)
        if threshold is not None:
            voice_log(f"KWS 监听中（阈值={threshold}，唤醒后 VAD 录音）")
        if self._dual_capture:
            voice_log(
                f"VoiceRuntime 已启动（双 capture：唤醒={self._wake_alsa}，用户={self._user_alsa}）"
            )
        else:
            voice_log("VoiceRuntime 已启动（单 capture + 单 TTS 队列）")

    def stop(self) -> None:
        self._running = False
        self._mic_watch_stop.set()
        if self._mic_watch_thread is not None:
            self._mic_watch_thread.join(timeout=2.0)
            self._mic_watch_thread = None
        if self._wake_input is not None:
            self._wake_input.remove_pcm_listener(self._on_pcm_chunk)
            self._wake_input.stop()
        else:
            self._input.remove_pcm_listener(self._on_pcm_chunk)
        self._input.stop()
        self._tts.stop()
        self._arbiter.reminder_buffer.clear_autonomous()
        self._probe.set_user_speak_active(False)
        voice_log("VoiceRuntime 已停止")

    def is_running(self) -> bool:
        return self._running

    def is_tts_running(self) -> bool:
        return self._tts.is_busy() or self._state.is_speaking()

    def execute(self, action: Any) -> None:
        action_type = str(getattr(action, "type", ""))
        payload = dict(getattr(action, "payload", {}))

        if action_type == "set_tts_volume":
            volume = int(payload.get("volume", 5))
            if self._tts_backend is not None:
                self._tts_backend.set_volume(volume)
            voice_log(f"TTS 音量设置为 {volume}")
            return

        if action_type != "speak":
            return

        text = str(payload.get("text", "")).strip()
        if not text:
            return

        kind = str(payload.get("kind", "") or "")
        reason = str(payload.get("reason", "") or "")
        priority, source = self._classify_speak(payload)
        spec = self._arbiter.classify(source=source, reason=reason, kind=kind)

        decision = self._arbiter.decide_enqueue(
            spec,
            text=text,
            source=source,
            reason=reason,
            priority=int(priority),
            payload=payload,
        )
        if decision.action == ArbiterAction.BUFFER:
            voice_log(f"自主提醒仲裁→缓冲（{decision.reason}）：{text[:40]}")
            return
        if decision.action == ArbiterAction.DROP:
            voice_log(f"自主提醒仲裁→丢弃（{decision.reason}）：{text[:40]}")
            return

        self._enqueue_speak(
            text,
            priority=priority,
            source=source,
            reason=reason,
            kind=kind,
            payload=payload,
            spec=spec,
        )

    def _classify_speak(self, payload: dict[str, Any]) -> tuple[TTSJobPriority, str]:
        kind = str(payload.get("kind", "") or "")
        reason = str(payload.get("reason", "") or "")
        if reason == "media_play_ack":
            return TTSJobPriority.MEDIA_ACK, "media_ack"
        if kind == "notification" or reason in _AUTONOMOUS_REASONS:
            spec = resolve_job_spec(source=reason or "autonomous", reason=reason, kind=kind)
            return TTSJobPriority(spec.priority), reason or "autonomous"
        return TTSJobPriority.USER_REPLY, "agent_reply"

    def _media_idle_check(self) -> bool:
        mc = self._media_controller
        if mc is None:
            return True
        occupies = getattr(mc, "occupies_audio_output", None)
        if not callable(occupies):
            return True
        try:
            return not bool(occupies())
        except Exception:
            return True

    def _signal_tts_worker(self) -> None:
        signal = getattr(self._tts, "_signal", None)
        if signal is not None:
            signal.set()

    def schedule_deferred_play_media(self, action: Any) -> None:
        """等前置 media_ack speak 播完后再启动媒体播放。"""
        with self._deferred_play_lock:
            self._deferred_play_media = action
        voice_log(f"媒体播放已排队（待选曲播报完成后）：{action.payload.get('title', '')}")

    def _flush_deferred_play_media(self) -> None:
        with self._deferred_play_lock:
            action = self._deferred_play_media
            self._deferred_play_media = None
        if action is None:
            return
        mc = self._media_controller
        if mc is None:
            return
        from src.agent.media.media_models import MediaSource, MediaTrack

        payload = dict(getattr(action, "payload", {}) or {})
        track = MediaTrack(
            id=str(payload.get("track_id", "")),
            title=str(payload.get("title", "")),
            path=str(payload.get("path", "")),
            media_type=str(payload.get("media_type", "unknown")),
            category=str(payload.get("category", "default")),
        )
        source_raw = str(payload.get("source", "user_explicit"))
        source = (
            MediaSource.AGENT_SUGGESTION
            if source_raw == MediaSource.AGENT_SUGGESTION.value
            else MediaSource.USER_EXPLICIT
        )
        mc.play_track(track, source=source)
        voice_log(
            f"媒体开始播放：{track.title}｜/root/Embeded-Agent/data/music 曲目｜{track.path}"
        )

    def _enqueue_speak(
        self,
        text: str,
        *,
        priority: TTSJobPriority,
        source: str,
        reason: str,
        kind: str,
        payload: dict[str, Any],
        spec: Any,
    ) -> None:
        self._current_text = text
        self._active_speak_is_user = not spec.is_autonomous
        if self._active_speak_is_user:
            self._probe.set_user_speak_active(True)
        if self._state.state in {VoiceState.AGENT_THINKING, VoiceState.IDLE}:
            if self._active_speak_is_user or not self._state.is_listening():
                self._state.transition(VoiceState.SPEAKING, f"播报入队：{source}")

        def _on_done() -> None:
            if self._active_speak_is_user:
                self._probe.set_user_speak_active(False)
                self._probe.mark_session_ended()
            self._on_speak_finished(text, reason=reason, kind=kind)
            if source == "media_ack":
                self._flush_deferred_play_media()

        self._tts.enqueue(
            text,
            priority=priority,
            source=source,
            reason=reason,
            kind=kind,
            payload=payload,
            voice=payload.get("voice"),
            volume=payload.get("volume"),
            speed=payload.get("speed"),
            on_started=lambda: self._bridge.emit_tts_started(text),
            on_finished=_on_done,
        )

    def _on_speak_finished(self, text: str, *, reason: str = "", kind: str = "") -> None:
        if self._state.state == VoiceState.SPEAKING:
            self._state.transition(VoiceState.IDLE, "播报完成")
        self._sync_wake_competing_audio()
        self._reset_wake_kws_stream()
        self._flush_pending_reminders()

    def _on_tts_started(self, _job_id: str, text: str) -> None:
        # listening 期间禁止自主提醒把状态改成 speaking（防御性：仲裁层应已拦截）。
        if self._state.state == VoiceState.LISTENING and not self._active_speak_is_user:
            voice_log("listening 中忽略自主提醒 TTS 状态迁移")
        elif self._state.state != VoiceState.ACK_PLAYING:
            if self._state.state != VoiceState.LISTENING or self._active_speak_is_user:
                self._state.transition(VoiceState.SPEAKING, "TTS 开始播放")
        self._sync_wake_competing_audio()
        self._bridge.emit_tts_started(text)

    def _on_tts_finished(
        self,
        _job_id: str,
        text: str,
        reason: str = "",
        kind: str = "",
        cancelled: bool = False,
    ) -> None:
        self._bridge.emit_tts_finished(
            text,
            reason=reason,
            kind=kind,
            cancelled=cancelled,
        )
        self._sync_wake_competing_audio()
        self._reset_wake_kws_stream()

    def _on_tts_cancelled(self, _job_id: str, reason: str) -> None:
        if self._current_text:
            self._bridge.emit_tts_cancelled(self._current_text, reason=reason)
        if self._state.state == VoiceState.SPEAKING:
            self._state.transition(VoiceState.IDLE, f"播报被打断：{reason}")
        self._sync_wake_competing_audio()
        self._reset_wake_kws_stream()

    def _on_pcm_chunk(self, chunk: bytes) -> None:
        if self._wake_detector is None or not self._running:
            return
        if self._wake_flow_active and self._state.state in {
            VoiceState.LISTENING,
            VoiceState.ASR_RUNNING,
        }:
            return
        feed = getattr(self._wake_detector, "feed_audio", None)
        if callable(feed):
            feed(chunk)

    def _on_wake_detected(self, wake_event: Any) -> None:
        with self._wake_flow_lock:
            if self._wake_flow_active:
                now = time.time()
                if now - self._last_wake_hint >= 3.0:
                    self._last_wake_hint = now
                    voice_log("已听到唤醒，请直接说您的问题")
                return
            self._wake_flow_active = True

        # 媒体播放中：唤醒词必须立即停止媒体，释放扬声器后再播唤醒应答。
        mc = self._media_controller
        media_was_playing = mc is not None and mc.is_playing()
        if media_was_playing:
            voice_log("播放中检测到唤醒词，正在停止音乐…")
            mc.stop_for_wake_word()
            wait = getattr(mc, "wait_until_stopped", None)
            if callable(wait):
                wait(timeout=2.0)
            time.sleep(0.1)

        # 唤醒最高优先级：清空自主提醒、取消可打断 TTS。
        self._tts.prepare_for_wake()
        self._probe.set_user_speak_active(False)

        if (
            self._state.allows_wake_barge_in()
            or media_was_playing
            or (mc is not None and mc.get_agent_media_state().value in {
            "listening_user_command",
            "interrupting_media",
        })
        ):
            self._state.transition(VoiceState.WAKE_DETECTED, "播放中检测到唤醒词")

        threading.Thread(
            target=self._run_wake_session,
            args=(wake_event,),
            name="VoiceRuntimeWakeFlow",
            daemon=True,
        ).start()

    def _run_wake_session(self, wake_event: Any) -> None:
        keyword = getattr(wake_event, "keyword", "wake")
        session_id = ""
        try:
            with self._session_lock:
                self._session_counter += 1
                session_id = f"session_{self._session_counter}_{int(time.time())}"
            session = VoiceSession(session_id=session_id, start_time=time.time(), is_active=True)

            voice_log(f"唤醒词命中：{keyword}")
            self._state.transition(VoiceState.WAKE_DETECTED, f"唤醒词：{keyword}")

            self._play_wake_ack()
            self._state.transition(VoiceState.LISTENING, "等待用户说话")
            self._bridge.emit_voice_input_started(session_id)

            vad_cfg = VadConfig(
                sample_rate=self._input.sample_rate,
                silence_duration_sec=self._silence_sec,
                max_duration_sec=self._max_capture_sec,
                initial_timeout_sec=max(12.0, self._max_capture_sec + 2.0),
                min_elapsed_before_end_sec=max(1.0, self._post_ack_window),
            )
            raw_path = self._audio_dir / self.WAKE_LATEST_RAW_WAV
            audio_path, _duration, stop_reason = self._input.record_until_silence(
                raw_path,
                config=vad_cfg,
            )

            self._bridge.emit_voice_input_stopped(session_id)
            session.is_active = False

            if audio_path is None:
                voice_log(f"唤醒会话结束：未获得有效录音（{stop_reason}）")
                self._state.transition(VoiceState.IDLE, "录音无效")
                return

            self._state.transition(VoiceState.ASR_RUNNING, "开始语音识别")
            threading.Thread(
                target=self._run_asr_and_agent,
                args=(session, audio_path),
                name=f"WakeASR-{session_id}",
                daemon=True,
            ).start()
        finally:
            with self._wake_flow_lock:
                self._wake_flow_active = False

    def _play_wake_ack(self) -> None:
        if self._wake_ack_mode == "off":
            return
        self._state.transition(VoiceState.ACK_PLAYING, "播放唤醒应答")

        wav_path = None
        ack_text = self._wake_ack_text
        if self._wake_ack_mode == "local" and self._local_wake_ack is not None:
            clip = self._local_wake_ack.pick_clip(strategy="random")
            if clip is not None:
                wav_path = clip.path
                ack_text = clip.text or ack_text
            else:
                voice_log(f"本地唤醒应答无可用 WAV，回退文案 TTS：{ack_text!r}")
        elif self._wake_ack_mode == "local" and self._local_wake_ack is None:
            voice_log(f"本地唤醒应答未预加载，回退文案 TTS：{ack_text!r}")

        done = threading.Event()

        def _on_ack_done() -> None:
            done.set()

        if wav_path is not None:
            self._tts.enqueue(
                ack_text,
                priority=TTSJobPriority.WAKE_ACK,
                source="wake_ack",
                wav_path=wav_path,
                on_finished=_on_ack_done,
            )
        else:
            self._tts.enqueue(
                ack_text,
                priority=TTSJobPriority.WAKE_ACK,
                source="wake_ack",
                on_finished=_on_ack_done,
            )
        done.wait(timeout=30.0)

    def _run_asr_and_agent(self, session: VoiceSession, audio_path: Path) -> None:
        voice_log("ASR 开始")
        text = ""
        try:
            if self._recognizer is None:
                voice_log("ASR 完成：未配置识别后端")
                self._state.transition(VoiceState.IDLE, "无 ASR")
                self._flush_pending_reminders()
                return
            result = self._recognizer.recognize_file(audio_path)
            text = str(getattr(result, "text", result) or "").strip()
            voice_log(f"ASR 完成：{text[:60] if text else '（空）'}")
        except Exception as exc:
            voice_log(f"ASR 失败：{exc}")
            self._state.transition(VoiceState.IDLE, "ASR 失败")
            self._flush_pending_reminders()
            return

        if not text or self._is_wake_ack_echo(text):
            voice_log("ASR 结果为唤醒应答回声，跳过 Agent")
            self._state.transition(VoiceState.IDLE, "回声过滤")
            self._flush_pending_reminders()
            return

        self._state.transition(VoiceState.AGENT_THINKING, "等待 Agent 回复")
        self._bridge.emit_speech_recognized(text=text, session_id=session.session_id)
        # Agent 异步生成 speak；若无回复，会话在 TTS 事件或超时后回到 IDLE。
        threading.Timer(2.0, self._maybe_idle_after_agent).start()

    def _maybe_idle_after_agent(self) -> None:
        if self._state.state == VoiceState.AGENT_THINKING and not self._tts.is_busy():
            self._state.transition(VoiceState.IDLE, "Agent 无语音回复")
            self._flush_pending_reminders()

    def _flush_pending_reminders(self) -> None:
        """会话结束后最多释放一条缓冲提醒，避免提醒连播轰炸。"""
        if not self._state.is_idle() or self._state.defer_autonomous_playback():
            return
        if self._tts.is_busy():
            return

        item = self._arbiter.try_pop_one_buffered_reminder()
        if item is None:
            return

        spec = self._arbiter.classify(source=item.source, reason=item.reason)
        decision = self._arbiter.decide_enqueue(
            spec,
            text=item.text,
            source=item.source,
            reason=item.reason,
            priority=item.priority,
            payload=item.payload,
        )
        if decision.action != ArbiterAction.PLAY:
            self._arbiter.reminder_buffer.offer(
                text=item.text,
                source=item.source,
                reason=item.reason,
                priority=item.priority,
                payload=item.payload,
                spec=spec,
                created_at=item.created_at,
            )
            return

        self._enqueue_speak(
            item.text,
            priority=TTSJobPriority(spec.priority),
            source=item.source,
            reason=item.reason,
            kind=str(item.payload.get("kind", "")),
            payload=item.payload,
            spec=spec,
        )

    @staticmethod
    def _normalize_utterance(text: str) -> str:
        return re.sub(r"[\s\W_]+", "", text.lower())

    def _is_wake_ack_echo(self, text: str) -> bool:
        normalized = self._normalize_utterance(text)
        if normalized in _WAKE_ACK_ECHO_NORMALIZED:
            return True
        local_ack = self._local_wake_ack
        if local_ack is not None:
            for clip in getattr(local_ack, "_clips", []):
                if self._normalize_utterance(getattr(clip, "text", "")) == normalized:
                    return True
        return self._normalize_utterance(self._wake_ack_text) == normalized
