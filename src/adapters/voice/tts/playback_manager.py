from __future__ import annotations

"""TTSPlaybackManager：全系统唯一语音播放入口（优先级队列 + 播放前复检 + 唤醒抢占）。"""

import heapq
import itertools
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from src.adapters.voice.arbitration.tts_job_policy import TTSJobPriority, TTSJobSpec, resolve_job_spec
from src.adapters.voice.arbitration.voice_arbiter import ArbiterAction, ArbiterDecision, VoiceInteractionArbiter
from src.adapters.voice.runtime.logger import voice_log
from src.adapters.voice.tts.audio_playback import play_wav_file, play_wav_sounddevice_first

PrePlayResult = Literal["play", "requeue", "buffer", "drop"]


@dataclass(order=True)
class _QueuedTTSJob:
    priority: int
    seq: int
    job_id: str = field(compare=False)
    text: str = field(compare=False, default="")
    source: str = field(compare=False, default="")
    reason: str = field(compare=False, default="")
    kind: str = field(compare=False, default="")
    spec: TTSJobSpec = field(compare=False, default=None)  # type: ignore[assignment]
    voice: Any = field(compare=False, default=None)
    volume: Any = field(compare=False, default=None)
    speed: Any = field(compare=False, default=None)
    wav_path: Path | None = field(compare=False, default=None)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)
    created_at: float = field(compare=False, default=0.0)
    on_started: Callable[[], None] | None = field(compare=False, default=None)
    on_finished: Callable[[], None] | None = field(compare=False, default=None)


class TTSPlaybackManager:
    """统一 TTS 播放队列：支持播放前二次检查、唤醒抢占、取消后统一 finalize。"""

    def __init__(
        self,
        *,
        tts_backend: Any | None,
        alsa_playback_device: str | None = None,
        prefer_capture_device: str | None = None,
        arbiter: VoiceInteractionArbiter | None = None,
        on_queued: Callable[[str, str], None] | None = None,
        on_started: Callable[[str, str], None] | None = None,
        on_finished: Callable[[str, str, str, str, bool], None] | None = None,
        on_cancelled: Callable[[str, str], None] | None = None,
        on_job_deferred: Callable[[_QueuedTTSJob, str], None] | None = None,
        on_before_play: Callable[[_QueuedTTSJob], None] | None = None,
        on_after_play: Callable[[_QueuedTTSJob], None] | None = None,
    ) -> None:
        self._tts_backend = tts_backend
        self._alsa_playback = alsa_playback_device
        self._prefer_capture = prefer_capture_device
        self._arbiter = arbiter or VoiceInteractionArbiter()
        self._on_queued = on_queued
        self._on_started = on_started
        self._on_finished = on_finished
        self._on_cancelled = on_cancelled
        self._on_job_deferred = on_job_deferred
        self._on_before_play = on_before_play
        self._on_after_play = on_after_play
        self._seq = itertools.count()
        self._heap: list[_QueuedTTSJob] = []
        self._heap_lock = threading.Lock()
        self._signal = threading.Event()
        self._worker: threading.Thread | None = None
        self._running = False
        self._current_job: _QueuedTTSJob | None = None
        self._cancel_flag = threading.Event()
        self._play_proc: subprocess.Popen[Any] | None = None
        self._state_lock = threading.Lock()
        self._media_idle_check: Callable[[], bool] | None = None
        self._waiting_media = False

    @property
    def arbiter(self) -> VoiceInteractionArbiter:
        return self._arbiter

    def set_media_idle_check(self, checker: Callable[[], bool] | None) -> None:
        self._media_idle_check = checker

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, name="TTSPlaybackManager", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        self._signal.set()
        with self._heap_lock:
            self._heap.clear()
        self.cancel_current("shutdown")
        if self._worker is not None:
            self._worker.join(timeout=10.0)
            self._worker = None

    def enqueue(
        self,
        text: str,
        *,
        priority: TTSJobPriority = TTSJobPriority.USER_REPLY,
        source: str = "agent",
        reason: str = "",
        kind: str = "",
        voice: Any = None,
        volume: Any = None,
        speed: Any = None,
        wav_path: Path | None = None,
        payload: dict[str, Any] | None = None,
        on_started: Callable[[], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> str:
        spec = resolve_job_spec(source=source, reason=reason, kind=kind)
        job_id = uuid.uuid4().hex[:12]
        job = _QueuedTTSJob(
            priority=int(priority) if isinstance(priority, TTSJobPriority) else int(priority),
            seq=next(self._seq),
            job_id=job_id,
            text=text.strip(),
            source=source,
            reason=reason,
            kind=kind,
            spec=spec,
            voice=voice,
            volume=volume,
            speed=speed,
            wav_path=wav_path,
            payload=dict(payload or {}),
            created_at=time.time(),
            on_started=on_started,
            on_finished=on_finished,
        )
        with self._heap_lock:
            heapq.heappush(self._heap, job)
        voice_log(f"已入队：{text[:50]}（来源={source}，优先级={spec.kind}）")
        if self._on_queued is not None:
            self._on_queued(job_id, text)
        self._signal.set()
        return job_id

    def prepare_for_wake(self) -> None:
        """唤醒抢占：取消可打断当前任务，移除堆内全部自主提醒。"""
        self._arbiter.on_wake_preempt()
        self.cancel_current_interruptible("唤醒词打断")
        removed = 0
        with self._heap_lock:
            kept: list[_QueuedTTSJob] = []
            for _ in range(len(self._heap)):
                job = heapq.heappop(self._heap)
                if job.spec.is_autonomous:
                    removed += 1
                    self._defer_job(job, "wake_preempt_purge")
                else:
                    kept.append(job)
            for job in kept:
                heapq.heappush(self._heap, job)
        if removed:
            voice_log(f"唤醒抢占：已从 TTS 队列移除 {removed} 条自主提醒")
        self._waiting_media = False
        self._signal.set()

    def cancel_current(self, reason: str) -> bool:
        with self._state_lock:
            job = self._current_job
            if job is None:
                return False
            self._cancel_flag.set()
            proc = self._play_proc
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            voice_log(f"被打断：{job.text[:40]}（{reason}）")
        return True

    def cancel_current_interruptible(self, reason: str) -> bool:
        with self._state_lock:
            job = self._current_job
            if job is None:
                return False
            if not job.spec.interruptible:
                return False
        cancelled = self.cancel_current(reason)
        if cancelled and self._on_cancelled is not None:
            self._on_cancelled(job.job_id, reason)
        return cancelled

    def is_busy(self) -> bool:
        with self._heap_lock:
            queued = bool(self._heap)
        with self._state_lock:
            playing = self._current_job is not None
        return queued or playing or self._waiting_media

    def _worker_loop(self) -> None:
        while self._running:
            self._signal.wait(timeout=0.2)
            self._signal.clear()
            while self._running:
                job = self._select_next_playable_job()
                if job is None:
                    break
                self._play_job(job)

    def _select_next_playable_job(self) -> _QueuedTTSJob | None:
        """从堆中选取当前可播的最高优先级任务；不可播的自主提醒转缓冲而非占住 worker。"""
        deferred: list[_QueuedTTSJob] = []
        chosen: _QueuedTTSJob | None = None
        now = time.time()

        with self._heap_lock:
            while self._heap:
                job = heapq.heappop(self._heap)
                decision = self._arbiter.decide_pre_play(job.spec, created_at=job.created_at, now=now)
                result = self._map_decision(decision)
                if result == "play":
                    chosen = job
                    break
                if result == "requeue":
                    deferred.append(job)
                    continue
                if result == "buffer":
                    self._defer_job(job, decision.reason)
                    continue
                voice_log(f"提醒丢弃（{decision.reason}）：{job.text[:40]}")
            for job in deferred:
                heapq.heappush(self._heap, job)
        if chosen is not None:
            with self._heap_lock:
                for job in deferred:
                    heapq.heappush(self._heap, job)
        return chosen

    @staticmethod
    def _map_decision(decision: ArbiterDecision) -> PrePlayResult:
        if decision.action == ArbiterAction.PLAY:
            return "play"
        if decision.action == ArbiterAction.REQUEUE:
            return "requeue"
        if decision.action == ArbiterAction.BUFFER:
            return "buffer"
        return "drop"

    def _defer_job(self, job: _QueuedTTSJob, reason: str) -> None:
        self._arbiter.reminder_buffer.offer(
            text=job.text,
            source=job.source,
            reason=job.reason or job.source,
            priority=job.priority,
            payload=job.payload,
            spec=job.spec,
            created_at=job.created_at,
        )
        if self._on_job_deferred is not None:
            self._on_job_deferred(job, reason)

    def _media_blocks_job(self, job: _QueuedTTSJob) -> bool:
        if job.spec.allow_during_media:
            return False
        checker = self._media_idle_check
        if checker is None:
            return False
        try:
            return not bool(checker())
        except Exception:
            return False

    def _play_job(self, job: _QueuedTTSJob) -> None:
        # 不在 worker 内长等媒体：媒体忙则回堆，由 _select_next_playable_job 转缓冲。
        if self._media_blocks_job(job):
            voice_log(f"媒体占用，延后处理（来源={job.source}）")
            with self._heap_lock:
                heapq.heappush(self._heap, job)
            self._waiting_media = True
            return
        self._waiting_media = False

        with self._state_lock:
            self._current_job = job
            self._cancel_flag.clear()

        voice_log(f"开始播放：{job.text[:50]}（来源={job.source}）")
        if job.on_started is not None:
            job.on_started()
        if self._on_started is not None:
            self._on_started(job.job_id, job.text)

        cancelled = False
        error: Exception | None = None
        if self._on_before_play is not None:
            try:
                self._on_before_play(job)
            except Exception as exc:
                voice_log(f"播放前回调异常：{exc}")
        try:
            if job.wav_path is not None and job.wav_path.is_file():
                prefer_sounddevice = job.source == "wake_ack"
                cancelled = self._play_wav_cancellable(
                    job.wav_path,
                    prefer_sounddevice=prefer_sounddevice,
                )
            elif self._tts_backend is not None and job.text:
                self._tts_backend.speak(
                    job.text,
                    voice=job.voice,
                    volume=job.volume,
                    speed=job.speed,
                )
            elif job.text:
                voice_log(f"[TTS] {job.text}")
        except Exception as exc:
            error = exc
            voice_log(f"播放失败：{exc}")
        finally:
            if self._on_after_play is not None:
                try:
                    self._on_after_play(job)
                except Exception as exc:
                    voice_log(f"播放后回调异常：{exc}")
            if self._cancel_flag.is_set():
                cancelled = True
            self._finalize_job(job, cancelled=cancelled, error=error)

    def _finalize_job(
        self,
        job: _QueuedTTSJob,
        *,
        cancelled: bool,
        error: Exception | None = None,
    ) -> None:
        """任意结束路径（完成/取消/失败）均触发 on_finished，避免 wake_ack 空等。"""
        with self._state_lock:
            self._current_job = None
            self._play_proc = None
            self._cancel_flag.clear()

        if cancelled:
            voice_log(f"播放取消：{job.text[:50]}（来源={job.source}）")
            if self._on_cancelled is not None:
                self._on_cancelled(job.job_id, "cancelled")
        elif error is None:
            voice_log(f"播放完成：{job.text[:50]}")
        else:
            voice_log(f"播放异常结束：{job.text[:50]}（{error}）")

        if job.on_finished is not None:
            try:
                job.on_finished()
            except Exception as exc:
                voice_log(f"on_finished 回调异常：{exc}")
        if self._on_finished is not None:
            try:
                self._on_finished(
                    job.job_id,
                    job.text,
                    str(job.reason or ""),
                    str(job.kind or ""),
                    bool(cancelled),
                )
            except Exception as exc:
                voice_log(f"全局 on_finished 回调异常：{exc}")

    def _play_wav_cancellable(self, path: Path, *, prefer_sounddevice: bool = False) -> bool:
        import platform

        from src.adapters.voice.input.alsa_audio_devices import (
            playback_device_for_tts,
            prepare_playback_device,
        )

        if platform.system() != "Linux":
            play_wav_file(
                path,
                alsa_playback_device=self._alsa_playback,
                prefer_capture_device=self._prefer_capture,
                prefer_sounddevice_first=prefer_sounddevice,
            )
            return self._cancel_flag.is_set()

        if prefer_sounddevice:
            voice_log("唤醒应答优先经 sounddevice 播放（与 arecord 并行）")
            cancelled = play_wav_sounddevice_first(
                path,
                alsa_playback_device=self._alsa_playback,
                prefer_capture_device=self._prefer_capture,
                cancel_flag=self._cancel_flag,
            )
            return cancelled

        alsa_device = prepare_playback_device(
            playback_device_for_tts(
                explicit=self._alsa_playback,
                prefer_capture_device=self._prefer_capture,
            )
        )
        cmd = ["aplay", str(path)]
        if alsa_device:
            cmd = ["aplay", "-D", alsa_device, str(path)]
        device = alsa_device or self._alsa_playback or "default"
        last_error = ""
        aplay_failed = False
        for attempt in range(3):
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with self._state_lock:
                self._play_proc = proc
            try:
                while proc.poll() is None:
                    if self._cancel_flag.is_set():
                        proc.terminate()
                        proc.wait(timeout=2.0)
                        return True
                    time.sleep(0.05)
                if proc.returncode in {0, None}:
                    return self._cancel_flag.is_set()
                stderr = ""
                if proc.stderr is not None:
                    stderr = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                last_error = stderr
                if "busy" in stderr.lower() and attempt < 2:
                    voice_log(f"aplay 设备忙，{50 * (attempt + 1)}ms 后重试（{device}）")
                    time.sleep(0.05 * (attempt + 1))
                    continue
                aplay_failed = True
                break
            finally:
                with self._state_lock:
                    if self._play_proc is proc:
                        self._play_proc = None

        if aplay_failed:
            voice_log(f"aplay 失败，尝试 sounddevice 回退（{device}）")
            try:
                play_wav_file(
                    path,
                    alsa_playback_device=self._alsa_playback,
                    prefer_capture_device=self._prefer_capture,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"本地 WAV 播放失败（device={device!r}）：{last_error or exc}"
                ) from exc
            return self._cancel_flag.is_set()

        raise RuntimeError(f"aplay 播放失败（device={device!r}）：{last_error}")
