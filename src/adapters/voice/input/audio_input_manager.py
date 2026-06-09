from __future__ import annotations

"""AudioInputManager：全系统唯一的 capture 输入流（单 arecord + ring buffer + tap）。

唤醒词、VAD 录音、调试录音均通过本管理器消费 PCM，禁止其他模块自行启动 arecord。
"""

import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from src.adapters.voice.runtime.logger import voice_log
from src.adapters.voice.vad.recorder import (
    VadConfig,
    detect_end_frame_index,
    frames_contain_speech,
    write_pcm_wav,
)


class AudioInputManager:
    """后台保持唯一 arecord 进程，向监听者分发 PCM，并支持 VAD tap 录音。"""

    def __init__(
        self,
        *,
        alsa_device: str,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        ring_sec: float = 2.0,
    ) -> None:
        self._alsa_device = alsa_device
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._frame_bytes = max(int(sample_rate * 2 * frame_ms / 1000), 960)
        self._ring_max = max(1, int(round(ring_sec / (frame_ms / 1000.0))))
        self._ring: deque[bytes] = deque(maxlen=self._ring_max)
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._tap_lock = threading.Lock()
        self._tap_active = False
        self._tap_chunks: list[bytes] = []
        self._tap_target_bytes = 0
        self._tap_vad_mode = False
        self._tap_done = threading.Event()
        self._last_error = ""
        self._pcm_listeners: list[Callable[[bytes], None]] = []
        self._listeners_lock = threading.Lock()

    @property
    def alsa_device(self) -> str:
        return self._alsa_device

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        return self._running and self._proc is not None and self._proc.poll() is None

    @property
    def last_error(self) -> str:
        return self._last_error

    def add_pcm_listener(self, listener: Callable[[bytes], None]) -> None:
        with self._listeners_lock:
            if listener not in self._pcm_listeners:
                self._pcm_listeners.append(listener)

    def remove_pcm_listener(self, listener: Callable[[bytes], None]) -> None:
        with self._listeners_lock:
            if listener in self._pcm_listeners:
                self._pcm_listeners.remove(listener)

    def _notify_listeners(self, chunk: bytes) -> None:
        with self._listeners_lock:
            listeners = list(self._pcm_listeners)
        for listener in listeners:
            try:
                listener(chunk)
            except Exception:
                pass

    def start(self) -> bool:
        if self.is_running:
            return True
        self._running = True
        self._last_error = ""
        try:
            self._proc = subprocess.Popen(
                [
                    "arecord",
                    "-D",
                    self._alsa_device,
                    "-f",
                    "S16_LE",
                    "-r",
                    str(self._sample_rate),
                    "-c",
                    "1",
                    "-t",
                    "raw",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self._last_error = "arecord not found"
            self._running = False
            return False

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="AudioInputManager",
            daemon=True,
        )
        self._reader_thread.start()
        time.sleep(0.05)
        if self.is_running:
            voice_log(f"录音开始：唯一 capture 流已打开（{self._alsa_device}）")
        return self.is_running

    def _cleanup_dead_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        thread = self._reader_thread
        self._reader_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def suspend_capture(self) -> None:
        """临时释放 arecord（同卡播报时避免 Device busy），保留 listeners 与 _running。"""
        if self._proc is None:
            return
        thread = self._reader_thread
        self._cleanup_dead_proc()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._reader_thread = None

    def resume_capture(self) -> bool:
        """suspend_capture 之后重新拉起 capture。"""
        if not self._running:
            return False
        if self.is_running:
            return True
        self._last_error = ""
        try:
            self._proc = subprocess.Popen(
                [
                    "arecord",
                    "-D",
                    self._alsa_device,
                    "-f",
                    "S16_LE",
                    "-r",
                    str(self._sample_rate),
                    "-c",
                    "1",
                    "-t",
                    "raw",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self._last_error = "arecord not found"
            return False
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="AudioInputManager",
            daemon=True,
        )
        self._reader_thread.start()
        time.sleep(0.05)
        if self.is_running:
            voice_log(f"录音恢复：capture 流已重新打开（{self._alsa_device}）")
        return self.is_running

    def restart_capture(self) -> bool:
        """arecord 异常退出后重新拉起 capture（唤醒监听依赖持续 PCM）。"""
        if self.is_running:
            return True
        err = self._last_error or "capture_eof"
        voice_log(f"capture 流已断开，尝试重启（原因：{err}）")
        self._cleanup_dead_proc()
        was_running = self._running
        self._running = True
        ok = self.start()
        if not ok and not was_running:
            self._running = False
        return ok

    def stop(self) -> None:
        self._running = False
        self._tap_done.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._proc = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        voice_log("录音结束：capture 流已关闭")

    def _reader_loop(self) -> None:
        while self._running:
            proc = self._proc
            stdout = proc.stdout if proc is not None else None
            if stdout is None:
                break
            chunk = stdout.read(self._frame_bytes)
            if not chunk:
                stderr = proc.stderr if proc is not None else None
                if stderr is not None:
                    err = stderr.read().decode("utf-8", errors="replace").strip()
                    if err:
                        self._last_error = err
                if self._running and self._proc is proc:
                    voice_log(f"capture 读流异常结束：{self._last_error or 'EOF'}")
                break
            if len(chunk) < self._frame_bytes:
                chunk = chunk + b"\x00" * (self._frame_bytes - len(chunk))
            with self._lock:
                self._ring.append(chunk)
            self._notify_listeners(chunk)
            with self._lock:
                if self._tap_active:
                    self._tap_chunks.append(chunk)
                    if (
                        not self._tap_vad_mode
                        and sum(len(c) for c in self._tap_chunks) >= self._tap_target_bytes
                    ):
                        self._tap_done.set()

    def record_seconds(
        self,
        output_path: str | Path,
        duration_sec: float,
        *,
        pre_roll_sec: float = 0.45,
    ) -> tuple[Path | None, float]:
        """从 ring buffer 截取固定时长（兼容旧 PersistentMicCapture API）。"""
        if not self.is_running and not self.start():
            return None, 0.0
        output_path = Path(output_path)
        pre_frames = max(1, int(round(pre_roll_sec / (self._frame_ms / 1000.0))))
        target_bytes = int(max(0.2, float(duration_sec)) * self._sample_rate * 2)
        with self._tap_lock:
            with self._lock:
                pre = b"".join(list(self._ring)[-pre_frames:])
                self._tap_chunks = [pre] if pre else []
                self._tap_target_bytes = target_bytes
                self._tap_vad_mode = False
                self._tap_active = True
                self._tap_done.clear()
            finished = self._tap_done.wait(timeout=float(duration_sec) + 2.0)
            with self._lock:
                self._tap_active = False
                pcm = b"".join(self._tap_chunks)
                self._tap_chunks.clear()
        if not finished and len(pcm) < target_bytes // 2:
            return None, 0.0
        if len(pcm) > target_bytes:
            pcm = pcm[:target_bytes]
        if len(pcm) < 2000:
            return None, 0.0
        write_pcm_wav(output_path, pcm, sample_rate=self._sample_rate)
        return output_path, len(pcm) / (self._sample_rate * 2)

    def record_until_silence(
        self,
        output_path: str | Path,
        *,
        config: VadConfig | None = None,
        pre_roll_sec: float = 0.7,
    ) -> tuple[Path | None, float, str]:
        """从 ring buffer VAD 截断录音（不新开 arecord）。"""
        cfg = config or VadConfig()
        if not self.is_running and not self.start():
            return None, 0.0, "mic_not_running"

        output_path = Path(output_path)
        pre_frames = max(1, int(round(pre_roll_sec / (self._frame_ms / 1000.0))))
        frame_sec = cfg.frame_ms / 1000.0
        max_frames = max(1, int(round(cfg.max_duration_sec / frame_sec)))
        initial_timeout_frames = max(1, int(round(cfg.initial_timeout_sec / frame_sec)))
        stop_reason = "unknown"
        pcm = b""

        voice_log(
            f"录音开始：VAD 采集（最长 {cfg.max_duration_sec:.0f}s，静音 {cfg.silence_duration_sec:.1f}s）"
        )

        with self._tap_lock:
            with self._lock:
                pre = b"".join(list(self._ring)[-pre_frames:])
                self._tap_chunks = [pre] if pre else []
                self._tap_target_bytes = 0
                self._tap_vad_mode = True
                self._tap_active = True
                self._tap_done.clear()

            deadline = time.monotonic() + cfg.max_duration_sec + 3.0
            last_frame_count = -1
            while time.monotonic() < deadline:
                if not self.is_running:
                    stop_reason = "mic_stopped"
                    break
                with self._lock:
                    frames = list(self._tap_chunks)
                if len(frames) == last_frame_count:
                    time.sleep(frame_sec * 0.5)
                    continue
                last_frame_count = len(frames)

                end_index = detect_end_frame_index(frames, config=cfg)
                if end_index is not None:
                    pcm = b"".join(frames[: end_index + 1])
                    stop_reason = "silence_after_speech"
                    break

                if len(frames) >= max_frames:
                    pcm = b"".join(frames)
                    stop_reason = "max_duration"
                    break

                if (
                    not frames_contain_speech(frames, config=cfg)
                    and len(frames) >= initial_timeout_frames
                ):
                    stop_reason = "initial_timeout_no_speech"
                    break

            with self._lock:
                self._tap_active = False
                self._tap_vad_mode = False
                if not pcm and self._tap_chunks:
                    pcm = b"".join(self._tap_chunks)
                self._tap_chunks.clear()
                self._tap_done.set()

        if stop_reason == "initial_timeout_no_speech" or not pcm:
            self._last_error = stop_reason
            voice_log(f"录音结束：未采集到有效语音（{stop_reason}）")
            return None, 0.0, stop_reason

        if len(pcm) < 2000:
            self._last_error = "pcm_too_short"
            voice_log("录音结束：PCM 过短")
            return None, 0.0, "pcm_too_short"

        if not frames_contain_speech(
            [pcm[i : i + self._frame_bytes] for i in range(0, len(pcm), self._frame_bytes)],
            config=cfg,
        ):
            self._last_error = "no_speech_detected"
            voice_log("录音结束：未检测到说话")
            return None, 0.0, "no_speech_detected"

        write_pcm_wav(output_path, pcm, sample_rate=self._sample_rate)
        duration = len(pcm) / (self._sample_rate * 2)
        voice_log(f"录音结束：{duration:.1f}s（{stop_reason}）")
        return output_path, duration, stop_reason
