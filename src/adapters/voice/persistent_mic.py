from __future__ import annotations

"""摄像头麦常驻采集：启动后 arecord 常开，唤醒时直接切流录音，避免反复 open 设备。"""

import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from src.adapters.voice.vad_recorder import (
    VadConfig,
    detect_end_frame_index,
    frames_contain_speech,
    write_pcm_wav,
)


class PersistentMicCapture:
    """后台保持 arecord 进程，按需提供 pre-roll + 固定时长 PCM。"""

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
        self._record_lock = threading.Lock()
        self._tap_active = False
        self._tap_chunks: list[bytes] = []
        self._tap_target_bytes = 0
        self._tap_vad_mode = False
        self._tap_done = threading.Event()
        self._last_error = ""

    @property
    def is_running(self) -> bool:
        return self._running and self._proc is not None and self._proc.poll() is None

    @property
    def last_error(self) -> str:
        return self._last_error

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
            name="PersistentMicReader",
            daemon=True,
        )
        self._reader_thread.start()
        time.sleep(0.05)
        return self.is_running

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

    def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        while self._running:
            chunk = stdout.read(self._frame_bytes)
            if not chunk:
                if self._proc.stderr is not None:
                    err = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
                    if err:
                        self._last_error = err
                break
            if len(chunk) < self._frame_bytes:
                chunk = chunk + b"\x00" * (self._frame_bytes - len(chunk))
            with self._lock:
                self._ring.append(chunk)
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
        """从常驻流截取 pre-roll + duration_sec 写入 WAV（不再 spawn arecord）。"""
        if not self.is_running and not self.start():
            return None, 0.0

        output_path = Path(output_path)
        pre_frames = max(1, int(round(pre_roll_sec / (self._frame_ms / 1000.0))))
        target_bytes = int(max(0.2, float(duration_sec)) * self._sample_rate * 2)

        with self._record_lock:
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
            self._last_error = "persistent mic tap timeout"
            return None, 0.0

        if len(pcm) > target_bytes:
            pcm = pcm[:target_bytes]
        if len(pcm) < 2000:
            return None, 0.0

        write_pcm_wav(output_path, pcm, sample_rate=self._sample_rate)
        duration = len(pcm) / (self._sample_rate * 2)
        return output_path, duration

    def record_until_silence(
        self,
        output_path: str | Path,
        *,
        config: VadConfig | None = None,
        pre_roll_sec: float = 0.45,
    ) -> tuple[Path | None, float, str]:
        """从常驻流 VAD 截断：检测到说话后，静音持续一段时间自动结束。"""
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

        with self._record_lock:
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
            return None, 0.0, stop_reason

        if len(pcm) < 2000:
            self._last_error = "pcm_too_short"
            return None, 0.0, "pcm_too_short"

        if not frames_contain_speech(
            [pcm[i : i + self._frame_bytes] for i in range(0, len(pcm), self._frame_bytes)],
            config=cfg,
        ):
            self._last_error = "no_speech_detected"
            return None, 0.0, "no_speech_detected"

        write_pcm_wav(output_path, pcm, sample_rate=self._sample_rate)
        duration = len(pcm) / (self._sample_rate * 2)
        return output_path, duration, stop_reason
