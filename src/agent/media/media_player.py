from __future__ import annotations

"""实际音频播放（与业务决策解耦）。"""

import logging
import subprocess
import threading
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from src.agent.media.media_models import MediaPlaybackState, MediaTrack

logger = logging.getLogger(__name__)


class MediaPlayerBackend(ABC):
    @abstractmethod
    def play(self, track: MediaTrack, *, on_finished: Callable[[], None] | None = None) -> None:
        ...

    @abstractmethod
    def stop(self, reason: str = "user") -> None:
        ...

    @abstractmethod
    def pause(self) -> None:
        ...

    @abstractmethod
    def resume(self) -> None:
        ...

    @abstractmethod
    def is_playing(self) -> bool:
        ...


class LocalMediaPlayer(MediaPlayerBackend):
    """基于子进程的可取消本地播放器（wav 优先 aplay，其他格式尝试 ffplay/mpg123）。"""

    def __init__(
        self,
        *,
        alsa_device: str | None = None,
        prefer_capture_device: str | None = None,
        mock_duration_sec: float | None = None,
        playback_volume: float = 0.4,
    ) -> None:
        self._alsa_device = alsa_device
        self._prefer_capture_device = prefer_capture_device
        self._mock_duration_sec = mock_duration_sec
        self._playback_volume = max(0.05, min(1.0, float(playback_volume)))
        self._proc: subprocess.Popen | None = None
        self._decoder_proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = False
        self._playing = False
        self._current_track: MediaTrack | None = None
        self._on_finished: Callable[[], None] | None = None
        self._lock = threading.RLock()

    def play(self, track: MediaTrack, *, on_finished: Callable[[], None] | None = None) -> None:
        with self._lock:
            self.stop(reason="replace")
            self._current_track = track
            self._on_finished = on_finished
            self._stop_event.clear()
            self._paused = False
            self._playing = True
            self._thread = threading.Thread(
                target=self._play_loop,
                args=(track,),
                name=f"MediaPlayer-{track.id}",
                daemon=True,
            )
            self._thread.start()
        logger.info("[媒体播放] 开始：%s (%s)", track.title, track.path)

    def stop(self, reason: str = "user") -> None:
        with self._lock:
            self._stop_event.set()
            self._playing = False
            self._paused = False
            proc = self._proc
            decoder = self._decoder_proc
            self._proc = None
            self._decoder_proc = None
        for child in (proc, decoder):
            if child is not None and child.poll() is None:
                try:
                    child.terminate()
                    child.wait(timeout=2.0)
                except Exception:
                    try:
                        child.kill()
                    except Exception:
                        pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        logger.info("[媒体播放] 停止：reason=%s track=%s", reason, getattr(self._current_track, "id", None))

    def pause(self) -> None:
        if not self._playing:
            return
        self._paused = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(subprocess.signal.SIGSTOP)  # type: ignore[attr-defined]
            except Exception:
                pass
        logger.info("[媒体播放] 暂停")

    def resume(self) -> None:
        if not self._paused:
            return
        self._paused = False
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(subprocess.signal.SIGCONT)  # type: ignore[attr-defined]
            except Exception:
                pass
        logger.info("[媒体播放] 继续")

    def is_playing(self) -> bool:
        return self._playing and not self._stop_event.is_set()

    def _play_loop(self, track: MediaTrack) -> None:
        try:
            if self._mock_duration_sec is not None:
                self._mock_play(track, self._mock_duration_sec)
            else:
                self._real_play(track)
        except Exception as exc:
            logger.warning("[媒体播放] 播放异常：%s", exc)
        finally:
            with self._lock:
                finished = not self._stop_event.is_set()
                self._playing = False
                self._proc = None
                self._decoder_proc = None
                cb = self._on_finished
            if finished and cb is not None:
                cb()
            if finished:
                logger.info("[媒体播放] 自然结束：%s", track.id)

    def _mock_play(self, track: MediaTrack, duration: float) -> None:
        """测试/无音频环境：模拟播放时长。"""
        end = time.time() + duration
        while time.time() < end:
            if self._stop_event.is_set():
                return
            while self._paused and not self._stop_event.is_set():
                time.sleep(0.1)
            time.sleep(0.05)

    def _real_play(self, track: MediaTrack) -> None:
        path = Path(track.path)
        suffix = path.suffix.lower()
        if suffix == ".wav":
            self._play_wav(path)
        else:
            self._play_external(path)

    def configure_devices(
        self,
        *,
        alsa_device: str | None = None,
        prefer_capture_device: str | None = None,
    ) -> None:
        self._alsa_device = alsa_device
        self._prefer_capture_device = prefer_capture_device

    def _resolved_alsa_device(self) -> str | None:
        from src.adapters.voice.input.alsa_audio_devices import (
            playback_device_for_tts,
            prepare_playback_device,
        )

        return prepare_playback_device(
            playback_device_for_tts(
                explicit=self._alsa_device,
                prefer_capture_device=self._prefer_capture_device,
            )
        )

    def _wait_playback_proc(self, *, decoder: subprocess.Popen | None = None) -> None:
        """等待播放子进程结束；stop() 将 _proc 置 None 时安全退出。"""
        while not self._stop_event.is_set():
            proc = self._proc
            if proc is None:
                return
            if proc.poll() is not None:
                break
            while self._paused and not self._stop_event.is_set():
                time.sleep(0.1)
            time.sleep(0.05)
        if decoder is not None and decoder.poll() is None:
            try:
                decoder.wait(timeout=1.0)
            except Exception:
                pass

    def _play_wav(self, path: Path) -> None:
        alsa = self._resolved_alsa_device()
        vol = self._playback_volume
        # 压低媒体音量，避免扬声器灌麦导致唤醒词 KWS 失效，也为唤醒应答留出扬声器。
        if vol < 0.99 and self._try_play_wav_ducked(path, alsa_device=alsa, volume=vol):
            return
        cmd = ["aplay", "-q", str(path)]
        if alsa:
            cmd = ["aplay", "-q", "-D", alsa, str(path)]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._wait_playback_proc()
        proc = self._proc
        if proc is not None and proc.returncode not in (0, None) and not self._stop_event.is_set():
            err = ""
            try:
                err = (proc.stderr.read() or b"").decode("utf-8", errors="replace").strip()
            except Exception:
                pass
            logger.warning(
                "[媒体播放] aplay 失败 code=%s device=%s path=%s err=%s",
                proc.returncode,
                alsa,
                path,
                err[:200],
            )

    def _try_play_wav_ducked(
        self,
        path: Path,
        *,
        alsa_device: str | None,
        volume: float,
    ) -> bool:
        """ffmpeg 降音量后 pipe 给 aplay；失败则回退直连 aplay。"""
        import shutil

        if shutil.which("ffmpeg") is None:
            return False
        decode = [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-i",
            str(path),
            "-filter:a",
            f"volume={volume:.2f}",
            "-f",
            "wav",
            "pipe:1",
        ]
        play = ["aplay", "-q"]
        if alsa_device:
            play = ["aplay", "-q", "-D", alsa_device]
        try:
            decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._decoder_proc = decoder
            self._proc = subprocess.Popen(
                play,
                stdin=decoder.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if decoder.stdout is not None:
                decoder.stdout.close()
        except Exception:
            return False
        self._wait_playback_proc(decoder=decoder)
        return True

    def _play_external(self, path: Path) -> None:
        for player in (["ffplay", "-nodisp", "-autoexit", str(path)], ["mpg123", "-q", str(path)]):
            try:
                self._proc = subprocess.Popen(player, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                while self._proc.poll() is None:
                    if self._stop_event.is_set():
                        return
                    time.sleep(0.05)
                if self._proc.returncode == 0:
                    return
            except FileNotFoundError:
                continue
        # 回退：按 wav 头估算时长并 mock
        duration = _estimate_duration(path) or 30.0
        self._mock_play(MediaTrack(id="", title="", path=str(path), media_type="", category=""), duration)


def _estimate_duration(path: Path) -> float | None:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / rate
        except Exception:
            return None
    return None


class MediaPlayer:
    """播放器门面：同步更新 MediaPlaybackState。"""

    def __init__(self, backend: MediaPlayerBackend | None = None) -> None:
        self._backend = backend or LocalMediaPlayer()
        self._state = MediaPlaybackState()
        self._finished_callback: Callable[[], None] | None = None

    @property
    def playback_state(self) -> MediaPlaybackState:
        return self._state

    def set_finished_callback(self, callback: Callable[[], None] | None) -> None:
        self._finished_callback = callback

    def play(self, track: MediaTrack) -> None:
        self._state.interrupted_by_wake_word = False
        self._backend.play(track, on_finished=self._on_play_finished)

    def stop(self, reason: str = "user") -> None:
        if reason == "wake_word":
            self._state.interrupted_by_wake_word = True
        self._backend.stop(reason=reason)
        self._state.is_playing = False

    def pause(self) -> None:
        self._backend.pause()

    def resume(self) -> None:
        self._backend.resume()

    def next(self, track: MediaTrack) -> None:
        self.play(track)

    def is_playing(self) -> bool:
        return self._backend.is_playing()

    def _on_play_finished(self) -> None:
        self._state.is_playing = False
        self._state.last_finished_at = int(time.time())
        if self._finished_callback:
            self._finished_callback()
