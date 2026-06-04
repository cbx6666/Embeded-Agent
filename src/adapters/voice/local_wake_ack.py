"""本地预加载唤醒应答音频：命中唤醒词后立即 aplay，不走云端 TTS。"""

from __future__ import annotations

import json
import random
import subprocess
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.adapters.voice.alsa_audio_devices import (
    playback_device_for_tts,
    prepare_playback_device,
)

DEFAULT_WAKE_ACK_DIR = Path("assets/voice/wake_ack")
DEFAULT_WAKE_ACK_PHRASES: tuple[tuple[str, str], ...] = (
    ("wozai_qingshuo", "我在，请说。"),
    ("zai_de", "在的，请说。"),
    ("ni_shuo", "嗯，你说。"),
    ("wozai", "我在。"),
)


@dataclass(frozen=True)
class WakeAckClip:
    clip_id: str
    text: str
    path: Path
    pcm_bytes: bytes
    duration_sec: float


class LocalWakeAckPlayer:
    """从本地 WAV 池随机/轮播播放唤醒应答。"""

    def __init__(
        self,
        *,
        ack_dir: str | Path = DEFAULT_WAKE_ACK_DIR,
        alsa_playback_device: str | None = None,
        prefer_capture_device: str | None = None,
        player_command: str = "aplay",
    ) -> None:
        self._ack_dir = Path(ack_dir).expanduser()
        self._alsa_playback_device = (alsa_playback_device or "").strip() or None
        self._prefer_capture_device = (prefer_capture_device or "").strip() or None
        self._player_command = player_command
        self._clips: list[WakeAckClip] = []
        self._round_robin = 0
        self._lock = threading.Lock()
        self._preloaded = False

    @property
    def clip_count(self) -> int:
        return len(self._clips)

    def preload(self) -> int:
        """把 manifest 中的 WAV 读入内存，返回成功加载条数。"""
        with self._lock:
            self._clips.clear()
            manifest = self._load_manifest()
            for item in manifest:
                path = self._ack_dir / item["file"]
                if not path.is_file():
                    continue
                pcm_bytes, duration_sec = _read_wav_pcm(path)
                self._clips.append(
                    WakeAckClip(
                        clip_id=str(item.get("id", path.stem)),
                        text=str(item.get("text", path.stem)),
                        path=path.resolve(),
                        pcm_bytes=pcm_bytes,
                        duration_sec=duration_sec,
                    )
                )
            self._preloaded = bool(self._clips)
            return len(self._clips)

    def play(self, *, strategy: str = "random") -> str | None:
        """播放一条本地应答，返回对应文案；失败返回 None。"""
        with self._lock:
            if not self._clips:
                return None
            if strategy == "round_robin":
                clip = self._clips[self._round_robin % len(self._clips)]
                self._round_robin += 1
            else:
                clip = random.choice(self._clips)

        device = prepare_playback_device(
            playback_device_for_tts(
                explicit=self._alsa_playback_device,
                prefer_capture_device=self._prefer_capture_device,
            )
        )
        ok = _play_pcm_via_aplay(
            clip.path,
            alsa_device=device,
            player_command=self._player_command,
        )
        if ok:
            print(f"[LocalWakeAck] 播放：{clip.text!r} ({clip.path.name})", flush=True)
            return clip.text
        return None

    def play_async(self, *, strategy: str = "random") -> tuple[threading.Thread | None, float, str | None]:
        """后台线程播放应答，返回 (线程, 预估时长秒, 文案)。"""
        with self._lock:
            if not self._clips:
                return None, 0.0, None
            if strategy == "round_robin":
                clip = self._clips[self._round_robin % len(self._clips)]
                self._round_robin += 1
            else:
                clip = random.choice(self._clips)

        device = prepare_playback_device(
            playback_device_for_tts(
                explicit=self._alsa_playback_device,
                prefer_capture_device=self._prefer_capture_device,
            )
        )

        def _run() -> None:
            ok = _play_pcm_via_aplay(
                clip.path,
                alsa_device=device,
                player_command=self._player_command,
            )
            if ok:
                print(f"[LocalWakeAck] 播放：{clip.text!r} ({clip.path.name})", flush=True)

        thread = threading.Thread(target=_run, name="LocalWakeAck", daemon=True)
        thread.start()
        return thread, clip.duration_sec, clip.text

    def _load_manifest(self) -> list[dict[str, Any]]:
        manifest_path = self._ack_dir / "manifest.json"
        if manifest_path.is_file():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict) and item.get("file")]

        clips: list[dict[str, Any]] = []
        for clip_id, text in DEFAULT_WAKE_ACK_PHRASES:
            for ext in (".wav", ".WAV"):
                candidate = self._ack_dir / f"{clip_id}{ext}"
                if candidate.is_file():
                    clips.append({"id": clip_id, "text": text, "file": candidate.name})
                    break
        return clips


def _read_wav_pcm(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate() or 16000
        duration = wf.getnframes() / float(rate)
    return frames, duration


def _play_pcm_via_aplay(
    wav_path: Path,
    *,
    alsa_device: str | None,
    player_command: str,
) -> bool:
    import platform

    if platform.system() != "Linux":
        try:
            subprocess.run([player_command, str(wav_path)], check=True, capture_output=True)
            return True
        except Exception:
            return False

    cmd = ["aplay", str(wav_path)]
    if alsa_device:
        cmd = ["aplay", "-D", alsa_device, str(wav_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=15)
        return True
    except Exception:
        return False


def default_ack_dir() -> Path:
    return DEFAULT_WAKE_ACK_DIR.resolve()


def missing_ack_files(ack_dir: Path | None = None) -> list[str]:
    """返回尚未生成的应答文件名列表。"""
    base = ack_dir or DEFAULT_WAKE_ACK_DIR
    missing: list[str] = []
    for clip_id, _ in DEFAULT_WAKE_ACK_PHRASES:
        if not (base / f"{clip_id}.wav").is_file():
            missing.append(f"{clip_id}.wav")
    return missing
