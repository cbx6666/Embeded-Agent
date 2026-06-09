"""WAV 播放工具（TTS / 本地应答共用）。"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

from src.adapters.voice.input.alsa_audio_devices import (
    card_from_alsa_device,
    find_sounddevice_output_index,
    playback_device_for_tts,
    prepare_playback_device,
)


def write_generated_audio_wav(
    *,
    output_path: Path,
    samples,
    sample_rate: int,
) -> None:
    """将 sherpa GeneratedAudio.samples 写入 16-bit mono WAV。"""
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = np.clip(arr, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def play_wav_file(
    wav_path: str | Path,
    *,
    alsa_playback_device: str | None = None,
    prefer_capture_device: str | None = None,
    player_command: str = "aplay",
    prefer_sounddevice_first: bool = False,
) -> None:
    """Linux 板端默认 aplay；``prefer_sounddevice_first`` 时先走 PortAudio（适合唤醒应答）。"""
    import platform

    path = Path(wav_path)
    if not path.is_file():
        raise RuntimeError(f"音频文件不存在：{path}")

    system = platform.system()
    alsa_device = prepare_playback_device(
        playback_device_for_tts(
            explicit=alsa_playback_device,
            prefer_capture_device=prefer_capture_device,
        )
    )
    prefer_card = _sounddevice_prefer_card(
        alsa_playback_device=alsa_playback_device,
        prefer_capture_device=prefer_capture_device,
    )

    if system == "Linux" and prefer_sounddevice_first:
        if _play_with_sounddevice(path, prefer_card=prefer_card):
            return
        if _play_with_aplay(path, player_command, alsa_device):
            return
    elif system == "Linux" and _play_with_aplay(path, player_command, alsa_device):
        return
    if _play_with_sounddevice(path, prefer_card=prefer_card):
        return
    if system == "Windows":
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return
        except Exception:
            pass
    if system == "Darwin":
        try:
            subprocess.run(["afplay", str(path)], check=True, capture_output=True)
            return
        except Exception:
            pass
    if system == "Linux":
        for cmd_name in ("aplay", "paplay", "ffplay"):
            if _play_with_aplay(path, cmd_name, alsa_device):
                return

    hint = f"（已探测播放设备={alsa_device!r}）" if alsa_device else ""
    raise RuntimeError(
        f"无法播放音频文件 {path}{hint}，请安装 alsa-utils 或 sounddevice。"
    )


def _play_with_aplay(path: Path, player_cmd: str, alsa_device: str | None) -> bool:
    if not player_cmd or player_cmd == "auto":
        player_cmd = "aplay"
    if player_cmd != "aplay":
        try:
            subprocess.run([player_cmd, str(path)], check=True, capture_output=True, timeout=120)
            return True
        except Exception:
            return False
    import time

    cmd = ["aplay", str(path)]
    if alsa_device:
        cmd = ["aplay", "-D", alsa_device, str(path)]
    for attempt in range(3):
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
            if result.returncode == 0:
                return True
            stderr = (result.stderr or b"").decode("utf-8", errors="ignore")
            if "busy" in stderr.lower() and attempt < 2:
                time.sleep(0.05 * (attempt + 1))
                continue
        except Exception:
            if attempt >= 2:
                return False
            time.sleep(0.05 * (attempt + 1))
    return False


def _sounddevice_prefer_card(
    *,
    alsa_playback_device: str | None,
    prefer_capture_device: str | None,
) -> int | None:
    """sounddevice 输出应优先对齐扬声器声卡，而非用户麦克风声卡。"""
    return card_from_alsa_device(alsa_playback_device) or card_from_alsa_device(
        prefer_capture_device
    )


def _load_wav_for_sounddevice(path: Path) -> tuple[object, int, int] | None:
    try:
        import numpy as np

        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            data = wf.readframes(wf.getnframes())

        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
        return audio, rate, channels
    except Exception:
        return None


def _play_with_sounddevice(path: Path, *, prefer_card: int | None) -> bool:
    try:
        import sounddevice as sd

        loaded = _load_wav_for_sounddevice(path)
        if loaded is None:
            return False
        audio, rate, channels = loaded

        output_device = find_sounddevice_output_index(prefer_card=prefer_card)
        if output_device is None:
            return False

        device_info = sd.query_devices(output_device)
        device_rate = float(device_info.get("default_samplerate") or rate)
        play_rate = rate
        if abs(device_rate - rate) > 1.0:
            import numpy as np

            target_len = max(1, int(len(audio) * device_rate / rate))
            if channels > 1:
                resampled = np.empty((target_len, channels), dtype=np.float32)
                for ch in range(channels):
                    resampled[:, ch] = np.interp(
                        np.linspace(0, len(audio) - 1, target_len),
                        np.arange(len(audio)),
                        audio[:, ch],
                    )
                audio = resampled
            else:
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, target_len),
                    np.arange(len(audio)),
                    audio,
                )
            play_rate = int(device_rate)

        sd.play(audio, samplerate=play_rate, device=output_device)
        sd.wait()
        return True
    except Exception:
        return False


def play_wav_sounddevice_first(
    wav_path: str | Path,
    *,
    alsa_playback_device: str | None = None,
    prefer_capture_device: str | None = None,
    cancel_flag: object | None = None,
) -> bool:
    """唤醒应答等短音频：优先 PortAudio，失败再回退 aplay。返回是否被取消。"""
    import platform

    path = Path(wav_path)
    if not path.is_file():
        raise RuntimeError(f"音频文件不存在：{path}")

    alsa_device = prepare_playback_device(
        playback_device_for_tts(
            explicit=alsa_playback_device,
            prefer_capture_device=prefer_capture_device,
        )
    )
    prefer_card = _sounddevice_prefer_card(
        alsa_playback_device=alsa_playback_device,
        prefer_capture_device=prefer_capture_device,
    )

    if platform.system() == "Linux" and _play_with_sounddevice_cancellable(
        path,
        prefer_card=prefer_card,
        cancel_flag=cancel_flag,
    ):
        return bool(getattr(cancel_flag, "is_set", lambda: False)())

    play_wav_file(
        path,
        alsa_playback_device=alsa_playback_device,
        prefer_capture_device=prefer_capture_device,
        prefer_sounddevice_first=False,
    )
    return bool(getattr(cancel_flag, "is_set", lambda: False)())


def _play_with_sounddevice_cancellable(
    path: Path,
    *,
    prefer_card: int | None,
    cancel_flag: object | None = None,
) -> bool:
    try:
        import time

        import sounddevice as sd

        loaded = _load_wav_for_sounddevice(path)
        if loaded is None:
            return False
        audio, rate, channels = loaded

        output_device = find_sounddevice_output_index(prefer_card=prefer_card)
        if output_device is None:
            return False

        device_info = sd.query_devices(output_device)
        device_rate = float(device_info.get("default_samplerate") or rate)
        play_rate = rate
        if abs(device_rate - rate) > 1.0:
            import numpy as np

            target_len = max(1, int(len(audio) * device_rate / rate))
            if channels > 1:
                resampled = np.empty((target_len, channels), dtype=np.float32)
                for ch in range(channels):
                    resampled[:, ch] = np.interp(
                        np.linspace(0, len(audio) - 1, target_len),
                        np.arange(len(audio)),
                        audio[:, ch],
                    )
                audio = resampled
            else:
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, target_len),
                    np.arange(len(audio)),
                    audio,
                )
            play_rate = int(device_rate)

        sd.play(audio, samplerate=play_rate, device=output_device)
        while True:
            if cancel_flag is not None and getattr(cancel_flag, "is_set", lambda: False)():
                sd.stop()
                return True
            stream = sd.get_stream()
            if stream is None or not stream.active:
                break
            time.sleep(0.05)
        return True
    except Exception:
        return False
