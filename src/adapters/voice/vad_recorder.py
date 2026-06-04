from __future__ import annotations

"""基于能量检测的 VAD 录音：检测到说话后，静音持续一段时间自动结束。"""

import math
import struct
import subprocess
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VadConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    speech_energy_threshold: float = 450.0
    silence_duration_sec: float = 0.8
    max_duration_sec: float = 15.0
    min_speech_duration_sec: float = 0.35
    initial_timeout_sec: float = 8.0
    pre_roll_ms: int = 200
    min_elapsed_before_end_sec: float = 0.0


def pcm_rms(pcm_chunk: bytes) -> float:
    if len(pcm_chunk) < 2:
        return 0.0
    count = len(pcm_chunk) // 2
    samples = struct.unpack(f"<{count}h", pcm_chunk[: count * 2])
    if not samples:
        return 0.0
    mean_sq = sum(sample * sample for sample in samples) / len(samples)
    return math.sqrt(mean_sq)


def frames_contain_speech(frames: list[bytes], *, config: VadConfig) -> bool:
    """是否已出现足够长的语音段（用于 initial_timeout 判断）。"""
    if not frames:
        return False
    frame_sec = config.frame_ms / 1000.0
    min_speech_frames = max(1, int(round(config.min_speech_duration_sec / frame_sec)))
    speech_run = 0
    for frame in frames:
        if pcm_rms(frame) >= config.speech_energy_threshold:
            speech_run += 1
            if speech_run >= min_speech_frames:
                return True
        else:
            speech_run = 0
    return False


def detect_end_frame_index(
    frames: list[bytes],
    *,
    config: VadConfig,
) -> int | None:
    """对已分帧 PCM 做离线 VAD，返回结束帧 index（含）。"""
    if not frames:
        return None

    frame_sec = config.frame_ms / 1000.0
    silence_frames_needed = max(1, int(round(config.silence_duration_sec / frame_sec)))
    min_speech_frames = max(1, int(round(config.min_speech_duration_sec / frame_sec)))
    max_frames = max(1, int(round(config.max_duration_sec / frame_sec)))
    initial_timeout_frames = max(1, int(round(config.initial_timeout_sec / frame_sec)))

    pre_roll_frames = max(0, int(round(config.pre_roll_ms / config.frame_ms)))
    ring: list[bytes] = []
    collected: list[bytes] = []
    speech_started = False
    speech_frames = 0
    silence_run = 0

    for index, frame in enumerate(frames):
        energy = pcm_rms(frame)
        is_speech = energy >= config.speech_energy_threshold

        if not speech_started:
            ring.append(frame)
            if len(ring) > pre_roll_frames:
                ring.pop(0)
            if is_speech:
                speech_started = True
                collected.extend(ring)
                ring.clear()
                speech_frames = 1
                silence_run = 0
            elif index + 1 >= initial_timeout_frames:
                return None
            continue

        collected.append(frame)
        if is_speech:
            speech_frames += 1
            silence_run = 0
        else:
            silence_run += 1

        if speech_frames >= min_speech_frames and silence_run >= silence_frames_needed:
            elapsed_sec = (index + 1) * frame_sec
            if elapsed_sec >= config.min_elapsed_before_end_sec:
                return index

        if len(collected) >= max_frames:
            return index

    return None


def write_pcm_wav(output_path: Path, pcm_data: bytes, *, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def record_audio_vad_wav(
    output_path: str | Path,
    *,
    alsa_device: str,
    config: VadConfig | None = None,
    prepare_device: bool = True,
    on_first_frame: Callable[[], None] | None = None,
    debug: Any | None = None,
) -> tuple[Path | None, float, str]:
    """Linux arecord 流式采集 + VAD 截断，返回 (wav路径, 秒数, 结束原因)。"""
    cfg = config or VadConfig()
    import platform

    def _log(level: str, event: str, **fields: Any) -> None:
        if debug is None:
            return
        fn = getattr(debug, level, None)
        if callable(fn):
            fn(event, **fields)

    output_path = Path(output_path)
    stop_reason = "unknown"
    if platform.system() == "Linux" and prepare_device:
        from src.adapters.voice.baidu_asr_backend import release_capture_device

        release_capture_device(alsa_device, settle_ms=80)

    frame_bytes = int(cfg.sample_rate * 2 * cfg.frame_ms / 1000)
    if frame_bytes <= 0:
        frame_bytes = 960

    stderr_target = subprocess.PIPE
    try:
        proc = subprocess.Popen(
            [
                "arecord",
                "-D",
                alsa_device,
                "-f",
                "S16_LE",
                "-r",
                str(cfg.sample_rate),
                "-c",
                "1",
                "-t",
                "raw",
            ],
            stdout=subprocess.PIPE,
            stderr=stderr_target,
        )
    except FileNotFoundError:
        stop_reason = "arecord_not_found"
        _log("error", "arecord_not_found")
        print("[vad_recorder] arecord 未找到，请安装 alsa-utils", flush=True)
        return None, 0.0, stop_reason

    frames: list[bytes] = []
    arecord_err = ""
    started = False
    frame_sec = cfg.frame_ms / 1000.0
    max_frames = max(1, int(round(cfg.max_duration_sec / frame_sec)))
    initial_timeout_frames = max(1, int(round(cfg.initial_timeout_sec / frame_sec)))
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                stop_reason = "arecord_eof"
                break
            if not started:
                started = True
                if on_first_frame is not None:
                    on_first_frame()
            if len(chunk) < frame_bytes:
                chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
            frames.append(chunk)

            end_index = detect_end_frame_index(frames, config=cfg)
            if end_index is not None:
                frames = frames[: end_index + 1]
                stop_reason = "silence_after_speech"
                break

            if len(frames) >= max_frames:
                stop_reason = "max_duration"
                _log("warn", "vad_max_duration", seconds=cfg.max_duration_sec)
                print(
                    f"[vad_recorder] 已达最长录音 {cfg.max_duration_sec:.1f}s，自动结束",
                    flush=True,
                )
                break

            if (
                not frames_contain_speech(frames, config=cfg)
                and len(frames) >= initial_timeout_frames
            ):
                stop_reason = "initial_timeout_no_speech"
                _log(
                    "warn",
                    "vad_initial_timeout",
                    seconds=cfg.initial_timeout_sec,
                    frames=len(frames),
                )
                print(
                    f"[vad_recorder] {cfg.initial_timeout_sec:.1f}s 内未检测到语音，结束采集",
                    flush=True,
                )
                break
    finally:
        if proc.stderr is not None:
            try:
                arecord_err = proc.stderr.read().decode("utf-8", errors="replace").strip()
            except Exception:
                arecord_err = ""
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    if not frames:
        stop_reason = "arecord_no_data"
        if arecord_err:
            _log("error", "arecord_no_data", stderr=arecord_err)
            print(f"[vad_recorder] arecord 无数据：{arecord_err}", flush=True)
        return None, 0.0, stop_reason

    pcm = b"".join(frames)
    peak = 0.0
    for frame in frames:
        peak = max(peak, pcm_rms(frame))

    if len(pcm) < 2000:
        stop_reason = "pcm_too_short"
        _log("error", "pcm_too_short", bytes=len(pcm), peak_rms=round(peak, 1))
        if debug is not None:
            debug.save_bytes("failed_too_short.pcm", pcm)
        return None, 0.0, stop_reason

    has_speech = frames_contain_speech(frames, config=cfg)
    if not has_speech:
        stop_reason = "no_speech_detected"
        _log(
            "warn",
            "no_speech_detected",
            peak_rms=round(peak, 1),
            threshold=cfg.speech_energy_threshold,
            frames=len(frames),
        )
        if debug is not None:
            debug.save_bytes("failed_no_speech.pcm", pcm)
            debug.save_json(
                "vad_metrics.json",
                {
                    "stop_reason": stop_reason,
                    "peak_rms": peak,
                    "threshold": cfg.speech_energy_threshold,
                    "frame_count": len(frames),
                    "config": cfg.__dict__,
                },
            )
        return None, 0.0, stop_reason

    write_pcm_wav(output_path, pcm, sample_rate=cfg.sample_rate)
    duration_sec = len(pcm) / (cfg.sample_rate * 2)
    _log(
        "info",
        "vad_success",
        stop_reason=stop_reason,
        duration_sec=round(duration_sec, 2),
        peak_rms=round(peak, 1),
    )
    return output_path, duration_sec, stop_reason


def trim_wav_leading(
    input_path: str | Path,
    trim_sec: float,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """裁掉 WAV 开头若干秒（用于去掉并行播放的唤醒应答回声）。"""
    src = Path(input_path)
    dst = Path(output_path) if output_path is not None else src
    trim_sec = max(0.0, float(trim_sec))
    with wave.open(str(src), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    skip = int(trim_sec * rate) * channels * width
    if skip >= len(frames):
        trimmed = b""
    else:
        trimmed = frames[skip:]

    if len(trimmed) < width * channels:
        trimmed = b"\x00" * (width * channels)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(trimmed)
    return dst


def wav_duration_sec(path: str | Path) -> float:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate() or 16000
        return wf.getnframes() / float(rate)


def wav_peak_rms(path: str | Path) -> float:
    """返回 WAV 内最高帧 RMS，用于判断麦是否几乎没录到声。"""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate() or 16000
        pcm = wf.readframes(wf.getnframes())
    if len(pcm) < 4:
        return 0.0
    frame_bytes = max(int(rate * 0.03 * 2), 960)
    peak = 0.0
    for index in range(0, len(pcm), frame_bytes):
        peak = max(peak, pcm_rms(pcm[index : index + frame_bytes]))
    return peak
