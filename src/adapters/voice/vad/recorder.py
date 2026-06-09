from __future__ import annotations

"""VAD 算法与 WAV 工具；实际录音由 AudioInputManager tap 完成。"""

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path


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
