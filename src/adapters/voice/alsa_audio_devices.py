"""ALSA 录音/播放设备探测与路由。

典型板端拓扑：摄像头 USB 麦（仅 capture）+ 板载/外接扬声器（仅 playback）。
默认录放分离，避免同一张声卡既播 TTS 又被当作麦克风。
"""
from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

_ARECORD_LINE = re.compile(
    r"^card\s+(\d+):\s*(.+?),\s*device\s+(\d+):\s*(.+)$",
    re.IGNORECASE,
)
_CAPTURE_KEYWORDS = (
    "camera",
    "webcam",
    "c920",
    "uvc",
    "video",
    "logitech",
    "hd pro",
)
_USB_KEYWORDS = ("usb",)
_APLAY_LINE = re.compile(
    r"^card\s+(\d+):\s*(.+?),\s*device\s+(\d+):\s*(.+)$",
    re.IGNORECASE,
)
_ALSA_DEVICE = re.compile(
    r"^(?:plug(?:hw)?|hw|default|sysdefault):(\d+)(?:,(\d+))?$",
    re.IGNORECASE,
)

_RESOLVED_PLAYBACK_DEVICE: str | None | object = object()


def parse_arecord_list_output(text: str) -> list[dict[str, str | int]]:
    """解析 `arecord -l` 输出，返回 capture 设备列表。"""
    devices: list[dict[str, str | int]] = []
    for line in text.splitlines():
        match = _ARECORD_LINE.match(line.strip())
        if not match:
            continue
        card = int(match.group(1))
        device = int(match.group(3))
        devices.append(
            {
                "card": card,
                "device": device,
                "card_name": match.group(2).strip(),
                "device_name": match.group(4).strip(),
                "alsa_device": format_alsa_device(card, device),
            }
        )
    return devices


def parse_aplay_list_output(text: str) -> list[dict[str, str | int]]:
    """解析 `aplay -l` 输出，返回 playback 设备列表。"""
    devices: list[dict[str, str | int]] = []
    for line in text.splitlines():
        match = _APLAY_LINE.match(line.strip())
        if not match:
            continue
        card = int(match.group(1))
        device = int(match.group(3))
        devices.append(
            {
                "card": card,
                "device": device,
                "card_name": match.group(2).strip(),
                "device_name": match.group(4).strip(),
                "alsa_device": format_alsa_device(card, device),
            }
        )
    return devices


def parse_alsa_device(device: str | None) -> tuple[int, int] | None:
    """从 plughw:1,0 / hw:1,0 解析 (card, device)。"""
    if not device:
        return None
    stripped = device.strip()
    match = _ALSA_DEVICE.match(stripped)
    if not match:
        return None
    card = int(match.group(1))
    dev = int(match.group(2) or "0")
    return card, dev


def format_alsa_device(card: int, device: int, *, plug: bool = True) -> str:
    prefix = "plughw" if plug else "hw"
    return f"{prefix}:{card},{device}"


def list_capture_devices() -> list[dict[str, str | int]]:
    """列出系统上 `arecord -l` 报告的可录音 ALSA 设备。"""
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return parse_arecord_list_output(result.stdout)


def _score_capture_device(item: dict[str, str | int]) -> int:
    haystack = f"{item.get('card_name', '')} {item.get('device_name', '')}".lower()
    score = 0
    for keyword in _CAPTURE_KEYWORDS:
        if keyword in haystack:
            score += 10
    for keyword in _USB_KEYWORDS:
        if keyword in haystack:
            score += 2
    if "mic" in haystack or "capture" in haystack:
        score += 4
    # 板载语音板名称里常带 demo/uac，默认不作为摄像头麦
    if "uacdemo" in haystack or "voice" in haystack:
        score -= 8
    return score


def capture_device_node_exists(alsa_device: str | None) -> bool:
    """检查 ALSA 录音 PCM 节点是否存在。"""
    parsed = parse_alsa_device(alsa_device or "")
    if parsed is None:
        return False
    card, dev = parsed
    return Path(f"/dev/snd/pcmC{card}D{dev}c").exists()


def wait_for_capture_device(
    alsa_device: str,
    *,
    timeout_sec: float = 2.5,
    poll_ms: int = 100,
) -> bool:
    """等待指定录音设备节点就绪（USB 麦释放后可能需要短暂恢复）。"""
    import time

    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    while time.monotonic() < deadline:
        if capture_device_node_exists(alsa_device):
            return True
        time.sleep(max(0.05, poll_ms / 1000.0))
    return capture_device_node_exists(alsa_device)


def resolve_capture_device(
    *,
    explicit: str | None = None,
    env_var: str = "EMBED_VOICE_CAPTURE_ALSA_DEVICE",
    exclude_card: int | None = None,
) -> str:
    """解析麦克风录音设备。

    优先级：显式参数 > 环境变量 > 摄像头/USB 关键词匹配 > 第一个非 exclude 的设备 > plughw:0,0。
    """
    chosen = (explicit or os.environ.get(env_var) or "").strip()
    if chosen and chosen.lower() not in {"auto", "default"}:
        return chosen

    devices = list_capture_devices()
    if not devices:
        return "plughw:0,0"

    ranked: list[tuple[int, dict[str, str | int]]] = []
    for item in devices:
        if exclude_card is not None and int(item["card"]) == exclude_card:
            continue
        ranked.append((_score_capture_device(item), item))
    if not ranked:
        ranked = [(_score_capture_device(item), item) for item in devices]

    ranked.sort(key=lambda pair: (-pair[0], int(pair[1]["card"]), int(pair[1]["device"])))
    return str(ranked[0][1]["alsa_device"])


def list_playback_devices() -> list[dict[str, str | int]]:
    """列出系统上 `aplay -l` 报告的可播放 ALSA 设备。"""
    try:
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return parse_aplay_list_output(result.stdout)


def resolve_playback_device(
    *,
    explicit: str | None = None,
    prefer_card: int | None = None,
    exclude_card: int | None = None,
    env_var: str = "EMBED_TTS_ALSA_DEVICE",
) -> str | None:
    """解析 TTS 播放设备。

    优先级：显式参数 > 环境变量 > prefer_card 同 card > 第一个可播放且非 exclude_card 的设备。
    """
    chosen = (explicit or os.environ.get(env_var) or "").strip()
    if chosen and chosen.lower() not in {"auto", "default"}:
        return chosen

    devices = list_playback_devices()
    if not devices:
        return None

    filtered = [
        item
        for item in devices
        if exclude_card is None or int(item["card"]) != exclude_card
    ] or list(devices)

    if prefer_card is not None:
        for item in filtered:
            if int(item["card"]) == prefer_card:
                return str(item["alsa_device"])

    return str(filtered[0]["alsa_device"])


@lru_cache(maxsize=8)
def get_cached_playback_device(
    explicit: str | None,
    prefer_card: int | None,
    exclude_card: int | None,
) -> str | None:
    """带进程内缓存的播放设备解析。"""
    return resolve_playback_device(
        explicit=explicit,
        prefer_card=prefer_card,
        exclude_card=exclude_card,
    )


def invalidate_playback_device_cache() -> None:
    get_cached_playback_device.cache_clear()


def resolve_voice_pipeline_devices(
    *,
    capture_explicit: str | None = None,
    playback_explicit: str | None = None,
    split_input_output: bool = True,
) -> tuple[str, str | None]:
    """一次性解析「摄像头麦 + 独立扬声器」路由。"""
    playback_card = card_from_alsa_device(
        (playback_explicit or os.environ.get("EMBED_TTS_ALSA_DEVICE") or "").strip()
        if (playback_explicit or os.environ.get("EMBED_TTS_ALSA_DEVICE") or "").strip().lower()
        not in {"auto", "default", ""}
        else None
    )
    capture = resolve_capture_device(
        explicit=capture_explicit,
        exclude_card=playback_card if split_input_output else None,
    )
    capture_card = card_from_alsa_device(capture)
    playback = resolve_playback_device(
        explicit=playback_explicit,
        exclude_card=capture_card if split_input_output else None,
        prefer_card=None if split_input_output else capture_card,
    )
    return capture, playback


def find_sounddevice_output_index(*, prefer_card: int | None = None) -> int | None:
    """在 sounddevice 设备列表中选取可输出设备，尽量与 prefer_card 对齐。"""
    try:
        import sounddevice as sd
    except ImportError:
        return None

    devices = sd.query_devices()
    candidates: list[tuple[int, int]] = []
    for index, info in enumerate(devices):
        if int(info.get("max_output_channels") or 0) <= 0:
            continue
        name = str(info.get("name", "")).lower()
        score = 0
        if prefer_card is not None:
            if f"hw:{prefer_card}," in name or f"(hw:{prefer_card}," in name:
                score += 10
            if f"card {prefer_card}" in name:
                score += 8
        candidates.append((score, index))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def card_from_alsa_device(device: str | None) -> int | None:
    parsed = parse_alsa_device(device)
    return parsed[0] if parsed is not None else None


def ensure_playback_volume(card: int, *, control: str = "PCM") -> None:
    """播放前将指定声卡 PCM 音量拉高并取消静音（板载 USB 扬声器默认常偏低）。"""
    if card < 0:
        return
    try:
        subprocess.run(
            ["amixer", "-c", str(card), "set", control, "100%", "unmute"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def prepare_playback_device(alsa_device: str | None) -> str | None:
    """播放前解析设备并设置硬件音量。"""
    if not alsa_device:
        return None
    card = card_from_alsa_device(alsa_device)
    if card is not None:
        ensure_playback_volume(card)
    return alsa_device


def playback_device_for_tts(
    *,
    explicit: str | None = None,
    prefer_capture_device: str | None = None,
    split_input_output: bool = True,
) -> str | None:
    """供 TTS 使用的播放设备解析入口。

    split_input_output=True（默认）时：播放设备与麦克风 card 分离，避免「同卡既录又播」。
    """
    prefer_card: int | None = None
    exclude_card: int | None = None
    capture_card = card_from_alsa_device(prefer_capture_device)
    if split_input_output:
        exclude_card = capture_card
    else:
        prefer_card = capture_card
    explicit_norm = (explicit or "").strip()
    cached_explicit = None if explicit_norm.lower() in {"auto", "default", ""} else explicit_norm
    return get_cached_playback_device(cached_explicit, prefer_card, exclude_card)
