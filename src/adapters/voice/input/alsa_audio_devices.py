"""ALSA 录音/播放设备探测与路由。

按设备名称（UAC 盒子 / C920 摄像头等）解析，避免 USB 插拔或重启后 card 编号变化导致路由错乱。
典型拓扑：盒子负责唤醒监听 + 扬声器播放；摄像头麦负责用户说话录音。
"""
from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

_ARECORD_LINE = re.compile(
    r"^card\s+(\d+):\s*(.+?),\s*device\s+(\d+):\s*(.+)$",
    re.IGNORECASE,
)
_APLAY_LINE = re.compile(
    r"^card\s+(\d+):\s*(.+?),\s*device\s+(\d+):\s*(.+)$",
    re.IGNORECASE,
)
_ALSA_DEVICE = re.compile(
    r"^(?:plug(?:hw)?|hw|default|sysdefault):(\d+)(?:,(\d+))?$",
    re.IGNORECASE,
)

_CAMERA_KEYWORDS = (
    "camera",
    "webcam",
    "c920",
    "uvc",
    "video",
    "logitech",
    "hd pro",
)
_BOX_KEYWORDS = (
    "uacdemo",
    "uac",
    "voice",
    "speaker",
    "box",
    "demo",
    "audio gadget",
)
_USB_KEYWORDS = ("usb",)

_DEVICE_ALIASES: dict[str, tuple[str, ...]] = {
    "camera": ("camera", "webcam", "c920"),
    "webcam": ("camera", "webcam", "c920"),
    "c920": ("c920", "camera", "webcam"),
    "box": ("box", "uac", "speaker"),
    "speaker": ("speaker", "box", "uac"),
    "uac": ("uac", "uacdemo", "box"),
    "uacdemo": ("uacdemo", "uac", "box"),
}

AudioRole = Literal["user_capture", "wake_capture", "playback"]


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


def _device_haystack(item: dict[str, str | int]) -> str:
    return f"{item.get('card_name', '')} {item.get('device_name', '')}".lower()


def _matches_keywords(item: dict[str, str | int], keywords: tuple[str, ...]) -> bool:
    haystack = _device_haystack(item)
    return any(keyword in haystack for keyword in keywords)


def _score_user_capture_device(item: dict[str, str | int]) -> int:
    haystack = _device_haystack(item)
    score = 0
    for keyword in _CAMERA_KEYWORDS:
        if keyword in haystack:
            score += 12
    for keyword in _USB_KEYWORDS:
        if keyword in haystack:
            score += 2
    if "mic" in haystack or "capture" in haystack:
        score += 4
    if _matches_keywords(item, _BOX_KEYWORDS):
        score -= 10
    return score


def _score_wake_capture_device(item: dict[str, str | int]) -> int:
    haystack = _device_haystack(item)
    score = 0
    for keyword in _BOX_KEYWORDS:
        if keyword in haystack:
            score += 12
    for keyword in _USB_KEYWORDS:
        if keyword in haystack:
            score += 2
    if "mic" in haystack or "capture" in haystack:
        score += 4
    if _matches_keywords(item, _CAMERA_KEYWORDS):
        score -= 10
    return score


def _score_playback_device(item: dict[str, str | int]) -> int:
    haystack = _device_haystack(item)
    score = 0
    for keyword in _BOX_KEYWORDS:
        if keyword in haystack:
            score += 12
    for keyword in _USB_KEYWORDS:
        if keyword in haystack:
            score += 2
    if _matches_keywords(item, _CAMERA_KEYWORDS):
        score -= 20
    return score


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


def capture_device_node_exists(alsa_device: str | None) -> bool:
    """检查 ALSA 录音 PCM 节点是否存在。"""
    parsed = parse_alsa_device(alsa_device or "")
    if parsed is None:
        return False
    card, dev = parsed
    return Path(f"/dev/snd/pcmC{card}D{dev}c").exists()


def playback_device_node_exists(alsa_device: str | None) -> bool:
    """检查 ALSA 播放 PCM 节点是否存在。"""
    parsed = parse_alsa_device(alsa_device or "")
    if parsed is None:
        return False
    card, dev = parsed
    return Path(f"/dev/snd/pcmC{card}D{dev}p").exists()


def playback_card_available(card: int) -> bool:
    return any(int(item["card"]) == card for item in list_playback_devices())


def capture_card_available(card: int) -> bool:
    return any(int(item["card"]) == card for item in list_capture_devices())


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


def _pick_best_device(
    devices: list[dict[str, str | int]],
    *,
    scorer: Callable[[dict[str, str | int]], int],
    exclude_card: int | None = None,
) -> dict[str, str | int] | None:
    ranked: list[tuple[int, dict[str, str | int]]] = []
    for item in devices:
        if exclude_card is not None and int(item["card"]) == exclude_card:
            continue
        ranked.append((scorer(item), item))
    if not ranked and exclude_card is not None:
        ranked = [(scorer(item), item) for item in devices]
    if not ranked:
        return None
    ranked.sort(key=lambda pair: (-pair[0], int(pair[1]["card"]), int(pair[1]["device"])))
    return ranked[0][1]


def _find_device_by_alias(
    alias: str,
    devices: list[dict[str, str | int]],
    *,
    scorer: Callable[[dict[str, str | int]], int],
) -> dict[str, str | int] | None:
    keywords = _DEVICE_ALIASES.get(alias.lower())
    if keywords is None:
        return None
    for keyword in keywords:
        for item in devices:
            if keyword in _device_haystack(item):
                return item
    return _pick_best_device(devices, scorer=scorer)


def _normalize_explicit(
    explicit: str | None,
    *,
    env_var: str,
) -> str:
    return (explicit or os.environ.get(env_var) or "").strip()


def _resolve_explicit_device(
    explicit: str,
    *,
    role: AudioRole,
) -> str | None:
    lowered = explicit.lower()
    if lowered in {"", "auto", "default"}:
        return None
    if lowered in {"same", "capture"} and role == "wake_capture":
        return "__same_as_user__"

    if role == "playback":
        devices = list_playback_devices()
        scorer = _score_playback_device
    else:
        devices = list_capture_devices()
        scorer = _score_user_capture_device if role == "user_capture" else _score_wake_capture_device

    if lowered in _DEVICE_ALIASES:
        item = _find_device_by_alias(lowered, devices, scorer=scorer)
        return str(item["alsa_device"]) if item is not None else None

    parsed = parse_alsa_device(explicit)
    if parsed is None:
        return explicit

    card, _dev = parsed
    if role == "playback":
        if playback_card_available(card) or playback_device_node_exists(explicit):
            return explicit
        return None

    if capture_card_available(card) or capture_device_node_exists(explicit):
        return explicit
    return None


def resolve_user_capture_device(
    *,
    explicit: str | None = None,
    env_var: str = "EMBED_VOICE_CAPTURE_ALSA_DEVICE",
    exclude_card: int | None = None,
) -> str:
    """解析用户说话录音设备（优先摄像头麦）。"""
    chosen = _normalize_explicit(explicit, env_var=env_var)
    resolved = _resolve_explicit_device(chosen, role="user_capture") if chosen else None
    if resolved:
        return resolved

    devices = list_capture_devices()
    if not devices:
        return "plughw:0,0"

    item = _pick_best_device(
        devices,
        scorer=_score_user_capture_device,
        exclude_card=exclude_card,
    )
    return str(item["alsa_device"]) if item is not None else "plughw:0,0"


def resolve_wake_capture_device(
    *,
    explicit: str | None = None,
    env_var: str = "EMBED_WAKE_CAPTURE_ALSA_DEVICE",
    fallback: str | None = None,
) -> str:
    """解析唤醒词监听设备（优先盒子麦）。"""
    chosen = _normalize_explicit(explicit, env_var=env_var)
    if chosen.lower() in {"same", "capture"}:
        return fallback or resolve_user_capture_device()
    resolved = _resolve_explicit_device(chosen, role="wake_capture") if chosen else None
    if resolved == "__same_as_user__":
        return fallback or resolve_user_capture_device()
    if resolved:
        return resolved

    devices = list_capture_devices()
    if not devices:
        return fallback or "plughw:0,0"

    item = _pick_best_device(devices, scorer=_score_wake_capture_device)
    if item is None:
        return fallback or "plughw:0,0"
    return str(item["alsa_device"])


def resolve_capture_device(
    *,
    explicit: str | None = None,
    env_var: str = "EMBED_VOICE_CAPTURE_ALSA_DEVICE",
    exclude_card: int | None = None,
) -> str:
    """兼容旧调用：等同 resolve_user_capture_device。"""
    return resolve_user_capture_device(
        explicit=explicit,
        env_var=env_var,
        exclude_card=exclude_card,
    )


def resolve_playback_device(
    *,
    explicit: str | None = None,
    prefer_card: int | None = None,
    exclude_card: int | None = None,
    env_var: str = "EMBED_TTS_ALSA_DEVICE",
) -> str | None:
    """解析 TTS 播放设备（优先盒子扬声器，忽略无播放能力的摄像头卡）。"""
    chosen = _normalize_explicit(explicit, env_var=env_var)
    resolved = _resolve_explicit_device(chosen, role="playback") if chosen else None
    if resolved:
        return resolved

    devices = list_playback_devices()
    if not devices:
        return None

    if prefer_card is not None:
        for item in devices:
            if int(item["card"]) == prefer_card:
                return str(item["alsa_device"])

    item = _pick_best_device(
        devices,
        scorer=_score_playback_device,
        exclude_card=exclude_card,
    )
    return str(item["alsa_device"]) if item is not None else str(devices[0]["alsa_device"])


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
    wake_explicit: str | None = None,
    playback_explicit: str | None = None,
    split_input_output: bool = True,
) -> tuple[str, str, str | None]:
    """解析「摄像头用户麦 + 盒子唤醒麦 + 盒子扬声器」路由（均按名称匹配，不依赖 card 编号）。"""
    del split_input_output  # 保留参数兼容；名称路由不再用 card 互斥硬排除

    user_capture = resolve_user_capture_device(explicit=capture_explicit)
    wake_capture = resolve_wake_capture_device(
        explicit=wake_explicit,
        fallback=user_capture,
    )
    playback = resolve_playback_device(
        explicit=playback_explicit,
        prefer_card=card_from_alsa_device(wake_capture),
    )
    return user_capture, wake_capture, playback


def describe_device(alsa_device: str | None) -> str:
    """返回 plughw:X,Y (card_name) 便于启动日志阅读。"""
    if not alsa_device:
        return "(未检测到)"
    parsed = parse_alsa_device(alsa_device)
    if parsed is None:
        return alsa_device
    card, _dev = parsed
    for item in list_capture_devices():
        if int(item["card"]) == card:
            return f"{alsa_device} ({item['card_name']})"
    for item in list_playback_devices():
        if int(item["card"]) == card:
            return f"{alsa_device} ({item['card_name']})"
    return alsa_device


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
    """播放前解析设备并设置硬件音量；无效设备返回 None。"""
    if not alsa_device:
        return None
    if not playback_device_node_exists(alsa_device) and not playback_card_available(
        card_from_alsa_device(alsa_device) or -1
    ):
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
    """供 TTS 使用的播放设备解析入口。"""
    del prefer_capture_device, split_input_output
    explicit_norm = (explicit or "").strip()
    cached_explicit = None if explicit_norm.lower() in {"auto", "default", ""} else explicit_norm
    return get_cached_playback_device(cached_explicit, None, None)
