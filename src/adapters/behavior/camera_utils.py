from __future__ import annotations

import glob
import re
import subprocess
import threading
from pathlib import Path

import cv2


class LatestFrameBus:
    """单摄像头多消费者的最新帧广播（避免 vision/behavior 重复 open 同一设备）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: object | None = None
        self._seq = 0

    def publish(self, frame) -> None:
        with self._lock:
            self._frame = frame.copy()
            self._seq += 1

    def get_latest_copy(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq


def _v4l2_device_has_video_capture(device_path: str) -> bool:
    """仅保留 Device Caps 含 Video Capture 的节点，跳过 C920 的 Metadata 节点。"""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--all"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True

    in_device_caps = False
    for line in result.stdout.splitlines():
        if line.strip().startswith("Device Caps"):
            in_device_caps = True
            continue
        if not in_device_caps:
            continue
        if line.startswith("\t") or line.startswith(" "):
            if line.strip() == "Video Capture":
                return True
            continue
        break
    return False


def _camera_label_for_node(node: str) -> str:
    name = Path(f"/sys/class/video4linux/{Path(node).name}/name")
    if not name.is_file():
        return ""
    return name.read_text(encoding="utf-8", errors="replace").strip()


def _camera_priority(label: str) -> int:
    upper = label.upper()
    if "C920" in upper or "WEBCAM" in upper:
        return 3
    if "CAMERA" in upper or "UVC" in upper:
        return 2
    return 1


def _enumerate_capture_candidates() -> list[tuple[int, str, str]]:
    """返回 [(opencv_index, device_path, label)]，已过滤 metadata-only 节点。"""
    candidates: list[tuple[int, str, str, int]] = []
    for node in sorted(glob.glob("/dev/video*"), key=lambda p: int(re.search(r"(\d+)$", p).group(1)) if re.search(r"(\d+)$", p) else 0):
        match = re.search(r"video(\d+)$", node)
        if not match:
            continue
        idx = int(match.group(1))
        if not _v4l2_device_has_video_capture(node):
            continue
        label = _camera_label_for_node(node)
        candidates.append((_camera_priority(label), idx, node, label))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [(idx, node, label) for _, idx, node, label in candidates]


def _opencv_log_level_silent() -> int | None:
    for name in ("LOG_LEVEL_SILENT", "LOG_LEVEL_ERROR"):
        level = getattr(cv2, name, None)
        if level is not None:
            return int(level)
    return None


def _probe_capture(*, index: int | None = None, device_path: str | None = None) -> bool:
    """确认设备能读到真实画面，而非 metadata 节点误报 isOpened。"""
    silent = _opencv_log_level_silent()
    old_level = None
    if silent is not None and hasattr(cv2, "setLogLevel"):
        old_level = cv2.getLogLevel()
        cv2.setLogLevel(silent)
    try:
        if device_path:
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        elif index is not None:
            cap = cv2.VideoCapture(index)
        else:
            return False
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        cap.release()
        return bool(ok and frame is not None)
    except Exception:
        return False
    finally:
        if old_level is not None and hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(old_level)


def resolve_camera_index(explicit: int | str | None = None) -> int:
    """解析 OpenCV 摄像头索引；auto 时优先 C920 的真实采集节点（跳过 metadata）。"""
    if explicit is not None:
        raw = str(explicit).strip().lower()
        if raw and raw not in {"auto", "none"}:
            try:
                idx = int(raw)
            except ValueError:
                idx = 0
            if _probe_capture(index=idx):
                return idx

    for idx, device_path, _label in _enumerate_capture_candidates():
        if _probe_capture(device_path=device_path):
            return idx

    for idx in (0, 1, 2, 3):
        if _probe_capture(index=idx):
            return idx
    return 0


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """打开摄像头并尽量减小缓冲，避免循环读到过期帧。"""
    for _prio, _idx, device_path, _label in _enumerate_capture_candidates():
        if _idx == index:
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            break
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def grab_latest_frame(cap: cv2.VideoCapture, *, flush: int = 4) -> tuple[bool, object | None]:
    """连读若干帧并返回最后一帧（丢掉队列里的旧画面）。"""
    ok = False
    frame = None
    for _ in range(max(1, flush)):
        ok, frame = cap.read()
    return ok, frame


def warmup_camera(cap: cv2.VideoCapture, frames: int = 8) -> None:
    for _ in range(frames):
        cap.read()
