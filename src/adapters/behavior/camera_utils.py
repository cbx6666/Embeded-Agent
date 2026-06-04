from __future__ import annotations

import glob
import re
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


def resolve_camera_index(explicit: int | str | None = None) -> int:
    """解析 OpenCV 摄像头索引；explicit=auto/None 时扫描首个可打开的 /dev/video*。"""
    if explicit is not None:
        raw = str(explicit).strip().lower()
        if raw and raw not in {"auto", "none"}:
            try:
                idx = int(raw)
            except ValueError:
                idx = 0
            cap = cv2.VideoCapture(idx)
            opened = cap.isOpened()
            cap.release()
            if opened:
                return idx

    preferred: list[int] = []
    for node in sorted(glob.glob("/dev/video*")):
        name = Path(f"/sys/class/video4linux/{Path(node).name}/name")
        label = name.read_text(encoding="utf-8", errors="replace").strip() if name.is_file() else ""
        m = re.search(r"video(\d+)$", node)
        if not m:
            continue
        idx = int(m.group(1))
        if "C920" in label or "Webcam" in label:
            preferred.insert(0, idx)
        elif idx not in preferred:
            preferred.append(idx)

    for idx in preferred or (0, 1, 2, 3):
        cap = cv2.VideoCapture(idx)
        opened = cap.isOpened()
        cap.release()
        if opened:
            return idx
    return 0


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """打开摄像头并尽量减小缓冲，避免循环读到过期帧。"""
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        # 部分后端支持；不支持时静默忽略
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
