from __future__ import annotations

import cv2


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
