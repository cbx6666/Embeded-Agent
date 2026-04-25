"""几何与统计纯函数：EAR、PERCLOS 窗、疲劳档位滞回、人脸 bbox。

仅供 `vision_affect` 适配器内部使用；内核不 import 本模块。
不依赖 YOLO；人脸区域来自 MediaPipe Face Mesh 关键点外接框。
"""

from __future__ import annotations

import time
from collections import deque
from typing import Literal

import numpy as np

# 与 `user_fatigue_updated` payload 对齐；定义在适配器侧，内核接入状态时可再迁入 agent.state。
FatigueLevel = Literal["none", "mild", "moderate", "high"]

# MediaPipe Face Mesh 常用眼部六点（与 OpenCV + MediaPipe 困倦检测教程一致）
_RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)
_LEFT_EYE_IDX = (362, 385, 387, 263, 373, 380)
# 口部外轮廓六点（与常见 MAR / 哈欠检测教程一致，顺序同 EAR 六边形）
_MOUTH_SIX = (78, 81, 13, 311, 308, 14)


def eye_aspect_ratio(landmarks, eye_indices: tuple[int, ...], frame_w: int, frame_h: int) -> float:
    """计算单眼 EAR；眼睛闭合时 EAR 明显变小。"""
    pts = np.array(
        [[landmarks.landmark[i].x * frame_w, landmarks.landmark[i].y * frame_h] for i in eye_indices],
        dtype=np.float64,
    )
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    if h < 1e-6:
        return 1.0
    return float((v1 + v2) / (2.0 * h))


def mean_ear(landmarks, frame_w: int, frame_h: int) -> float:
    r = eye_aspect_ratio(landmarks, _RIGHT_EYE_IDX, frame_w, frame_h)
    l = eye_aspect_ratio(landmarks, _LEFT_EYE_IDX, frame_w, frame_h)
    return (r + l) / 2.0


def mouth_aspect_ratio(landmarks, frame_w: int, frame_h: int) -> float:
    """MAR：打哈欠/张口时口部「垂直」张量相对「水平」宽度变大，同 EAR 式六比。"""
    return eye_aspect_ratio(landmarks, _MOUTH_SIX, frame_w, frame_h)


def mean_mar(landmarks, frame_w: int, frame_h: int) -> float:
    """与 `mean_ear` 命名对称，单帧口部纵横向比。"""
    return mouth_aspect_ratio(landmarks, frame_w, frame_h)


def combined_fatigue_score(
    eye_perclos: float,
    yawn_frame_ratio: float,
    *,
    eye_weight: float = 0.55,
    mouth_weight: float = 0.45,
) -> float:
    """将眼部 PERCLOS 与口部打哈欠/张口帧比例融合为 0~1 标量，供滞回分档。"""
    w0 = max(0.0, min(1.0, float(eye_weight)))
    w1 = max(0.0, min(1.0, float(mouth_weight)))
    s = w0 + w1
    if s < 1e-6:
        return 0.0
    return (w0 * max(0.0, min(1.0, eye_perclos)) + w1 * max(0.0, min(1.0, yawn_frame_ratio))) / s


def face_bbox_from_landmarks(landmarks, frame_w: int, frame_h: int, margin: float = 0.08) -> tuple[int, int, int, int]:
    """由全脸关键点得到整数 bbox (x1, y1, x2, y2)，含边界裁剪。"""
    xs = [landmarks.landmark[i].x * frame_w for i in range(len(landmarks.landmark))]
    ys = [landmarks.landmark[i].y * frame_h for i in range(len(landmarks.landmark))]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    pad_x = int((x2 - x1) * margin)
    pad_y = int((y2 - y1) * margin)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w - 1, x2 + pad_x)
    y2 = min(frame_h - 1, y2 + pad_y)
    return x1, y1, x2, y2


class PercLosWindow:
    """滑动时间窗内「闭眼帧」比例，作为 PERCLOS 近似。"""

    def __init__(self, window_sec: float) -> None:
        self._window_sec = window_sec
        self._samples: deque[tuple[float, bool]] = deque()

    @property
    def window_sec(self) -> float:
        return self._window_sec

    def push(self, now: float, eye_closed: bool) -> None:
        self._samples.append((now, eye_closed))
        cutoff = now - self._window_sec
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def ratio(self) -> float:
        if not self._samples:
            return 0.0
        closed = sum(1 for _, c in self._samples if c)
        return closed / len(self._samples)


def map_fatigue_with_hysteresis(
    perclos: float,
    previous: FatigueLevel,
    *,
    up_mild: float = 0.18,
    up_mod: float = 0.30,
    up_high: float = 0.45,
    down_high: float = 0.38,
    down_mod: float = 0.22,
    down_mild: float = 0.12,
) -> FatigueLevel:
    """简单滞回，减少档位抖动。"""
    if previous == "none":
        if perclos >= up_high:
            return "high"
        if perclos >= up_mod:
            return "moderate"
        if perclos >= up_mild:
            return "mild"
        return "none"
    if previous == "mild":
        if perclos >= up_high:
            return "high"
        if perclos >= up_mod:
            return "moderate"
        if perclos < down_mild:
            return "none"
        return "mild"
    if previous == "moderate":
        if perclos >= up_high:
            return "high"
        if perclos < down_mod:
            return "mild" if perclos >= down_mild else "none"
        return "moderate"
    # high
    if perclos < down_high:
        if perclos < down_mod:
            return "mild" if perclos >= down_mild else "none"
        return "moderate"
    return "high"


def monotonic_ts() -> float:
    return time.monotonic()
