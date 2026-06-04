"""全栈感知统一节拍（视觉 / 行为 / 情绪 OM 对齐）。"""

from __future__ import annotations

import os

# 主时钟：摄像头采集 + 行为 YOLO 双 OM 对齐到同一 tick
DEFAULT_PERCEPTION_HZ = 4.0


def perception_hz() -> float:
    raw = os.environ.get("EMBED_PERCEPTION_HZ", "").strip()
    if raw:
        try:
            return max(1.0, min(15.0, float(raw)))
        except ValueError:
            pass
    return DEFAULT_PERCEPTION_HZ


def perception_interval_sec() -> float:
    return 1.0 / perception_hz()


def vision_target_fps() -> float:
    return perception_hz()


def behavior_inference_interval_sec() -> float:
    return perception_interval_sec()


def emotion_every_n_frames() -> int:
    """4 Hz 下每 4 帧 ≈ 1 次/秒 WuJie OM 情绪推理。"""
    hz = perception_hz()
    return max(1, int(round(hz)))
