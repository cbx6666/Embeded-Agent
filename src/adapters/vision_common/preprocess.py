from __future__ import annotations

"""视觉前处理：YOLO letterbox 与表情灰度人脸块。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class YoloLetterboxMeta:
    """与 ultralytics scale_boxes / scale_coords 兼容的元数据。"""

    orig_shape: tuple[int, int]  # (h, w)
    ratio_pad: tuple[tuple[float, float], tuple[float, float]]


def letterbox_bgr_for_yolo(bgr: np.ndarray, imgsz: int = 320) -> tuple[np.ndarray, YoloLetterboxMeta]:
    """BGR 帧 → NCHW float32 [1,3,imgsz,imgsz]，与 Ultralytics 预测一致。"""
    from ultralytics.data.augment import LetterBox

    if bgr is None or bgr.size == 0:
        raise ValueError("empty frame")
    h, w = bgr.shape[:2]
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    params = lb.get_params({"img": bgr.copy()})
    padded = lb.apply_image({"img": bgr}, params)["img"]
    ratio_pad = (params["ratio"], (params["left"], params["top"]))
    # Ultralytics 模型输入为 RGB、/255
    rgb = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    meta = YoloLetterboxMeta(orig_shape=(h, w), ratio_pad=ratio_pad)
    return np.ascontiguousarray(rgb), meta


def resize_gray_face_patch(crop_bgr: np.ndarray, out_size: int = 48, center_crop: int = 44) -> np.ndarray:
    """与 WuJie OM 一致：灰度 48×48 → 中心 44×44 → [1,3,44,44] float32。"""
    import cv2

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)
    img = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    margin = (out_size - center_crop) // 2
    c = img[margin : margin + center_crop, margin : margin + center_crop, :]
    x = np.transpose(c, (2, 0, 1)).astype(np.float32)
    return np.expand_dims(x, axis=0)
