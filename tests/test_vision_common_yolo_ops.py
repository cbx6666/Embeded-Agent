import numpy as np

from src.adapters.vision_common.preprocess import letterbox_bgr_for_yolo
from src.adapters.vision_common.yolo_ultralytics_ops import (
    YoloLetterboxMeta,
    decode_yolo_detect_output,
)


def test_letterbox_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, meta = letterbox_bgr_for_yolo(img, imgsz=320)
    assert tensor.shape == (1, 3, 320, 320)
    assert meta.orig_shape == (480, 640)


def test_decode_detect_empty():
    meta = YoloLetterboxMeta(orig_shape=(100, 100), ratio_pad=((1.0, 1.0), (0.0, 0.0)))
    flat = np.zeros(84 * 100, dtype=np.float32)
    boxes = decode_yolo_detect_output(flat, meta, conf_thres=0.99, imgsz=320)
    assert boxes == []
