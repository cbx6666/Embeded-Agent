from __future__ import annotations

"""YOLO OM 原始输出 → 框/关键点（复用 Ultralytics NMS 与坐标还原）。"""

from dataclasses import dataclass

import numpy as np

from src.adapters.vision_common.preprocess import YoloLetterboxMeta

# COCO pose 17 点：左/右腕
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10


@dataclass
class DetectBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


def _reshape_yolo_head(flat: np.ndarray, channels: int) -> np.ndarray:
    n = flat.size // channels
    if n <= 0 or channels * n != flat.size:
        raise ValueError(f"cannot reshape om output size={flat.size} channels={channels}")
    return flat.reshape(1, channels, n)


def _is_e2e_detect(flat: np.ndarray) -> bool:
    return flat.size == 1 * 300 * 6


def _is_e2e_pose(flat: np.ndarray) -> bool:
    return flat.size == 1 * 300 * 57


def decode_yolo_detect_output(
    flat_output: np.ndarray,
    meta: YoloLetterboxMeta,
    *,
    nc: int = 80,
    conf_thres: float = 0.35,
    iou_thres: float = 0.45,
    classes: list[int] | None = None,
    imgsz: int = 320,
) -> list[DetectBox]:
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    if _is_e2e_detect(flat_output):
        pred = torch.from_numpy(flat_output.reshape(1, 300, 6).copy())
        dets = non_max_suppression(
            pred,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            end2end=True,
        )
        if not dets or dets[0] is None or len(dets[0]) == 0:
            return []
        det = dets[0]
        det[:, :4] = scale_boxes((imgsz, imgsz), det[:, :4], meta.orig_shape, ratio_pad=meta.ratio_pad)
        boxes: list[DetectBox] = []
        for row in det.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = row[:6]
            boxes.append(
                DetectBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf),
                    class_id=int(cls_id),
                )
            )
        return boxes

    channels = 4 + nc
    pred = torch.from_numpy(_reshape_yolo_head(flat_output, channels))
    dets = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=classes,
        nc=nc,
    )
    if not dets or dets[0] is None or len(dets[0]) == 0:
        return []
    det = dets[0]
    det[:, :4] = scale_boxes((imgsz, imgsz), det[:, :4], meta.orig_shape, ratio_pad=meta.ratio_pad)
    boxes: list[DetectBox] = []
    for row in det.cpu().numpy():
        x1, y1, x2, y2, conf, cls_id = row[:6]
        boxes.append(
            DetectBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(conf),
                class_id=int(cls_id),
            )
        )
    return boxes


@dataclass
class PosePerson:
    keypoints_xy: np.ndarray  # (17, 2)
    keypoints_conf: np.ndarray  # (17,)
    box_conf: float


def decode_yolo_pose_output(
    flat_output: np.ndarray,
    meta: YoloLetterboxMeta,
    *,
    nc: int = 1,
    nkpt: int = 17,
    ndim: int = 3,
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    imgsz: int = 320,
) -> list[PosePerson]:
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes, scale_coords

    if _is_e2e_pose(flat_output):
        pred = torch.from_numpy(flat_output.reshape(1, 300, 57).copy())
        dets = non_max_suppression(
            pred,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nc=nc,
            end2end=True,
        )
        if not dets or dets[0] is None or len(dets[0]) == 0:
            return []
        det = dets[0]
        det[:, :4] = scale_boxes((imgsz, imgsz), det[:, :4], meta.orig_shape, ratio_pad=meta.ratio_pad)
        people: list[PosePerson] = []
        kpt_shape = (nkpt, ndim)
        for row in det.cpu().numpy():
            kpts = row[6:].reshape(kpt_shape)
            kpts = scale_coords((imgsz, imgsz), kpts, meta.orig_shape, ratio_pad=meta.ratio_pad)
            conf = np.ones(nkpt, dtype=np.float32)
            if ndim == 3:
                conf = kpts[:, 2].astype(np.float32)
                kpts = kpts[:, :2]
            people.append(
                PosePerson(
                    keypoints_xy=kpts.astype(np.float32),
                    keypoints_conf=conf,
                    box_conf=float(row[4]),
                )
            )
        return people

    channels = 4 + nc + nkpt * ndim
    pred = torch.from_numpy(_reshape_yolo_head(flat_output, channels))
    dets = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        nc=nc,
    )
    if not dets or dets[0] is None or len(dets[0]) == 0:
        return []
    det = dets[0]
    det[:, :4] = scale_boxes((imgsz, imgsz), det[:, :4], meta.orig_shape, ratio_pad=meta.ratio_pad)
    people: list[PosePerson] = []
    kpt_shape = (nkpt, ndim)
    for row in det.cpu().numpy():
        kpts = row[6:].reshape(kpt_shape)
        kpts = scale_coords((imgsz, imgsz), kpts, meta.orig_shape, ratio_pad=meta.ratio_pad)
        conf = np.ones(nkpt, dtype=np.float32)
        if ndim == 3:
            conf = kpts[:, 2].astype(np.float32)
            kpts = kpts[:, :2]
        people.append(
            PosePerson(
                keypoints_xy=kpts.astype(np.float32),
                keypoints_conf=conf,
                box_conf=float(row[4]),
            )
        )
    return people
