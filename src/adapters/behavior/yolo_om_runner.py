from __future__ import annotations

"""YOLO26 detect + pose 双 OM 在 Ascend 上推理。"""

from pathlib import Path

import numpy as np

from src.adapters.behavior.phone_hand_detector import (
    COCO_CELL_PHONE_CLASS,
    KP_LEFT_WRIST,
    KP_RIGHT_WRIST,
    PhoneBox,
)
from src.adapters.vision_common.acl_runtime import AscendOmSession
from src.adapters.vision_common.preprocess import letterbox_bgr_for_yolo
from src.adapters.vision_common.yolo_ultralytics_ops import (
    PosePerson,
    decode_yolo_detect_output,
    decode_yolo_pose_output,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DETECT_OM = PROJECT_ROOT / "models" / "yolo26" / "yolo26n.om"
DEFAULT_POSE_OM = PROJECT_ROOT / "models" / "yolo26" / "yolo26n-pose.om"


def om_models_available(
    detect_om: Path | None = None,
    pose_om: Path | None = None,
) -> bool:
    d = detect_om or DEFAULT_DETECT_OM
    p = pose_om or DEFAULT_POSE_OM
    return d.is_file() and p.is_file()


class YoloOmPhonePoseRunner:
    """同一 letterbox 输入，分别跑 detect / pose 两个 OM。"""

    def __init__(
        self,
        detect_om: str | Path | None = None,
        pose_om: str | Path | None = None,
        device_id: int = 0,
        imgsz: int = 320,
        phone_conf: float = 0.35,
        pose_conf: float = 0.5,
        min_kpt_conf: float = 0.3,
    ) -> None:
        self.imgsz = int(imgsz)
        self.phone_conf = float(phone_conf)
        self.pose_conf = float(pose_conf)
        self.min_kpt_conf = float(min_kpt_conf)
        self._detect = AscendOmSession(detect_om or DEFAULT_DETECT_OM, device_id=device_id)
        self._pose = AscendOmSession(pose_om or DEFAULT_POSE_OM, device_id=device_id)

    def load(self) -> bool:
        return self._detect.load() and self._pose.load()

    @property
    def loaded(self) -> bool:
        return self._detect.loaded and self._pose.loaded

    def infer_phones_and_wrists(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[PhoneBox], list[tuple[float, float]], list[float], int, PosePerson | None]:
        tensor, meta = letterbox_bgr_for_yolo(frame_bgr, self.imgsz)
        det_flat = self._detect.execute(tensor)
        pose_flat = self._pose.execute(tensor)
        if det_flat is None or pose_flat is None:
            return [], [], [], 0, None

        # detect OM 只解手机；人在场由 pose OM 判断
        decode_conf = min(0.2, self.phone_conf)
        dets = decode_yolo_detect_output(
            det_flat,
            meta,
            conf_thres=decode_conf,
            classes=None,
            imgsz=self.imgsz,
        )
        phones = [
            PhoneBox(b.x1, b.y1, b.x2, b.y2, b.confidence)
            for b in dets
            if b.class_id == COCO_CELL_PHONE_CLASS and b.confidence >= self.phone_conf
        ]
        people = decode_yolo_pose_output(
            pose_flat,
            meta,
            conf_thres=self.pose_conf,
            imgsz=self.imgsz,
        )
        wrists: list[tuple[float, float]] = []
        confs: list[float] = []
        for person in people:
            self._append_wrist(person, KP_LEFT_WRIST, wrists, confs, self.min_kpt_conf)
            self._append_wrist(person, KP_RIGHT_WRIST, wrists, confs, self.min_kpt_conf)
        primary = max(people, key=lambda p: p.box_conf) if people else None
        return phones, wrists, confs, len(people), primary

    @staticmethod
    def _append_wrist(
        person: PosePerson,
        idx: int,
        wrists: list[tuple[float, float]],
        confs: list[float],
        min_kpt_conf: float,
    ) -> None:
        if idx >= len(person.keypoints_xy):
            return
        x, y = float(person.keypoints_xy[idx][0]), float(person.keypoints_xy[idx][1])
        if x <= 0 and y <= 0:
            return
        c = float(person.keypoints_conf[idx]) if idx < len(person.keypoints_conf) else 1.0
        if c < min_kpt_conf:
            return
        wrists.append((x, y))
        confs.append(c)

    def infer_detect_boxes(
        self,
        frame_bgr: np.ndarray,
        *,
        conf_thres: float = 0.2,
        classes: list[int] | None = None,
    ) -> list:
        """排障：解码检测 OM 全部框（不过滤手机类）。"""
        from src.adapters.vision_common.yolo_ultralytics_ops import DetectBox, decode_yolo_detect_output

        if not self.loaded:
            return []
        tensor, meta = letterbox_bgr_for_yolo(frame_bgr, self.imgsz)
        det_flat = self._detect.execute(tensor)
        if det_flat is None:
            return []
        boxes: list[DetectBox] = decode_yolo_detect_output(
            det_flat,
            meta,
            conf_thres=conf_thres,
            classes=classes,
            imgsz=self.imgsz,
        )
        return boxes
