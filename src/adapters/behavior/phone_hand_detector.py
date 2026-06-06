from __future__ import annotations

"""YOLO26 手机检测 + 手腕邻近 → 是否手持手机。"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

InferenceBackend = Literal["auto", "pt", "om"]

# COCO detect（仅手机类）
COCO_CELL_PHONE_CLASS = 67
PresencePhase = Literal["present", "absent_grace", "left"]
# COCO pose keypoint indices
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DETECT_MODEL = PROJECT_ROOT / "models" / "yolo26" / "yolo26n.pt"
DEFAULT_POSE_MODEL = PROJECT_ROOT / "models" / "yolo26" / "yolo26n-pose.pt"


@dataclass
class PhoneBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass
class PhoneHandFrameResult:
    """单帧检测结果。"""

    phone_in_hand: bool
    confidence: float
    phones: list[PhoneBox] = field(default_factory=list)
    wrist_near_phone: bool = False
    raw_phone_count: int = 0
    person_count: int = 0
    """兼容字段：有人在画内为 1，否则 0（由 pose 判断）。"""
    person_count_pose: int = 0
    person_visible: bool = False
    absent_frames: int = 0
    presence_phase: PresencePhase = "present"
    wrist_count: int = 0
    looking_down: bool = False
    head_down_assist: bool = False
    """未检出手机时，靠 Face Mesh 低头+手腕区域辅助判分心。"""
    posture: str = "unknown"
    activity: str = "unknown"
    posture_confidence: float = 0.0


def _point_to_bbox_distance(px: float, py: float, box: PhoneBox) -> float:
    dx = max(box.x1 - px, 0.0, px - box.x2)
    dy = max(box.y1 - py, 0.0, py - box.y2)
    return math.hypot(dx, dy)


def _bbox_scale(box: PhoneBox) -> float:
    w = max(box.x2 - box.x1, 1.0)
    h = max(box.y2 - box.y1, 1.0)
    return max(w, h)


def _wrists_near_any_phone(
    wrists: list[tuple[float, float]],
    phones: list[PhoneBox],
    *,
    distance_ratio: float,
    min_kpt_conf: float,
    kpt_conf: list[float] | None,
) -> bool:
    if not phones or not wrists:
        return False
    for i, (wx, wy) in enumerate(wrists):
        if kpt_conf is not None and i < len(kpt_conf) and kpt_conf[i] < min_kpt_conf:
            continue
        for phone in phones:
            thresh = distance_ratio * _bbox_scale(phone)
            if _point_to_bbox_distance(wx, wy, phone) <= thresh:
                return True
    return False


def _phone_likely_held_up(
    phones: list[PhoneBox],
    frame_bgr: np.ndarray,
    *,
    person_count: int,
    min_conf: float,
    max_center_y_ratio: float = 0.78,
) -> bool:
    """手机在画面上半部且置信度够高 → 视为手持（单手场景，不强制手腕邻近）。"""
    if not phones or person_count <= 0:
        return False
    h = float(frame_bgr.shape[0])
    for phone in phones:
        cy = (phone.y1 + phone.y2) * 0.5
        if phone.confidence >= min_conf and cy < max_center_y_ratio * h:
            return True
    return False


def _collect_visible_wrists(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray | None,
    *,
    min_kpt_conf: float,
) -> tuple[list[tuple[float, float]], list[float]]:
    """只收集置信度达标、坐标有效的前臂关键点（左/右腕）。"""
    wrists: list[tuple[float, float]] = []
    confs: list[float] = []
    for kp_idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
        if kp_idx >= len(keypoints_xy):
            continue
        x, y = float(keypoints_xy[kp_idx][0]), float(keypoints_xy[kp_idx][1])
        if x <= 0 and y <= 0:
            continue
        c = 1.0
        if keypoints_conf is not None and kp_idx < len(keypoints_conf):
            c = float(keypoints_conf[kp_idx])
        if c < min_kpt_conf:
            continue
        wrists.append((x, y))
        confs.append(c)
    return wrists, confs


class PersonPresenceTracker:
    """连续无人帧计数：≤grace 且仅手机 → 仍可判分心；>grace → 离开。"""

    def __init__(self, grace_frames: int = 10) -> None:
        self.grace_frames = max(1, int(grace_frames))
        self.absent_frames = 0
        self.phase: PresencePhase = "present"

    def update(self, person_visible: bool) -> PresencePhase:
        if person_visible:
            self.absent_frames = 0
            self.phase = "present"
            return self.phase
        self.absent_frames += 1
        if self.absent_frames > self.grace_frames:
            self.phase = "left"
        else:
            self.phase = "absent_grace"
        return self.phase


class PhoneHandProximityDetector:
    """YOLO26 detect（手机）+ YOLO26-pose（手腕）邻近判定。"""

    def __init__(
        self,
        detect_model: str | Path | None = None,
        pose_model: str | Path | None = None,
        device: str = "cpu",
        phone_conf: float = 0.25,
        pose_conf: float = 0.5,
        min_kpt_conf: float = 0.3,
        distance_ratio: float = 1.0,
        loose_distance_mult: float = 1.75,
        phone_solo_conf: float = 0.45,
        absent_grace_frames: int = 10,
        enable_head_down_fusion: bool = True,
        head_down_ratio_thresh: float = 0.10,
        hold_seconds: float = 1.0,
        imgsz: int = 640,
        verbose: bool = False,
        inference_backend: InferenceBackend = "auto",
        om_device_id: int = 0,
        detect_om: str | Path | None = None,
        pose_om: str | Path | None = None,
    ) -> None:
        self.detect_model_path = Path(detect_model or DEFAULT_DETECT_MODEL)
        self.pose_model_path = Path(pose_model or DEFAULT_POSE_MODEL)
        self.inference_backend = inference_backend
        self.om_device_id = int(om_device_id)
        self.device = device
        self.phone_conf = float(phone_conf)
        self.pose_conf = float(pose_conf)
        self.min_kpt_conf = float(min_kpt_conf)
        self.distance_ratio = float(distance_ratio)
        self.loose_distance_mult = float(loose_distance_mult)
        self.phone_solo_conf = float(phone_solo_conf)
        self.absent_grace_frames = int(absent_grace_frames)
        self.hold_seconds = float(hold_seconds)
        self._presence = PersonPresenceTracker(grace_frames=self.absent_grace_frames)
        self.enable_head_down_fusion = bool(enable_head_down_fusion)
        self.head_down_ratio_thresh = float(head_down_ratio_thresh)
        self._head_down: Any = None
        if self.enable_head_down_fusion:
            from src.adapters.behavior.head_down_hint import FaceMeshHeadDownEstimator

            est = FaceMeshHeadDownEstimator(down_ratio_thresh=self.head_down_ratio_thresh)
            self._head_down = est if est.available else None
        self.imgsz = int(imgsz)
        self.verbose = bool(verbose)
        self._detect_model: Any = None
        self._pose_model: Any = None
        self._om_runner: Any = None
        self._backend_resolved: str = ""
        self._positive_since: float | None = None
        self._detect_om_path = Path(detect_om) if detect_om else None
        self._pose_om_path = Path(pose_om) if pose_om else None
        self._last_primary_person: Any = None

    def _resolve_backend(self) -> str:
        if self._backend_resolved:
            return self._backend_resolved
        if self.inference_backend == "pt":
            self._backend_resolved = "pt"
        elif self.inference_backend == "om":
            self._backend_resolved = "om"
        else:
            from src.adapters.behavior.yolo_om_runner import om_models_available

            self._backend_resolved = "om" if om_models_available(self._detect_om_path, self._pose_om_path) else "pt"
        return self._backend_resolved

    @property
    def active_backend(self) -> str:
        return self._resolve_backend()

    def load_models(self) -> None:
        backend = self._resolve_backend()
        if backend == "om":
            if self._om_runner is not None and self._om_runner.loaded:
                return
            from src.adapters.behavior.yolo_om_runner import YoloOmPhonePoseRunner

            om_imgsz = self.imgsz if self.imgsz <= 416 else 320
            self._om_runner = YoloOmPhonePoseRunner(
                detect_om=self._detect_om_path,
                pose_om=self._pose_om_path,
                device_id=self.om_device_id,
                imgsz=om_imgsz,
                phone_conf=self.phone_conf,
                pose_conf=self.pose_conf,
                min_kpt_conf=self.min_kpt_conf,
            )
            if not self._om_runner.load():
                raise RuntimeError("YOLO OM 加载失败，请检查 .om 文件与 Ascend 环境")
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            self._analyze_frame_impl(blank)
            return

        from ultralytics import YOLO

        det_path = self._resolve_weights(self.detect_model_path, "yolo26n.pt")
        pose_path = self._resolve_weights(self.pose_model_path, "yolo26n-pose.pt")
        self._detect_model = YOLO(str(det_path))
        self._pose_model = YOLO(str(pose_path))
        self._warmup_models()

    def _warmup_models(self) -> None:
        """首帧推理往往很慢，启动时先跑一小图避免误以为卡死。"""
        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        self._detect_phones(blank)
        self._detect_wrists(blank)

    @staticmethod
    def _resolve_weights(path: Path, official_name: str) -> Path:
        if path.is_file():
            return path
        from ultralytics.utils.downloads import attempt_download_asset

        dest = DEFAULT_DETECT_MODEL.parent / official_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file():
            attempt_download_asset(official_name, file=dest)
        return dest

    def _ensure_loaded(self) -> None:
        if self._resolve_backend() == "om":
            if self._om_runner is None or not self._om_runner.loaded:
                self.load_models()
            return
        if self._detect_model is None or self._pose_model is None:
            self.load_models()

    def _analyze_frame_impl(self, frame_bgr: np.ndarray) -> PhoneHandFrameResult:
        primary_person = None
        if self._resolve_backend() == "om":
            assert self._om_runner is not None
            phones, wrists, kpt_conf, person_pose, primary_person = self._om_runner.infer_phones_and_wrists(
                frame_bgr
            )
        else:
            phones = self._detect_phones(frame_bgr)
            wrists, kpt_conf, person_pose, primary_person = self._detect_wrists(frame_bgr)
        person_visible = person_pose > 0
        phase = self._presence.update(person_visible)
        near_strict = _wrists_near_any_phone(
            wrists,
            phones,
            distance_ratio=self.distance_ratio,
            min_kpt_conf=self.min_kpt_conf,
            kpt_conf=kpt_conf,
        )
        near_loose = _wrists_near_any_phone(
            wrists,
            phones,
            distance_ratio=self.distance_ratio * self.loose_distance_mult,
            min_kpt_conf=self.min_kpt_conf,
            kpt_conf=kpt_conf,
        )
        held_up = _phone_likely_held_up(
            phones,
            frame_bgr,
            person_count=1 if person_visible else 0,
            min_conf=self.phone_solo_conf,
        )
        near = near_strict or near_loose
        present_instant = bool(phones) and (near_strict or near_loose or held_up)
        head_hint = self._analyze_head_down(frame_bgr)
        head_down_assist = self._head_down_phone_assist(
            frame_bgr,
            person_visible=person_visible,
            wrists=wrists,
            head_hint=head_hint,
            phase=phase,
        )
        if phase == "left":
            instant = False
        elif phase == "absent_grace":
            instant = bool(phones) or head_down_assist
        else:
            instant = present_instant or head_down_assist
        conf = 0.0
        if phones and instant:
            conf = min(p.confidence for p in phones)
        elif head_down_assist and instant:
            conf = max(0.45, min(0.75, 0.45 + head_hint.down_ratio))
        from src.adapters.behavior.pose_inference import infer_posture_and_activity

        posture, activity, posture_conf = infer_posture_and_activity(
            person=primary_person,
            person_visible=person_visible,
            presence_phase=phase,
            phone_in_hand=instant,
            looking_down=bool(getattr(head_hint, "looking_down", False)),
        )
        self._last_primary_person = primary_person
        return PhoneHandFrameResult(
            phone_in_hand=instant,
            confidence=conf,
            phones=phones,
            wrist_near_phone=near,
            raw_phone_count=len(phones),
            person_count=1 if person_visible else 0,
            person_count_pose=person_pose,
            person_visible=person_visible,
            absent_frames=self._presence.absent_frames,
            presence_phase=phase,
            wrist_count=len(wrists),
            looking_down=head_hint.looking_down,
            head_down_assist=head_down_assist,
            posture=posture,
            activity=activity,
            posture_confidence=posture_conf,
        )

    def _analyze_head_down(self, frame_bgr: np.ndarray) -> Any:
        from src.adapters.behavior.head_down_hint import HeadDownHint

        if self._head_down is None:
            return HeadDownHint()
        return self._head_down.analyze(frame_bgr)

    @staticmethod
    def _head_down_phone_assist(
        frame_bgr: np.ndarray,
        *,
        person_visible: bool,
        wrists: list[tuple[float, float]],
        head_hint: Any,
        phase: str,
    ) -> bool:
        """低头看屏 + 人在 + 手腕在画面上半区 → 辅助分心（弥补 COCO 手机漏检）。"""
        if phase == "left" or not person_visible:
            return False
        if not getattr(head_hint, "face_detected", False):
            return False
        if not getattr(head_hint, "looking_down", False):
            return False
        if not wrists:
            return False
        h = float(frame_bgr.shape[0])
        # 手腕在画面上半 65%：举手/持机姿态
        raised = sum(1 for _, wy in wrists if wy < h * 0.65)
        return raised >= 1

    def analyze_frame(self, frame_bgr: np.ndarray) -> PhoneHandFrameResult:
        """分析单帧 BGR 图像。"""
        self._ensure_loaded()
        return self._analyze_frame_impl(frame_bgr)

    def diagnose_frame(self, frame_bgr: np.ndarray, phone_conf: float | None = None) -> dict[str, Any]:
        """单帧诊断：人体、手机及全类别检测（排障用）。"""
        self._ensure_loaded()
        conf = self.phone_conf if phone_conf is None else float(phone_conf)
        result = self._analyze_frame_impl(frame_bgr)
        all_dets: list[str] = []
        if self._resolve_backend() == "pt":
            assert self._detect_model is not None
            out = self._detect_model.predict(
                source=frame_bgr,
                conf=conf,
                device=self.device,
                imgsz=self.imgsz,
                verbose=False,
            )
            if out and out[0].boxes is not None and len(out[0].boxes):
                names = out[0].names or {}
                for box in out[0].boxes:
                    cls_id = int(box.cls[0])
                    name = names.get(cls_id, str(cls_id))
                    all_dets.append(f"{name}:{float(box.conf[0]):.2f}")
        else:
            assert self._om_runner is not None
            all_dets.append("backend=om")
            boxes = self._om_runner.infer_detect_boxes(frame_bgr, conf_thres=min(0.2, conf))
            for b in boxes[:12]:
                all_dets.append(f"cls{b.class_id}:{b.confidence:.2f}")
            if result.phones and not any(b.class_id == COCO_CELL_PHONE_CLASS for b in boxes):
                all_dets.append(f"cell phone(filtered):{result.phones[0].confidence:.2f}")
        return {
            "backend": self.active_backend,
            "person_count": result.person_count,
            "person_count_pose": result.person_count_pose,
            "presence_phase": result.presence_phase,
            "absent_frames": result.absent_frames,
            "wrist_count": result.wrist_count,
            "phone_count": result.raw_phone_count,
            "all_detections": all_dets[:12],
        }

    def analyze_frame_stable(self, frame_bgr: np.ndarray) -> PhoneHandFrameResult:
        """带 hold_seconds 时间窗的稳定判定。"""
        from src.adapters.behavior.pose_inference import infer_posture_and_activity

        result = self.analyze_frame(frame_bgr)
        now = time.time()
        if result.phone_in_hand:
            if self._positive_since is None:
                self._positive_since = now
            held = (now - self._positive_since) >= self.hold_seconds
            result.phone_in_hand = held
            if held:
                result.confidence = max(result.confidence, 0.85)
        else:
            self._positive_since = None
            result.phone_in_hand = False
        posture, activity, posture_conf = infer_posture_and_activity(
            person=self._last_primary_person,
            person_visible=result.person_visible,
            presence_phase=result.presence_phase,
            phone_in_hand=result.phone_in_hand,
            looking_down=result.looking_down,
        )
        result.posture = posture
        result.activity = activity
        result.posture_confidence = posture_conf
        return result

    def _detect_phones(self, frame_bgr: np.ndarray) -> list[PhoneBox]:
        """yolo26n detect：仅手机 (COCO cls67)。"""
        assert self._detect_model is not None
        out = self._detect_model.predict(
            source=frame_bgr,
            classes=[COCO_CELL_PHONE_CLASS],
            conf=self.phone_conf,
            device=self.device,
            imgsz=self.imgsz,
            verbose=self.verbose,
        )
        phones: list[PhoneBox] = []
        if not out:
            return phones
        boxes = out[0].boxes
        if boxes is None or len(boxes) == 0:
            return phones
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            phones.append(
                PhoneBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(confs[i]))
            )
        return phones

    def _detect_wrists(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[tuple[float, float]], list[float] | None, int, Any | None]:
        from src.adapters.vision_common.yolo_ultralytics_ops import PosePerson

        assert self._pose_model is not None
        out = self._pose_model.predict(
            source=frame_bgr,
            conf=self.pose_conf,
            device=self.device,
            imgsz=self.imgsz,
            verbose=self.verbose,
        )
        wrists: list[tuple[float, float]] = []
        confs: list[float] = []
        person_count = 0
        primary: PosePerson | None = None
        best_conf = -1.0
        if not out:
            return wrists, None, person_count, primary
        for r in out:
            if r.keypoints is None:
                continue
            kxy = r.keypoints.xy.cpu().numpy()
            kcf = r.keypoints.conf
            if kcf is not None:
                kcf_np = kcf.cpu().numpy()
            else:
                kcf_np = None
            for person_idx in range(len(kxy)):
                person_count += 1
                pts = kxy[person_idx]
                kc = None
                if kcf_np is not None and person_idx < len(kcf_np):
                    kc = kcf_np[person_idx]
                box_conf = 0.5
                if getattr(r, "boxes", None) is not None and r.boxes is not None and person_idx < len(r.boxes):
                    box_conf = float(r.boxes.conf[person_idx])
                if box_conf > best_conf:
                    best_conf = box_conf
                    kc_arr = kc if kc is not None else np.ones(len(pts))
                    primary = PosePerson(keypoints_xy=pts, keypoints_conf=kc_arr, box_conf=box_conf)
                w, c = _collect_visible_wrists(pts, kc, min_kpt_conf=self.min_kpt_conf)
                wrists.extend(w)
                confs.extend(c)
        return wrists, confs if confs else None, person_count, primary


def dependencies_met() -> bool:
    try:
        import cv2  # noqa: F401
        from ultralytics import YOLO  # noqa: F401

        return True
    except ImportError:
        return False
