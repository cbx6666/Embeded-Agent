from __future__ import annotations

"""由 YOLO26-pose OM 关键点推断 posture / activity（与行为同帧，不单独起线程）。"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.adapters.vision_common.yolo_ultralytics_ops import PosePerson

# COCO 17 点
KP_NOSE = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14


def _kpt_y(person: PosePerson, idx: int, *, min_conf: float = 0.3) -> float | None:
    if idx >= len(person.keypoints_xy):
        return None
    x, y = float(person.keypoints_xy[idx][0]), float(person.keypoints_xy[idx][1])
    if x <= 0 and y <= 0:
        return None
    if idx < len(person.keypoints_conf) and float(person.keypoints_conf[idx]) < min_conf:
        return None
    return y


def _kpt_x(person: PosePerson, idx: int, *, min_conf: float = 0.3) -> float | None:
    if idx >= len(person.keypoints_xy):
        return None
    x, y = float(person.keypoints_xy[idx][0]), float(person.keypoints_xy[idx][1])
    if x <= 0 and y <= 0:
        return None
    if idx < len(person.keypoints_conf) and float(person.keypoints_conf[idx]) < min_conf:
        return None
    return x


def _mean_y(person: PosePerson, indices: tuple[int, ...], *, min_conf: float = 0.3) -> float | None:
    vals = [_kpt_y(person, i, min_conf=min_conf) for i in indices]
    ok = [v for v in vals if v is not None]
    if not ok:
        return None
    return sum(ok) / len(ok)


def _posture_from_keypoints(person: PosePerson) -> tuple[str, float]:
    shoulder_y = _mean_y(person, (KP_L_SHOULDER, KP_R_SHOULDER))
    hip_y = _mean_y(person, (KP_L_HIP, KP_R_HIP))
    knee_y = _mean_y(person, (KP_L_KNEE, KP_R_KNEE))
    nose_x = _kpt_x(person, KP_NOSE)
    hip_x_vals = [_kpt_x(person, i) for i in (KP_L_HIP, KP_R_HIP)]
    hip_x_ok = [v for v in hip_x_vals if v is not None]

    if shoulder_y is None or hip_y is None:
        return "sitting", 0.55

    torso_vert = hip_y - shoulder_y
    if abs(torso_vert) < 25:
        return "lying", 0.65

    if knee_y is not None and (knee_y - hip_y) < max(15.0, abs(torso_vert) * 0.35):
        return "standing", 0.7

    if nose_x is not None and hip_x_ok:
        hip_center_x = sum(hip_x_ok) / len(hip_x_ok)
        if abs(nose_x - hip_center_x) > 40:
            return "leaning", 0.75

    return "sitting", 0.8


def infer_posture_and_activity(
    *,
    person: PosePerson | None,
    person_visible: bool,
    presence_phase: str,
    phone_in_hand: bool,
    looking_down: bool,
) -> tuple[str, str, float]:
    """与 `PhoneHandFrameResult` 同帧输出 posture / activity。"""
    if presence_phase == "left" or not person_visible:
        return "unknown", "unknown", 0.0

    if person is not None:
        posture, conf = _posture_from_keypoints(person)
    else:
        posture, conf = "sitting", 0.55

    if phone_in_hand:
        activity = "phone_use"
        if looking_down:
            posture = "leaning"
    elif looking_down:
        activity = "studying"
        posture = "leaning"
    else:
        activity = "working"

    return posture, activity, conf
