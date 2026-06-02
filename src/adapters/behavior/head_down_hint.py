from __future__ import annotations

"""MediaPipe Face Mesh 低头启发式（与 vision_affect 同源，供手机分心融合）。"""

from dataclasses import dataclass
from typing import Any

import numpy as np

# Face Mesh 常用索引：鼻尖、双眼中心、下巴
_NOSE_TIP = 1
_CHIN = 152
_RIGHT_EYE = (33, 133)
_LEFT_EYE = (362, 263)


@dataclass
class HeadDownHint:
    face_detected: bool = False
    looking_down: bool = False
    """鼻尖低于双眼连线一定比例 → 低头看屏。"""

    down_ratio: float = 0.0
    """鼻尖相对眼线的垂直偏移 / 脸高，越大越低头。"""


class FaceMeshHeadDownEstimator:
    """轻量 Face Mesh，仅判断是否在低头（不跑情绪 OM）。"""

    def __init__(
        self,
        *,
        down_ratio_thresh: float = 0.10,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.down_ratio_thresh = float(down_ratio_thresh)
        self._mesh: Any = None
        self._mp_face_mesh: Any = None
        self._init_error: str | None = None
        try:
            import mediapipe as mp

            self._mp_face_mesh = mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as exc:
            self._init_error = str(exc)

    @property
    def available(self) -> bool:
        return self._mesh is not None

    def analyze(self, frame_bgr: np.ndarray) -> HeadDownHint:
        if self._mesh is None:
            return HeadDownHint()
        h, w = frame_bgr.shape[:2]
        if h < 32 or w < 32:
            return HeadDownHint()
        rgb = frame_bgr[:, :, ::-1]
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return HeadDownHint()
        lm = res.multi_face_landmarks[0].landmark
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        face_h = max(max(ys) - min(ys), 1.0)
        nose_y = lm[_NOSE_TIP].y * h
        chin_y = lm[_CHIN].y * h
        eye_y = sum(lm[i].y for i in _RIGHT_EYE + _LEFT_EYE) / 4.0 * h
        down_ratio = (nose_y - eye_y) / face_h
        # 低头：鼻尖明显低于眼线，且下巴低于鼻尖（避免仰头误判）
        looking_down = (
            down_ratio >= self.down_ratio_thresh
            and chin_y > nose_y + face_h * 0.05
        )
        return HeadDownHint(
            face_detected=True,
            looking_down=looking_down,
            down_ratio=float(down_ratio),
        )
