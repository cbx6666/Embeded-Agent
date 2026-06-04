"""DeepFace 情绪：人脸 crop 上分析 dominant emotion，并映射为 Agent 闭集。

依赖 `deepface`（会间接安装 TensorFlow 等，体积较大）。对内核只暴露
`EmotionPredictResult`。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.adapters.vision_affect.backends.protocols import EmotionPredictResult

_log = logging.getLogger(__name__)

# DeepFace 返回的 dominant 标签 -> Agent 闭集
_DEEPFACE_TO_AGENT: dict[str, str] = {
    "happy": "happy",
    "neutral": "neutral",
    "sad": "stressed",
    "fear": "stressed",
    "disgust": "stressed",
    "angry": "stressed",
    "surprise": "neutral",
}


@dataclass
class DeepFaceEmotionBackend:
    """`model_name` 与 DeepFace `analyze` 的 `model_name` 一致（如 VGG-Face, Facenet）。"""

    deepface_model: str = "VGG-Face"

    def available(self) -> bool:
        try:
            import deepface  # noqa: F401
        except ImportError:
            return False
        return True

    def predict(self, crop_bgr: np.ndarray) -> EmotionPredictResult:
        if not self.available() or crop_bgr is None or crop_bgr.size == 0:
            return EmotionPredictResult()
        try:
            from deepface import DeepFace
        except ImportError:
            return EmotionPredictResult()

        # analyze 的 emotion 子模型不通过 model_name 选择（与 represent/verify 不同）；
        # `deepface_model` 仅作文档与后续扩展。
        _ = self.deepface_model
        try:
            res = DeepFace.analyze(
                img_path=crop_bgr,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
                align=False,
                silent=True,
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("deepface analyze failed: %s", exc)
            return EmotionPredictResult()

        if isinstance(res, dict):
            objs = [res]
        else:
            objs = res
        if not objs:
            return EmotionPredictResult()

        em = objs[0].get("emotion") or {}
        em_lower: dict[str, float] = {}
        for k, v in em.items():
            try:
                em_lower[str(k).lower()] = float(v)
            except (TypeError, ValueError):
                continue
        dom = objs[0].get("dominant_emotion")
        if not dom and em_lower:
            dom = max(em_lower, key=em_lower.get)  # type: ignore[misc, arg-type]
        if not dom or not em_lower:
            return EmotionPredictResult()
        dkey = str(dom).lower()
        c = float(em_lower.get(dkey, 0.0))
        if c > 1.5:
            c = c / 100.0
        c = max(0.0, min(1.0, c))

        agent = _DEEPFACE_TO_AGENT.get(dkey, "neutral")
        return EmotionPredictResult(agent_emotion=agent, confidence=c)
