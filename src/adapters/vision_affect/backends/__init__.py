"""视觉底层推理实现：ResNet/RAF、DeepFace 预留等。内核不得引用本包。"""

from __future__ import annotations

from src.adapters.vision_affect.backends.deepface_emotion import DeepFaceEmotionBackend
from src.adapters.vision_affect.backends.factory import build_emotion_backend
from src.adapters.vision_affect.backends.noop_emotion import NoEmotionBackend
from src.adapters.vision_affect.backends.protocols import EmotionInferenceBackend, EmotionPredictResult
from src.adapters.vision_affect.backends.raf_resnet import RafEmotionBackend

__all__ = [
    "DeepFaceEmotionBackend",
    "EmotionInferenceBackend",
    "EmotionPredictResult",
    "NoEmotionBackend",
    "RafEmotionBackend",
    "build_emotion_backend",
]
