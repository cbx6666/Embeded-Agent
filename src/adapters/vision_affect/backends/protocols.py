"""底层推理后端协议：仅被 `vision_affect` 包使用，内核不得 import。

实现类放在本包同目录下（如 `raf_resnet.py`、`deepface_emotion.py`）；
`VisionAffectInputAdapter` 只依赖协议与 `EmotionPredictResult`，不依赖具体库。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EmotionPredictResult:
    """单帧人脸 crop 的推理结果，供适配器选择事件工厂。"""

    # RAF-DB 等：上报 label_id，由 `user_emotion_updated_from_rafdb` 映射到闭集 emotion
    raf_label_id: int | None = None
    # DeepFace 等：已在后端映射到闭集：neutral / tired / stressed / happy
    agent_emotion: str | None = None
    confidence: float | None = None

    @property
    def is_empty(self) -> bool:
        return self.raf_label_id is None and self.agent_emotion is None


@runtime_checkable
class EmotionInferenceBackend(Protocol):
    """情绪推理后端：与采集、landmark 解耦，仅消费人脸区域图像。"""

    def available(self) -> bool:
        """依赖与权重已就绪时返回 True。"""

    def predict(self, crop_bgr: np.ndarray) -> EmotionPredictResult:
        """对 BGR uint8 人脸图推理；无结果时返回 `is_empty`。"""
