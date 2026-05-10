"""不输出情绪事件（仅保留疲劳等几何管线）。"""

from __future__ import annotations

import numpy as np

from src.adapters.vision_affect.backends.protocols import EmotionPredictResult


class NoEmotionBackend:
    def available(self) -> bool:
        return False

    def predict(self, crop_bgr: np.ndarray) -> EmotionPredictResult:
        return EmotionPredictResult()
