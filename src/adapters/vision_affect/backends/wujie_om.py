"""WuJie OM backend (Ascend NPU via ACL runtime)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.adapters.vision_affect.backends.protocols import EmotionPredictResult
from src.adapters.vision_common.acl_runtime import shared_om_session
from src.adapters.vision_common.preprocess import resize_gray_face_patch


@dataclass
class WuJieOmBackend:
    """加载 WuJie 导出的 OM 模型，在 Ascend NPU 上推理（ACL Runtime）。"""

    model_path: str | Path | None
    device_id: int = 0

    _session: Any | None = field(default=None, repr=False)

    def available(self) -> bool:
        if self.model_path is None:
            return False
        return Path(self.model_path).is_file()

    def _ensure_loaded(self) -> bool:
        if self._session is not None and self._session.loaded:
            return True
        if not self.available():
            return False
        self._session = shared_om_session(self.model_path, device_id=self.device_id)
        return self._session.load()

    def predict(self, crop_bgr: np.ndarray) -> EmotionPredictResult:
        if not self._ensure_loaded():
            return EmotionPredictResult()
        if crop_bgr is None or crop_bgr.size == 0 or self._session is None:
            return EmotionPredictResult()

        x = resize_gray_face_patch(crop_bgr)
        out = self._session.execute(x)
        if out is None or out.size == 0:
            return EmotionPredictResult()

        logits = out.reshape(1, -1) if out.ndim == 1 else out
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(z)
        prob = exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)
        idx = int(np.argmax(prob, axis=1)[0])
        conf = float(np.max(prob, axis=1)[0])
        return EmotionPredictResult(raf_label_id=idx + 1, confidence=conf)
