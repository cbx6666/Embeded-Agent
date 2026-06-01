"""WuJie1010 FER2013 VGG19 backend.

兼容 `WuJie1010/Facial-Expression-Recognition.Pytorch` 导出的
`FER2013_VGG19/PrivateTest_model.t7` 检查点结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.adapters.vision_affect.backends.protocols import EmotionPredictResult


class _WuJieVGG19:  # pragma: no cover - thin wrapper for optional torch import
    def __init__(self) -> None:
        import torch.nn as nn
        import torch.nn.functional as F

        cfg = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ]
        layers: list[Any] = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend(
                    [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = x
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))

        class _Net(nn.Module):
            def __init__(self, features: Any) -> None:
                super().__init__()
                self.features = features
                self.classifier = nn.Linear(512, 7)

            def forward(self, x: Any) -> Any:
                out = self.features(x)
                out = out.view(out.size(0), -1)
                out = F.dropout(out, p=0.5, training=self.training)
                return self.classifier(out)

        self.net = _Net(nn.Sequential(*layers))


@dataclass
class WuJieVGG19Backend:
    """加载 WuJie1010 VGG19 `PrivateTest_model.t7`。"""

    checkpoint_path: str | Path | None
    device: str = "cpu"

    _model: Any = field(default=None, repr=False)

    def available(self) -> bool:
        if self.checkpoint_path is None:
            return False
        return Path(self.checkpoint_path).is_file()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.available():
            return False
        try:
            import torch
        except ImportError:
            return False

        m = _WuJieVGG19().net
        try:
            try:
                blob = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:
                blob = torch.load(self.checkpoint_path, map_location=self.device)
        except OSError:
            return False

        state = blob["net"] if isinstance(blob, dict) and "net" in blob else blob
        if isinstance(state, dict):
            stripped = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                stripped[nk] = v
            state = stripped
        m.load_state_dict(state, strict=False)
        m.eval()
        m.to(self.device)
        self._model = m
        return True

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """按原仓库流程预处理：灰度->48x48->3通道，再做10-crop 44x44。"""
        import cv2

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        img = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0

        # 10-crop: 5个位置 + mirror
        size = 44
        h, w = 48, 48
        coords = [
            (0, 0),
            (0, w - size),
            (h - size, 0),
            (h - size, w - size),
            ((h - size) // 2, (w - size) // 2),
        ]
        crops: list[np.ndarray] = []
        for y, x in coords:
            c = img[y : y + size, x : x + size, :]
            crops.append(c)
            crops.append(c[:, ::-1, :])  # mirrored
        stacked = np.stack(crops, axis=0)  # [10, 44, 44, 3]
        return np.transpose(stacked, (0, 3, 1, 2))  # [10, 3, 44, 44]

    def predict(self, crop_bgr: np.ndarray) -> EmotionPredictResult:
        if not self._ensure_loaded():
            return EmotionPredictResult()
        import torch

        x = self._preprocess(crop_bgr)
        tensor = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            logits = logits.mean(dim=0, keepdim=True)  # 平均 10-crop
            prob = torch.softmax(logits, dim=1)
            conf, idx = prob.max(dim=1)
        label_id = int(idx.item()) + 1
        return EmotionPredictResult(raf_label_id=label_id, confidence=float(conf.item()))

