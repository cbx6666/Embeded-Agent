"""可选：ResNet18 + RAF-DB 七类权重，从人脸 crop 预测 label_id（1–7）。

仅由 `vision_affect` 适配器使用；内核不加载、不依赖本模块。

未安装 torch 或未提供权重文件时，`RafEmotionBackend` 不可用，视觉管线仅输出疲劳事件。

**注意**：权重训练时的类别顺序须与 `src/agent/event/factories.py` 中 `RAF_DB_LABELS` 的 1–7 编号一致，否则需自行调整映射。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RafEmotionBackend:
    """延迟加载 torchvision ResNet18；`predict` 返回 (raf_label_id 1–7, confidence) 或 (None, None)。"""

    checkpoint_path: str | Path | None
    device: str = "cpu"

    _model: Any = field(default=None, repr=False)
    _transform: Any = field(default=None, repr=False)

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
            from torchvision import models, transforms
        except ImportError:
            return False

        num_classes = 7
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        try:
            try:
                blob = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:
                blob = torch.load(self.checkpoint_path, map_location=self.device)
        except OSError:
            return False
        state = blob
        if isinstance(blob, dict):
            if "state_dict" in blob:
                state = blob["state_dict"]
            elif "model" in blob:
                state = blob["model"]
        if isinstance(state, dict):
            stripped = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                stripped[nk] = v
            state = stripped
        m.load_state_dict(state, strict=False)
        m.eval()
        m.to(self.device)
        self._model = m
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return True

    def predict(self, crop_bgr: np.ndarray) -> tuple[int | None, float | None]:
        """crop_bgr: HWC BGR uint8。"""
        if not self._ensure_loaded():
            return None, None
        import torch

        rgb = crop_bgr[:, :, ::-1].copy()
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            prob = torch.softmax(logits, dim=1)
            conf, idx = prob.max(dim=1)
        label_id = int(idx.item()) + 1
        return label_id, float(conf.item())
