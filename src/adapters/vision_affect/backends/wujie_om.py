"""WuJie OM backend (Ascend NPU via ACL runtime)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.adapters.vision_affect.backends.protocols import EmotionPredictResult


def _import_acl() -> Any:
    """导入 ACL Python 包；必要时注入 Ascend site-packages 路径。"""
    try:
        import acl  # type: ignore
        return acl
    except Exception:
        pass

    import sys
    for p in (
        "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
        "/usr/local/Ascend/ascend-toolkit/7.0.RC1/python/site-packages",
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
    import acl  # type: ignore
    return acl


@dataclass
class WuJieOmBackend:
    """加载 WuJie 导出的 OM 模型，在 Ascend NPU 上推理（ACL Runtime）。"""

    model_path: str | Path | None
    device_id: int = 0

    _acl: Any = field(default=None, repr=False)
    _ctx: Any = field(default=None, repr=False)
    _model_id: int | None = field(default=None, repr=False)
    _model_desc: Any = field(default=None, repr=False)
    _input_dataset: Any = field(default=None, repr=False)
    _output_dataset: Any = field(default=None, repr=False)
    _input_dev: int | None = field(default=None, repr=False)
    _output_dev: int | None = field(default=None, repr=False)
    _input_size: int = field(default=0, repr=False)
    _output_size: int = field(default=0, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def available(self) -> bool:
        if self.model_path is None:
            return False
        return Path(self.model_path).is_file()

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if not self.available():
            return False
        try:
            acl = _import_acl()
        except Exception:
            return False
        try:
            ret = acl.init()
            # 0: success; repeated init may return non-zero in some envs, ignore if later steps succeed.
            _ = ret
            if acl.rt.set_device(int(self.device_id)) != 0:
                return False
            ctx, ret = acl.rt.create_context(int(self.device_id))
            if ret != 0:
                return False
            if acl.rt.set_context(ctx) != 0:
                return False

            model_id, ret = acl.mdl.load_from_file(str(self.model_path))
            if ret != 0:
                return False
            model_desc = acl.mdl.create_desc()
            if acl.mdl.get_desc(model_desc, model_id) != 0:
                return False

            in_size = int(acl.mdl.get_input_size_by_index(model_desc, 0))
            out_size = int(acl.mdl.get_output_size_by_index(model_desc, 0))
            in_dev, ret = acl.rt.malloc(in_size, 0)
            if ret != 0:
                return False
            out_dev, ret = acl.rt.malloc(out_size, 0)
            if ret != 0:
                return False

            in_ds = acl.mdl.create_dataset()
            out_ds = acl.mdl.create_dataset()
            in_db = acl.create_data_buffer(in_dev, in_size)
            out_db = acl.create_data_buffer(out_dev, out_size)
            if acl.mdl.add_dataset_buffer(in_ds, in_db) is None:
                return False
            if acl.mdl.add_dataset_buffer(out_ds, out_db) is None:
                return False

            self._acl = acl
            self._ctx = ctx
            self._model_id = int(model_id)
            self._model_desc = model_desc
            self._input_dataset = in_ds
            self._output_dataset = out_ds
            self._input_dev = int(in_dev)
            self._output_dev = int(out_dev)
            self._input_size = in_size
            self._output_size = out_size
            self._loaded = True
        except Exception:
            self._loaded = False
            return False
        return self._loaded

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """为固定输入 1x3x44x44 的 OM 做预处理，输出 [1, 3, 44, 44] float32。"""
        import cv2

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        img = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0

        # OM 是固定 batch=1 输入，这里使用中心 44x44 裁剪。
        c = img[2:46, 2:46, :]
        x = np.transpose(c, (2, 0, 1)).astype(np.float32)
        return np.expand_dims(x, axis=0)  # [1, 3, 44, 44]

    def predict(self, crop_bgr: np.ndarray) -> EmotionPredictResult:
        if not self._ensure_loaded():
            return EmotionPredictResult()
        if crop_bgr is None or crop_bgr.size == 0:
            return EmotionPredictResult()

        x = self._preprocess(crop_bgr)
        if x.nbytes != self._input_size:
            return EmotionPredictResult()
        try:
            raw = x.tobytes()
            host_ptr = self._acl.util.bytes_to_ptr(raw)
            if self._acl.rt.memcpy(self._input_dev, self._input_size, host_ptr, len(raw), 1) != 0:
                return EmotionPredictResult()
            if self._acl.mdl.execute(self._model_id, self._input_dataset, self._output_dataset) != 0:
                return EmotionPredictResult()
            host_out, ret = self._acl.rt.malloc_host(self._output_size)
            if ret != 0:
                return EmotionPredictResult()
            try:
                if self._acl.rt.memcpy(host_out, self._output_size, self._output_dev, self._output_size, 2) != 0:
                    return EmotionPredictResult()
                out_bytes = self._acl.util.ptr_to_bytes(host_out, self._output_size)
            finally:
                self._acl.rt.free_host(host_out)
        except Exception:
            return EmotionPredictResult()
        logits = np.frombuffer(out_bytes, dtype=np.float32)
        if logits.size == 0:
            return EmotionPredictResult()
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        # softmax
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(z)
        prob = exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)
        idx = int(np.argmax(prob, axis=1)[0])
        conf = float(np.max(prob, axis=1)[0])
        label_id = idx + 1  # RAF-DB 1..7
        return EmotionPredictResult(raf_label_id=label_id, confidence=conf)
