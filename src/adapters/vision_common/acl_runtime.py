from __future__ import annotations

"""Ascend ACL OM 推理会话（表情 WuJie、YOLO OM 等共用）。"""

from pathlib import Path
from typing import Any

import numpy as np


def import_acl() -> Any:
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


class _AclDeviceRuntime:
    """同一 device 上多路 OM 共享 ACL 上下文（避免第二个 load 冲掉第一个）。"""

    _by_device: dict[int, "_AclDeviceRuntime"] = {}

    def __init__(self, device_id: int) -> None:
        self.device_id = int(device_id)
        self.acl: Any = None
        self.ctx: Any = None
        self._ready = False

    @classmethod
    def get(cls, device_id: int) -> "_AclDeviceRuntime":
        device_id = int(device_id)
        if device_id not in cls._by_device:
            cls._by_device[device_id] = cls(device_id)
        return cls._by_device[device_id]

    def ensure_ready(self) -> bool:
        if self._ready:
            return True
        try:
            acl = import_acl()
        except Exception:
            return False
        try:
            _ = acl.init()
            if acl.rt.set_device(self.device_id) != 0:
                return False
            ctx, ret = acl.rt.create_context(self.device_id)
            if ret != 0:
                return False
            if acl.rt.set_context(ctx) != 0:
                return False
            self.acl = acl
            self.ctx = ctx
            self._ready = True
        except Exception:
            self._ready = False
            return False
        return True

    def activate(self) -> bool:
        if not self.ensure_ready():
            return False
        return self.acl.rt.set_context(self.ctx) == 0


class AscendOmSession:
    """加载单个 .om，执行固定字节大小的输入张量推理。"""

    def __init__(self, model_path: str | Path, device_id: int = 0) -> None:
        self.model_path = Path(model_path)
        self.device_id = int(device_id)
        self._runtime = _AclDeviceRuntime.get(self.device_id)
        self._model_id: int | None = None
        self._model_desc: Any = None
        self._input_dataset: Any = None
        self._output_dataset: Any = None
        self._input_dev: int | None = None
        self._output_dev: int | None = None
        self._input_size: int = 0
        self._output_size: int = 0
        self._loaded = False

    def available(self) -> bool:
        return self.model_path.is_file()

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        if self._loaded:
            return True
        if not self.available():
            return False
        if not self._runtime.ensure_ready():
            return False
        acl = self._runtime.acl
        try:
            if not self._runtime.activate():
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
        return True

    def execute(self, input_tensor: np.ndarray) -> np.ndarray | None:
        """input_tensor 须为 C 连续 float32，字节数与 OM 输入一致。"""
        if not self.load():
            return None
        acl = self._runtime.acl
        x = np.ascontiguousarray(input_tensor, dtype=np.float32)
        if x.nbytes != self._input_size:
            return None
        try:
            if not self._runtime.activate():
                return None
            raw = x.tobytes()
            host_ptr = acl.util.bytes_to_ptr(raw)
            if acl.rt.memcpy(self._input_dev, self._input_size, host_ptr, len(raw), 1) != 0:
                return None
            if acl.mdl.execute(self._model_id, self._input_dataset, self._output_dataset) != 0:
                return None
            host_out, ret = acl.rt.malloc_host(self._output_size)
            if ret != 0:
                return None
            try:
                if (
                    acl.rt.memcpy(host_out, self._output_size, self._output_dev, self._output_size, 2)
                    != 0
                ):
                    return None
                out_bytes = acl.util.ptr_to_bytes(host_out, self._output_size)
            finally:
                acl.rt.free_host(host_out)
        except Exception:
            return None
        return np.frombuffer(out_bytes, dtype=np.float32).copy()
