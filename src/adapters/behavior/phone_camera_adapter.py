from __future__ import annotations

"""摄像头线程：手机+手腕邻近 → BehaviorAdapter 事件。"""

import threading
import time
from typing import Any

import cv2

from src.adapters.behavior.camera_utils import grab_latest_frame, open_camera, warmup_camera
from src.adapters.behavior.phone_hand_detector import (
    PhoneHandProximityDetector,
    dependencies_met,
)
from src.adapters.behavior_adapter import BehaviorAdapter


class PhoneHandCameraAdapter:
    """从摄像头读取帧，检测手持手机并上报注意力事件。"""

    def __init__(
        self,
        core: Any,
        *,
        camera_index: int = 0,
        detector: PhoneHandProximityDetector | None = None,
        behavior_adapter: BehaviorAdapter | None = None,
        inference_interval: float = 0.25,
        source: str = "yolo26_phone_hand_v1",
    ) -> None:
        self.core = core
        self.camera_index = int(camera_index)
        self.detector = detector or PhoneHandProximityDetector()
        self.behavior = behavior_adapter or BehaviorAdapter(core)
        self.inference_interval = float(inference_interval)
        self.source = source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start_background(self) -> None:
        if not dependencies_met():
            raise RuntimeError(
                "缺少依赖：pip install -r requirements-behavior.txt"
            )
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run_loop(self) -> None:
        self.detector.load_models()
        cap = open_camera(self.camera_index)
        if not cap.isOpened():
            print(f"[PhoneHandCameraAdapter] 无法打开摄像头 index={self.camera_index}")
            return
        warmup_camera(cap)
        last_infer = 0.0
        try:
            while not self._stop.is_set():
                ok, frame = grab_latest_frame(cap, flush=3)
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                now = time.time()
                if now - last_infer < self.inference_interval:
                    continue
                last_infer = now
                result = self.detector.analyze_frame_stable(frame)
                if result.person_visible:
                    self.behavior.publish_presence("present", confidence=0.9, source=self.source)
                elif result.presence_phase == "left":
                    self.behavior.publish_presence("away", confidence=0.9, source=self.source)

                if result.presence_phase == "left":
                    self.behavior.publish_attention(
                        attention="focused",
                        behavior="away",
                        confidence=0.9,
                        source=self.source,
                    )
                elif result.phone_in_hand:
                    self.behavior.publish_attention(
                        attention="distracted",
                        behavior="phone_use",
                        confidence=result.confidence,
                        source=self.source,
                    )
                else:
                    self.behavior.publish_attention(
                        attention="focused",
                        behavior="working",
                        confidence=0.9,
                        source=self.source,
                    )
        finally:
            cap.release()
