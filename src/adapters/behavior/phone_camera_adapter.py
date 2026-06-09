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
from src.adapters.perception_config import behavior_inference_interval_sec
from src.adapters.perception_debug_log import perception_debug
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
        inference_interval: float | None = None,
        source: str = "yolo26_phone_hand_om_v1",
        frame_bus: Any | None = None,
        pose_source: str = "yolo26_pose_om_v1",
    ) -> None:
        self.core = core
        self.camera_index = int(camera_index)
        self.detector = detector or PhoneHandProximityDetector()
        self.behavior = behavior_adapter or BehaviorAdapter(core)
        self.inference_interval = (
            behavior_inference_interval_sec() if inference_interval is None else float(inference_interval)
        )
        self.source = source
        self.pose_source = pose_source
        self.frame_bus = frame_bus
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_behavior_sig: tuple[Any, ...] | None = None
        self._last_stable_phone_ts: float = 0.0
        self._working_publish_grace_sec: float = 4.0

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
        perception_debug().log(
            "behavior",
            "detector_ready",
            camera_index=self.camera_index,
            backend=self.detector.active_backend,
            shared_frame_bus=self.frame_bus is not None,
        )
        own_cap = None
        if self.frame_bus is None:
            cap = open_camera(self.camera_index)
            if not cap.isOpened():
                print(f"[PhoneHandCameraAdapter] 无法打开摄像头 index={self.camera_index}")
                return
            warmup_camera(cap)
            own_cap = cap
        last_infer = 0.0
        last_bus_seq = -1
        try:
            while not self._stop.is_set():
                frame = None
                if self.frame_bus is not None:
                    seq = self.frame_bus.seq
                    if seq == last_bus_seq:
                        time.sleep(0.02)
                        continue
                    frame = self.frame_bus.get_latest_copy()
                    last_bus_seq = seq
                    if frame is None:
                        time.sleep(0.02)
                        continue
                else:
                    assert own_cap is not None
                    ok, frame = grab_latest_frame(own_cap, flush=3)
                    if not ok or frame is None:
                        time.sleep(0.05)
                        continue
                now = time.time()
                if now - last_infer < self.inference_interval:
                    continue
                last_infer = now
                result = self.detector.analyze_frame_stable(frame)
                perception_debug().log(
                    "behavior",
                    "frame_result",
                    backend=self.detector.active_backend,
                    phone_in_hand=result.phone_in_hand,
                    confidence=round(float(result.confidence), 4),
                    person_visible=result.person_visible,
                    presence_phase=result.presence_phase,
                    posture=result.posture,
                    activity=result.activity,
                    looking_down=result.looking_down,
                    head_down_assist=result.head_down_assist,
                    raw_phone_count=result.raw_phone_count,
                    person_count_pose=result.person_count_pose,
                    face_detected=result.face_detected,
                    phone_signal=result.phone_signal,
                )
                self._log_behavior_to_console(result)
                # 玩手机优先于 pose 的 left/away：举机时人体常被遮挡，pose OM 漏检不能盖掉 phone_use。
                phone_signal = str(result.phone_signal)
                has_phone_evidence = (
                    result.raw_phone_count > 0
                    or phone_signal in {"phone", "phone_wrist", "phone_face", "sustained"}
                )
                if result.phone_in_hand and has_phone_evidence:
                    self.behavior.publish_presence("present", confidence=0.85, source=self.source)
                    self._last_stable_phone_ts = now
                    self.behavior.publish_attention(
                        attention="distracted",
                        behavior="phone_use",
                        confidence=max(float(result.confidence), 0.85),
                        source=self.source,
                        yolo_phone_detected=result.raw_phone_count > 0,
                    )
                elif result.presence_phase == "left":
                    self.behavior.publish_presence("away", confidence=0.9, source=self.source)
                    self.behavior.publish_attention(
                        attention="idle",
                        behavior="away",
                        confidence=0.9,
                        source=self.source,
                    )
                    self.behavior.publish_posture(
                        "unknown",
                        confidence=0.9,
                        source=self.pose_source,
                    )
                    self.behavior.publish_activity(
                        "unknown",
                        confidence=0.9,
                        source=self.pose_source,
                    )
                elif result.person_visible:
                    self.behavior.publish_presence("present", confidence=0.9, source=self.source)
                elif (now - self._last_stable_phone_ts) >= self._working_publish_grace_sec:
                    # 短暂掉检不立刻刷成 working，避免稀释 30s 窗口里的 phone_use 占比。
                    self.behavior.publish_attention(
                        attention="focused",
                        behavior="working",
                        confidence=0.9,
                        source=self.source,
                        yolo_phone_detected=False,
                    )
                if result.posture != "unknown":
                    pose_conf = max(float(result.posture_confidence), 0.85)
                    self.behavior.publish_posture(
                        result.posture,
                        confidence=pose_conf,
                        source=self.pose_source,
                    )
                if result.activity != "unknown":
                    pose_conf = max(float(result.posture_confidence), 0.85)
                    self.behavior.publish_activity(
                        result.activity,
                        confidence=pose_conf,
                        source=self.pose_source,
                    )
        finally:
            if own_cap is not None:
                own_cap.release()

    def _log_behavior_to_console(self, result: Any) -> None:
        """行为检测变化时打印到控制台（仅在关键状态变化时打印，避免 4Hz 刷屏）。"""

        sig = (
            bool(result.phone_in_hand),
            str(result.presence_phase),
            str(result.activity),
            str(result.posture),
        )
        if sig == self._last_behavior_sig:
            return
        self._last_behavior_sig = sig
        phone = "玩手机" if result.phone_in_hand else "无手机"
        signal = str(getattr(result, "phone_signal", "none"))
        print(
            f"[行为] {phone}｜依据={signal}｜在位={result.presence_phase}｜"
            f"人脸={getattr(result, 'face_detected', False)}｜"
            f"raw_phone={result.raw_phone_count}｜conf={float(result.confidence):.2f}",
            flush=True,
        )
