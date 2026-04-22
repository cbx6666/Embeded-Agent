"""视觉输入适配器：摄像头 + MediaPipe +（可选）ResNet。

职责（对内核不可见）：
- 采集、人脸网格、EAR/PERCLOS、疲劳档位与滞回、人脸裁剪与情绪推理；
- 仅通过标准 Event 向上游投递结果。

唯一依赖的内核接口：`handle_event(Event)`（由注入的 sink 提供）。
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Any, Protocol

from src.agent.event import Event, user_emotion_updated_from_rafdb

from .config import VisionAffectConfig
from .emotion_torch import RafEmotionBackend
from .pipeline import (
    FatigueLevel,
    PercLosWindow,
    face_bbox_from_landmarks,
    map_fatigue_with_hysteresis,
    mean_ear,
    monotonic_ts,
)


class EventEmitSink(Protocol):
    """只要能接收标准事件即可（通常为 `AgentCore.handle_event`）。"""

    def handle_event(self, event: Event) -> Any:
        ...


def vision_dependencies_met() -> bool:
    try:
        import cv2  # noqa: F401
        import mediapipe as mp  # noqa: F401

        return True
    except ImportError:
        return False


class VisionAffectInputAdapter:
    """后台线程跑检测逻辑，只向 sink 发 `user_fatigue_updated` / `user_emotion_updated`。"""

    def __init__(self, sink: EventEmitSink, config: VisionAffectConfig) -> None:
        self._sink = sink
        self._cfg = config
        self._ear_threshold = config.ear_threshold
        self._perclos = PercLosWindow(config.perclos_window_sec)
        self._min_frame_interval = 1.0 / max(config.target_fps, 1.0)
        self._emotion_backend = RafEmotionBackend(config.raf_checkpoint)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_fatigue_level: FatigueLevel = "none"
        self._last_fatigue_emit_mon: float = 0.0
        self._frame_counter = 0
        self._emotion_every_n = max(1, config.emotion_every_n_frames)
        self._last_emotion_emit_mon: float = 0.0
        self._last_emotion_label_id: int | None = None

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="vision-affect", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run_loop(self) -> None:
        try:
            self._run_capture_session()
        except Exception:
            print("[vision_affect] 线程异常退出:\n" + traceback.format_exc(), file=sys.stderr)

    def _run_capture_session(self) -> None:
        import cv2
        import mediapipe as mp

        cap = cv2.VideoCapture(self._cfg.camera_index)
        if not cap.isOpened():
            print(
                f"[vision_affect] 无法打开摄像头 index={self._cfg.camera_index}",
                file=sys.stderr,
            )
            return
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self._cfg.face_mesh_max_faces,
            refine_landmarks=self._cfg.face_mesh_refine_landmarks,
            min_detection_confidence=self._cfg.face_mesh_min_detection_confidence,
            min_tracking_confidence=self._cfg.face_mesh_min_tracking_confidence,
        )
        try:
            while not self._stop.is_set():
                t0 = monotonic_ts()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if not res.multi_face_landmarks:
                    self._sleep_remainder(t0)
                    continue
                lm = res.multi_face_landmarks[0]
                ear = mean_ear(lm, w, h)
                now = monotonic_ts()
                eye_closed = ear < self._ear_threshold
                self._perclos.push(now, eye_closed)
                perclos = self._perclos.ratio()
                new_level = map_fatigue_with_hysteresis(perclos, self._last_fatigue_level)
                periodic = (now - self._last_fatigue_emit_mon) >= self._cfg.fatigue_periodic_emit_sec
                level_changed = new_level != self._last_fatigue_level
                if level_changed or periodic:
                    self._emit_fatigue(new_level, perclos)
                    self._last_fatigue_level = new_level
                    self._last_fatigue_emit_mon = now

                self._frame_counter += 1
                if self._frame_counter % self._emotion_every_n == 0:
                    self._maybe_emit_emotion(frame, lm, w, h, now)

                self._sleep_remainder(t0)
        finally:
            face_mesh.close()
            cap.release()

    def _sleep_remainder(self, t0: float) -> None:
        elapsed = monotonic_ts() - t0
        sleep_s = self._min_frame_interval - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)

    def _emit_fatigue(self, level: FatigueLevel, perclos: float) -> None:
        ts = int(time.time())
        self._sink.handle_event(
            Event(
                type="user_fatigue_updated",
                timestamp=ts,
                payload={
                    "fatigue_level": level,
                    "perclos": round(float(perclos), 4),
                    "yawn_in_window": False,
                    "window_sec": int(self._perclos.window_sec),
                    "source": self._cfg.fatigue_event_source,
                },
            )
        )

    def _maybe_emit_emotion(
        self,
        frame_bgr,
        landmarks,
        w: int,
        h: int,
        now_mon: float,
    ) -> None:
        if not self._emotion_backend.available():
            return
        x1, y1, x2, y2 = face_bbox_from_landmarks(landmarks, w, h)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return
        label_id, conf = self._emotion_backend.predict(crop)
        if label_id is None:
            return
        # 防抖：同类标签且间隔过短则不上报，减轻内核与记忆压力
        interval_ok = (now_mon - self._last_emotion_emit_mon) >= self._cfg.emotion_min_emit_interval_sec
        label_changed = label_id != self._last_emotion_label_id
        if not label_changed and not interval_ok:
            return
        self._last_emotion_label_id = label_id
        self._last_emotion_emit_mon = now_mon
        ts = int(time.time())
        self._sink.handle_event(
            user_emotion_updated_from_rafdb(
                timestamp=ts,
                label_id=label_id,
                confidence=conf,
                source=self._cfg.emotion_event_source,
            )
        )
