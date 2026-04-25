"""视觉输入适配器：摄像头 + MediaPipe Face Mesh + 疲劳 (EAR+MAR) + 情绪 (DeepFace 等)。

职责（对内核不可见）：
- EAR 与眼部滑动窗得 PERCLOS；MAR 与口部窗得打哈欠/张口占比；
- 两路融合后经滞回得到疲劳档位；人脸 crop 上跑情绪模型；
- 仅通过标准 Event 向上游投递结果。

唯一依赖的内核接口：`handle_event(Event)`（由注入的 sink 提供）。
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Any, Protocol

from src.agent.event import (
    Event,
    make_fatigue_event,
    user_emotion_updated_from_rafdb,
    user_emotion_updated_standard,
)

from .backends.factory import build_emotion_backend
from .backends.protocols import EmotionInferenceBackend, EmotionPredictResult
from .config import VisionAffectConfig
from .pipeline import (
    FatigueLevel,
    PercLosWindow,
    combined_fatigue_score,
    face_bbox_from_landmarks,
    map_fatigue_with_hysteresis,
    mean_ear,
    mean_mar,
    monotonic_ts,
)


class EventEmitSink(Protocol):
    """只要能接收标准事件即可（通常为 `AgentCore.handle_event`）。"""

    def handle_event(self, event: Event) -> Any:
        ...


def _emotion_fingerprint(pr: EmotionPredictResult) -> tuple[int, str]:
    if pr.raf_label_id is not None:
        return (pr.raf_label_id, "")
    return (0, pr.agent_emotion or "")


def vision_dependencies_met() -> bool:
    try:
        import cv2  # noqa: F401
        import mediapipe as mp  # noqa: F401

        return True
    except ImportError:
        return False


def vision_emotion_backend_ready(config: VisionAffectConfig) -> bool:
    """与 `build_emotion_backend` 的可用性预期一致，用于启动时提示。"""
    b = (config.emotion_backend or "deepface").strip().lower()
    if b in {"none", "off", "disabled"}:
        return True
    if b in ("raf", "raf-db"):
        from pathlib import Path

        p = config.raf_checkpoint
        if not p or not Path(p).is_file():
            return False
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return True
    if b == "deepface":
        try:
            import deepface  # noqa: F401
        except ImportError:
            return False
        return True
    return True


class VisionAffectInputAdapter:
    """后台线程跑检测逻辑，只向 sink 发 `user_fatigue_updated` / `user_emotion_updated`。"""

    def __init__(self, sink: EventEmitSink, config: VisionAffectConfig) -> None:
        self._sink = sink
        self._cfg = config
        self._ear_threshold = config.ear_threshold
        self._mar_yawn_threshold = config.mar_yawn_threshold
        self._perclos = PercLosWindow(config.perclos_window_sec)
        self._yawn_w = PercLosWindow(config.perclos_window_sec)
        self._min_frame_interval = 1.0 / max(config.target_fps, 1.0)
        self._emotion_backend: EmotionInferenceBackend = build_emotion_backend(config)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_fatigue_level: FatigueLevel = "none"
        self._last_fatigue_emit_mon: float = 0.0
        self._frame_counter = 0
        self._emotion_every_n = max(1, config.emotion_every_n_frames)
        self._last_emotion_emit_mon: float = 0.0
        self._last_emotion_fingerprint: tuple[int, str] | None = None

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
                mar = mean_mar(lm, w, h)
                now = monotonic_ts()
                eye_closed = ear < self._ear_threshold
                self._perclos.push(now, eye_closed)
                yawn_mouth = mar > self._mar_yawn_threshold
                self._yawn_w.push(now, yawn_mouth)
                eye_p = self._perclos.ratio()
                yawn_p = self._yawn_w.ratio()
                combined = combined_fatigue_score(
                    eye_p,
                    yawn_p,
                    eye_weight=self._cfg.fatigue_eye_weight,
                    mouth_weight=self._cfg.fatigue_mouth_weight,
                )
                new_level = map_fatigue_with_hysteresis(combined, self._last_fatigue_level)
                periodic = (now - self._last_fatigue_emit_mon) >= self._cfg.fatigue_periodic_emit_sec
                level_changed = new_level != self._last_fatigue_level
                if level_changed or periodic:
                    yawn_flag = yawn_p >= self._cfg.yawn_flag_min_ratio
                    self._emit_fatigue(
                        new_level,
                        perclos=eye_p,
                        yawn_ratio=yawn_p,
                        yawn_in_window=yawn_flag,
                    )
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

    def _emit_fatigue(
        self,
        level: FatigueLevel,
        *,
        perclos: float,
        yawn_ratio: float,
        yawn_in_window: bool,
    ) -> None:
        ts = int(time.time())
        self._sink.handle_event(
            make_fatigue_event(
                fatigue_level=level,
                perclos=round(float(perclos), 4),
                yawn_ratio=round(float(yawn_ratio), 4),
                yawn_in_window=yawn_in_window,
                window_sec=int(self._perclos.window_sec),
                source=self._cfg.fatigue_event_source,
                timestamp=ts,
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
        pr = self._emotion_backend.predict(crop)
        if pr.is_empty:
            return
        fp = _emotion_fingerprint(pr)
        # 防抖：同类结果且间隔过短则不上报，减轻内核与记忆压力
        interval_ok = (now_mon - self._last_emotion_emit_mon) >= self._cfg.emotion_min_emit_interval_sec
        label_changed = fp != self._last_emotion_fingerprint
        if not label_changed and not interval_ok:
            return
        self._last_emotion_fingerprint = fp
        self._last_emotion_emit_mon = now_mon
        ts = int(time.time())
        if pr.raf_label_id is not None:
            self._sink.handle_event(
                user_emotion_updated_from_rafdb(
                    timestamp=ts,
                    label_id=pr.raf_label_id,
                    confidence=pr.confidence,
                    source=self._cfg.emotion_event_source,
                )
            )
        elif pr.agent_emotion is not None:
            self._sink.handle_event(
                user_emotion_updated_standard(
                    timestamp=ts,
                    emotion=pr.agent_emotion,
                    confidence=pr.confidence,
                    source=self._cfg.emotion_event_source,
                    model="deepface" if (self._cfg.emotion_backend or "").lower() == "deepface" else None,
                )
            )
