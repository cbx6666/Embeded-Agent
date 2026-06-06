"""视觉输入适配器：摄像头 + MediaPipe Face Mesh + 疲劳 (EAR+MAR) + 情绪 (DeepFace 等)。

职责（对内核不可见）：
- EAR 与眼部滑动窗得 PERCLOS；MAR 与口部窗得打哈欠/张口占比；
- 两路融合后经滞回得到疲劳档位；人脸 crop 上跑情绪模型；
- 仅通过标准 Event 向上游投递结果。

唯一依赖的内核接口：`handle_event(Event)`（由注入的 sink 提供）。
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
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
from .emotion_second_stats import EmotionSecondStats, EmotionSecondSummary
from .emotion_state_store import EmotionStateStore
from .fatigue_second_stats import FatigueSecondStats, FatigueSecondSummary
from .fatigue_state_store import FatigueStateStore
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


def vision_dependencies_met() -> bool:
    try:
        import cv2  # noqa: F401
        import mediapipe as mp  # noqa: F401

        return True
    except ImportError:
        return False


def vision_emotion_backend_ready(config: VisionAffectConfig) -> bool:
    """与 `build_emotion_backend` 的可用性预期一致，用于启动时提示。"""
    b = (config.emotion_backend or "wujie-om").strip().lower()
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
    if b in {"wujie-vgg19", "wujie", "fer-vgg19"}:
        from pathlib import Path

        p = config.wujie_checkpoint
        if not p or not Path(p).is_file():
            return False
        try:
            import torch  # noqa: F401
            import cv2  # noqa: F401
        except ImportError:
            return False
        return True
    if b in {"wujie-om", "om", "wujie_om"}:
        from pathlib import Path

        p = config.wujie_om_model
        if not p or not Path(p).is_file():
            return False
        try:
            from src.adapters.vision_common.acl_runtime import import_acl

            import_acl()
            import cv2  # noqa: F401
        except Exception:
            return False
        return True
    if b == "deepface":
        try:
            import deepface  # noqa: F401
        except ImportError:
            return False
        return True
    return True


@dataclass
class _EmotionJob:
    """后台情绪推理任务（人脸 crop 副本，避免与采集线程共享内存）。"""

    crop: Any
    timestamp_sec: int


class VisionAffectInputAdapter:
    """后台线程跑检测逻辑，只向 sink 发 `user_fatigue_updated` / `user_emotion_updated`。"""

    def __init__(self, sink: EventEmitSink, config: VisionAffectConfig, *, frame_bus: object | None = None) -> None:
        self._sink = sink
        self._cfg = config
        self._frame_bus = frame_bus
        self._ear_threshold = config.ear_threshold
        self._mar_yawn_threshold = config.mar_yawn_threshold
        self._perclos = PercLosWindow(config.perclos_window_sec)
        self._yawn_w = PercLosWindow(config.perclos_window_sec)
        self._min_frame_interval = 1.0 / max(config.target_fps, 1.0)
        self._emotion_backend: EmotionInferenceBackend = build_emotion_backend(config)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._emotion_worker: threading.Thread | None = None
        self._emotion_queue: queue.Queue[_EmotionJob | None] = queue.Queue(maxsize=1)
        self._emotion_lock = threading.Lock()
        self._last_emotion_submit_mono: float = 0.0
        self._last_fatigue_level: FatigueLevel = "none"
        self._last_frame_monotonic: float | None = None
        self._eye_closed_streak_sec: float = 0.0
        self._last_eye_perclos: float = 0.0
        self._last_yawn_ratio: float = 0.0
        self._frame_counter = 0
        self._emotion_every_n = max(1, config.emotion_every_n_frames)
        self._emotion_async_enabled = self._emotion_backend.available()
        # 每秒聚合按模块拆分，便于后续扩展行为统计
        self._fatigue_second_stats = FatigueSecondStats()
        self._emotion_second_stats = EmotionSecondStats()
        self._fatigue_state_store: FatigueStateStore | None = None
        self._emotion_state_store: EmotionStateStore | None = None
        if config.enable_state_storage:
            try:
                self._fatigue_state_store = FatigueStateStore(
                    config.state_stats_db_path,
                    second_state_retention_days=config.second_state_retention_days,
                    cleanup_interval_sec=config.state_cleanup_interval_sec,
                )
                self._emotion_state_store = EmotionStateStore(
                    config.state_stats_db_path,
                    second_state_retention_days=config.second_state_retention_days,
                    cleanup_interval_sec=config.state_cleanup_interval_sec,
                )
            except Exception:
                if self._fatigue_state_store is not None:
                    self._fatigue_state_store.close()
                    self._fatigue_state_store = None
                if self._emotion_state_store is not None:
                    self._emotion_state_store.close()
                    self._emotion_state_store = None
                print(
                    f"[vision_affect] 状态存储初始化失败，已降级为仅事件上报: db={config.state_stats_db_path}",
                    file=sys.stderr,
                )

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._start_emotion_worker()
        self._thread = threading.Thread(target=self._run_loop, name="vision-affect", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._stop_emotion_worker(timeout=timeout)
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._fatigue_state_store is not None:
            self._fatigue_state_store.close()
            self._fatigue_state_store = None
        if self._emotion_state_store is not None:
            self._emotion_state_store.close()
            self._emotion_state_store = None

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
            from src.adapters.behavior.camera_utils import open_camera

            cap = open_camera(self._cfg.camera_index)
        if not cap.isOpened():
            print(
                f"[vision_affect] 无法打开摄像头 index={self._cfg.camera_index}",
                file=sys.stderr,
            )
            from src.adapters.perception_debug_log import perception_debug

            perception_debug().log(
                "vision",
                "camera_open_failed",
                camera_index=self._cfg.camera_index,
            )
            return
        from src.adapters.perception_debug_log import perception_debug

        perception_debug().log(
            "vision",
            "camera_opened",
            camera_index=self._cfg.camera_index,
            emotion_backend=self._cfg.emotion_backend,
        )
        last_no_face_log_mono = 0.0
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
                if self._frame_bus is not None:
                    try:
                        self._frame_bus.publish(frame)
                    except Exception:
                        pass
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if not res.multi_face_landmarks:
                    no_face_now = monotonic_ts()
                    if no_face_now - last_no_face_log_mono >= 5.0:
                        last_no_face_log_mono = no_face_now
                        perception_debug().log(
                            "vision",
                            "no_face",
                            camera_index=self._cfg.camera_index,
                            frame_size=f"{w}x{h}",
                        )
                    self._sleep_remainder(t0)
                    continue
                lm = res.multi_face_landmarks[0]
                ear = mean_ear(lm, w, h)
                mar = mean_mar(lm, w, h)
                now = monotonic_ts()
                now_sec = int(time.time())
                dt = self._min_frame_interval if self._last_frame_monotonic is None else max(
                    0.0, min(0.5, now - self._last_frame_monotonic)
                )
                self._last_frame_monotonic = now
                eye_closed = ear < self._ear_threshold
                if eye_closed:
                    self._eye_closed_streak_sec += dt
                else:
                    self._eye_closed_streak_sec = 0.0
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
                if (
                    self._eye_closed_streak_sec >= float(self._cfg.force_high_eye_closed_sec)
                    or eye_p >= float(self._cfg.force_high_eye_perclos)
                ):
                    new_level = "high"
                elif eye_p >= float(self._cfg.force_moderate_eye_perclos) and new_level == "mild":
                    # 眼部闭合占比明显升高时，至少给到 moderate，避免长闭眼仅 mild。
                    new_level = "moderate"
                self._last_fatigue_level = new_level
                self._last_eye_perclos = eye_p
                self._last_yawn_ratio = yawn_p
                for summary in self._fatigue_second_stats.push(now_sec, new_level, combined):
                    self._emit_fatigue_second_summary(
                        summary,
                        eye_perclos=self._last_eye_perclos,
                        yawn_ratio=self._last_yawn_ratio,
                    )

                self._frame_counter += 1
                if self._emotion_async_enabled and self._frame_counter % self._emotion_every_n == 0:
                    self._submit_emotion_job(frame, lm, w, h, now_sec)

                self._sleep_remainder(t0)
        finally:
            last_fatigue = self._fatigue_second_stats.flush()
            if last_fatigue is not None:
                self._emit_fatigue_second_summary(last_fatigue)
            last_emotion = self._emotion_second_stats.flush()
            if last_emotion is not None:
                self._emit_emotion_second_summary(last_emotion)
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
        perclos: float | None,
        yawn_ratio: float | None,
        yawn_in_window: bool | None,
        confidence: float | None = None,
        timestamp: int | None = None,
    ) -> None:
        ts = int(time.time()) if timestamp is None else int(timestamp)
        self._sink.handle_event(
            make_fatigue_event(
                fatigue_level=level,
                perclos=round(float(perclos), 4) if perclos is not None else None,
                yawn_ratio=round(float(yawn_ratio), 4) if yawn_ratio is not None else None,
                yawn_in_window=yawn_in_window,
                window_sec=int(self._perclos.window_sec),
                confidence=confidence,
                source=self._cfg.fatigue_event_source,
                timestamp=ts,
            )
        )

    def _start_emotion_worker(self) -> None:
        if not self._emotion_async_enabled:
            return
        if self._emotion_worker is not None and self._emotion_worker.is_alive():
            return
        self._emotion_worker = threading.Thread(
            target=self._emotion_worker_loop,
            name="vision-affect-emotion",
            daemon=True,
        )
        self._emotion_worker.start()

    def _stop_emotion_worker(self, timeout: float = 3.0) -> None:
        if self._emotion_worker is None:
            return
        try:
            self._emotion_queue.put_nowait(None)
        except queue.Full:
            try:
                self._emotion_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._emotion_queue.put_nowait(None)
            except queue.Full:
                pass
        self._emotion_worker.join(timeout=timeout)
        self._emotion_worker = None

    def _submit_emotion_job(self, frame_bgr, landmarks, w: int, h: int, timestamp_sec: int) -> None:
        if not self._emotion_async_enabled:
            return
        now_mono = monotonic_ts()
        min_gap = float(self._cfg.emotion_min_emit_interval_sec)
        if now_mono - self._last_emotion_submit_mono < min_gap:
            return
        x1, y1, x2, y2 = face_bbox_from_landmarks(landmarks, w, h)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return
        self._last_emotion_submit_mono = now_mono
        job = _EmotionJob(crop=crop.copy(), timestamp_sec=int(timestamp_sec))
        try:
            self._emotion_queue.put_nowait(job)
        except queue.Full:
            try:
                self._emotion_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._emotion_queue.put_nowait(job)
            except queue.Full:
                pass

    def _emotion_worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._emotion_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                pr = self._emotion_backend.predict(job.crop)
            except Exception as exc:
                print(f"[vision_affect] 情绪推理失败: {exc}", file=sys.stderr)
                from src.adapters.perception_debug_log import perception_debug

                perception_debug().log("emotion", "infer_failed", error=str(exc))
                continue
            if pr.is_empty:
                continue
            if pr.raf_label_id is not None:
                key = f"raf:{pr.raf_label_id}"
            else:
                key = f"emo:{pr.agent_emotion}"
            result_sec = int(time.time())
            with self._emotion_lock:
                for summary in self._emotion_second_stats.push(result_sec, key, pr.confidence):
                    self._emit_emotion_second_summary(summary)

    def _emit_fatigue_second_summary(
        self,
        summary: FatigueSecondSummary,
        *,
        eye_perclos: float | None = None,
        yawn_ratio: float | None = None,
    ) -> None:
        if self._fatigue_state_store is not None:
            try:
                self._fatigue_state_store.record_second_state(
                    timestamp=summary.timestamp,
                    fatigue_level=summary.fatigue_level,
                    confidence=summary.avg_confidence,
                    source=self._cfg.fatigue_event_source,
                )
            except Exception as exc:
                print(f"[vision_affect] 疲劳状态写入 SQLite 失败: {exc}", file=sys.stderr)
        yawn_flag = None
        if yawn_ratio is not None:
            yawn_flag = float(yawn_ratio) >= float(self._cfg.yawn_flag_min_ratio)
        self._emit_fatigue(
            summary.fatigue_level,
            perclos=eye_perclos,
            yawn_ratio=yawn_ratio,
            yawn_in_window=yawn_flag,
            confidence=summary.avg_confidence,
            timestamp=summary.timestamp,
        )
        from src.adapters.perception_debug_log import perception_debug

        perception_debug().log(
            "fatigue",
            "detected",
            level=summary.fatigue_level,
            perclos=round(float(eye_perclos), 4) if eye_perclos is not None else None,
            yawn_ratio=round(float(yawn_ratio), 4) if yawn_ratio is not None else None,
            yawn_in_window=yawn_flag,
            confidence=round(float(summary.avg_confidence), 4),
            timestamp=summary.timestamp,
        )

    def _emit_emotion_second_summary(self, summary: EmotionSecondSummary) -> None:
        emo_key = summary.emotion_key
        if emo_key.startswith("raf:"):
            label_id = int(emo_key.split(":", 1)[1])
            event = user_emotion_updated_from_rafdb(
                timestamp=summary.timestamp,
                label_id=label_id,
                confidence=summary.avg_confidence,
                source=self._cfg.emotion_event_source,
            )
            if self._emotion_state_store is not None:
                try:
                    self._emotion_state_store.record_second_state(
                        timestamp=summary.timestamp,
                        emotion=str(event.payload.get("emotion", "neutral")),
                        confidence=summary.avg_confidence,
                        source=self._cfg.emotion_event_source,
                        model="raf-db",
                    )
                except Exception as exc:
                    print(f"[vision_affect] 情绪状态写入 SQLite 失败: {exc}", file=sys.stderr)
            self._sink.handle_event(event)
            from src.adapters.perception_debug_log import perception_debug

            perception_debug().log(
                "emotion",
                "detected",
                emotion=str(event.payload.get("emotion", "unknown")),
                raf_emotion=event.payload.get("raf_emotion"),
                confidence=round(float(summary.avg_confidence), 4),
                backend="raf-db",
                timestamp=summary.timestamp,
            )
            return

        if emo_key.startswith("emo:"):
            emotion = emo_key.split(":", 1)[1]
            model_name = (
                "deepface"
                if (self._cfg.emotion_backend or "").lower() == "deepface"
                else ("wujie-om" if (self._cfg.emotion_backend or "").lower() in {"wujie-om", "om", "wujie_om"} else "wujie-vgg19")
            )
            if self._emotion_state_store is not None:
                try:
                    self._emotion_state_store.record_second_state(
                        timestamp=summary.timestamp,
                        emotion=emotion,
                        confidence=summary.avg_confidence,
                        source=self._cfg.emotion_event_source,
                        model=model_name,
                    )
                except Exception as exc:
                    print(f"[vision_affect] 情绪状态写入 SQLite 失败: {exc}", file=sys.stderr)
            self._sink.handle_event(
                user_emotion_updated_standard(
                    timestamp=summary.timestamp,
                    emotion=emotion,
                    confidence=summary.avg_confidence,
                    source=self._cfg.emotion_event_source,
                    model=model_name,
                )
            )
            from src.adapters.perception_debug_log import perception_debug

            perception_debug().log(
                "emotion",
                "detected",
                emotion=emotion,
                confidence=round(float(summary.avg_confidence), 4),
                backend=model_name,
                timestamp=summary.timestamp,
            )

    def query_state_seconds(self, *, start_ts: int, end_ts: int) -> dict[str, dict[str, int]]:
        """按时间区间统计 emotion / fatigue 的秒数分布。"""
        fatigue = (
            self._fatigue_state_store.query_seconds_by_state(start_ts=start_ts, end_ts=end_ts)
            if self._fatigue_state_store is not None
            else {}
        )
        emotion = (
            self._emotion_state_store.query_seconds_by_state(start_ts=start_ts, end_ts=end_ts)
            if self._emotion_state_store is not None
            else {}
        )
        return {"fatigue_seconds": fatigue, "emotion_seconds": emotion}

    def query_daily_summary(self, *, day: str) -> dict[str, dict[str, dict[str, float | int]]]:
        """查询某天的状态汇总（秒数与占比）。"""
        fatigue = self._fatigue_state_store.query_daily_summary(day=day) if self._fatigue_state_store else {}
        emotion = self._emotion_state_store.query_daily_summary(day=day) if self._emotion_state_store else {}
        return {"fatigue_daily": fatigue, "emotion_daily": emotion}
