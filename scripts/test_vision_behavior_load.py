#!/usr/bin/env python3
"""表情识别 + 行为识别 同机并行负载测试（单摄像头共享帧）。

用法（项目根目录）:
  source /opt/ai-envs/shared/bin/activate
  python scripts/test_vision_behavior_load.py --duration 60
  python scripts/test_vision_behavior_load.py --duration 90 --emotion-backend deepface --behavior-backend om
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class LoadStats:
    vision_frames: int = 0
    vision_errors: int = 0
    emotion_infers: int = 0
    emotion_ms: list[float] = field(default_factory=list)
    fatigue_events: int = 0
    emotion_events: int = 0
    behavior_infers: int = 0
    behavior_ms: list[float] = field(default_factory=list)
    behavior_events: int = 0
    cpu_samples: list[float] = field(default_factory=list)
    rss_mb_samples: list[float] = field(default_factory=list)


class PrintSink:
    def handle_event(self, event) -> None:
        p = event.payload
        if event.type == "user_emotion_updated":
            self.stats.emotion_events += 1
            if self.stats.emotion_events <= 5 or self.stats.emotion_events % 10 == 0:
                print(
                    f"[表情] {p.get('emotion')} conf={p.get('confidence')}",
                    flush=True,
                )
        elif event.type == "user_fatigue_updated":
            self.stats.fatigue_events += 1
            if self.stats.fatigue_events <= 3 or self.stats.fatigue_events % 15 == 0:
                print(
                    f"[疲劳] {p.get('fatigue_level')} perclos={p.get('perclos')}",
                    flush=True,
                )
        elif event.type == "user_attention_updated":
            self.stats.behavior_events += 1
            if self.stats.behavior_events <= 5 or self.stats.behavior_events % 8 == 0:
                print(
                    f"[行为] attention={p.get('attention')} behavior={p.get('behavior')}",
                    flush=True,
                )

    def __init__(self, stats: LoadStats) -> None:
        self.stats = stats


def _read_proc_rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _read_cpu_percent(interval: float = 0.1) -> float:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=interval))
    except ImportError:
        return -1.0


def run_load_test(args: argparse.Namespace) -> int:
    import cv2

    from src.adapters.behavior.camera_utils import grab_latest_frame, open_camera, warmup_camera
    from src.adapters.behavior.phone_hand_detector import PhoneHandProximityDetector, dependencies_met as behavior_deps
    from src.adapters.behavior_adapter import BehaviorAdapter
    from src.adapters.vision_affect import (
        VisionAffectConfig,
        vision_dependencies_met,
        vision_emotion_backend_ready,
    )

    if not vision_dependencies_met():
        print("缺少视觉依赖：opencv-python-headless、mediapipe", flush=True)
        return 1
    if not behavior_deps():
        print("缺少行为依赖：pip install -r requirements-behavior.txt", flush=True)
        return 1

    cfg = VisionAffectConfig(
        camera_index=args.camera,
        emotion_backend=args.emotion_backend,
        deepface_model=args.deepface_model,
        target_fps=args.vision_fps,
        emotion_every_n_frames=args.emotion_every_n,
        enable_state_storage=False,
        state_stats_db_path="",
    )
    if not vision_emotion_backend_ready(cfg) and args.emotion_backend.lower() not in {"none", "off"}:
        print(f"情绪后端 {args.emotion_backend} 未就绪，可改用 --emotion-backend none", flush=True)
        return 1

    stats = LoadStats()
    sink = PrintSink(stats)

    print("加载行为 YOLO 模型...", flush=True)
    behavior_detector = PhoneHandProximityDetector(
        inference_backend=args.behavior_backend,
        device=args.behavior_device,
        imgsz=args.imgsz,
        om_device_id=args.om_device_id,
    )
    behavior_detector.load_models()
    print(f"行为后端: {behavior_detector.active_backend}", flush=True)

    behavior = BehaviorAdapter(sink)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}", flush=True)
        return 1
    warmup_camera(cap, frames=8)

    import mediapipe as mp
    from src.adapters.vision_affect.adapter import (
        combined_fatigue_score,
        face_bbox_from_landmarks,
        map_fatigue_with_hysteresis,
        mean_ear,
        mean_mar,
        monotonic_ts,
    )
    from src.adapters.vision_affect.pipeline import PercLosWindow

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    perclos = PercLosWindow(window_sec=cfg.perclos_window_sec)
    yawn_w = PercLosWindow(window_sec=cfg.perclos_window_sec)
    last_fatigue = "none"
    frame_counter = 0
    min_frame_interval = 1.0 / max(1.0, cfg.target_fps)
    last_behavior_infer = 0.0
    behavior_interval = args.behavior_interval

    from src.adapters.vision_affect.backends.factory import build_emotion_backend

    emotion_backend = build_emotion_backend(cfg)
    emotion_enabled = args.emotion_backend.lower() not in {"none", "off", "disabled"}

    print(
        f"开始负载测试 {args.duration}s | 视觉 fps≈{cfg.target_fps} "
        f"情绪={args.emotion_backend} 行为间隔={behavior_interval}s",
        flush=True,
    )
    t_end = time.time() + args.duration
    t_report = time.time()
    last_heartbeat = time.time()

    try:
        while time.time() < t_end:
            t0 = monotonic_ts()
            ok, frame = grab_latest_frame(cap, flush=2)
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            stats.vision_frames += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                res = face_mesh.process(rgb)
            except Exception as exc:
                stats.vision_errors += 1
                print(f"[视觉] face_mesh 异常: {exc}", flush=True)
                res = None

            if res and res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                ear = mean_ear(lm, w, h)
                mar = mean_mar(lm, w, h)
                now = monotonic_ts()
                eye_closed = ear < cfg.ear_threshold
                perclos.push(now, eye_closed)
                yawn_w.push(now, mar > cfg.mar_yawn_threshold)
                combined = combined_fatigue_score(
                    perclos.ratio(),
                    yawn_w.ratio(),
                    eye_weight=cfg.fatigue_eye_weight,
                    mouth_weight=cfg.fatigue_mouth_weight,
                )
                new_level = map_fatigue_with_hysteresis(combined, last_fatigue)
                last_fatigue = new_level
                sink.handle_event(
                    __import__("src.agent.event", fromlist=["make_fatigue_event"]).make_fatigue_event(
                        fatigue_level=new_level,
                        perclos=perclos.ratio(),
                        source="load_test",
                    )
                )

                frame_counter += 1
                if emotion_enabled and frame_counter % max(1, cfg.emotion_every_n_frames) == 0:
                    x1, y1, x2, y2 = face_bbox_from_landmarks(lm, w, h)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        t_inf = time.perf_counter()
                        try:
                            pr = emotion_backend.predict(crop)
                            stats.emotion_infers += 1
                            stats.emotion_ms.append((time.perf_counter() - t_inf) * 1000.0)
                            if not pr.is_empty:
                                from src.agent.event import user_emotion_updated_standard

                                sink.handle_event(
                                    user_emotion_updated_standard(
                                        timestamp=int(time.time()),
                                        emotion=pr.agent_emotion or "neutral",
                                        confidence=pr.confidence,
                                        source="load_test",
                                        model=args.emotion_backend,
                                    )
                                )
                        except Exception as exc:
                            stats.vision_errors += 1
                            print(f"[表情] 推理失败: {exc}", flush=True)

            now_wall = time.time()
            if now_wall - last_behavior_infer >= behavior_interval:
                last_behavior_infer = now_wall
                t_inf = time.perf_counter()
                try:
                    result = behavior_detector.analyze_frame_stable(frame)
                    stats.behavior_infers += 1
                    stats.behavior_ms.append((time.perf_counter() - t_inf) * 1000.0)
                    if result.presence_phase == "left":
                        behavior.publish_attention("focused", "away", 0.9, source="load_test")
                    elif result.phone_in_hand:
                        behavior.publish_attention("distracted", "phone_use", result.confidence, source="load_test")
                    else:
                        behavior.publish_attention("focused", "working", 0.9, source="load_test")
                except Exception as exc:
                    stats.vision_errors += 1
                    print(f"[行为] 推理失败: {exc}", flush=True)

            elapsed = monotonic_ts() - t0
            sleep_s = min_frame_interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

            if now_wall - t_report >= 1.0:
                stats.cpu_samples.append(_read_cpu_percent(0.05))
                stats.rss_mb_samples.append(_read_proc_rss_mb())
                t_report = now_wall

            if now_wall - last_heartbeat >= args.heartbeat:
                last_heartbeat = now_wall
                b_ms = stats.behavior_ms[-1] if stats.behavior_ms else 0.0
                e_ms = stats.emotion_ms[-1] if stats.emotion_ms else 0.0
                print(
                    f"[心跳] 视觉帧={stats.vision_frames} 行为推理={stats.behavior_infers} "
                    f"({b_ms:.0f}ms) 表情推理={stats.emotion_infers} ({e_ms:.0f}ms) "
                    f"RSS={stats.rss_mb_samples[-1] if stats.rss_mb_samples else 0:.0f}MB",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("用户中断", flush=True)
    finally:
        face_mesh.close()
        cap.release()

    _print_summary(stats, args.duration)
    return 0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


def _print_summary(stats: LoadStats, duration: float) -> None:
    vfps = stats.vision_frames / duration if duration > 0 else 0.0
    b_avg = sum(stats.behavior_ms) / len(stats.behavior_ms) if stats.behavior_ms else 0.0
    e_avg = sum(stats.emotion_ms) / len(stats.emotion_ms) if stats.emotion_ms else 0.0
    cpu_avg = sum(stats.cpu_samples) / len(stats.cpu_samples) if stats.cpu_samples else -1.0
    rss_max = max(stats.rss_mb_samples) if stats.rss_mb_samples else 0.0

    print("\n========== 负载测试汇总 ==========", flush=True)
    print(f"时长: {duration:.0f}s", flush=True)
    print(f"视觉采集帧: {stats.vision_frames} ({vfps:.1f} fps)", flush=True)
    print(f"表情推理: {stats.emotion_infers} 次, 平均 {e_avg:.0f}ms, P95 {_percentile(stats.emotion_ms, 0.95):.0f}ms", flush=True)
    print(f"行为推理: {stats.behavior_infers} 次, 平均 {b_avg:.0f}ms, P95 {_percentile(stats.behavior_ms, 0.95):.0f}ms", flush=True)
    print(f"事件: 疲劳={stats.fatigue_events} 表情={stats.emotion_events} 行为={stats.behavior_events}", flush=True)
    print(f"错误次数: {stats.vision_errors}", flush=True)
    if cpu_avg >= 0:
        print(f"CPU 采样均值: {cpu_avg:.0f}% (3 核机器 300% 为满载)", flush=True)
    print(f"进程 RSS 峰值: {rss_max:.0f} MB", flush=True)

    ok = stats.vision_errors == 0 and vfps >= 3.0 and stats.behavior_infers > 0
    if stats.emotion_infers > 0 and e_avg > 8000:
        print("结论: 能跑但表情推理偏慢，建议降低 emotion_every_n 或换 wujie-om/none", flush=True)
        ok = ok and e_avg < 15000
    elif ok:
        print("结论: 并行负载可接受", flush=True)
    else:
        print("结论: 负载偏高或链路异常，建议错开 NPU/降帧/减推理频率", flush=True)
    print("================================\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="表情+行为并行负载测试")
    parser.add_argument("--duration", type=int, default=60, help="测试时长（秒）")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--emotion-backend", default="deepface", help="deepface | none")
    parser.add_argument("--behavior-backend", default="auto", choices=("auto", "pt", "om"))
    parser.add_argument("--behavior-device", default="cpu")
    parser.add_argument("--behavior-interval", type=float, default=0.3, help="行为推理最小间隔秒")
    parser.add_argument("--vision-fps", type=float, default=6.0, help="视觉主循环目标 fps（降载）")
    parser.add_argument("--emotion-every-n", type=int, default=6, help="每 N 帧做一次表情推理")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--om-device-id", type=int, default=0)
    parser.add_argument("--heartbeat", type=float, default=5.0)
    parser.add_argument("--deepface-model", default="VGG-Face")
    args = parser.parse_args()
    raise SystemExit(run_load_test(args))


if __name__ == "__main__":
    main()
