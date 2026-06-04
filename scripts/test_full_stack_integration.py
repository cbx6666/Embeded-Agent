#!/usr/bin/env python3
"""全栈感知联调：视觉(疲劳/情绪) + 行为(手机/在场) + 姿势，并统计 Event 与性能。

不启桌宠/语音硬件（无 DISPLAY/ALSA 也可跑）；语音相关 Event 用 mock 注入验证协议。

用法（项目根目录，推荐共享环境）:
  source /opt/ai-envs/shared/bin/activate
  python scripts/test_full_stack_integration.py --duration 45
  python scripts/test_full_stack_integration.py --duration 60 --emotion-backend none
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 全栈应能见到的 Event 类型（语音/显示用 mock 或说明跳过）
EXPECTED_EVENT_GROUPS: dict[str, list[str]] = {
    "视觉-疲劳": ["user_fatigue_updated"],
    "视觉-情绪": ["user_emotion_updated"],
    "行为-在场/注意力": ["user_presence_updated", "user_attention_updated"],
    "姿势-姿态/活动": ["user_posture_updated", "user_activity_updated"],
    "交互-mock": ["speech_recognized", "user_text_input"],
}


@dataclass
class StackStats:
    event_counts: Counter[str] = field(default_factory=Counter)
    event_samples: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    behavior_ms: list[float] = field(default_factory=list)
    emotion_ms: list[float] = field(default_factory=list)
    vision_frames: int = 0
    vision_errors: int = 0
    behavior_infers: int = 0
    emotion_infers: int = 0
    cpu_samples: list[float] = field(default_factory=list)
    rss_mb_samples: list[float] = field(default_factory=list)
    handle_event_ms: list[float] = field(default_factory=list)


class CollectingSink:
    """收集 Event；可选记录 handle_event 耗时。"""

    def __init__(self, stats: StackStats, *, track_latency: bool = True) -> None:
        self.stats = stats
        self.track_latency = track_latency

    def handle_event(self, event: Any) -> None:
        t0 = time.perf_counter()
        et = str(event.type)
        self.stats.event_counts[et] += 1
        samples = self.stats.event_samples[et]
        if len(samples) < 3:
            samples.append(dict(event.payload))
        if self.track_latency:
            self.stats.handle_event_ms.append((time.perf_counter() - t0) * 1000.0)


def _read_proc_rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _read_cpu_percent(interval: float = 0.05) -> float:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=interval))
    except ImportError:
        return -1.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def _inject_mock_interaction(sink: CollectingSink) -> None:
    from src.agent.event import Event

    now = int(time.time())
    sink.handle_event(
        Event(type="user_text_input", timestamp=now, payload={"text": "全栈联调测试", "source": "integration_test"})
    )
    sink.handle_event(
        Event(
            type="speech_recognized",
            timestamp=now + 1,
            payload={"text": "你好小助", "source": "integration_test", "confidence": 0.95},
        )
    )


def _print_coverage(stats: StackStats, *, emotion_enabled: bool, pose_from_behavior: bool) -> None:
    print("\n========== Event 功能覆盖 ==========", flush=True)
    checks: list[tuple[str, str, bool, str]] = []

    def add(group: str, et: str, required: bool) -> None:
        n = stats.event_counts.get(et, 0)
        ok = n > 0
        note = f"{n} 条" if ok else "未收到"
        checks.append((group, et, required and not ok, note))

    add("视觉-疲劳", "user_fatigue_updated", True)
    add("视觉-情绪", "user_emotion_updated", emotion_enabled)
    add("行为", "user_presence_updated", True)
    add("行为", "user_attention_updated", True)
    if pose_from_behavior:
        add("姿势(pose OM)", "user_posture_updated", True)
        add("姿势(pose OM)", "user_activity_updated", True)
    add("mock-文本", "user_text_input", True)
    add("mock-语音", "speech_recognized", True)

    for group, et, failed, note in checks:
        mark = "FAIL" if failed else "OK  "
        print(f"  [{mark}] {group:16} {et:28} {note}", flush=True)
        if stats.event_samples.get(et):
            sample = stats.event_samples[et][0]
            keys = ", ".join(list(sample.keys())[:8])
            print(f"         样例 payload 字段: {keys}", flush=True)

    missing_required = [et for _, et, failed, _ in checks if failed]
    if missing_required:
        print(f"\n  缺少必需 Event: {', '.join(missing_required)}", flush=True)
        print("  提示: 情绪需人脸在画面内；行为/姿势需人体在摄像头内（yolo26n-pose.om）。", flush=True)
    else:
        print("\n  必需 Event 均已收到。", flush=True)
    print("====================================\n", flush=True)


def _print_performance(stats: StackStats, duration: float) -> None:
    fatigue_n = stats.event_counts.get("user_fatigue_updated", 0)
    vfps = stats.vision_frames / duration if duration > 0 and stats.vision_frames else fatigue_n / duration
    b_avg = sum(stats.behavior_ms) / len(stats.behavior_ms) if stats.behavior_ms else 0.0
    e_avg = sum(stats.emotion_ms) / len(stats.emotion_ms) if stats.emotion_ms else 0.0
    cpu_avg = sum(stats.cpu_samples) / len(stats.cpu_samples) if stats.cpu_samples else -1.0
    rss_max = max(stats.rss_mb_samples) if stats.rss_mb_samples else 0.0
    he_avg = sum(stats.handle_event_ms) / len(stats.handle_event_ms) if stats.handle_event_ms else 0.0

    total_events = sum(stats.event_counts.values())
    ev_per_sec = total_events / duration if duration > 0 else 0.0

    print("========== 性能汇总 ==========", flush=True)
    print(f"时长: {duration:.0f}s", flush=True)
    print(f"视觉 fatigue Event: {fatigue_n} ({vfps:.1f}/s)", flush=True)
    print(
        f"行为 YOLO: {stats.behavior_infers} 次, "
        f"avg {b_avg:.0f}ms, P95 {_percentile(stats.behavior_ms, 0.95):.0f}ms",
        flush=True,
    )
    if stats.emotion_infers:
        print(
            f"表情推理: {stats.emotion_infers} 次, "
            f"avg {e_avg:.0f}ms, P95 {_percentile(stats.emotion_ms, 0.95):.0f}ms",
            flush=True,
        )
    else:
        print("表情推理: 0 次（--emotion-backend none 或未检测到人脸）", flush=True)
    print(f"Event 总计: {total_events} ({ev_per_sec:.1f}/s)", flush=True)
    for et, n in stats.event_counts.most_common():
        print(f"  - {et}: {n}", flush=True)
    print(f"handle_event 回调: avg {he_avg:.3f}ms (仅收集，无 AgentCore/LLM)", flush=True)
    if cpu_avg >= 0:
        print(f"CPU 采样均值: {cpu_avg:.0f}%", flush=True)
    print(f"进程 RSS 峰值: {rss_max:.0f} MB", flush=True)
    print(f"错误: {stats.vision_errors}", flush=True)

    # 粗算若接 full AgentCore + LLM 的压力
    perception_types = {
        "user_fatigue_updated",
        "user_emotion_updated",
        "user_presence_updated",
        "user_attention_updated",
    }
    perception_n = sum(stats.event_counts.get(t, 0) for t in perception_types)
    est_llm_per_event = 5  # 1 memory_observer + 4 decision roles (当前默认)
    print(
        f"\n[估算] 若每条感知 Event 都进完整 AgentCore+LLM: "
        f"~{perception_n}×{est_llm_per_event}≈{perception_n * est_llm_per_event} 次 LLM/本段测试",
        flush=True,
    )
    print(
        "  建议: 感知 Event 跳过 Decision LLM，或合并/节流后再决策（否则语音问答会被堵住）。",
        flush=True,
    )

    ok = stats.vision_errors == 0 and vfps >= 0.5 and stats.event_counts.get("user_attention_updated", 0) > 0
    if ok and (not stats.emotion_infers or e_avg < 12000):
        print("\n结论: 全栈并行负载可接受。", flush=True)
    elif vfps < 2.0 or b_avg > 500:
        print("\n结论: CPU/帧率吃紧，建议降 vision_fps、加大 behavior_interval、emotion_every_n。", flush=True)
    else:
        print("\n结论: 能跑；表情推理偏慢时可换 none/wujie-om 或降低频率。", flush=True)
    print("==============================\n", flush=True)


def run_integration(args: argparse.Namespace) -> int:
    import os
    import time

    from src.adapters.behavior.phone_camera_adapter import PhoneHandCameraAdapter
    from src.adapters.behavior.phone_hand_detector import PhoneHandProximityDetector, dependencies_met
    from src.adapters.behavior.yolo_om_runner import om_models_available
    from src.adapters.behavior_adapter import BehaviorAdapter
    from src.adapters.vision_affect import (
        VisionAffectConfig,
        VisionAffectInputAdapter,
        vision_dependencies_met,
        vision_emotion_backend_ready,
    )

    if not vision_dependencies_met():
        print("缺少视觉依赖: opencv + mediapipe (见 requirements.txt)", flush=True)
        return 1
    if not dependencies_met():
        print("缺少行为依赖: pip install -r requirements-behavior.txt", flush=True)
        return 1

    emotion_be = args.emotion_backend.strip().lower()
    emotion_enabled = emotion_be not in {"none", "off", "disabled"}
    from src.adapters.vision_affect.config import DEFAULT_WUJIE_OM_MODEL

    from src.adapters.perception_config import (
        behavior_inference_interval_sec,
        emotion_every_n_frames,
        perception_hz,
        vision_target_fps,
    )

    if args.perception_hz is not None:
        os.environ["EMBED_PERCEPTION_HZ"] = str(args.perception_hz)

    vision_fps = args.vision_fps if args.vision_fps is not None else vision_target_fps()
    emotion_every_n = args.emotion_every_n if args.emotion_every_n is not None else emotion_every_n_frames()
    behavior_interval = (
        args.behavior_interval
        if args.behavior_interval is not None
        else behavior_inference_interval_sec()
    )

    wujie_om_path = args.wujie_om or os.environ.get("WUJIE_OM_MODEL") or DEFAULT_WUJIE_OM_MODEL
    cfg = VisionAffectConfig(
        camera_index=args.camera,
        emotion_backend=args.emotion_backend,
        wujie_om_model=wujie_om_path,
        wujie_om_device_id=args.wujie_device_id,
        deepface_model=args.deepface_model,
        target_fps=vision_fps,
        emotion_every_n_frames=emotion_every_n,
        enable_state_storage=False,
        state_stats_db_path="",
    )
    if emotion_enabled and not vision_emotion_backend_ready(cfg):
        print(
            f"情绪后端 {args.emotion_backend} 未就绪（OM: {wujie_om_path}），仅测疲劳+行为。",
            flush=True,
        )
        emotion_enabled = False

    stats = StackStats()
    sink = CollectingSink(stats)

    from src.adapters.behavior.camera_utils import LatestFrameBus

    shared_frame_bus = LatestFrameBus()

    behavior_backend = args.behavior_backend.strip().lower()
    if behavior_backend == "auto":
        backend_hint = "om(NPU)" if om_models_available() else "pt(CPU)"
    else:
        backend_hint = behavior_backend

    print("加载行为 YOLO...", flush=True)
    behavior_detector = PhoneHandProximityDetector(
        inference_backend=behavior_backend,
        om_device_id=args.behavior_om_device_id,
        imgsz=args.imgsz,
        hold_seconds=args.hold_seconds,
    )
    behavior_detector.load_models()
    print(f"行为后端: {behavior_detector.active_backend} ({backend_hint})", flush=True)

    vision_adapter: VisionAffectInputAdapter | None = None
    behavior_adapter: PhoneHandCameraAdapter | None = None

    print("启动视觉适配器（MediaPipe 疲劳 + OM 情绪）...", flush=True)
    vision_adapter = VisionAffectInputAdapter(sink, cfg, frame_bus=shared_frame_bus)
    vision_adapter.start_background()

    behavior_adapter = PhoneHandCameraAdapter(
        sink,
        camera_index=args.camera,
        detector=behavior_detector,
        behavior_adapter=BehaviorAdapter(sink, debounce_seconds=args.behavior_debounce),
        inference_interval=behavior_interval,
        source="integration_test_om",
        frame_bus=shared_frame_bus,
    )
    behavior_adapter.start_background()

    print(
        f"全栈联调 {args.duration}s | camera={args.camera} "
        f"tick≈{perception_hz():.0f}Hz vision_fps≈{cfg.target_fps} "
        f"emotion={args.emotion_backend} behavior={behavior_detector.active_backend} "
        f"(pose 同帧 yolo26n-pose.om)",
        flush=True,
    )
    _inject_mock_interaction(sink)

    t_end = time.time() + args.duration
    last_heartbeat = time.time()
    t_report = time.time()

    try:
        while time.time() < t_end:
            now_wall = time.time()
            if now_wall - t_report >= 1.0:
                stats.cpu_samples.append(_read_cpu_percent(0.05))
                stats.rss_mb_samples.append(_read_proc_rss_mb())
                t_report = now_wall
            if now_wall - last_heartbeat >= args.heartbeat:
                last_heartbeat = now_wall
                print(
                    f"[心跳] events={sum(stats.event_counts.values())} "
                    f"fatigue={stats.event_counts.get('user_fatigue_updated', 0)} "
                    f"emotion={stats.event_counts.get('user_emotion_updated', 0)} "
                    f"behavior={stats.event_counts.get('user_attention_updated', 0)} "
                    f"RSS={stats.rss_mb_samples[-1] if stats.rss_mb_samples else 0:.0f}MB",
                    flush=True,
                )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("用户中断", flush=True)
    finally:
        if behavior_adapter is not None:
            behavior_adapter.stop()
        if vision_adapter is not None:
            vision_adapter.stop()

    _print_coverage(stats, emotion_enabled=emotion_enabled, pose_from_behavior=True)
    _print_performance(stats, args.duration)
    required = {"user_fatigue_updated", "user_attention_updated", "user_text_input", "speech_recognized"}
    if emotion_enabled:
        required.add("user_emotion_updated")
    required |= {"user_posture_updated", "user_activity_updated"}
    missing = [e for e in required if stats.event_counts.get(e, 0) == 0]
    return 1 if missing else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="全栈感知 + Event 覆盖 + 性能测试")
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--emotion-backend",
        default="wujie-om",
        help="默认 wujie-om（Ascend NPU）；无 .om 时可 none",
    )
    parser.add_argument("--behavior-backend", default="auto", choices=("auto", "om", "pt"))
    parser.add_argument("--behavior-interval", type=float, default=None)
    parser.add_argument("--behavior-debounce", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--vision-fps", type=float, default=None, help="默认与 perception tick 对齐（4 Hz）")
    parser.add_argument("--emotion-every-n", type=int, default=None, help="默认 tick 下每 4 帧 ≈ 1 次/秒")
    parser.add_argument("--perception-hz", type=float, default=None, help="统一感知 tick，默认 4")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--behavior-om-device-id", type=int, default=0)
    parser.add_argument("--wujie-om", default=None, help="WuJie 情绪 OM 路径")
    parser.add_argument("--wujie-device-id", type=int, default=0)
    parser.add_argument("--heartbeat", type=float, default=8.0)
    parser.add_argument("--deepface-model", default="VGG-Face")
    args = parser.parse_args()
    raise SystemExit(run_integration(args))


if __name__ == "__main__":
    main()
