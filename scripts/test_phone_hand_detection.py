#!/usr/bin/env python3
"""YOLO26 手机框 + 手腕邻近 — 摄像头联调。

准备:
  pip install -r requirements-behavior.txt
  python scripts/download_yolo26_models.py

运行:
  python scripts/test_phone_hand_detection.py --camera 0
  # SSH/板子无桌面（headless OpenCV）会自动 --no-gui；也可显式:
  python scripts/test_phone_hand_detection.py --camera 0 --no-gui
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.adapters.behavior.camera_utils import grab_latest_frame, open_camera, warmup_camera
from src.adapters.behavior.phone_hand_detector import (
    PhoneHandProximityDetector,
    dependencies_met,
)


def opencv_gui_available() -> bool:
    """headless 版 OpenCV（无 GTK）不支持 imshow。"""
    try:
        info = cv2.getBuildInformation()
        if "GUI:           NONE" in info or "GUI:\n    NONE" in info:
            return False
    except Exception:
        pass
    if sys.platform != "win32" and not os.environ.get("DISPLAY"):
        return False
    try:
        cv2.namedWindow("__gui_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__gui_probe__")
        return True
    except cv2.error:
        return False


def _safe_destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


class PrintSink:
    def handle_event(self, event) -> None:
        p = event.payload
        if event.type == "user_attention_updated":
            print(
                f"[行为] attention={p.get('attention')} behavior={p.get('behavior')} "
                f"conf={p.get('confidence')}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26 手机+手腕邻近检测联调")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="PyTorch 后端设备 (cpu/npu:0)")
    parser.add_argument(
        "--backend",
        choices=("auto", "pt", "om"),
        default="auto",
        help="auto=有 .om 则用 Ascend，否则 ultralytics .pt",
    )
    parser.add_argument("--om-device-id", type=int, default=0)
    parser.add_argument("--phone-conf", type=float, default=0.25)
    parser.add_argument(
        "--distance-ratio",
        type=float,
        default=None,
        help="手腕与手机框邻近阈值系数，默认用检测器内置 1.0",
    )
    parser.add_argument(
        "--phone-solo-conf",
        type=float,
        default=None,
        help="仅检出手机、上半区且置信≥该值也判手持，默认 0.45",
    )
    parser.add_argument(
        "--absent-grace-frames",
        type=int,
        default=None,
        help="连续无人帧宽限，≤该值时仅手机仍算分心，默认 10",
    )
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument(
        "--no-head-down-fusion",
        action="store_true",
        help="关闭 Face Mesh 低头+手腕辅助（仅 YOLO 手机框）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="YOLO 输入边长，板子 CPU 建议 320（默认）或 416",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=3.0,
        help="无 GUI 时每隔多少秒打印一次心跳（0=关闭，仅状态变化时打印）",
    )
    parser.add_argument(
        "--infer-interval",
        type=float,
        default=0.25,
        help="两次 YOLO 推理的最小间隔(秒)，默认 0.25≈每秒 4 次；板子慢可改 0.5~1.0",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="在达到 infer-interval 的帧上，每 N 帧才推理一次（默认 1=不跳帧）",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="拍一帧并打印人体/手机/全类别检测后退出",
    )
    parser.add_argument(
        "--save-frame",
        type=Path,
        default=None,
        help="保存首帧到 jpg，确认摄像头画面",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="无窗口，仅控制台输出（SSH/板子默认推荐）",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="强制尝试 OpenCV 窗口（需非 headless 的 opencv 与 DISPLAY）",
    )
    parser.add_argument("--publish-events", action="store_true", help="经 BehaviorAdapter 上报")
    args = parser.parse_args()

    use_gui = bool(args.gui) or (not args.no_gui and opencv_gui_available())
    if not args.gui and not args.no_gui and not use_gui:
        print(
            "当前 OpenCV 无 GUI（多为 opencv-python-headless），已自动使用 --no-gui 模式。\n"
            "  仅打印检测结果；需要窗口请加 --gui 并安装带 GTK 的 opencv-python。",
            flush=True,
        )

    if not dependencies_met():
        print("缺少依赖: pip install -r requirements-behavior.txt")
        sys.exit(1)

    print("加载 YOLO26 模型（首次会自动从官方 GitHub 下载）...")
    det_kw: dict = {
        "device": args.device,
        "phone_conf": args.phone_conf,
        "hold_seconds": args.hold_seconds,
        "imgsz": args.imgsz,
        "inference_backend": args.backend,
        "om_device_id": args.om_device_id,
    }
    if args.distance_ratio is not None:
        det_kw["distance_ratio"] = args.distance_ratio
    if args.phone_solo_conf is not None:
        det_kw["phone_solo_conf"] = args.phone_solo_conf
    if args.absent_grace_frames is not None:
        det_kw["absent_grace_frames"] = args.absent_grace_frames
    if args.no_head_down_fusion:
        det_kw["enable_head_down_fusion"] = False
    detector = PhoneHandProximityDetector(**det_kw)
    print("正在加载并预热模型（首帧会较慢）...", flush=True)
    detector.load_models()
    print(f"模型就绪。推理后端={detector.active_backend}", flush=True)

    adapter = None
    if args.publish_events:
        from src.adapters.behavior.phone_camera_adapter import PhoneHandCameraAdapter

        sink = PrintSink()
        adapter = PhoneHandCameraAdapter(
            sink,
            camera_index=args.camera,
            detector=detector,
            inference_interval=0.2,
        )
        adapter.start_background()
        print("事件上报已启动。Ctrl+C 退出。")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            adapter.stop()
        return

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}")
        sys.exit(1)

    warmup_camera(cap, frames=8)

    if args.diagnose or args.save_frame:
        ok, frame = grab_latest_frame(cap, flush=4)
        cap.release()
        if not ok:
            print("无法从摄像头读取画面")
            sys.exit(1)
        if args.save_frame:
            path = args.save_frame.expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)
            print(f"已保存画面: {path} ({frame.shape[1]}x{frame.shape[0]})")
        if args.diagnose:
            info = detector.diagnose_frame(frame)
            print("--- 单帧诊断 ---")
            print(f"  推理后端: {info.get('backend', '?')}")
            print(
                f"  人(pose): {info.get('person_count_pose', '?')} "
                f"phase={info.get('presence_phase', '?')} "
                f"absent_frames={info.get('absent_frames', '?')}"
            )
            print(
                f"  手腕关键点(置信≥阈值,非举手): {info['wrist_count']}  "
                f"（人在画内时姿态模型通常给出左右腕 0~2 个）"
            )
            print(f"  手机(cell phone, COCO-67): {info['phone_count']}")
            if info["all_detections"]:
                print(f"  本帧其它检测( conf≥{args.phone_conf} ): {', '.join(info['all_detections'])}")
            else:
                print(f"  本帧无任何目标 conf≥{args.phone_conf}")
            print("---")
            if info["person_count"] == 0:
                print("建议: 调整摄像头对准头肩；确认不是反了/盖了镜头。")
            if info["phone_count"] == 0:
                print(
                    "建议: 手机举到胸前、屏幕朝镜头、占画面约 1/10 以上；"
                    "或降低阈值: --phone-conf 0.2 。"
                    "COCO 预训练对「手持小手机」召回率偏低，属常见情况。"
                )
        return

    if use_gui:
        print("对着摄像头举起手机。绿框=手机，[PHONE]=手持。Esc 或 Ctrl+C 退出。", flush=True)
    else:
        hb = args.heartbeat
        if hb > 0:
            print(
                f"对着摄像头举起手机。[PHONE]=手持。"
                f" 推理约每 {args.infer_interval:.2f}s 一次；"
                f" 日志在状态变化时打印，另每 {hb:.0f}s 心跳。Ctrl+C 退出。",
                flush=True,
            )
        else:
            print("对着摄像头举起手机。[PHONE]=手持，仅状态变化时打印。Ctrl+C 退出。", flush=True)
    last_label: str | None = None
    frame_idx = 0
    last_heartbeat = time.time()
    last_infer_ts = 0.0
    print(
        f"推理节流: 至少间隔 {args.infer_interval:.2f}s（OM 每次含 detect+pose 共 2 路）",
        flush=True,
    )
    try:
        while True:
            ok, frame = grab_latest_frame(cap, flush=3)
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            frame_idx += 1
            now_loop = time.time()
            if now_loop - last_infer_ts < args.infer_interval:
                time.sleep(0.02)
                continue
            if args.infer_every > 1 and (frame_idx % args.infer_every) != 0:
                continue
            last_infer_ts = now_loop

            t0 = time.perf_counter()
            result = detector.analyze_frame_stable(frame)
            ms = (time.perf_counter() - t0) * 1000.0
            # phone_in_hand 需连续 hold_seconds；raw 为本帧是否检出手机
            if result.presence_phase == "left":
                label = "AWAY"
            else:
                label = "PHONE" if result.phone_in_hand else "FOCUS"
            pending = (
                result.raw_phone_count > 0
                and not result.phone_in_hand
                and result.wrist_near_phone
            )

            if not use_gui:
                now = time.time()
                state_changed = label != last_label
                heartbeat_due = args.heartbeat > 0 and (now - last_heartbeat) >= args.heartbeat
                if state_changed or heartbeat_due:
                    tag = "状态" if state_changed else "心跳"
                    pending_note = " (已检出手机+手腕，等待稳定…)" if pending else ""
                    print(
                        f"[{label}][{tag}] phones={result.raw_phone_count} "
                        f"pose={result.person_count_pose} "
                        f"phase={result.presence_phase} absent={result.absent_frames} "
                        f"wrist_kpts={result.wrist_count} wrist_near={result.wrist_near_phone} "
                        f"phones={result.raw_phone_count} look_down={result.looking_down} "
                        f"head_assist={result.head_down_assist} conf={result.confidence:.2f} "
                        f"infer={ms:.0f}ms{pending_note}",
                        flush=True,
                    )
                    last_label = label
                    last_heartbeat = now
            else:
                vis = frame.copy()
                for p in result.phones:
                    color = (0, 255, 0) if result.wrist_near_phone else (0, 180, 255)
                    cv2.rectangle(
                        vis,
                        (int(p.x1), int(p.y1)),
                        (int(p.x2), int(p.y2)),
                        color,
                        2,
                    )
                cv2.putText(
                    vis,
                    f"{label} phones={result.raw_phone_count} {ms:.0f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255) if result.phone_in_hand else (0, 255, 0),
                    2,
                )
                cv2.imshow("phone_hand_yolo26", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if use_gui:
            _safe_destroy_windows()


if __name__ == "__main__":
    main()
