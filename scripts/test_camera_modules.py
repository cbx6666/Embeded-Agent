#!/usr/bin/env python3
"""仅用摄像头 + 桌宠窗口做联调（不依赖语音/环境传感器）。

用法（在项目根目录）：
  # 1) 只看视觉事件（疲劳 + 表情），控制台打印，不走 LLM
  python scripts/test_camera_modules.py --vision-only

  # 2) 只测桌宠窗口状态切换
  python scripts/test_camera_modules.py --screen-only

  # 3) 视觉 + 桌宠：视觉事件驱动控制台，桌宠随 /mock 或手动改状态
  python scripts/test_camera_modules.py --vision --screen

  # 笔记本无 NPU 时推荐 DeepFace 表情（首次会较慢）
  python scripts/test_camera_modules.py --vision-only --emotion-backend deepface

  # 只测疲劳（不跑表情模型）
  python scripts/test_camera_modules.py --vision-only --emotion-backend none
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.event import Event

_SHARED_ENV_PREFIX = "/opt/ai-envs/shared"
_SHARED_PYTHON = Path("/opt/ai-envs/shared/bin/python")


def _vision_import_ok(python_exe: str | Path) -> bool:
    proc = subprocess.run(
        [str(python_exe), "-c", "import cv2, mediapipe"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _using_shared_site_packages() -> bool:
    """共享环境的 python 常 symlink 到系统解释器，用 prefix 判断更可靠。"""
    return _SHARED_ENV_PREFIX in str(Path(sys.prefix).resolve())


def _maybe_reexec_with_shared_python() -> None:
    """若当前是项目 .venv 且缺视觉依赖，自动改用 /opt/ai-envs/shared/bin/python。"""
    if os.environ.get("EMBED_NO_SHARED_REEXEC") == "1":
        return
    if os.environ.get("EMBED_SHARED_REEXEC_DONE") == "1":
        return
    if not _SHARED_PYTHON.is_file():
        return
    if _using_shared_site_packages():
        return
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401

        return
    except ImportError:
        pass
    if not _vision_import_ok(_SHARED_PYTHON):
        return
    print(
        f"检测到当前 Python（{sys.executable}）无法加载 cv2/mediapipe，\n"
        f"将改用共享环境：{_SHARED_PYTHON}\n"
        f"（若不想自动切换，可设置 EMBED_NO_SHARED_REEXEC=1）",
        flush=True,
    )
    os.environ["EMBED_SHARED_REEXEC_DONE"] = "1"
    os.execv(str(_SHARED_PYTHON), [str(_SHARED_PYTHON), *sys.argv])


def _warn_python_env(emotion_backend: str) -> None:
    """提醒使用共享环境，避免 .venv 缺包。"""
    exe = Path(sys.executable).resolve()
    print(f"当前 Python: {exe}", flush=True)
    if _SHARED_ENV_PREFIX not in str(exe):
        print(
            "提示：未使用共享环境。视觉测试请优先执行：\n"
            "  deactivate   # 若提示无此命令可忽略\n"
            "  source /opt/ai-envs/shared/bin/activate\n"
            "或在当前环境安装： pip install -r requirements.txt",
            flush=True,
        )
    if emotion_backend.strip().lower() == "deepface":
        try:
            import tf_keras  # noqa: F401
        except ImportError:
            print(
                "错误：deepface 需要 tf-keras。请执行：\n"
                "  pip install tf-keras\n"
                "或改用共享环境： source /opt/ai-envs/shared/bin/activate",
                flush=True,
            )
            sys.exit(1)


class PrintEventSink:
    """将视觉事件打印到控制台，不调用 LLM。"""

    def handle_event(self, event: Event) -> None:
        p = event.payload
        if event.type == "user_fatigue_updated":
            score = p.get("confidence", p.get("perclos", "-"))
            print(
                f"[疲劳] level={p.get('fatigue_level')} "
                f"perclos={p.get('perclos', '-')} score={score} source={p.get('source')}",
                flush=True,
            )
        elif event.type == "user_emotion_updated":
            print(
                f"[表情] emotion={p.get('emotion')} conf={p.get('confidence')} source={p.get('source')}",
                flush=True,
            )
        else:
            print(f"[Event] {event.type} {p}", flush=True)


def run_vision_only(args: argparse.Namespace) -> None:
    from src.adapters.vision_affect import (
        VisionAffectConfig,
        VisionAffectInputAdapter,
        vision_dependencies_met,
        vision_emotion_backend_ready,
    )

    if not vision_dependencies_met():
        print("缺少依赖：pip install opencv-python-headless mediapipe")
        if _SHARED_PYTHON.is_file():
            print(f"或直接使用共享 Python：\n  {_SHARED_PYTHON} {' '.join(sys.argv)}")
        sys.exit(1)

    _warn_python_env(args.emotion_backend)

    cfg = VisionAffectConfig(
        camera_index=args.camera,
        emotion_backend=args.emotion_backend,
        deepface_model=args.deepface_model,
        raf_checkpoint=args.raf_ckpt,
        wujie_checkpoint=args.wujie_ckpt,
        wujie_om_model=args.wujie_om,
        wujie_om_device_id=args.wujie_device_id,
        state_stats_db_path=args.state_stats_db or "data/runtime/state_stats.db",
    )
    sink = PrintEventSink()
    adapter = VisionAffectInputAdapter(sink, cfg)
    em_ok = vision_emotion_backend_ready(cfg)
    print(f"摄像头索引: {args.camera}")
    print(f"情绪后端: {args.emotion_backend}（就绪={em_ok}）")
    print("对着摄像头：闭眼/打哈欠测疲劳；换表情测情绪。Ctrl+C 退出。")
    adapter.start_background()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        adapter.stop()
        print("视觉检测已停止。")


def _create_screen_window(args: argparse.Namespace):
    from src.adapters.screen import create_screen_window

    window = create_screen_window(
        fullscreen=getattr(args, "screen_fullscreen", False),
        size_arg=getattr(args, "screen_size", None),
    )
    print(
        f"桌宠窗口：{window.size[0]}x{window.size[1]}"
        + (" 全屏" if window.fullscreen else ""),
        flush=True,
    )
    return window


def run_screen_only(args: argparse.Namespace) -> None:
    from src.adapters.screen import ScreenDisplayAdapter
    from src.agent.action import display, render_pet_expression

    window = _create_screen_window(args)
    window.start()
    adapter = ScreenDisplayAdapter(hardware=window)
    sequence = [
        ("idle", display("空闲", kind="idle")),
        ("listening", render_pet_expression("listening")),
        ("thinking", render_pet_expression("thinking")),
        ("speaking", render_pet_expression("happy")),
        ("focus", display("专注模式", kind="focus_mode")),
    ]
    for name, action in sequence:
        print(f"桌宠状态 -> {name}")
        adapter.execute(action)
        if name == "focus":
            adapter.update_focus_timer(25 * 60, 25 * 60)
        time.sleep(2)
    print("按 Enter 关闭窗口...")
    input()
    window.stop()


def run_vision_and_screen(args: argparse.Namespace) -> None:
    from src.adapters.screen import ScreenDisplayAdapter
    from src.adapters.vision_affect import (
        VisionAffectConfig,
        VisionAffectInputAdapter,
        vision_dependencies_met,
        vision_emotion_backend_ready,
    )
    from src.agent.action import display, render_pet_expression

    if not vision_dependencies_met():
        print("缺少依赖：pip install opencv-python-headless mediapipe pygame")
        if _SHARED_PYTHON.is_file():
            print(f"或直接使用共享 Python：\n  {_SHARED_PYTHON} {' '.join(sys.argv)}")
        sys.exit(1)

    _warn_python_env(args.emotion_backend)

    window = _create_screen_window(args)
    window.start()
    screen = ScreenDisplayAdapter(hardware=window)

    class ScreenSink:
        def handle_event(self, event: Event) -> None:
            p = event.payload
            if event.type == "user_fatigue_updated":
                level = str(p.get("fatigue_level", "none"))
                print(f"[疲劳] {level}", flush=True)
                expr = {"none": "neutral", "mild": "neutral", "moderate": "tired", "high": "tired"}.get(
                    level, "neutral"
                )
                screen.execute(render_pet_expression(expr))
                screen.execute(display(f"疲劳: {level}", kind="status"))
            elif event.type == "user_emotion_updated":
                emo = str(p.get("emotion", "neutral"))
                print(f"[表情] {emo}", flush=True)
                mapped = emo if emo in {"happy", "neutral", "tired", "stressed"} else "neutral"
                screen.execute(render_pet_expression(mapped))

    cfg = VisionAffectConfig(
        camera_index=args.camera,
        emotion_backend=args.emotion_backend,
        deepface_model=args.deepface_model,
        state_stats_db_path=args.state_stats_db or "data/runtime/state_stats.db",
    )
    adapter = VisionAffectInputAdapter(ScreenSink(), cfg)
    print(f"视觉+桌宠已启动。情绪后端={args.emotion_backend} 就绪={vision_emotion_backend_ready(cfg)}")
    print("另开终端可运行 python -m src.main --screen --vision 做完整 Agent 链路（需 DEEPSEEK_API_KEY）。")
    adapter.start_background()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        adapter.stop()
        window.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="摄像头模块联调（疲劳/表情/桌宠）")
    parser.add_argument("--vision-only", action="store_true", help="仅视觉事件打印")
    parser.add_argument("--screen-only", action="store_true", help="仅桌宠窗口演示")
    parser.add_argument("--vision", action="store_true", help="视觉驱动桌宠表情（无 LLM）")
    parser.add_argument("--screen", action="store_true", help="与 --vision 合用")
    parser.add_argument("--screen-fullscreen", action="store_true", help="桌宠全屏（适配 VNC/HDMI 分辨率）")
    parser.add_argument("--screen-size", type=str, default=None, help="桌宠窗口 WxH，非全屏时有效")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--emotion-backend", default="deepface", help="笔记本推荐 deepface 或 none")
    parser.add_argument("--deepface-model", default="VGG-Face")
    parser.add_argument("--raf-ckpt", default=None)
    parser.add_argument("--wujie-ckpt", default=None)
    parser.add_argument("--wujie-om", default=None)
    parser.add_argument("--wujie-device-id", type=int, default=0)
    parser.add_argument("--state-stats-db", default=None)
    args = parser.parse_args()

    if args.screen_only:
        run_screen_only(args)
    elif args.vision_only or (args.vision and not args.screen):
        run_vision_only(args)
    elif args.vision and args.screen:
        run_vision_and_screen(args)
    else:
        parser.print_help()
        print("\n示例: python scripts/test_camera_modules.py --vision-only --emotion-backend deepface")


if __name__ == "__main__":
    _maybe_reexec_with_shared_python()
    main()
