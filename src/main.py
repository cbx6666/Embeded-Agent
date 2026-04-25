from __future__ import annotations

import argparse
import os

from src.adapters.cli_input import CLIInputAdapter, HELP_TEXT, parse_cli_event
from src.adapters.console_output import ConsoleOutput
from src.adapters.mock_input import parse_mock_command
from src.agent.core import build_default_core


def main() -> None:
    parser = argparse.ArgumentParser(description="Embeded-Agent CLI MVP")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="启用 MediaPipe 摄像头管线（疲劳 EAR+MAR；情绪默认 DeepFace，可选 --emotion-backend raf）",
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备索引")
    parser.add_argument(
        "--emotion-backend",
        type=str,
        default="deepface",
        help="情绪后端：deepface（默认，需安装 deepface）| raf | none；可用环境变量 EMBED_EMOTION_BACKEND",
    )
    parser.add_argument(
        "--deepface-model",
        type=str,
        default="VGG-Face",
        help="保留字段：DeepFace 当前情绪模型由库内置，该参数备后续扩展。",
    )
    parser.add_argument(
        "--raf-ckpt",
        type=str,
        default=None,
        help="仅当 --emotion-backend raf 时：RAF-ResNet18 权重路径",
    )
    args = parser.parse_args()
    raf_path = args.raf_ckpt or os.environ.get("RAF_RESNET18_CKPT")
    emotion_be = (os.environ.get("EMBED_EMOTION_BACKEND") or args.emotion_backend or "deepface").strip()

    output = ConsoleOutput()
    cli = CLIInputAdapter()
    core = build_default_core(output=output)

    vision_adapter = None
    if args.vision:
        from src.adapters.vision_affect import (
            VisionAffectConfig,
            VisionAffectInputAdapter,
            vision_dependencies_met,
            vision_emotion_backend_ready,
        )

        if vision_dependencies_met():
            cfg = VisionAffectConfig(
                camera_index=args.camera,
                raf_checkpoint=raf_path,
                emotion_backend=emotion_be,
                deepface_model=args.deepface_model,
            )
            vision_adapter = VisionAffectInputAdapter(core, cfg)
            vision_adapter.start_background()
            em_ok = vision_emotion_backend_ready(cfg)
            raf_h = f" 情绪：RAF+权重。" if (emotion_be.lower() in {"raf", "raf-db"} and raf_path) else ""
            df_h = f" 情绪：DeepFace。" if emotion_be.lower() == "deepface" and em_ok else ""
            none_h = f" 情绪已关闭。" if emotion_be.lower() in {"none", "off", "disabled"} else ""
            output.show_text(
                "已启动视觉适配器（adapters/vision_affect，内核只收标准 Event；疲劳 EAR+MAR+融合）。"
                + (df_h or raf_h or none_h)
                + (
                    " 未安装 deepface 或无法导入，仅疲劳/几何事件上报。"
                    if (emotion_be.lower() == "deepface" and not em_ok)
                    else ""
                )
                + (
                    " RAF 需有效 --raf-ckpt 与 PyTorch，否则无情绪事件。"
                    if (emotion_be.lower() in {"raf", "raf-db"} and not em_ok)
                    else ""
                )
            )
        else:
            output.show_text(
                "无法启动视觉适配器：请安装 opencv-python-headless、mediapipe（见 requirements.txt）。"
            )

    output.show_text("Agent MVP 已启动，输入 /help 查看可用命令。")
    try:
        while True:
            line = cli.readline()
            if line is None:
                break
            command = line.strip()
            if not command:
                continue

            if command == "/exit":
                break
            if command == "/help":
                output.show_text(HELP_TEXT.rstrip())
                continue
            if command == "/state":
                output.show_text(core.render_state())
                continue
            if command == "/history":
                output.show_text(core.render_history())
                continue

            try:
                mock_event = parse_mock_command(command)
            except ValueError as exc:
                output.show_text(f"[Error] {exc}")
                continue

            if mock_event is not None:
                core.handle_event(mock_event)
                continue

            core.handle_event(parse_cli_event(command))
    finally:
        if vision_adapter is not None:
            vision_adapter.stop()
        core.shutdown()
        output.show_text("Agent MVP 已退出。")


if __name__ == "__main__":
    main()
