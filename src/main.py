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
        help="启用 MediaPipe 摄像头管线（疲劳 EAR/PERCLOS；情绪需 --raf-ckpt 与 PyTorch）",
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备索引")
    parser.add_argument(
        "--raf-ckpt",
        type=str,
        default=None,
        help="可选：RAF-DB ResNet18 权重文件；不设则仅上报疲劳、不上报情绪",
    )
    args = parser.parse_args()
    raf_path = args.raf_ckpt or os.environ.get("RAF_RESNET18_CKPT")

    output = ConsoleOutput()
    cli = CLIInputAdapter()
    core = build_default_core(output=output)

    vision_adapter = None
    if args.vision:
        from src.adapters.vision_affect import (
            VisionAffectConfig,
            VisionAffectInputAdapter,
            vision_dependencies_met,
        )

        if vision_dependencies_met():
            cfg = VisionAffectConfig(camera_index=args.camera, raf_checkpoint=raf_path)
            vision_adapter = VisionAffectInputAdapter(core, cfg)
            vision_adapter.start_background()
            output.show_text(
                "已启动视觉适配器（检测逻辑在 adapters/vision_affect，内核仅收 Event）。"
                + (" 已配置 RAF 权重。" if raf_path else " 未配置 RAF 权重，仅疲劳事件。")
            )
        else:
            output.show_text(
                "无法启动视觉适配器：请安装 opencv-python-headless、mediapipe "
                "（见 requirements-vision.txt）。"
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
