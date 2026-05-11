from __future__ import annotations

import argparse
import os

from src.adapters.cli_input import CLIInputAdapter, HELP_TEXT, parse_cli_event
from src.adapters.console_output import ConsoleOutput
from src.adapters.mock_input import parse_mock_command
from src.adapters.profile_cli import handle_profile_command
from src.agent.core import build_default_core


def main() -> None:
    parser = argparse.ArgumentParser(description="Embeded-Agent CLI MVP")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="启用 MediaPipe 摄像头管线（疲劳 EAR+MAR；情绪默认 WuJie-OM/NPU）",
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备索引")
    parser.add_argument(
        "--emotion-backend",
        type=str,
        default="wujie-om",
        help="情绪后端：wujie-om（默认）| wujie-vgg19 | raf | deepface | none；可用环境变量 EMBED_EMOTION_BACKEND",
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
    parser.add_argument(
        "--wujie-ckpt",
        type=str,
        default=None,
        help="仅当 --emotion-backend wujie-vgg19 时：WuJie1010 的 PrivateTest_model.t7 路径",
    )
    parser.add_argument(
        "--wujie-om",
        type=str,
        default=None,
        help="仅当 --emotion-backend wujie-om 时：WuJie OM 模型路径（*.om）",
    )
    parser.add_argument(
        "--wujie-device-id",
        type=int,
        default=0,
        help="仅当 --emotion-backend wujie-om 时：Ascend 设备ID",
    )
    parser.add_argument(
        "--state-stats-db",
        type=str,
        default=None,
        help="视觉状态统计 SQLite 路径（每秒明细 + 日/周汇总）",
    )
    args = parser.parse_args()
    raf_path = args.raf_ckpt or os.environ.get("RAF_RESNET18_CKPT")
    wujie_path = args.wujie_ckpt or os.environ.get("WUJIE_VGG19_CKPT")
    default_om = "external/fer_wujie1010/FER2013_VGG19/wujie_vgg19_static.om"
    wujie_om_path = args.wujie_om or os.environ.get("WUJIE_OM_MODEL") or default_om
    wujie_device_id = int(os.environ.get("WUJIE_OM_DEVICE_ID", str(args.wujie_device_id)))
    state_stats_db = args.state_stats_db or os.environ.get("EMBED_STATE_STATS_DB")
    emotion_be = (os.environ.get("EMBED_EMOTION_BACKEND") or args.emotion_backend or "wujie-om").strip()

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
                wujie_checkpoint=wujie_path,
                wujie_om_model=wujie_om_path,
                wujie_om_device_id=wujie_device_id,
                emotion_backend=emotion_be,
                deepface_model=args.deepface_model,
                state_stats_db_path=state_stats_db or "data/state_stats.db",
            )
            vision_adapter = VisionAffectInputAdapter(core, cfg)
            vision_adapter.start_background()
            em_ok = vision_emotion_backend_ready(cfg)
            raf_h = f" 情绪：RAF+权重。" if (emotion_be.lower() in {"raf", "raf-db"} and raf_path) else ""
            wj_h = (
                f" 情绪：WuJie VGG19+权重。"
                if (emotion_be.lower() in {"wujie-vgg19", "wujie", "fer-vgg19"} and wujie_path)
                else ""
            )
            om_h = (
                f" 情绪：WuJie OM+NPU。"
                if (emotion_be.lower() in {"wujie-om", "om", "wujie_om"} and wujie_om_path)
                else ""
            )
            df_h = f" 情绪：DeepFace。" if emotion_be.lower() == "deepface" and em_ok else ""
            none_h = f" 情绪已关闭。" if emotion_be.lower() in {"none", "off", "disabled"} else ""
            output.show_text(
                "已启动视觉适配器（adapters/vision_affect，内核只收标准 Event；疲劳 EAR+MAR+融合）。"
                + (df_h or raf_h or om_h or wj_h or none_h)
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
                + (
                    " WuJie 需有效 --wujie-ckpt 与 PyTorch/OpenCV，否则无情绪事件。"
                    if (emotion_be.lower() in {"wujie-vgg19", "wujie", "fer-vgg19"} and not em_ok)
                    else ""
                )
                + (
                    " WuJie-OM 需有效 --wujie-om 且 ACL 运行时可用（先 source Ascend set_env.sh），否则无情绪事件。"
                    if (emotion_be.lower() in {"wujie-om", "om", "wujie_om"} and not em_ok)
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
            if handle_profile_command(core, output, command):
                continue

            try:
                mock_event = parse_mock_command(command)
            except ValueError as exc:
                output.show_text(f"[Error] {exc}")
                continue

            if mock_event is not None:
                core.handle_event_with_results(mock_event)
                continue

            core.handle_event_with_results(parse_cli_event(command))
    finally:
        if vision_adapter is not None:
            vision_adapter.stop()
        core.shutdown()
        output.show_text("Agent MVP 已退出。")


if __name__ == "__main__":
    main()
