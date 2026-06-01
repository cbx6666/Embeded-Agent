from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.adapters.cli_input import CLIInputAdapter, HELP_TEXT, parse_cli_event
from src.adapters.console_output import ConsoleOutput
from src.adapters.mock_input import parse_mock_command
from src.adapters.profile_cli import handle_profile_command
from src.agent.core import build_default_core


class MultiOutput:
    """将动作同时发往控制台与可选语音输出。"""

    def __init__(self, console: ConsoleOutput, voice_adapter=None) -> None:
        self.console = console
        self.voice_adapter = voice_adapter

    def execute(self, action) -> None:
        self.console.execute(action)
        if self.voice_adapter is not None:
            self.voice_adapter.execute(action)

    def show_text(self, text: str) -> None:
        self.console.show_text(text)


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
    parser.add_argument(
        "--screen",
        action="store_true",
        help="启用 pygame 桌宠显示窗口",
    )
    parser.add_argument("--voice", action="store_true", help="启用百度 ASR/TTS 板级语音模式")
    parser.add_argument("--voice-once", action="store_true", help="启动后立即执行一次语音识别")
    parser.add_argument("--voice-loop", action="store_true", help="后台持续执行语音识别并注入 Agent")
    parser.add_argument("--voice-debug", action="store_true", help="打印每次语音识别结果与错误，便于联调")
    parser.add_argument("--voice-loop-interval", type=float, default=1.0, help="两次语音识别之间的间隔秒数")
    parser.add_argument("--voice-duration", type=int, default=10, help="单次语音录音时长，单位秒")
    parser.add_argument("--voice-alsa-device", type=str, default="plughw:0,0", help="ALSA 录音设备")
    parser.add_argument("--voice-sample-rate", type=int, default=16000, help="语音录音采样率")
    parser.add_argument("--tts-output-path", type=str, default="data/tts_output.wav", help="TTS 输出音频路径")
    parser.add_argument("--wake-backend", type=str, default="energy", help="唤醒词检测引擎：energy（默认）| porcupine | mock")
    parser.add_argument("--wake-model-path", type=str, default="models/porcupine_params_zh.pv", help="Porcupine 模型路径")
    parser.add_argument("--wake-keyword-path", type=str, default="models/hey_assistant_zh.ppn", help="Porcupine 唤醒词路径")
    parser.add_argument("--wake-word", type=str, default="小助", help="能量检测模式下的唤醒关键词（用于日志标识）")
    parser.add_argument("--wake-sensitivities", type=str, default="0.5", help="Porcupine 灵敏度，逗号分隔")
    parser.add_argument("--wake-energy-threshold", type=float, default=0.2, help="能量检测阈值，越小越灵敏")
    parser.add_argument("--no-wake", action="store_true", help="禁用唤醒词检测，仅使用 --voice-loop 持续识别模式")
    parser.add_argument("--pose", action="store_true", help="启动时自动开启 YOLO 姿势检测")
    parser.add_argument("--pose-model", type=str, default="yolov8n-pose.pt", help="YOLO 姿势模型路径")
    parser.add_argument("--pose-device", type=str, default="cpu", help="姿势检测推理设备：cpu | cuda | npu")
    parser.add_argument("--pose-interval", type=float, default=5.0, help="姿势检测间隔（秒）")
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
    voice_adapter = None
    detector = None

    screen_adapter = None
    if args.screen:
        from src.adapters.screen import ScreenDisplayAdapter, ScreenWindow

        screen_window = ScreenWindow()
        screen_window.start()
        screen_adapter = ScreenDisplayAdapter(
            hardware=screen_window,
            console_output=output
        )
        output = screen_adapter

    if args.voice:
        from src.adapters.voice import (
            BaiduShortASRBackend,
            BaiduTTSBackend,
            BoardVoiceAdapter,
            build_wake_word_detector,
        )

        recognizer = BaiduShortASRBackend(sample_rate=args.voice_sample_rate)
        tts_backend = BaiduTTSBackend(output_path=args.tts_output_path)

        if not args.no_wake:
            try:
                sensitivities = [float(s) for s in args.wake_sensitivities.split(",")]
            except ValueError:
                sensitivities = [0.5]

            if args.wake_backend == "porcupine":
                detector = build_wake_word_detector(
                    backend="porcupine",
                    model_path=args.wake_model_path,
                    keyword_path=args.wake_keyword_path,
                    sensitivities=sensitivities,
                    alsa_device=args.voice_alsa_device,
                    sink=None,
                )
            elif args.wake_backend == "mock":
                detector = build_wake_word_detector(backend="mock", sink=None)
            else:
                detector = build_wake_word_detector(
                    backend="energy",
                    wake_word=args.wake_word,
                    alsa_device=args.voice_alsa_device,
                    sink=None,
                    energy_threshold=args.wake_energy_threshold,
                )

        voice_adapter = BoardVoiceAdapter(
            sink=None,
            detector=detector,
            recognizer=recognizer,
            tts_backend=tts_backend,
            alsa_device=args.voice_alsa_device,
            sample_rate=args.voice_sample_rate,
            capture_duration_sec=args.voice_duration,
        )
        voice_adapter.debug = bool(args.voice_debug)
        output = MultiOutput(output, voice_adapter)

    core = build_default_core(output=output)

    if voice_adapter is not None:
        voice_adapter._sink = core
        if detector is not None:
            detector._sink = core
        if args.voice_loop:
            voice_adapter.start_background_loop(interval_sec=args.voice_loop_interval)
            output.show_text(
                f"已启动语音后台识别：duration={args.voice_duration}s interval={args.voice_loop_interval}s"
            )
        elif detector is not None:
            voice_adapter.start()
            output.show_text("已启动唤醒词监听模式，请说出唤醒词。")
        else:
            output.show_text("语音适配器已就绪（静默模式），使用 /voice_once 命令手动触发。")

        import platform
        output.show_text(f"[诊断] 操作系统：{platform.system()}，Python 版本：{platform.python_version()}")
        asr_configured = recognizer.is_configured()
        tts_configured = tts_backend.is_configured()
        output.show_text(
            f"[诊断] ASR 配置：{'已配置' if asr_configured else '未配置（请设置 BAIDU_ASR_API_KEY / BAIDU_ASR_SECRET_KEY）'}"
        )
        output.show_text(
            f"[诊断] TTS 配置：{'已配置' if tts_configured else '未配置（请设置 BAIDU_TTS_API_KEY / BAIDU_TTS_SECRET_KEY）'}"
        )

        if args.voice_once:
            output.show_text("立即执行一次语音识别，请在录音窗口内说话。")
            event = voice_adapter.run_recognize_once()
            if event is None:
                output.show_text("[Voice] 未识别到有效文本。")
            else:
                core.handle_event_with_results(event)
                output.show_text(f"[Voice] 已上报事件：{event.payload}")
    
    # 设置事件处理回调，自动更新屏幕
    if screen_adapter is not None:
        def on_event_handled(state):
            if state.focus.active:
                screen_adapter.update_focus_timer(
                    state.focus.remaining_sec,
                    state.focus.target_duration_sec
                )
        core.set_event_handled_callback(on_event_handled)

    pose_adapter = None
    pose_detection_active = False
    if args.pose:
        from src.adapters.pose import PoseDetectionAdapter, YOLOPoseDetector

        pose_detector = YOLOPoseDetector(
            model_path=args.pose_model,
            device=args.pose_device,
        )
        pose_adapter = PoseDetectionAdapter(
            detector=pose_detector,
            event_callback=lambda event: core.handle_event(event),
            detection_interval=args.pose_interval,
        )
        pose_adapter.start()
        pose_detection_active = True
        output.show_text(
            f"已启动姿势检测（model={args.pose_model}, device={args.pose_device}, interval={args.pose_interval}s）。"
        )

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
                state_stats_db_path=state_stats_db or "data/runtime/state_stats.db",
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
    if voice_adapter is not None:
        output.show_text("语音命令：/voice_once")
    output.show_text("姿势命令：/pose start | /pose stop（可用 --pose 启动时自动开启）")
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
            if args.voice and voice_adapter is not None and command == "/voice_once":
                output.show_text("开始一次语音识别，请在录音窗口内说话。")
                event = voice_adapter.run_recognize_once()
                if event is None:
                    output.show_text("[Voice] 未识别到有效文本。")
                else:
                    core.handle_event_with_results(event)
                    output.show_text(f"[Voice] 已上报事件：{event.payload}")
                continue
            if command == "/pose start":
                if pose_adapter is None:
                    from src.adapters.pose import PoseDetectionAdapter, YOLOPoseDetector

                    pose_detector = YOLOPoseDetector(
                        model_path=args.pose_model,
                        device=args.pose_device,
                    )
                    pose_adapter = PoseDetectionAdapter(
                        detector=pose_detector,
                        event_callback=lambda event: core.handle_event(event),
                        detection_interval=args.pose_interval,
                    )
                if not pose_detection_active:
                    pose_adapter.start()
                    pose_detection_active = True
                    output.show_text("姿势检测已启动")
                else:
                    output.show_text("姿势检测已经在运行中")
                continue
            if command == "/pose stop":
                if pose_adapter is None or not pose_detection_active:
                    output.show_text("姿势检测未在运行")
                    continue
                pose_adapter.stop()
                pose_detection_active = False
                output.show_text("姿势检测已停止")
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
        if voice_adapter is not None:
            voice_adapter.stop_background_loop()
            voice_adapter.stop()
        if pose_detection_active and pose_adapter is not None:
            pose_adapter.stop()
        if vision_adapter is not None:
            vision_adapter.stop()
        if screen_adapter is not None:
            screen_adapter._hardware.stop()
        core.shutdown()
        output.show_text("Agent MVP 已退出。")


if __name__ == "__main__":
    main()
