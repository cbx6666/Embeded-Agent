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


def _preload_npu_perception_models(
    args: argparse.Namespace,
    output: ConsoleOutput,
    *,
    wujie_om_path: str | Path,
    wujie_device_id: int,
    emotion_be: str,
) -> object | None:
    """主线程串行加载 NPU OM；须在桌宠、语音、视觉后台线程之前完成。"""
    behavior_detector = None
    if args.behavior:
        from src.adapters.behavior.phone_hand_detector import (
            PhoneHandProximityDetector,
            dependencies_met as behavior_dependencies_met,
        )
        from src.adapters.behavior.yolo_om_runner import om_models_available

        if not behavior_dependencies_met():
            output.show_text(
                "无法启动行为识别：请安装 requirements-behavior.txt（ultralytics、opencv 等）。"
            )
        else:
            behavior_om_device = int(
                os.environ.get("BEHAVIOR_OM_DEVICE_ID", str(args.behavior_om_device_id))
            )
            behavior_backend = args.behavior_backend.strip().lower()
            if behavior_backend == "auto" and not om_models_available():
                output.show_text(
                    "[Warn] 未找到 models/yolo26/*.om，行为识别将回退 PyTorch CPU。"
                    " 请先 bash scripts/export_yolo26_to_om.sh 或设置 BEHAVIOR_DETECT_OM。"
                )
            behavior_detector = PhoneHandProximityDetector(
                inference_backend=behavior_backend,
                om_device_id=behavior_om_device,
                imgsz=320,
            )
            try:
                output.show_text("正在主线程预加载行为 YOLO 模型（避免与语音/视觉 NPU 并发）…")
                sys.stdout.flush()
                behavior_detector.load_models()
                output.show_text(f"行为模型已就绪（后端={behavior_detector.active_backend}）。")
                sys.stdout.flush()
            except Exception as exc:
                behavior_detector = None
                output.show_text(f"行为模型预加载失败，跳过行为识别：{exc}")

    if (
        args.vision
        and emotion_be.lower() in {"wujie-om", "om", "wujie_om"}
        and wujie_om_path
        and Path(wujie_om_path).is_file()
    ):
        try:
            from src.adapters.vision_common.acl_runtime import shared_om_session

            output.show_text("正在主线程预加载情绪 WuJie OM…")
            sys.stdout.flush()
            if not shared_om_session(wujie_om_path, wujie_device_id).load():
                output.show_text("[Warn] WuJie OM 预加载失败，情绪事件可能不可用。")
            sys.stdout.flush()
        except Exception as exc:
            output.show_text(f"[Warn] WuJie OM 预加载异常：{exc}")

    return behavior_detector


def _apply_run_profile(args: argparse.Namespace) -> None:
    """默认全栈（桌宠+视觉+语音）；--llm 恢复仅 CLI Agent 的原有行为。"""
    if args.llm:
        return
    if not args.no_screen:
        args.screen = True
    if not args.no_vision:
        args.vision = True
    if not args.no_voice:
        args.voice = True
    if args.screen and not args.no_screen_fullscreen:
        args.screen_fullscreen = True
    if args.camera is None:
        raw_cam = os.environ.get("EMBED_CAMERA", "auto").strip()
        if raw_cam.lower() in {"", "auto"}:
            from src.adapters.behavior.camera_utils import resolve_camera_index

            args.camera = resolve_camera_index(None)
        else:
            try:
                from src.adapters.behavior.camera_utils import resolve_camera_index

                args.camera = resolve_camera_index(int(raw_cam))
            except ValueError:
                args.camera = 0
    if args.emotion_backend is None:
        args.emotion_backend = (
            os.environ.get("EMBED_EMOTION_BACKEND", "wujie-om").strip() or "wujie-om"
        )
    if args.no_behavior:
        args.behavior = False
    elif args.behavior:
        args.behavior = True
    elif not args.llm and not args.no_vision:
        args.behavior = True
    else:
        args.behavior = False
    if getattr(args, "no_environment", False):
        args.environment = False
    elif getattr(args, "environment", False):
        args.environment = True
    elif not args.llm:
        args.environment = True
    else:
        args.environment = False

    if getattr(args, "no_perception_debug", False):
        args.perception_debug = False
    elif getattr(args, "perception_debug", False):
        args.perception_debug = True
    elif not args.llm and (args.vision or args.behavior):
        args.perception_debug = True
    else:
        args.perception_debug = False

    preview_env = os.environ.get("EMBED_SCREEN_PREVIEW", "").strip().lower()
    if preview_env in {"1", "true", "yes", "on"}:
        args.screen_preview = True
    if getattr(args, "screen_preview", False):
        args.no_screen = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embeded-Agent：默认全栈（桌宠+视觉+语音+LLM）；--llm 仅 CLI Agent",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="仅 CLI Agent + LLM（不自动启桌宠/视觉/语音），等同原先裸 python -m src.main",
    )
    parser.add_argument(
        "--no-screen",
        action="store_true",
        help="全栈模式下不启桌宠（默认会启 --screen）",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="全栈模式下不启视觉（默认会启 --vision）",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="全栈模式下不启语音（默认会启 --voice）",
    )
    parser.add_argument(
        "--no-screen-fullscreen",
        action="store_true",
        help="全栈模式下桌宠窗口化（非全屏）",
    )
    parser.add_argument(
        "--screen-preview",
        action="store_true",
        help="桌宠 HTTP 预览（无需 VNC；在 Cursor 端口面板打开 http://127.0.0.1:8765/）",
    )
    parser.add_argument(
        "--screen-preview-port",
        type=int,
        default=int(os.environ.get("EMBED_SCREEN_PREVIEW_PORT", "8765")),
        help="--screen-preview 本地 HTTP 端口（默认 8765）",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="启用 MediaPipe 摄像头管线（全栈默认已开；--llm 时需显式指定）",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="摄像头 OpenCV 索引；未指定时 auto=扫描 C920/首个可用设备（默认 auto）",
    )
    parser.add_argument(
        "--emotion-backend",
        type=str,
        default=None,
        help="情绪后端：全栈默认 wujie-om（Ascend NPU）；可用 EMBED_EMOTION_BACKEND",
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
        help="启用 pygame 桌宠显示窗口（全栈默认已开；--llm 时需显式指定）",
    )
    parser.add_argument(
        "--screen-fullscreen",
        action="store_true",
        help="桌宠全屏显示（适配当前 DISPLAY 分辨率，如 HDMI/VNC）",
    )
    parser.add_argument(
        "--screen-size",
        type=str,
        default=None,
        help="桌宠窗口尺寸 WxH（非全屏时有效，默认 400x320）；也可用 EMBED_SCREEN_SIZE",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="启用百度 ASR/TTS 板级语音模式（全栈默认已开；--llm 时需显式指定）",
    )
    parser.add_argument("--voice-once", action="store_true", help="启动后立即执行一次语音识别")
    parser.add_argument("--voice-loop", action="store_true", help="后台持续执行语音识别并注入 Agent")
    parser.add_argument("--voice-debug", action="store_true", help="打印每次语音识别详细步骤，并写入 voice_debug 目录")
    parser.add_argument(
        "--voice-debug-dir",
        type=str,
        default="data/voice_debug",
        help="语音调试日志目录（仅保留 latest/voice.log，新唤醒覆盖上一次）",
    )
    parser.add_argument(
        "--no-voice-debug-log",
        action="store_true",
        help="关闭写入 --voice-debug-dir（仍会在终端打印关键里程碑）",
    )
    parser.add_argument("--voice-loop-interval", type=float, default=1.0, help="两次语音识别之间的间隔秒数")
    parser.add_argument("--voice-duration", type=int, default=10, help="单次语音录音时长，单位秒")
    parser.add_argument(
        "--post-wake-duration",
        type=int,
        default=6,
        help="唤醒应答后采集用户说话的时长（秒），默认 6",
    )
    parser.add_argument(
        "--post-ack-delay",
        type=float,
        default=0.5,
        help="仅 --wake-record-timing after_ack 时：应答后再延迟开录（秒）",
    )
    parser.add_argument(
        "--post-ack-user-window",
        type=float,
        default=2.5,
        help="sync 模式：应答播完后至少再录这么多秒才允许因静音停录（默认 2.5）",
    )
    parser.add_argument(
        "--wake-record-timing",
        type=str,
        default="sync",
        choices=("sync", "after_ack"),
        help="sync=听到唤醒词立即开录，应答并行播放（默认）| after_ack=应答结束后再录",
    )
    parser.add_argument(
        "--capture-mode",
        type=str,
        default="vad",
        choices=("vad", "fixed"),
        help="录音模式：vad=说完自动结束（默认）| fixed=固定时长",
    )
    parser.add_argument(
        "--max-capture-duration",
        type=float,
        default=15.0,
        help="VAD 模式最长录音秒数（上限），默认 15",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.8,
        help="VAD 检测到多少秒静音后结束录音，默认 0.8",
    )
    parser.add_argument(
        "--no-cloud-streaming",
        action="store_true",
        help="关闭 LLM 流式 + 逐句 TTS（默认开启以加快首字出声）",
    )
    parser.add_argument(
        "--keep-voice-recordings",
        action="store_true",
        help="保留每次唤醒录音到 data/voice_recordings/（便于回放排查）",
    )
    parser.add_argument(
        "--playback-recording",
        action="store_true",
        help="每次录完后自动播放一遍刚才的录音（联调 ASR 前可先听）",
    )
    parser.add_argument(
        "--voice-record-dir",
        type=str,
        default="data/voice_recordings",
        help="录音保存目录（配合 --keep-voice-recordings 或 --playback-recording）",
    )
    parser.add_argument(
        "--wake-ack",
        type=str,
        default="我在，请说。",
        help="云端回退时的唤醒短句（默认优先本地预加载 WAV）",
    )
    parser.add_argument(
        "--wake-ack-mode",
        type=str,
        default="local",
        choices=("local", "cloud", "off"),
        help="唤醒应答：local=本地预加载 WAV（最快）| cloud=百度 TTS | off=不应答",
    )
    parser.add_argument(
        "--wake-ack-dir",
        type=str,
        default="assets/voice/wake_ack",
        help="本地唤醒应答 WAV 目录（含 manifest.json）",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="列出 ALSA 录音/播放设备并退出",
    )
    parser.add_argument(
        "--voice-alsa-device",
        type=str,
        default="auto",
        help="用户说话录音 ALSA 设备；auto=优先摄像头/USB 麦（默认），例 plughw:0,0",
    )
    parser.add_argument(
        "--wake-alsa-device",
        type=str,
        default="auto",
        help="唤醒词监听 ALSA 设备；auto=与 --voice-alsa-device 相同；"
        "可与录音分离，例 plughw:1,0=扬声器盒子麦听唤醒，plughw:0,0 专录用户话",
    )
    parser.add_argument(
        "--wake-echo-trim",
        action="store_true",
        help="唤醒录音后裁掉应答回声再送 ASR（默认不裁，整段原样识别）",
    )
    parser.add_argument(
        "--no-persistent-capture",
        action="store_true",
        help="关闭摄像头麦常驻 arecord（唤醒后重新 open 设备，延迟更高）",
    )
    parser.add_argument(
        "--tts-alsa-device",
        type=str,
        default="plughw:1,0",
        help="扬声器播放 ALSA 设备（仅 TTS/应答，不用于录音）；auto=自动选非麦克风声卡",
    )
    parser.add_argument("--voice-sample-rate", type=int, default=16000, help="语音录音采样率")
    parser.add_argument("--tts-output-path", type=str, default="data/tts_output.wav", help="TTS 输出音频路径")
    parser.add_argument(
        "--tts-backend",
        type=str,
        default="baidu",
        choices=("sherpa-onnx", "baidu"),
        help="Agent 回复 TTS：baidu=百度云端（默认，音质更好）| sherpa-onnx=板端离线",
    )
    parser.add_argument(
        "--llm-mode",
        type=str,
        default="fast",
        choices=("fast", "full"),
        help="LLM 决策：fast=统一规划 1 次、必要时追加安全审查（默认）| full=四角色串行",
    )
    parser.add_argument(
        "--sherpa-tts-dir",
        type=str,
        default="models/vits-icefall-zh-aishell3",
        help="Sherpa 离线 TTS 模型目录",
    )
    parser.add_argument(
        "--tts-speaker-id",
        type=int,
        default=0,
        help="Sherpa VITS 说话人 ID（aishell3 模型可用 0~218）",
    )
    parser.add_argument(
        "--wake-backend",
        type=str,
        default="sherpa-onnx",
        help="唤醒引擎：sherpa-onnx（默认，固定中文词）| energy | porcupine | mock",
    )
    parser.add_argument(
        "--sherpa-kws-dir",
        type=str,
        default="models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
        help="Sherpa-ONNX KWS 模型目录（含 tokens.txt）",
    )
    parser.add_argument(
        "--wake-keywords-file",
        type=str,
        default="models/sherpa-kws-keywords.txt",
        help="Sherpa keywords.txt（不存在时按 --wake-word 自动生成）",
    )
    parser.add_argument(
        "--wake-keywords-threshold",
        type=float,
        default=0.25,
        help="Sherpa 触发阈值，越大越难唤醒",
    )
    parser.add_argument(
        "--wake-keywords-score",
        type=float,
        default=2.0,
        help="Sherpa 关键词 boosting 分数",
    )
    parser.add_argument("--wake-model-path", type=str, default="models/porcupine_params_zh.pv", help="Porcupine 模型路径")
    parser.add_argument("--wake-keyword-path", type=str, default="models/hey_assistant_zh.ppn", help="Porcupine 唤醒词路径")
    parser.add_argument("--wake-word", type=str, default="小助", help="Sherpa/energy 唤醒短语")
    parser.add_argument("--wake-sensitivities", type=str, default="0.5", help="Porcupine 灵敏度，逗号分隔")
    parser.add_argument("--wake-energy-threshold", type=float, default=0.2, help="能量检测阈值，越小越灵敏")
    parser.add_argument("--no-wake", action="store_true", help="禁用唤醒词检测，仅使用 --voice-loop 持续识别模式")
    parser.add_argument(
        "--behavior",
        action="store_true",
        help="启用 YOLO26 手机+手腕行为识别（全栈默认已开；--llm 时需显式指定）",
    )
    parser.add_argument(
        "--no-behavior",
        action="store_true",
        help="全栈模式下不启行为识别",
    )
    parser.add_argument(
        "--behavior-backend",
        type=str,
        default="auto",
        choices=("auto", "om", "pt"),
        help="行为推理：auto=有 .om 则 Ascend NPU，否则 PyTorch CPU",
    )
    parser.add_argument(
        "--behavior-interval",
        type=float,
        default=None,
        help="行为 YOLO 推理最小间隔（秒）；默认与感知 tick 对齐（4 Hz → 0.25s）",
    )
    parser.add_argument(
        "--behavior-debounce",
        type=float,
        default=2.0,
        help="行为 Event 去抖间隔（秒）",
    )
    parser.add_argument(
        "--behavior-om-device-id",
        type=int,
        default=0,
        help="行为 YOLO OM 的 Ascend 设备 ID（可用 BEHAVIOR_OM_DEVICE_ID 覆盖）",
    )
    parser.add_argument(
        "--perception-hz",
        type=float,
        default=None,
        help="统一感知 tick（Hz），覆盖视觉采集/行为 OM/情绪帧率；默认 4",
    )
    parser.add_argument(
        "--perception-debug",
        action="store_true",
        help="开启摄像头感知调试日志（行为/情绪/疲劳，写入 data/perception_debug/perception.log）",
    )
    parser.add_argument(
        "--no-perception-debug",
        action="store_true",
        help="关闭感知调试日志（全栈默认在启视觉/行为时开启）",
    )
    parser.add_argument(
        "--perception-debug-dir",
        type=str,
        default="data/perception_debug",
        help="感知调试日志目录",
    )
    parser.add_argument(
        "--environment",
        action="store_true",
        help="启用 ESP32 USB 环境传感器（温湿度/光照/噪声）",
    )
    parser.add_argument(
        "--no-environment",
        action="store_true",
        help="全栈模式下不读 ESP32 环境传感器",
    )
    parser.add_argument(
        "--esp32-sensor-port",
        default=None,
        help="ESP32 串口（默认 $EMBED_ESP32_SENSOR_PORT 或 /dev/ttyUSB0）",
    )
    parser.add_argument(
        "--esp32-sensor-baud",
        type=int,
        default=None,
        help="ESP32 波特率（默认 115200）",
    )
    parser.add_argument(
        "--env-low-light-lux",
        type=float,
        default=None,
        help="低光照阈值 lux（默认 120，可用 EMBED_ENV_LOW_LIGHT_LUX）",
    )
    parser.add_argument(
        "--env-low-temperature-c",
        type=float,
        default=None,
        help="低温阈值 ℃（默认 18）",
    )
    parser.add_argument(
        "--env-dry-humidity-pct",
        type=float,
        default=None,
        help="干燥阈值 %（默认 30）",
    )
    parser.add_argument(
        "--env-noisy-db",
        type=float,
        default=None,
        help="噪声超标阈值 dB（默认 65）",
    )
    args = parser.parse_args()
    if args.perception_hz is not None:
        os.environ["EMBED_PERCEPTION_HZ"] = str(args.perception_hz)
    args.behavior = False
    args.environment = False
    args.perception_debug = False
    if args.llm:
        if args.camera is None:
            args.camera = 0
        if args.emotion_backend is None:
            args.emotion_backend = "wujie-om"
    _apply_run_profile(args)

    if args.list_audio_devices:
        from src.adapters.voice.alsa_audio_devices import (
            list_capture_devices,
            list_playback_devices,
            resolve_voice_pipeline_devices,
        )

        output = ConsoleOutput()
        output.show_text("=== arecord -l（录音设备）===")
        for item in list_capture_devices():
            output.show_text(
                f"  {item['alsa_device']}  card={item['card']}  "
                f"{item['card_name']} / {item['device_name']}"
            )
        output.show_text("=== aplay -l（播放设备）===")
        for item in list_playback_devices():
            output.show_text(
                f"  {item['alsa_device']}  card={item['card']}  "
                f"{item['card_name']} / {item['device_name']}"
            )
        capture_dev, playback_dev = resolve_voice_pipeline_devices(
            capture_explicit=args.voice_alsa_device,
            playback_explicit=args.tts_alsa_device,
        )
        output.show_text("=== 推荐路由（录放分离）===")
        output.show_text(f"  麦克风：{capture_dev}")
        output.show_text(f"  扬声器：{playback_dev or '(未检测到)'}")
        return

    output = ConsoleOutput()
    cli = CLIInputAdapter()

    from src.adapters.perception_debug_log import configure_perception_debug

    configure_perception_debug(
        enabled=bool(args.perception_debug),
        log_dir=args.perception_debug_dir,
        session_note=(
            f"camera={args.camera} vision={args.vision} behavior={args.behavior} "
            f"emotion={args.emotion_backend}"
        ),
    )
    if args.perception_debug:
        output.show_text(
            f"[感知调试] 行为/情绪/疲劳日志：{args.perception_debug_dir}/perception.log"
        )

    if not args.llm and (args.screen or args.vision or args.voice):
        output.show_text(
            "全栈模式：桌宠"
            + ("（全屏）" if args.screen and args.screen_fullscreen else "")
            + " + 视觉(NPU 情绪 OM) + 行为(NPU YOLO OM) + 语音 + LLM Agent；仅 CLI 请加 --llm"
        )

    raf_path = args.raf_ckpt or os.environ.get("RAF_RESNET18_CKPT")
    wujie_path = args.wujie_ckpt or os.environ.get("WUJIE_VGG19_CKPT")
    from src.adapters.vision_affect.config import DEFAULT_WUJIE_OM_MODEL

    wujie_om_path = args.wujie_om or os.environ.get("WUJIE_OM_MODEL") or DEFAULT_WUJIE_OM_MODEL
    wujie_device_id = int(os.environ.get("WUJIE_OM_DEVICE_ID", str(args.wujie_device_id)))
    state_stats_db = args.state_stats_db or os.environ.get("EMBED_STATE_STATS_DB")
    emotion_be = (os.environ.get("EMBED_EMOTION_BACKEND") or args.emotion_backend or "wujie-om").strip()

    voice_adapter = None
    detector = None

    screen_adapter = None
    pet_preview_server = None
    if getattr(args, "screen_preview", False):
        from src.adapters.screen.headless_pet_display import HeadlessPetDisplay
        from src.adapters.screen.pet_preview_server import PetPreviewServer
        from src.adapters.screen.screen_adapter import ScreenDisplayAdapter

        pet_preview_server = PetPreviewServer(port=args.screen_preview_port)
        pet_preview_server.start()
        headless = HeadlessPetDisplay(pet_preview_server)
        headless.start()
        screen_adapter = ScreenDisplayAdapter(
            hardware=headless,
            console_output=output,
        )
        output = screen_adapter
        output.show_text(
            f"桌宠 HTTP 预览：{pet_preview_server.url} "
            "（Cursor：Ports 面板打开该端口；PNG 同步 data/runtime/pet_preview.png）"
        )
    elif args.screen:
        from src.adapters.screen import ScreenDisplayAdapter, create_screen_window

        try:
            screen_window = create_screen_window(
                fullscreen=args.screen_fullscreen,
                size_arg=args.screen_size,
            )
            screen_window.start()
            output.show_text(
                f"桌宠显示已启动：{screen_window.size[0]}x{screen_window.size[1]}"
                + ("（全屏）" if screen_window.fullscreen else "")
            )
            screen_adapter = ScreenDisplayAdapter(
                hardware=screen_window,
                console_output=output,
            )
            output = screen_adapter
        except Exception as exc:
            output.show_text(
                f"[Warn] 桌宠窗口启动失败（{exc}），继续无屏模式。"
                " VNC 终端请确认 export DISPLAY=:1，或加 --no-screen。"
            )

    # NPU OM 须在语音 start 之前、且须在 MediaPipe 大量初始化之前完成 load；
    # 桌宠 pygame 须先于 OM 预加载，否则 SDL 视频子系统 init 会失败。
    behavior_detector = _preload_npu_perception_models(
        args,
        output,
        wujie_om_path=wujie_om_path,
        wujie_device_id=wujie_device_id,
        emotion_be=emotion_be,
    )

    if args.voice:
        from src.adapters.voice import (
            BaiduShortASRBackend,
            BoardVoiceAdapter,
            build_tts_backend,
            build_wake_word_detector,
        )
        from src.adapters.voice.alsa_audio_devices import (
            list_capture_devices,
            list_playback_devices,
            resolve_voice_pipeline_devices,
        )

        capture_alsa, playback_alsa = resolve_voice_pipeline_devices(
            capture_explicit=args.voice_alsa_device,
            playback_explicit=args.tts_alsa_device,
            split_input_output=True,
        )
        from src.adapters.voice.alsa_audio_devices import resolve_capture_device

        wake_raw = (args.wake_alsa_device or "auto").strip().lower()
        if wake_raw in {"", "auto", "same", "default"}:
            wake_alsa = capture_alsa
        else:
            wake_alsa = resolve_capture_device(explicit=args.wake_alsa_device)

        recognizer = BaiduShortASRBackend(sample_rate=args.voice_sample_rate)
        tts_backend = build_tts_backend(
            backend=args.tts_backend,
            output_path=args.tts_output_path,
            alsa_playback_device=playback_alsa,
            prefer_capture_device=capture_alsa,
            sherpa_tts_dir=args.sherpa_tts_dir,
            speaker_id=args.tts_speaker_id,
        )
        if hasattr(tts_backend, "preload"):
            try:
                tts_backend.preload()
            except FileNotFoundError as exc:
                output.show_text(f"[Warn] 本地 TTS 未就绪：{exc}")

        if not args.no_wake:
            try:
                sensitivities = [float(s) for s in args.wake_sensitivities.split(",")]
            except ValueError:
                sensitivities = [0.5]

            if args.wake_backend in {"sherpa-onnx", "sherpa", "kws", "sherpa_onnx"}:
                from src.adapters.voice.sherpa_kws import ensure_keywords_file

                keywords_file = ensure_keywords_file(
                    model_dir=args.sherpa_kws_dir,
                    phrases=[args.wake_word],
                    keywords_file=args.wake_keywords_file,
                    keywords_threshold=args.wake_keywords_threshold,
                    keywords_score=args.wake_keywords_score,
                )
                detector = build_wake_word_detector(
                    backend="sherpa-onnx",
                    model_dir=args.sherpa_kws_dir,
                    keywords_file=keywords_file,
                    wake_word=args.wake_word,
                    keywords_threshold=args.wake_keywords_threshold,
                    keywords_score=args.wake_keywords_score,
                    alsa_device=wake_alsa,
                    sink=None,
                )
            elif args.wake_backend == "porcupine":
                detector = build_wake_word_detector(
                    backend="porcupine",
                    model_path=args.wake_model_path,
                    keyword_path=args.wake_keyword_path,
                    sensitivities=sensitivities,
                    alsa_device=wake_alsa,
                    sink=None,
                )
            elif args.wake_backend == "mock":
                detector = build_wake_word_detector(backend="mock", sink=None)
            elif args.wake_backend in {"energy", "ste", "simple"}:
                detector = build_wake_word_detector(
                    backend="energy",
                    wake_word=args.wake_word,
                    alsa_device=wake_alsa,
                    sink=None,
                    energy_threshold=args.wake_energy_threshold,
                )
            else:
                output.show_text(
                    f"[Error] 未知 --wake-backend={args.wake_backend!r}，"
                    "可选：sherpa-onnx / energy / porcupine / mock"
                )

        voice_adapter = BoardVoiceAdapter(
            sink=None,
            detector=detector,
            recognizer=recognizer,
            tts_backend=tts_backend,
            alsa_device=capture_alsa,
            wake_alsa_device=wake_alsa,
            persistent_capture=not args.no_persistent_capture,
            wake_echo_trim=args.wake_echo_trim,
            sample_rate=args.voice_sample_rate,
            capture_duration_sec=args.voice_duration,
            post_wake_capture_sec=args.post_wake_duration,
            post_ack_listen_delay_sec=args.post_ack_delay,
            post_ack_user_window_sec=args.post_ack_user_window,
            capture_mode=args.capture_mode,
            max_capture_duration_sec=args.max_capture_duration,
            silence_duration_sec=args.silence_duration,
            cloud_streaming=not args.no_cloud_streaming,
            keep_voice_recordings=args.keep_voice_recordings,
            playback_recording=args.playback_recording,
            voice_record_dir=args.voice_record_dir,
            wake_record_timing=args.wake_record_timing,
            wake_ack_text=args.wake_ack,
            wake_ack_mode=args.wake_ack_mode,
            wake_ack_dir=args.wake_ack_dir,
            voice_debug_dir=args.voice_debug_dir,
            voice_debug_log=not args.no_voice_debug_log,
        )
        voice_adapter.debug = bool(args.voice_debug)
        if not args.no_voice_debug_log:
            output.show_text(f"[语音调试] 日志：{args.voice_debug_dir}/latest/voice.log（每次唤醒覆盖）")
        output = MultiOutput(output, voice_adapter)

    from src.agent.config.policy_config import DecisionPolicyConfig

    core = build_default_core(
        output=output,
        decision_policy=DecisionPolicyConfig(llm_mode=args.llm_mode),
    )
    core.start_autonomous_scheduler()

    if voice_adapter is not None:
        voice_adapter._sink = core
        output.show_text(
            f"[音频] 唤醒监听：{wake_alsa}  |  用户录音：{capture_alsa}  |  "
            f"扬声器：{playback_alsa or 'auto'}"
        )

        import platform
        output.show_text(f"[诊断] 操作系统：{platform.system()}，Python 版本：{platform.python_version()}")
        asr_configured = recognizer.is_configured()
        tts_configured = tts_backend.is_configured()
        output.show_text(
            f"[诊断] ASR 配置：{'已配置' if asr_configured else '未配置（请设置 BAIDU_ASR_API_KEY / BAIDU_ASR_SECRET_KEY）'}"
        )
        output.show_text(
            f"[诊断] TTS 配置：{'已配置' if tts_configured else '未配置'}"
            f"（backend={args.tts_backend}）"
        )
    # 设置事件处理回调，自动更新屏幕
    if screen_adapter is not None:
        def on_event_handled(state):
            if state.focus.active:
                screen_adapter.update_focus_timer(
                    state.focus.remaining_sec,
                    state.focus.target_duration_sec
                )
        core.set_event_handled_callback(on_event_handled)

    vision_adapter = None
    behavior_adapter = None
    shared_frame_bus = None
    if args.vision and args.behavior:
        from src.adapters.behavior.camera_utils import LatestFrameBus

        shared_frame_bus = LatestFrameBus()

    if args.vision:
        from src.adapters.vision_affect import (
            VisionAffectConfig,
            VisionAffectInputAdapter,
            vision_dependencies_met,
            vision_emotion_backend_ready,
        )

        if vision_dependencies_met():
            from src.adapters.perception_config import (
                emotion_every_n_frames,
                perception_hz,
                vision_target_fps,
            )

            cfg = VisionAffectConfig(
                camera_index=args.camera,
                raf_checkpoint=raf_path,
                wujie_checkpoint=wujie_path,
                wujie_om_model=wujie_om_path,
                wujie_om_device_id=wujie_device_id,
                emotion_backend=emotion_be,
                deepface_model=args.deepface_model,
                target_fps=vision_target_fps(),
                emotion_every_n_frames=emotion_every_n_frames(),
                state_stats_db_path=state_stats_db or "data/runtime/state_stats.db",
            )
            vision_adapter = VisionAffectInputAdapter(core, cfg, frame_bus=shared_frame_bus)
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
                f"已启动视觉适配器（camera index={args.camera}；感知 tick≈{perception_hz():.0f}Hz；"
                f"疲劳 EAR+MAR+融合）。"
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

    if args.behavior and behavior_detector is not None:
        from src.adapters.behavior.phone_camera_adapter import PhoneHandCameraAdapter
        from src.adapters.behavior_adapter import BehaviorAdapter
        from src.adapters.perception_config import behavior_inference_interval_sec, perception_hz

        behavior_interval = (
            args.behavior_interval
            if args.behavior_interval is not None
            else behavior_inference_interval_sec()
        )
        behavior_adapter = PhoneHandCameraAdapter(
            core,
            camera_index=args.camera,
            detector=behavior_detector,
            behavior_adapter=BehaviorAdapter(
                core,
                debounce_seconds=args.behavior_debounce,
            ),
            inference_interval=behavior_interval,
            source="yolo26_phone_hand_om_v1",
            frame_bus=shared_frame_bus,
        )
        try:
            behavior_adapter.start_background()
            backend_label = behavior_detector.active_backend
            npu_note = "Ascend NPU" if backend_label == "om" else "CPU PyTorch"
            share_note = "共享视觉摄像头帧" if shared_frame_bus is not None else f"独立摄像头 index={args.camera}"
            output.show_text(
                f"已启动行为+姿势（{share_note}；后端={backend_label}/{npu_note}；"
                f"tick≈{perception_hz():.0f}Hz interval={behavior_interval:.2f}s；"
                f"pose 来自 yolo26n-pose.om 同帧推断）。"
            )
        except Exception as exc:
            behavior_adapter = None
            output.show_text(f"行为识别启动失败：{exc}")

    environment_adapter = None
    if args.environment:
        from src.adapters.environment import (
            Esp32EnvironmentAdapter,
            esp32_port_available,
            resolve_environment_thresholds,
            resolve_esp32_port,
        )

        env_port = resolve_esp32_port(args.esp32_sensor_port)
        env_thresholds = resolve_environment_thresholds(
            low_light_lux=args.env_low_light_lux,
            low_temperature_c=args.env_low_temperature_c,
            dry_humidity_pct=args.env_dry_humidity_pct,
            noisy_db=args.env_noisy_db,
        )
        if not esp32_port_available(env_port):
            output.show_text(
                f"[Warn] 未找到 ESP32 环境传感器串口 {env_port}，跳过 environment Event。"
                " 请确认 USB 已连接或设置 EMBED_ESP32_SENSOR_PORT。"
            )
        else:
            environment_adapter = Esp32EnvironmentAdapter(
                core,
                port=env_port,
                baudrate=args.esp32_sensor_baud,
                thresholds=env_thresholds,
            )
            try:
                environment_adapter.start_background()
                output.show_text(
                    f"已启动 ESP32 环境传感器（port={env_port}；"
                    f"temperature/humidity/lux/noise_db → 标准 Event）。"
                )
            except Exception as exc:
                environment_adapter = None
                output.show_text(f"ESP32 环境传感器启动失败：{exc}")

    if voice_adapter is not None:
        if args.voice_loop:
            voice_adapter.start_background_loop(interval_sec=args.voice_loop_interval)
            output.show_text(
                f"已启动语音后台识别：duration={args.voice_duration}s interval={args.voice_loop_interval}s"
            )
        elif detector is not None:
            voice_adapter.start()
            output.show_text(
                f"已启动唤醒词监听（{args.wake_backend}，词：{args.wake_word}）；"
                f"{'摄像头麦常驻+唤醒即录' if not args.no_persistent_capture else '唤醒即录'}，"
                f"VAD 说完停（静音 {args.silence_duration}s），"
                f"本地应答（{args.wake_ack_mode}），识别后再走 LLM。"
            )
        else:
            output.show_text("语音适配器已就绪（静默模式），使用 /voice_once 命令手动触发。")
        if args.voice_once:
            output.show_text("立即执行一次语音识别，请在录音窗口内说话。")
            event = voice_adapter.run_recognize_once()
            if event is None:
                output.show_text("[Voice] 未识别到有效文本。")
            else:
                core.handle_event_with_results(event)
                output.show_text(f"[Voice] 已上报事件：{event.payload}")

    output.show_text("Agent MVP 已启动，输入 /help 查看可用命令。")
    if voice_adapter is not None:
        hints = ["/voice_once"]
        if args.playback_recording or args.keep_voice_recordings:
            hints.append("/voice_replay")
        output.show_text("语音命令：" + " ".join(hints))
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
            if voice_adapter is not None and command == "/voice_once":
                output.show_text("开始一次语音识别，请在录音窗口内说话。")
                event = voice_adapter.run_recognize_once()
                if event is None:
                    output.show_text("[Voice] 未识别到有效文本。")
                else:
                    core.handle_event_with_results(event)
                    output.show_text(f"[Voice] 已上报事件：{event.payload}")
                continue
            if voice_adapter is not None and command == "/voice_replay":
                if voice_adapter.replay_last_recording():
                    output.show_text("[Voice] 正在回放最近一次录音。")
                else:
                    output.show_text(
                        "[Voice] 没有可回放的录音。请先用 --playback-recording 或 --keep-voice-recordings 录一次。"
                    )
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
        if vision_adapter is not None:
            vision_adapter.stop()
        if behavior_adapter is not None:
            behavior_adapter.stop()
        if environment_adapter is not None:
            environment_adapter.stop()
        if screen_adapter is not None:
            screen_adapter._hardware.stop()
        core.shutdown()
        output.show_text("Agent MVP 已退出。")


if __name__ == "__main__":
    main()
