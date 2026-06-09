#!/usr/bin/env python3
"""仅语音 + LLM Agent 联调（不启桌宠 / 视觉 / 姿势）。

用于在无其他负载时测试唤醒 → VAD 录音 → ASR → LLM → TTS。

示例：
  export DISPLAY=:1
  /opt/ai-envs/shared/bin/python scripts/voice_llm_only.py \\
    --voice-alsa-device plughw:0,0 \\
    --tts-alsa-device plughw:1,0 \\
    --voice-debug

  # 手动录一次（不依赖唤醒词）：
  /opt/ai-envs/shared/bin/python scripts/voice_llm_only.py --no-wake --voice-once
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_voice_stack(args: argparse.Namespace):
    from src.adapters.cli_input import CLIInputAdapter
    from src.adapters.console_output import ConsoleOutput
    from src.adapters.voice import (
        BaiduShortASRBackend,
        BoardVoiceAdapter,
        build_tts_backend,
        build_wake_word_detector,
    )
    from src.adapters.voice.input.alsa_audio_devices import (
        resolve_capture_device,
        resolve_voice_pipeline_devices,
    )
    from src.adapters.voice.wake.sherpa_kws import ensure_keywords_file
    from src.agent.config.policy_config import DecisionPolicyConfig
    from src.agent.core import build_default_core

    output = ConsoleOutput()
    cli = CLIInputAdapter()

    capture_alsa, _wake_alsa, playback_alsa = resolve_voice_pipeline_devices(
        capture_explicit=args.voice_alsa_device,
        playback_explicit=args.tts_alsa_device,
        split_input_output=True,
    )

    recognizer = BaiduShortASRBackend(sample_rate=args.voice_sample_rate)
    tts_backend = build_tts_backend(
        backend=args.tts_backend,
        output_path=args.tts_output_path,
        alsa_playback_device=playback_alsa,
        prefer_capture_device=capture_alsa,
        sherpa_tts_dir=args.sherpa_tts_dir,
        speaker_id=args.tts_speaker_id,
    )

    detector = None
    if not args.no_wake:
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
            alsa_device=capture_alsa,
            sink=None,
        )

    voice_adapter = BoardVoiceAdapter(
        sink=None,
        detector=detector,
        recognizer=recognizer,
        tts_backend=tts_backend,
        alsa_device=capture_alsa,
        sample_rate=args.voice_sample_rate,
        silence_duration_sec=args.silence_duration,
        max_capture_duration_sec=args.max_capture_duration,
        voice_debug_dir=args.voice_debug_dir,
        voice_debug_log=not args.no_voice_debug_log,
    )
    voice_adapter.debug = bool(args.voice_debug)

    core = build_default_core(
        output=output,
        decision_policy=DecisionPolicyConfig(llm_mode=args.llm_mode),
    )
    voice_adapter._sink = core

    output.show_text("=== 语音 + LLM 轻量模式（无桌宠/视觉）===")
    output.show_text(
        f"[音频] 录音：{capture_alsa}  |  扬声器：{playback_alsa or 'auto'}"
    )
    if not args.no_voice_debug_log:
        output.show_text(f"[调试] {args.voice_debug_dir}/latest/voice.log")

    if args.no_wake:
        output.show_text("唤醒词已关闭；可用 /voice_once 或 --voice-once。")
    else:
        voice_adapter.start()
        output.show_text(
            f"已启动唤醒词「{args.wake_word}」；AudioInputManager + VAD（静音 {args.silence_duration}s）。"
        )

    if args.voice_once:
        output.show_text("立即执行一次语音识别…")
        event = voice_adapter.run_recognize_once()
        if event is None:
            output.show_text("[Voice] 未识别到有效文本。")
        else:
            core.handle_event(event)

    return output, cli, core, voice_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description="仅语音 + LLM，不启桌宠/视觉/姿势")
    parser.add_argument("--voice-alsa-device", default="plughw:0,0")
    parser.add_argument("--tts-alsa-device", default="plughw:1,0")
    parser.add_argument("--wake-word", default="小助")
    parser.add_argument("--no-wake", action="store_true")
    parser.add_argument("--voice-sample-rate", type=int, default=16000)
    parser.add_argument("--max-capture-duration", type=float, default=15.0)
    parser.add_argument("--silence-duration", type=float, default=0.8)
    parser.add_argument("--voice-debug", action="store_true")
    parser.add_argument("--voice-debug-dir", default="data/voice_debug")
    parser.add_argument("--no-voice-debug-log", action="store_true")
    parser.add_argument("--tts-backend", default="baidu", choices=("baidu", "sherpa-onnx"))
    parser.add_argument("--tts-output-path", default="data/tts_output.wav")
    parser.add_argument("--sherpa-tts-dir", default="models/vits-icefall-zh-aishell3")
    parser.add_argument("--tts-speaker-id", type=int, default=0)
    parser.add_argument("--sherpa-kws-dir", default="models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01")
    parser.add_argument("--wake-keywords-file", default="models/sherpa-kws-keywords.txt")
    parser.add_argument("--wake-keywords-threshold", type=float, default=0.25)
    parser.add_argument("--wake-keywords-score", type=float, default=2.0)
    parser.add_argument("--llm-mode", default="fast", choices=("fast", "full"))
    parser.add_argument("--voice-once", action="store_true", help="启动后立即手动录一次")
    args = parser.parse_args()

    output, cli, core, voice_adapter = _build_voice_stack(args)
    output.show_text("Agent 已就绪。命令：/help /state /voice_once /voice_replay /exit")

    try:
        while True:
            line = cli.readline()
            if line is None:
                break
            command = line.strip()
            if not command:
                continue
            if command in {"/exit", "/quit"}:
                break
            if command == "/help":
                output.show_text("/voice_once  /voice_replay  /state  /exit")
                continue
            if command == "/state":
                output.show_text(core.render_state())
                continue
            if command == "/voice_once":
                output.show_text("开始一次语音识别…")
                event = voice_adapter.run_recognize_once()
                if event is None:
                    output.show_text("[Voice] 未识别到有效文本。")
                else:
                    core.handle_event(event)
                continue
            if command == "/voice_replay":
                if voice_adapter.replay_last_recording():
                    output.show_text("[Voice] 正在回放最近一次录音。")
                else:
                    output.show_text("[Voice] 没有可回放的录音（需先成功唤醒或 /voice_once）。")
                continue
            output.show_text(f"[Info] 未知命令：{command!r}，输入 /help")
    finally:
        voice_adapter.stop()
        output.show_text("已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
