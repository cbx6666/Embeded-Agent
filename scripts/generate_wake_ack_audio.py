#!/usr/bin/env python3
"""一次性生成本地唤醒应答 WAV（可用百度 TTS），供命中唤醒词后立即播放。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.adapters.voice.local_wake_ack import (  # noqa: E402
    DEFAULT_WAKE_ACK_DIR,
    DEFAULT_WAKE_ACK_PHRASES,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local wake-ack WAV clips")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_WAKE_ACK_DIR),
        help="输出目录（默认 assets/voice/wake_ack）",
    )
    parser.add_argument(
        "--backend",
        choices=("baidu", "espeak"),
        default="baidu",
        help="生成方式：baidu（需 API）| espeak（离线，音质一般）",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    for clip_id, text in DEFAULT_WAKE_ACK_PHRASES:
        wav_path = out_dir / f"{clip_id}.wav"
        if args.backend == "baidu":
            _synthesize_baidu(text, wav_path)
        else:
            _synthesize_espeak(text, wav_path)
        manifest.append({"id": clip_id, "text": text, "file": wav_path.name})
        print(f"已生成：{wav_path} ← {text!r}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest：{manifest_path}")
    print("启动后会在 voice adapter 启动时 preload 这些文件。")


def _synthesize_baidu(text: str, wav_path: Path) -> None:
    from src.adapters.voice.baidu_tts_backend import BaiduTTSBackend

    backend = BaiduTTSBackend(output_path=wav_path)
    if not backend.is_configured():
        raise RuntimeError("百度 TTS 未配置，无法生成 wake_ack WAV")
    backend.speak(text, voice=None, volume=None, speed=None)


def _synthesize_espeak(text: str, wav_path: Path) -> None:
    import subprocess

    subprocess.run(
        ["espeak-ng", "-v", "zh", "-s", "160", "-w", str(wav_path), text],
        check=True,
        capture_output=True,
    )


if __name__ == "__main__":
    main()
