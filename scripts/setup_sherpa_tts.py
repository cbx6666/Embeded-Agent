#!/usr/bin/env python3
"""下载 Sherpa-ONNX 中文离线 TTS 模型。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.adapters.voice.sherpa_tts_backend import (  # noqa: E402
    DEFAULT_MODEL_URL,
    DEFAULT_SHERPA_TTS_DIR,
    resolve_sherpa_tts_dir,
)


def download_model(model_dir: Path, url: str, force: bool) -> Path:
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if (model_dir / "model.onnx").is_file() and not force:
        print(f"模型已存在：{model_dir}")
        return resolve_sherpa_tts_dir(model_dir)

    print(f"下载 Sherpa TTS 模型：{url}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "tts.tar.bz2"
        subprocess.run(["wget", "-O", str(archive), url], check=True)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=model_dir.parent)
    resolved = resolve_sherpa_tts_dir(model_dir)
    print(f"模型已解压到：{resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Sherpa-ONNX offline TTS")
    parser.add_argument("--model-dir", default=str(DEFAULT_SHERPA_TTS_DIR))
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    model_dir = download_model(Path(args.model_dir), args.model_url, args.force_download)
    print("测试合成一句：")
    from src.adapters.voice.sherpa_tts_backend import SherpaOnnxTTSBackend

    backend = SherpaOnnxTTSBackend(model_dir=model_dir, output_path="data/tts_test.wav")
    backend.speak("本地语音合成测试。", voice=None, volume=None, speed=None)
    print("完成。启动主程序时加 --tts-backend sherpa-onnx 使用离线 TTS（默认已是 baidu）")


if __name__ == "__main__":
    main()
